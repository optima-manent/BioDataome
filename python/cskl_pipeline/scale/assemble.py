"""Assemble stage + update flow (plan sections 4.3, 5).

Fast enough to rerun end-to-end (no checkpointing):
  * pair-level shared-sample evidence from cached ``sample_hashes.json``,
  * observed cskl matrix over all kept datasets (batched; fast),
  * p-values from cached null profiles (interpolate - lever D), BH q-values,
  * write ``cskl_matrix`` + ``network_edges`` tables + ``manifest.json`` (+ compat
    TSV/JSON so the existing explainers / GEO / HTML tooling still works).

``update`` = ingest new inputs only, then assemble. Existing signatures and
profiles are never recomputed; BH q-values are global and re-sort over all pairs
(correct, and costs milliseconds).
"""

from __future__ import annotations

import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cskl
import numpy as np
import pandas as pd

from . import profile as profile_mod
from . import store as store_mod
from . import tabio
from .fastcore import observed_cskl_matrix
from .pool import PoolHandle

# ---------------------------------------------------------------------------
# Shared-sample evidence
# ---------------------------------------------------------------------------

def pair_overlap_evidence(
    store: store_mod.Store,
    ids: Sequence[str],
    *,
    major_overlap_coefficient: float = 0.5,
) -> Dict[Tuple[str, str], dict]:
    """Return evidence only for pairs sharing at least one molecular profile.

    An inverted index makes this proportional to genuinely repeated profiles,
    rather than repeatedly intersecting all pairwise sample sets. Duplicate
    hashes inside one dataset are counted with multiset semantics.
    """

    if not 0 < float(major_overlap_coefficient) <= 1:
        raise ValueError("major_overlap_coefficient must be in (0, 1]")
    counts: Dict[str, Counter[str]] = {}
    sample_counts: Dict[str, int] = {}
    for gse in ids:
        p = store.sample_hashes_path(gse)
        if p.exists():
            payload = store_mod.read_json(p)
            values = [str(value).strip().lower() for value in payload.get("hashes", [])]
            counts[gse] = Counter(value for value in values if value)
            sample_counts[gse] = len(values)
        else:
            counts[gse] = Counter()
            sample_counts[gse] = 0

    inverted: Dict[str, List[Tuple[str, int]]] = {}
    for gse, dataset_counts in counts.items():
        for expression_hash, count in dataset_counts.items():
            inverted.setdefault(expression_hash, []).append((gse, count))

    shared_counts: Dict[Tuple[str, str], int] = {}
    shared_hashes: Dict[Tuple[str, str], List[str]] = {}
    for expression_hash, appearances in inverted.items():
        if len(appearances) < 2:
            continue
        appearances.sort()
        for index, (left, left_count) in enumerate(appearances):
            for right, right_count in appearances[index + 1:]:
                pair = (left, right)
                shared_counts[pair] = shared_counts.get(pair, 0) + min(left_count, right_count)
                shared_hashes.setdefault(pair, []).append(expression_hash)

    evidence: Dict[Tuple[str, str], dict] = {}
    for pair, shared_count in shared_counts.items():
        left, right = pair
        left_count = sample_counts[left]
        right_count = sample_counts[right]
        union_count = left_count + right_count - shared_count
        smaller_count = min(left_count, right_count)
        coefficient = shared_count / smaller_count if smaller_count else 0.0
        if shared_count == left_count == right_count:
            classification = "exact"
        elif coefficient >= major_overlap_coefficient:
            classification = "major"
        else:
            classification = "minor"
        evidence[pair] = {
            "sample_count_a": left_count,
            "sample_count_b": right_count,
            "shared_sample_count": shared_count,
            "shared_expression_hashes": sorted(shared_hashes[pair]),
            "fraction_a": shared_count / left_count if left_count else 0.0,
            "fraction_b": shared_count / right_count if right_count else 0.0,
            "jaccard": shared_count / union_count if union_count else 0.0,
            "overlap_coefficient": coefficient,
            "overlap_classification": classification,
            # Every literal shared sample invalidates an independent-replication
            # claim even when the UI only emphasizes substantial overlap.
            "discovery_excluded": True,
            "edge_style": "dotted" if classification in {"major", "exact"} else "solid",
        }
    return evidence


# ---------------------------------------------------------------------------
# Assemble
# ---------------------------------------------------------------------------

def run_assemble(
    store: store_mod.Store,
    platform: str,
    pool_version: str,
    run_id: str,
    *,
    fdr_alpha: float = 0.05,
    generate_html: bool = False,
    run_explainers: bool = False,
    k: int = 10,
    probe2gene: Optional[str] = None,
    fetch_geo: bool = False,
    output_html: Optional[str] = None,
    only: Optional[Sequence[str]] = None,
    major_overlap_coefficient: float = 0.5,
) -> dict:
    t_start = datetime.now(timezone.utc)
    pool = PoolHandle(store, platform, pool_version)
    probes = store.read_probes(platform)

    # 1. Eligible datasets: valid signature + valid profile, not quarantined.
    candidates = only if only is not None else store.list_datasets()
    ids = [g for g in candidates
           if store.has_valid_signature(g, platform)
           and store.has_valid_profile(g, platform, pool_version)
           and not store.is_quarantined(g)]
    ids = sorted(ids)
    if len(ids) < 2:
        raise SystemExit(f"Need >=2 assembled datasets; found {len(ids)}. "
                         "Run `ingest` first.")

    # 2. Retain every eligible dataset. Overlap affects the corresponding
    # edge's interpretation, never either endpoint's existence.
    kept = ids
    overlap = pair_overlap_evidence(
        store, kept, major_overlap_coefficient=major_overlap_coefficient
    )

    # 3. Observed cskl matrix (batched).
    sigs = [store_mod.load_signature(store.signature_path(g)) for g in kept]
    sample_counts = {gse: int(signature.m_samples) for gse, signature in zip(kept, sigs, strict=True)}
    C = observed_cskl_matrix(sigs)

    # 4. p-values from cached profiles + BH.
    ptable = profile_mod.load_profile_table(store, kept, pool_version)
    P = ptable.all_pairs_pvalues(kept, C)

    iu, ju = np.triu_indices(len(kept), k=1)
    pairs_cskl = C[iu, ju]
    pairs_p = P[iu, ju]
    q_global = np.array(cskl.bh_qvalues(pairs_p.tolist()))
    independent_mask = np.array(
        [(kept[i], kept[j]) not in overlap for i, j in zip(iu, ju, strict=True)],
        dtype=bool,
    )
    q_independent = np.full(len(pairs_p), np.nan, dtype=np.float64)
    if independent_mask.any():
        q_independent[independent_mask] = np.asarray(
            cskl.bh_qvalues(pairs_p[independent_mask].tolist()), dtype=np.float64
        )

    overlap_rows = []
    for i, j in zip(iu, ju, strict=True):
        left, right = kept[i], kept[j]
        row = overlap.get((left, right))
        if row is None:
            left_count = sample_counts[left]
            right_count = sample_counts[right]
            row = {
                "sample_count_a": left_count,
                "sample_count_b": right_count,
                "shared_sample_count": 0,
                "shared_expression_hashes": [],
                "fraction_a": 0.0,
                "fraction_b": 0.0,
                "jaccard": 0.0,
                "overlap_coefficient": 0.0,
                "overlap_classification": "none",
                "discovery_excluded": False,
                "edge_style": "solid",
            }
        overlap_rows.append(row)

    edges_df = pd.DataFrame({
        "Dataset_A": [kept[i] for i in iu],
        "Dataset_B": [kept[j] for j in ju],
        "cSKL": pairs_cskl,
        "p_value": pairs_p,
        # q_value remains a compatibility alias for the complete family.
        "q_value": q_global,
        "q_global": q_global,
        "q_independent": q_independent,
        "sample_count_a": [row["sample_count_a"] for row in overlap_rows],
        "sample_count_b": [row["sample_count_b"] for row in overlap_rows],
        "shared_sample_count": [row["shared_sample_count"] for row in overlap_rows],
        "fraction_a": [row["fraction_a"] for row in overlap_rows],
        "fraction_b": [row["fraction_b"] for row in overlap_rows],
        "jaccard": [row["jaccard"] for row in overlap_rows],
        "overlap_coefficient": [row["overlap_coefficient"] for row in overlap_rows],
        "overlap_classification": [row["overlap_classification"] for row in overlap_rows],
        "discovery_excluded": [row["discovery_excluded"] for row in overlap_rows],
        "edge_style": [row["edge_style"] for row in overlap_rows],
    })
    matrix_df = edges_df[["Dataset_A", "Dataset_B", "cSKL"]].copy()

    # 5. Write run artifacts.
    run_dir = store.run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    tabio.write_table(matrix_df, run_dir, "cskl_matrix")
    tabio.write_table(edges_df, run_dir, "network_edges")

    # human-readable wide matrix + compat files for existing tooling
    _write_wide_matrix(run_dir, kept, C)
    _write_compat_edges(run_dir, edges_df)
    _write_pca_meta(store, run_dir, kept)

    n_edges = int((q_global <= fdr_alpha).sum())
    n_independent_edges = int(np.nansum(q_independent <= fdr_alpha))
    overlap_counts = Counter(row["overlap_classification"] for row in overlap_rows)
    manifest = {
        "run_id": run_id,
        "platform": platform,
        "pool_version": pool_version,
        "pool_hash": pool.pool_hash,
        "feature_hash": pool.feature_hash,
        "grid": pool.grid,
        "seed": pool.seed,
        "alpha": pool.alpha,
        "B": pool.B,
        "fdr_alpha": fdr_alpha,
        "n_datasets_eligible": len(ids),
        "n_datasets_kept": len(kept),
        "datasets_dropped_shared_profile": [],
        "overlap_policy": {
            "retains_all_datasets": True,
            "major_overlap_coefficient": major_overlap_coefficient,
            "independent_family_excludes_any_shared_sample": True,
            "dotted_edge_classes": ["major", "exact"],
        },
        "overlap_pair_counts": dict(sorted(overlap_counts.items())),
        "quarantined": [g for g in candidates if store.is_quarantined(g)],
        "n_pairs": int(len(pairs_cskl)),
        "n_significant_edges": n_edges,
        "n_significant_global_edges": n_edges,
        "n_significant_independent_edges": n_independent_edges,
        "code_version": _code_version(),
        "created": t_start.isoformat(),
        "finished": datetime.now(timezone.utc).isoformat(),
    }
    store_mod.atomic_write_json(run_dir / "manifest.json", manifest)
    print(f"[assemble] run={run_id}: {len(kept)} datasets, {len(pairs_cskl)} pairs, "
          f"{n_edges} global / {n_independent_edges} independent significant edges "
          f"(FDR<={fdr_alpha}). -> {run_dir}")

    # 6. Optional explainers / GEO / HTML (reuse existing modules).
    if run_explainers:
        _run_explainers(store, run_dir, kept, sigs, probes, k, probe2gene)
    if fetch_geo:
        _run_geo(run_dir)
    if generate_html:
        _run_html(run_dir, fdr_alpha, output_html)

    return manifest


def _write_wide_matrix(run_dir: Path, ids: List[str], C: np.ndarray) -> None:
    df = pd.DataFrame(C, index=ids, columns=ids)
    tmp = run_dir / "cskl_matrix.tsv.tmp"
    df.to_csv(tmp, sep="\t")
    import os
    os.replace(tmp, run_dir / "cskl_matrix.tsv")


def _write_compat_edges(run_dir: Path, edges_df: pd.DataFrame) -> None:
    import os
    tmp = run_dir / "cskl_network_edges.tsv.tmp"
    edges_df.to_csv(tmp, sep="\t", index=False)
    os.replace(tmp, run_dir / "cskl_network_edges.tsv")


def _write_pca_meta(store: store_mod.Store, run_dir: Path, ids: List[str]) -> dict:
    metas = {}
    for g in ids:
        m = store_mod.read_signature_meta(store.signature_path(g))
        metas[g] = {
            "n_features": m["n_features"],
            "n_features_used": m["n_features"],
            "n_samples": m["m_samples"],
            "c_components": m["c_components"],
            "alpha": m["alpha"],
        }
    store_mod.atomic_write_json(run_dir / "pca_meta.json", metas)
    return metas


def _code_version() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, cwd=str(Path(__file__).resolve().parent))
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    import hashlib
    here = Path(__file__).resolve().parent
    h = hashlib.sha1()
    for f in sorted(here.glob("*.py")):
        h.update(f.read_bytes())
    h.update((here.parent.parent / "cskl.py").read_bytes())
    return "sha1:" + h.hexdigest()[:12]


def _run_explainers(store, run_dir, kept, sigs, probes, k, probe2gene):
    try:
        from cskl_pipeline.explainers import write_edge_explainers
        sig_map = {g: s for g, s in zip(kept, sigs, strict=True)}
        write_edge_explainers(
            sig_map,
            run_dir / "cskl_network_edges.tsv",
            run_dir / "edge_explainers.json",
            feature_names=probes,
            k=k,
            probe2gene=probe2gene,
        )
        print("[assemble] wrote edge_explainers.json")
    except Exception as exc:
        print(f"[assemble] explainers skipped: {type(exc).__name__}: {exc}")


def _run_geo(run_dir):
    try:
        from cskl_pipeline.geo import write_geo_descriptions
        write_geo_descriptions(run_dir)
    except Exception as exc:
        print(f"[assemble] GEO fetch skipped: {type(exc).__name__}: {exc}")


def _run_html(run_dir, fdr_alpha, output_html):
    try:
        from cskl_pipeline.graph_html import generate_graph_html
        generate_graph_html(run_dir, output_html=output_html, q_threshold=fdr_alpha)
        print("[assemble] wrote interactive HTML network")
    except Exception as exc:
        print(f"[assemble] HTML generation skipped: {type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Update
# ---------------------------------------------------------------------------

def run_update(
    store: store_mod.Store,
    platform: str,
    pool_version: str,
    run_id: str,
    *,
    tars: Sequence[str] = (),
    csvs: Sequence[str] = (),
    workers: int = 1,
    input_orientation: str = "features-by-samples",
    rscript: str = "Rscript",
    fdr_alpha: float = 0.05,
    generate_html: bool = False,
    run_explainers: bool = False,
    **assemble_kwargs,
) -> dict:
    from .ingest import run_ingest

    ingest_summary = run_ingest(
        store, platform, pool_version, tars=tars, csvs=csvs, workers=workers,
        input_orientation=input_orientation, rscript=rscript,
    )
    manifest = run_assemble(
        store, platform, pool_version, run_id, fdr_alpha=fdr_alpha,
        generate_html=generate_html, run_explainers=run_explainers, **assemble_kwargs,
    )
    manifest["ingest"] = ingest_summary
    return manifest
