from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import cskl

from .io import (
    backup_file,
    compute_global_good_features,
    edge_id,
    edge_id_variants,
    expand_paths,
    load_expr,
    read_json,
    safe_name_from_file,
    write_json,
)


def load_probe_gene_map(path: Path | str | None) -> dict[str, str]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Probe-to-gene mapping not found: {p}")

    df = pd.read_csv(p).dropna(how="all")
    probe_candidates = [c for c in df.columns if "probe" in c.lower() or "id" in c.lower()]
    gene_candidates = [c for c in df.columns if "gene" in c.lower() or "symbol" in c.lower()]
    if not probe_candidates or not gene_candidates:
        raise ValueError(
            f"{p} must contain probe/id and gene/symbol columns. "
            f"Columns found: {list(df.columns)}"
        )

    probe_col = probe_candidates[0]
    gene_col = gene_candidates[0]
    probe2gene: dict[str, str] = {}
    for _, row in df.iterrows():
        probe = str(row[probe_col]).strip()
        gene = str(row[gene_col]).strip()
        if not probe or not gene or gene.lower() == "nan":
            continue
        probe2gene[probe] = gene.split("///")[0].strip()
    return probe2gene


def _feature_records(
    indices: np.ndarray,
    scores: np.ndarray,
    feature_names: list[str],
    probe_map: dict[str, str],
) -> list[dict[str, float | str]]:
    records: list[dict[str, float | str]] = []
    for idx, score in zip(indices, scores):
        feature = feature_names[int(idx)]
        records.append({"gene": probe_map.get(feature, feature), "score": float(score)})
    return records


def generate_edge_explainers(
    signatures: dict[str, cskl.PCASignature],
    edges: pd.DataFrame,
    *,
    feature_names: list[str],
    k: int = 10,
    probe2gene: Path | str | None = None,
    manual_analyses: Path | str | None = None,
) -> dict[str, dict[str, Any]]:
    probe_map = load_probe_gene_map(probe2gene)
    if probe_map:
        print(f"Loaded mapping for {len(probe_map)} probes.")

    required_cols = {"Dataset_A", "Dataset_B"}
    missing = required_cols - set(edges.columns)
    if missing:
        raise ValueError(f"Edges table is missing columns: {sorted(missing)}")

    explainers: dict[str, dict[str, Any]] = {}
    print(f"Running explain_topk for {len(edges)} edges (k={k})...")

    for idx, row in edges.iterrows():
        a = str(row["Dataset_A"])
        b = str(row["Dataset_B"])
        if a not in signatures or b not in signatures:
            raise KeyError(f"Cannot explain {a}_{b}: missing PCA signature.")

        idx_b, scores_b, _ = cskl.explain_topk(
            signatures[a],
            signatures[b],
            k=k,
            mode="B",
            return_scores=True,
            return_details=True,
        )
        idx_w, scores_w, _ = cskl.explain_topk(
            signatures[a],
            signatures[b],
            k=k,
            mode="W",
            return_scores=True,
            return_details=True,
        )

        explainers[edge_id(a, b)] = {
            "similar_features": _feature_records(idx_b, scores_b, feature_names, probe_map),
            "dissimilar_features": _feature_records(idx_w, scores_w, feature_names, probe_map),
        }

        if (idx + 1) % 20 == 0:
            print(f"  Processed {idx + 1}/{len(edges)} edges...")

    if manual_analyses:
        merge_manual_analyses(explainers, read_json(manual_analyses))

    return explainers


def write_edge_explainers(
    signatures: dict[str, cskl.PCASignature],
    edges_file: Path | str,
    output_file: Path | str,
    *,
    feature_names: list[str],
    k: int = 10,
    probe2gene: Path | str | None = None,
    manual_analyses: Path | str | None = None,
) -> dict[str, dict[str, Any]]:
    edges_path = Path(edges_file)
    if not edges_path.exists():
        raise FileNotFoundError(f"Missing edges file for explainers: {edges_path}")
    edges = pd.read_csv(edges_path, sep="\t")
    explainers = generate_edge_explainers(
        signatures,
        edges,
        feature_names=feature_names,
        k=k,
        probe2gene=probe2gene,
        manual_analyses=manual_analyses,
    )
    write_json(output_file, explainers)
    print(f"Wrote edge explainers to: {output_file}")
    return explainers


def recompute_explainers_from_matrices(
    csvs: list[str],
    edges_file: Path | str,
    output_file: Path | str,
    *,
    alpha: float = 0.5,
    k: int = 10,
    probe2gene: Path | str | None = None,
    manual_analyses: Path | str | None = None,
    seed: int = 0,
    orientation: str = "features-by-samples",
) -> dict[str, dict[str, Any]]:
    expanded = expand_paths(csvs)
    if not expanded:
        raise ValueError("At least one matrix path is required for explainer recomputation.")

    edges = pd.read_csv(edges_file, sep="\t")
    needed = set(edges["Dataset_A"].astype(str)).union(set(edges["Dataset_B"].astype(str)))

    expr: dict[str, pd.DataFrame] = {}
    for item in expanded:
        path = Path(item)
        gse = safe_name_from_file(path)
        if gse in needed:
            print(f"[{gse}] Loading matrix for explainers: {path}")
            expr[gse] = load_expr(path, orientation=orientation)

    missing = sorted(needed - set(expr))
    if missing:
        raise FileNotFoundError(
            "Missing matrices for datasets required by edges: " + ", ".join(missing)
        )

    common: set[str] | None = None
    for df in expr.values():
        idx = df.index.astype(str)
        common = set(idx) if common is None else common & set(idx)
    common_list = sorted(common or [])
    good_feats = compute_global_good_features(expr, common_list)
    print(f"Usable common features for explainers: {len(good_feats)}")

    rng = np.random.default_rng(seed)
    signatures: dict[str, cskl.PCASignature] = {}
    for name, df in expr.items():
        X = df.loc[good_feats].to_numpy(dtype=np.float64).T
        signatures[name] = cskl.fit_pca_signature(
            X,
            alpha=alpha,
            feature_names=good_feats,
            rng=rng,
            nan_policy="raise",
            r_compat_noise=True,
        )

    explainers = generate_edge_explainers(
        signatures,
        edges,
        feature_names=good_feats,
        k=k,
        probe2gene=probe2gene,
        manual_analyses=manual_analyses,
    )
    write_json(output_file, explainers)
    print(f"Wrote edge explainers to: {output_file}")
    return explainers


def merge_manual_analyses(explainers: dict[str, dict[str, Any]], analyses: dict[str, Any]) -> int:
    updates = 0
    for raw_edge, payload in analyses.items():
        if isinstance(payload, dict):
            llm_text = payload.get("llm_analysis") or payload.get("analysis") or payload.get("text")
        else:
            llm_text = payload
        if not llm_text:
            continue

        parts = str(raw_edge).split("_")
        keys = [str(raw_edge)]
        if len(parts) == 2:
            keys = list(edge_id_variants(parts[0], parts[1]))

        for key in keys:
            if key in explainers:
                explainers[key]["llm_analysis"] = str(llm_text)
                updates += 1
                break
    if updates:
        print(f"Attached manual/LLM analyses to {updates} edge(s).")
    return updates


def apply_probe_mapping_to_file(
    explainer_path: Path | str,
    mapping_path: Path | str,
    *,
    backup: bool = True,
) -> Path | None:
    if backup:
        backup_created = backup_file(explainer_path)
        if backup_created:
            print(f"Created backup: {backup_created}")

    explainers = read_json(explainer_path)
    probe_map = load_probe_gene_map(mapping_path)
    for data in explainers.values():
        for feat_type in ("similar_features", "dissimilar_features"):
            for item in data.get(feat_type, []):
                if isinstance(item, dict) and "gene" in item:
                    item["gene"] = probe_map.get(str(item["gene"]), item["gene"])
    write_json(explainer_path, explainers)
    return backup_created if backup else None
