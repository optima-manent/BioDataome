from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

import cskl

from .config import PipelineConfig
from .explainers import write_edge_explainers
from .geo import write_geo_descriptions
from .graph_html import generate_graph_html
from .io import (
    PIPELINE_OUTPUTS,
    compute_global_good_features,
    expand_paths,
    load_expr,
    output_path,
    safe_name_from_file,
    write_json,
)
from .normalize import normalize_affy_scan
from .specter import write_specter2_similarities


def load_datasets(config: PipelineConfig) -> tuple[dict[str, pd.DataFrame], list[str]]:
    outdir = Path(config.outdir).resolve()
    work_root = outdir / "work"
    work_root.mkdir(parents=True, exist_ok=True)

    expr: dict[str, pd.DataFrame] = {}
    names: list[str] = []

    for tar_item in expand_paths(config.tars):
        tar_path = Path(tar_item).resolve()
        if not tar_path.exists():
            raise FileNotFoundError(f"TAR input not found: {tar_path}")
        gse = safe_name_from_file(tar_path)
        if gse in expr:
            print(f"[{gse}] Skipping duplicate input: {tar_path}")
            continue
        names.append(gse)
        expr_path = normalize_affy_scan(tar_path, work_root, config.rscript)
        expr[gse] = load_expr(expr_path, orientation="features-by-samples")

    for csv_item in expand_paths(config.csvs):
        csv_path = Path(csv_item).resolve()
        if not csv_path.exists():
            raise FileNotFoundError(f"Matrix input not found: {csv_path}")
        gse = safe_name_from_file(csv_path)
        if gse in expr:
            print(f"[{gse}] Skipping duplicate input: {csv_path}")
            continue
        names.append(gse)
        print(f"[{gse}] Loading preprocessed matrix: {csv_path}")
        expr[gse] = load_expr(csv_path, orientation=config.input_orientation)

    if not names:
        raise ValueError("Must provide at least one dataset via --tars or --csvs.")
    return expr, names


def load_pool_matrix(config: PipelineConfig) -> pd.DataFrame | None:
    if not config.pool_matrix:
        return None
    path = Path(config.pool_matrix).resolve()
    print(f"Loading global background pool from {path}...")
    return load_expr(path, orientation=config.input_orientation)


def intersect_features(expr: dict[str, pd.DataFrame]) -> list[str]:
    common: set[str] | None = None
    for df in expr.values():
        idx = df.index.astype(str)
        common = set(idx) if common is None else common & set(idx)
    return sorted(common or [])


def sample_hashes(df: pd.DataFrame, features: list[str], decimals: int | None = 10) -> set[str]:
    X = df.loc[features].to_numpy(dtype=np.float64)
    if decimals is not None:
        X = np.round(X, decimals=decimals)

    hashes: set[str] = set()
    for j in range(X.shape[1]):
        col = np.ascontiguousarray(X[:, j])
        hashes.add(hashlib.sha1(col.view(np.uint8)).hexdigest())
    return hashes


def filter_shared_profile_datasets(
    expr: dict[str, pd.DataFrame],
    names: list[str],
    features: list[str],
) -> tuple[dict[str, pd.DataFrame], list[str], set[str]]:
    datasets_to_drop: set[str] = set()
    sample_hashes_by_dataset = {
        name: sample_hashes(expr[name], features, decimals=10)
        for name in names
    }

    for i, name_a in enumerate(names):
        for name_b in names[i + 1 :]:
            shared = sample_hashes_by_dataset[name_a] & sample_hashes_by_dataset[name_b]
            if shared:
                print(
                    f"[Filter] Found {len(shared)} shared molecular profile(s) "
                    f"between {name_a} and {name_b}."
                )
                datasets_to_drop.add(name_a)
                datasets_to_drop.add(name_b)

    if not datasets_to_drop:
        return expr, names, datasets_to_drop

    filtered_expr = dict(expr)
    filtered_names = list(names)
    for drop_name in sorted(datasets_to_drop):
        print(f"[Filter] Dropping {drop_name} from analysis to prevent artificial similarity artifacts.")
        filtered_expr.pop(drop_name, None)
        if drop_name in filtered_names:
            filtered_names.remove(drop_name)
    return filtered_expr, filtered_names, datasets_to_drop


def fit_signatures(
    expr: dict[str, pd.DataFrame],
    names: list[str],
    features: list[str],
    *,
    alpha: float,
    rng: np.random.Generator,
    n_features_common_before_filter: int,
) -> tuple[dict[str, cskl.PCASignature], dict[str, dict]]:
    signatures: dict[str, cskl.PCASignature] = {}
    metas: dict[str, dict] = {}

    for name in names:
        df = expr[name]
        X = df.loc[features].to_numpy(dtype=np.float64).T
        sig = cskl.fit_pca_signature(
            X,
            alpha=alpha,
            feature_names=features,
            rng=rng,
            nan_policy="raise",
            r_compat_noise=True,
        )
        signatures[name] = sig
        metas[name] = {
            "n_features_common_before_filter": int(n_features_common_before_filter),
            "n_features_used": sig.n_features,
            "n_samples": sig.m_samples,
            "c_components": len(sig.lam),
            "alpha": float(sig.alpha),
            "lambda_sum": float(np.sum(sig.lam)),
        }
        print(
            f"[{name}] samples={sig.m_samples}  "
            f"features_used={sig.n_features}  "
            f"c={len(sig.lam)}  "
            f"lambda_sum={np.sum(sig.lam):.6f}"
        )

    return signatures, metas


def write_pairwise_outputs(
    signatures: dict[str, cskl.PCASignature],
    names: list[str],
    outdir: Path,
) -> pd.DataFrame:
    mat = np.zeros((len(names), len(names)), dtype=np.float64)
    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            mat[i, j] = cskl.cskl(signatures[ni], signatures[nj])

    df_cskl = pd.DataFrame(mat, index=names, columns=names)
    df_cskl.to_csv(output_path(outdir, "cskl_matrix"), sep="\t")

    df_sim = 1.0 / (1.0 + df_cskl)
    df_sim.to_csv(output_path(outdir, "cskl_similarity"), sep="\t")

    print("\n=== c-SKL divergence matrix (lower = more similar) ===")
    print(df_cskl.round(8).to_string())
    return df_cskl


def write_network_output(
    signatures: dict[str, cskl.PCASignature],
    expr: dict[str, pd.DataFrame],
    names: list[str],
    features: list[str],
    pool_df: pd.DataFrame | None,
    config: PipelineConfig,
) -> pd.DataFrame | None:
    if len(names) <= 1:
        return None

    print("\nBuilding statistical significance network...")
    if pool_df is not None:
        X_pool_all = pool_df.loc[features].to_numpy(dtype=np.float64).T
    else:
        print(
            "\nWARNING: No --pool-matrix provided. Significance testing will use ONLY "
            "the queried datasets as the background."
        )
        print("This lacks statistical power. Please provide a global platform pool for accurate p-values.\n")
        X_pool_all = np.vstack([expr[name].loc[features].to_numpy(dtype=np.float64).T for name in names])

    pool = cskl.Pool(
        X_pool_all,
        alpha=config.alpha,
        feature_names=features,
        nan_policy="raise",
        r_compat_noise=True,
    )
    rng = np.random.default_rng(config.seed)
    all_pairs, kept_edges = cskl.build_dataset_network(
        signatures,
        pool,
        B=config.B,
        fdr_alpha=config.fdr_alpha,
        rng=rng,
    )
    network_df = pd.DataFrame(
        all_pairs,
        columns=["Dataset_A", "Dataset_B", "cSKL", "p_value", "q_value"],
    )
    network_df.to_csv(output_path(config.outdir, "network_edges"), sep="\t", index=False)
    print(f"Discovered {len(kept_edges)} statistically significant similarities (FDR <= {config.fdr_alpha}).")
    return network_df


def run_pipeline(config: PipelineConfig) -> dict[str, Path]:
    outdir = Path(config.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    config.outdir = outdir

    rng = np.random.default_rng(config.seed)
    expr, names = load_datasets(config)
    pool_df = load_pool_matrix(config)

    expr_for_feature_alignment = dict(expr)
    if pool_df is not None:
        expr_for_feature_alignment["__GLOBAL_POOL__"] = pool_df

    common = intersect_features(expr_for_feature_alignment)
    if len(common) < config.min_common_features:
        raise SystemExit(
            f"Too few common features across datasets ({len(common)}). "
            "Likely different platforms; map to a common gene set first."
        )

    good_feats = compute_global_good_features(expr_for_feature_alignment, common)
    if len(good_feats) < config.min_common_features:
        raise SystemExit(
            f"Too few usable common features after filtering ({len(good_feats)}). "
            "Check for missing values or mismatched platforms."
        )

    print(f"Common features across all datasets: {len(common)}")
    print(f"Usable common features (finite in all datasets): {len(good_feats)}")

    expr, names, dropped = filter_shared_profile_datasets(expr, names, good_feats)
    if not names:
        raise SystemExit(
            "All target datasets were removed by the shared-profile filter. "
            "No c-SKL analysis can be run."
        )

    signatures, metas = fit_signatures(
        expr,
        names,
        good_feats,
        alpha=config.alpha,
        rng=rng,
        n_features_common_before_filter=len(common),
    )
    if dropped:
        for meta in metas.values():
            meta["datasets_dropped_for_shared_profiles"] = sorted(dropped)

    write_json(output_path(outdir, "pca_meta"), metas)
    write_pairwise_outputs(signatures, names, outdir)
    network_df = write_network_output(signatures, expr, names, good_feats, pool_df, config)

    if config.run_explainers:
        edges_file = output_path(outdir, "network_edges")
        if network_df is None or not edges_file.exists():
            raise FileNotFoundError("Cannot run explainers because cskl_network_edges.tsv was not created.")
        write_edge_explainers(
            signatures,
            edges_file,
            output_path(outdir, "edge_explainers"),
            feature_names=good_feats,
            k=config.k,
            probe2gene=config.probe2gene,
            manual_analyses=config.manual_analyses,
        )

    if config.fetch_geo:
        write_geo_descriptions(outdir)

    if config.run_specter2:
        write_specter2_similarities(
            outdir,
            model_name=config.specter2_model,
            top_n=config.specter2_top_n,
            q_threshold=config.fdr_alpha,
        )

    if config.generate_html:
        generate_graph_html(
            outdir,
            output_html=config.output_html,
            q_threshold=config.fdr_alpha,
        )

    print("\nWrote:")
    for key in ("cskl_matrix", "cskl_similarity", "network_edges", "pca_meta"):
        path = output_path(outdir, key)
        if path.exists():
            print(" -", path)
    for key in ("edge_explainers", "geo_descriptions", "specter2_similarities", "html"):
        path = output_path(outdir, key)
        if path.exists():
            print(" -", path)
    print(" - normalized matrices under:", outdir / "work" / "<GSE>" / "expr.tsv.gz")

    return {key: output_path(outdir, key) for key in PIPELINE_OUTPUTS}
