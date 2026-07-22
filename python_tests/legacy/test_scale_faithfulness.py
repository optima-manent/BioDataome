"""Faithfulness gates (plan section 7).

These prove the fast/scalable paths reproduce the *current* ``cskl.py`` source of
truth. They use the 4 already-normalized CEL datasets under ``outreplica/work``;
if that sample data is absent the module is skipped.

NOTE on the golden files: ``outreplica/cskl_matrix.tsv`` and
``out_gpl570/pairwise.csv`` were produced by an OLDER, superseded c-SKL
implementation (values ~1e-6, inversely related to distance) and are NOT
reproducible by the current ``cskl.py`` (which yields ~40000-scale distances,
matching ``results/cskl_legacy_4gse/cskl_paper_matrix.tsv`` exactly). The binding
regression target is therefore the current ``cskl.py`` reference pipeline,
reproduced here in query-set-pool + exact-size mode. See RUN.md / STALE_GOLDENS.md.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import cskl
from cskl_pipeline.scale import (
    assemble as assemble_mod,
    fastcore as fc,
    grid as grid_mod,
    ingest as ingest_mod,
    pool as pool_mod,
    profile as profile_mod,
    reference as ref_mod,
    store as store_mod,
)

REPO = Path(__file__).resolve().parent.parent
CEL_NAMES = ["GSE16214", "GSE33630", "GSE35570", "GSE42057"]
CEL_DIR = REPO / "outreplica" / "work"

pytestmark = pytest.mark.skipif(
    not all((CEL_DIR / n / "expr.tsv.gz").exists() for n in CEL_NAMES),
    reason="CEL sample data (outreplica/work) not present",
)

B_TEST = 50


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cel_matrices():
    feats = None
    X = {}
    for n in CEL_NAMES:
        df = pd.read_csv(CEL_DIR / n / "expr.tsv.gz", sep="\t", index_col=0)
        if feats is None:
            feats = list(df.index.astype(str))
        X[n] = df.loc[feats].to_numpy(dtype=np.float64).T
    return X, feats


@pytest.fixture(scope="module")
def scale_store(cel_matrices):
    """A store with the 4 CEL datasets ingested against an exact-size query pool."""
    X, feats = cel_matrices
    root = Path(tempfile.mkdtemp(prefix="cskl_gate_"))
    store = store_mod.Store(root)
    for n in CEL_NAMES:
        dst = store.expr_path(n)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(CEL_DIR / n / "expr.tsv.gz", dst)
    pool_mod.build_pool(store, "GPL570", sample_from_corpus=True,
                        probes_from=str(store.expr_path(CEL_NAMES[0])),
                        M_samples=10_000, seed=0, alpha=0.5, B=B_TEST, exact_size=True)
    tars = [str(REPO / "raw" / f"{n}_RAW.tar") for n in CEL_NAMES]
    ingest_mod.run_ingest(store, "GPL570", "pool_v1", tars=tars, workers=1)
    yield store
    shutil.rmtree(root, ignore_errors=True)


# ---------------------------------------------------------------------------
# Gate 1 - fast fit vs fit_pca_signature
# ---------------------------------------------------------------------------

def test_gate1_fast_fit_reproduces_reference(cel_matrices):
    X, feats = cel_matrices
    ref = {n: cskl.fit_pca_signature(X[n], alpha=0.5, feature_names=feats,
                                     rng=np.random.default_rng(0)) for n in CEL_NAMES}
    fast = {n: fc.fast_fit_signature(X[n], alpha=0.5, feature_names=feats,
                                     rng=np.random.default_rng(0)) for n in CEL_NAMES}
    for n in CEL_NAMES:
        assert len(ref[n].lam) == len(fast[n].lam), f"c mismatch for {n}"
        assert np.allclose(ref[n].lam, fast[n].lam, rtol=1e-9, atol=1e-12), \
            f"lambda mismatch for {n}"
    # resulting cskl differences <= 1e-9
    maxd = max(abs(cskl.cskl(ref[a], ref[b]) - cskl.cskl(fast[a], fast[b]))
               for a in CEL_NAMES for b in CEL_NAMES)
    assert maxd <= 1e-9, f"cskl diff {maxd} > 1e-9"


# ---------------------------------------------------------------------------
# Gate 2 - batched null distances vs cskl loop
# ---------------------------------------------------------------------------

def test_gate2_batched_null_reproduces_loop(cel_matrices):
    X, feats = cel_matrices
    sigs = {n: fc.fast_fit_signature(X[n], alpha=0.5, feature_names=feats,
                                     rng=np.random.default_rng(0)) for n in CEL_NAMES}
    Xpool = np.vstack([X[n] for n in CEL_NAMES])
    lib_pool = cskl.Pool(Xpool, alpha=0.5, feature_names=feats)
    nulls = lib_pool.get_null_signatures(m=64, B=24, rng=np.random.default_rng(3))
    bank = fc.SignatureBank(nulls)
    for n in CEL_NAMES:
        loop = np.array([cskl.cskl(sigs[n], r) for r in nulls])
        batched = fc.batched_null_cskl(sigs[n], bank)
        assert np.max(np.abs(loop - batched)) <= 1e-9


# ---------------------------------------------------------------------------
# Gate 3 - grid interpolation vs exact-size profile
# ---------------------------------------------------------------------------

def _fast_profile(pool, sig, grid_pts):
    """(mu, sigma) over grid_pts via the fast path (fit-method agnostic)."""
    mu = np.zeros(len(grid_pts))
    sg = np.zeros(len(grid_pts))
    for i, g in enumerate(grid_pts):
        bank = pool.null_bank(int(g), B=B_TEST)
        d = fc.batched_null_cskl(sig, bank)
        mu[i], sg[i] = d.mean(), d.std(ddof=0)
    return mu, sg


def test_gate3_grid_interpolation_vs_exact(scale_store, cel_matrices):
    store = scale_store
    pool = pool_mod.PoolHandle(store, "GPL570", "pool_v1")  # exact-size grid
    sizes = {n: store_mod.read_signature_meta(store.signature_path(n))["m_samples"]
             for n in CEL_NAMES}
    exact = {n: store_mod.load_null_profile(store.null_profile_path(n, "pool_v1"))
             for n in CEL_NAMES}
    sigs = {n: store_mod.load_signature(store.signature_path(n)) for n in CEL_NAMES}
    C = np.array([[cskl.cskl(sigs[a], sigs[b]) for b in CEL_NAMES] for a in CEL_NAMES])

    # Coarse grid whose nodes do NOT coincide with the dataset sizes.
    coarse = grid_mod.build_size_grid(80, 300)
    coarse = [g for g in coarse if g not in set(sizes.values())]
    grid_prof = {n: (np.array(coarse), *_fast_profile(pool, sigs[n], coarse))
                 for n in CEL_NAMES}
    exact_prof = {n: (exact[n].grid, exact[n].mu, exact[n].sigma) for n in CEL_NAMES}

    # Interpolated mu must match the exact-size mu within bootstrap MC tolerance.
    # (The null sigma here is ~1e3 on a small query-set pool, so the Monte-Carlo
    # standard error of mu is ~sigma/sqrt(B); we allow a few of those.)
    worst_ratio = 0.0
    for n in CEL_NAMES:
        for target in sizes.values():
            gi = list(exact[n].grid).index(target)
            mu_interp = np.interp(target, grid_prof[n][0], grid_prof[n][1])
            mu_exact = exact_prof[n][1][gi]
            mc_se = max(exact[n].sigma[gi], 1.0) / np.sqrt(B_TEST)
            worst_ratio = max(worst_ratio, abs(mu_interp - mu_exact) / mc_se)
    assert worst_ratio < 8.0, f"interpolated mu off by {worst_ratio:.1f} MC standard errors"

    # p-values agree to ~1e-2 and edge sets are identical.
    def pvals_edges(prof):
        from scipy.stats import norm
        iu, ju = np.triu_indices(len(CEL_NAMES), 1)
        pv = []
        for i, j in zip(iu, ju):
            a, b = CEL_NAMES[i], CEL_NAMES[j]
            muA = np.interp(sizes[b], prof[a][0], prof[a][1])
            sgA = max(np.interp(sizes[b], prof[a][0], prof[a][2]), 1e-12)
            muB = np.interp(sizes[a], prof[b][0], prof[b][1])
            sgB = max(np.interp(sizes[a], prof[b][0], prof[b][2]), 1e-12)
            pv.append(max(norm.cdf(C[i, j], muA, sgA), norm.cdf(C[i, j], muB, sgB)))
        pv = np.array(pv)
        q = np.array(cskl.bh_qvalues(pv.tolist()))
        edges = {(CEL_NAMES[i], CEL_NAMES[j]) for i, j, qq in zip(iu, ju, q) if qq <= 0.05}
        return pv, edges

    pv_grid, e_grid = pvals_edges(grid_prof)
    pv_exact, e_exact = pvals_edges(exact_prof)
    assert np.max(np.abs(pv_grid - pv_exact)) <= 1e-2, "p-values diverged > 1e-2"
    assert e_grid == e_exact, f"edge sets differ: grid={e_grid} exact={e_exact}"


# ---------------------------------------------------------------------------
# Gate 4 - end-to-end observed matrix vs current cskl.py reference
# ---------------------------------------------------------------------------

def test_gate4_observed_matrix_reproduces_reference(scale_store, cel_matrices):
    store = scale_store
    X, feats = cel_matrices

    assemble_mod.run_assemble(store, "GPL570", "pool_v1", "gate4", fdr_alpha=0.05)
    got = pd.read_csv(store.run_dir("gate4") / "cskl_matrix.tsv", sep="\t", index_col=0)
    got = got.loc[CEL_NAMES, CEL_NAMES].to_numpy()

    rng = np.random.default_rng(0)
    ref_sigs = {n: cskl.fit_pca_signature(X[n], alpha=0.5, feature_names=feats, rng=rng)
                for n in CEL_NAMES}
    ref = ref_mod.reference_observed_matrix(ref_sigs, CEL_NAMES)
    assert np.max(np.abs(got - ref)) <= 1e-9, "observed cskl matrix diverged from reference"


# ---------------------------------------------------------------------------
# Gate 5 - profile p-values match the library cskl.pair_pvalue_vs_pool
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_gate5_pvalues_match_library(scale_store, cel_matrices):
    """The strongest external tie: the cached-profile p-values reproduce the
    library's semi-parametric bootstrap p-values within Monte-Carlo noise."""
    store = scale_store
    X, feats = cel_matrices
    pool = pool_mod.PoolHandle(store, "GPL570", "pool_v1")

    Xpool = np.ascontiguousarray(pool.X_pool, dtype=np.float64)
    lib_pool = cskl.Pool(Xpool, alpha=0.5, feature_names=feats)
    sigs = {n: store_mod.load_signature(store.signature_path(n), feature_names=feats)
            for n in CEL_NAMES}
    ptable = profile_mod.load_profile_table(store, CEL_NAMES, "pool_v1")

    rng = np.random.default_rng(999)
    maxdiff = 0.0
    for i, a in enumerate(CEL_NAMES):
        for b in CEL_NAMES[i + 1:]:
            c = cskl.cskl(sigs[a], sigs[b])
            p_scale = profile_mod.pair_pvalue_from_profiles(
                c, ptable.profiles[a], ptable.sizes[b],
                ptable.profiles[b], ptable.sizes[a])
            p_lib = cskl.pair_pvalue_vs_pool(sigs[a], sigs[b], lib_pool, B=B_TEST, rng=rng)
            maxdiff = max(maxdiff, abs(p_scale - p_lib))
    # Independent bootstrap draws -> agreement is within MC noise, not exact.
    assert maxdiff <= 0.05, f"scale vs library p-value max diff {maxdiff} > 0.05"
