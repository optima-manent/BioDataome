"""Unit tests for the scale pipeline building blocks (fast, no external data)."""
from __future__ import annotations

import numpy as np
import pytest

import cskl
from cskl_pipeline.scale import align, grid, profile, qc, store as store_mod
from cskl_pipeline.scale import fastcore as fc


# ---------------------------------------------------------------------------
# grid
# ---------------------------------------------------------------------------

def test_build_size_grid_brackets_range():
    g = grid.build_size_grid(5, 300)
    assert g[0] == 5 and g[-1] == 300
    assert g == sorted(g)
    assert all(5 <= x <= 300 for x in g)


def test_build_size_grid_extends_above_400():
    g = grid.build_size_grid(4, 1200)
    assert g[-1] == 1200
    assert max(g) >= 1000  # geometric extension covers large corpora


def test_exact_size_grid_dedups():
    assert grid.exact_size_grid([10, 10, 20, 5, 5]) == [5, 10, 20]


# ---------------------------------------------------------------------------
# QC
# ---------------------------------------------------------------------------

def test_qc_ok_and_imputes_stray_nan():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 100))
    X[3, 7] = np.nan
    res = qc.run_qc(X, max_nonfinite_fraction=0.01)
    assert res.status == "ok"
    assert res.n_imputed_entries == 1
    assert np.isfinite(res.X).all()


def test_qc_quarantines_all_nan_matrix():
    X = np.full((10, 50), np.nan)
    res = qc.run_qc(X)
    assert res.status == "quarantine"
    assert "nonfinite" in res.reason


def test_qc_quarantines_too_few_samples():
    res = qc.run_qc(np.ones((1, 50)))
    assert res.status == "quarantine"
    assert "too_few_samples" in res.reason


def test_qc_quarantines_zero_variance():
    res = qc.run_qc(np.ones((10, 50)))
    assert res.status == "quarantine"
    assert res.reason == "zero_variance"


# ---------------------------------------------------------------------------
# align
# ---------------------------------------------------------------------------

def test_align_reindexes_and_reports_missing(tmp_path):
    import pandas as pd
    df = pd.DataFrame(
        [[1.0, 2.0], [3.0, 4.0]],
        index=["p1", "p3"], columns=["s1", "s2"],
    )
    probes = ["p1", "p2", "p3"]  # p2 missing, order enforced
    X, info = align.align_to_probes(df, probes)
    assert X.shape == (2, 3)          # samples x probes
    assert info["n_missing"] == 1
    assert np.isnan(X[:, 1]).all()    # p2 column is missing -> NaN
    # X is samples x probes: row 0 = sample s1 -> [p1=1.0, p2=NaN, p3=3.0]
    assert X[0, 0] == 1.0 and X[0, 2] == 3.0
    assert X[1, 0] == 2.0 and X[1, 2] == 4.0


# ---------------------------------------------------------------------------
# store atomic IO + resume validation
# ---------------------------------------------------------------------------

def test_signature_roundtrip_and_feature_hash(tmp_path):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 200))
    sig = cskl.fit_pca_signature(X, alpha=0.5)
    path = tmp_path / "signature.npz"
    fh = "deadbeef"
    store_mod.save_signature(path, sig, fh)
    loaded = store_mod.load_signature(path)
    assert np.allclose(loaded.lam, sig.lam)
    assert np.allclose(loaded.P, sig.P)
    assert store_mod.read_signature_feature_hash(path) == fh
    meta = store_mod.read_signature_meta(path)
    assert meta["m_samples"] == 30 and meta["n_features"] == 200


def test_atomic_json_and_null_profile(tmp_path):
    p = tmp_path / "sub" / "x.json"
    store_mod.atomic_write_json(p, {"b": 1, "a": 2})
    assert store_mod.read_json(p) == {"b": 1, "a": 2}

    npz = tmp_path / "np.npz"
    store_mod.save_null_profile(
        npz, grid=np.array([4, 10]), mu=np.array([1.0, 2.0]),
        sigma=np.array([0.1, 0.2]), pool_version="pool_v1", pool_hash="ph",
        feature_hash="fh", signature_hash="a" * 64, alpha=0.5, mode="grid", B=100,
    )
    prof = store_mod.load_null_profile(npz)
    assert list(prof.grid) == [4, 10]
    assert prof.pool_version == "pool_v1" and prof.B == 100
    assert prof.alpha == 0.5
    assert prof.signature_hash == "a" * 64


def _minimal_signature(alpha=0.5):
    return cskl.PCASignature(
        P=np.array([[1.0], [0.0], [0.0]]),
        lam=np.array([alpha * 3.0]),
        n_features=3,
        m_samples=4,
        alpha=alpha,
    )


def _profile_store(tmp_path):
    store = store_mod.Store(tmp_path / "store")
    store.write_probes("GPL570", ["p1", "p2", "p3"])
    feature_hash = store.feature_hash("GPL570")
    meta = {
        "seed": 11,
        "grid": [4, 10],
        "M_samples": 20,
        "feature_hash": feature_hash,
        "matrix_sha1": "matrix-v1",
        "alpha": 0.5,
        "B": 100,
        "exact_size": False,
    }
    store_mod.atomic_write_json(store.pool_meta_path("GPL570", "pool_v1"), meta)
    store_mod.save_signature(
        store.signature_path("GSE1"), _minimal_signature(), feature_hash
    )
    return store, meta


def test_pool_hash_includes_profile_parameters(tmp_path):
    store, meta = _profile_store(tmp_path)
    hashes = {store.pool_hash("GPL570", "pool_v1")}

    for update in ({"alpha": 0.6}, {"B": 101}, {"exact_size": True}):
        store_mod.atomic_write_json(
            store.pool_meta_path("GPL570", "pool_v1"), {**meta, **update}
        )
        hashes.add(store.pool_hash("GPL570", "pool_v1"))

    assert len(hashes) == 4


@pytest.mark.parametrize(
    ("alpha", "B", "mode"),
    [(0.6, 100, "grid"), (0.5, 99, "grid"), (0.5, 100, "exact")],
)
def test_profile_validation_rejects_mismatched_parameters(tmp_path, alpha, B, mode):
    store, _ = _profile_store(tmp_path)
    store_mod.save_null_profile(
        store.null_profile_path("GSE1", "pool_v1"),
        grid=np.array([4, 10]),
        mu=np.array([1.0, 2.0]),
        sigma=np.array([0.1, 0.2]),
        pool_version="pool_v1",
        pool_hash=store.pool_hash("GPL570", "pool_v1"),
        feature_hash=store.feature_hash("GPL570"),
        signature_hash=store_mod.sha256_file(store.signature_path("GSE1")),
        alpha=alpha,
        mode=mode,
        B=B,
    )

    assert not store.has_valid_profile("GSE1", "GPL570", "pool_v1")


def test_profile_header_and_validation_bind_configuration(tmp_path):
    store, _ = _profile_store(tmp_path)
    path = store.null_profile_path("GSE1", "pool_v1")
    store_mod.save_null_profile(
        path,
        grid=np.array([4, 10]),
        mu=np.array([1.0, 2.0]),
        sigma=np.array([0.1, 0.2]),
        pool_version="pool_v1",
        pool_hash=store.pool_hash("GPL570", "pool_v1"),
        feature_hash=store.feature_hash("GPL570"),
        signature_hash=store_mod.sha256_file(store.signature_path("GSE1")),
        alpha=0.5,
        mode="grid",
        B=100,
    )

    header = store_mod.read_null_profile_header(path)
    assert header is not None
    assert (header["alpha"], header["B"], header["mode"]) == (0.5, 100, "grid")
    assert header["signature_hash"] == store_mod.sha256_file(store.signature_path("GSE1"))
    assert store.has_valid_profile("GSE1", "GPL570", "pool_v1")


def test_profile_validation_rejects_same_alpha_signature_replacement(tmp_path):
    store, _ = _profile_store(tmp_path)
    signature_path = store.signature_path("GSE1")
    profile_path = store.null_profile_path("GSE1", "pool_v1")
    store_mod.save_null_profile(
        profile_path,
        grid=np.array([4, 10]),
        mu=np.array([1.0, 2.0]),
        sigma=np.array([0.1, 0.2]),
        pool_version="pool_v1",
        pool_hash=store.pool_hash("GPL570", "pool_v1"),
        feature_hash=store.feature_hash("GPL570"),
        signature_hash=store_mod.sha256_file(signature_path),
        alpha=0.5,
        mode="grid",
        B=100,
    )
    assert store.has_valid_profile("GSE1", "GPL570", "pool_v1")

    replacement = cskl.PCASignature(
        P=np.array([[0.0], [1.0], [0.0]]),
        lam=np.array([1.5]),
        n_features=3,
        m_samples=4,
        alpha=0.5,
    )
    store_mod.save_signature(signature_path, replacement, store.feature_hash("GPL570"))

    assert not store.has_valid_profile("GSE1", "GPL570", "pool_v1")


def test_incomplete_legacy_profile_is_invalidated_safely(tmp_path):
    store, _ = _profile_store(tmp_path)
    path = store.null_profile_path("GSE1", "pool_v1")
    store_mod.atomic_save_npz(
        path,
        grid=np.array([4, 10]),
        mu=np.array([1.0, 2.0]),
        sigma=np.array([0.1, 0.2]),
        pool_version=np.array("pool_v1"),
        pool_hash=np.array(store.pool_hash("GPL570", "pool_v1")),
        feature_hash=np.array(store.feature_hash("GPL570")),
    )

    assert store_mod.read_null_profile_header(path) is None
    assert not store.has_valid_profile("GSE1", "GPL570", "pool_v1")


def test_profile_bootstrap_override_cannot_diverge_from_pool(tmp_path):
    store, _ = _profile_store(tmp_path)

    class Pool:
        B = 100

    with pytest.raises(ValueError, match="must match the frozen pool B"):
        profile.compute_profiles(store, Pool(), ["GSE1"], B=2, verbose=False)


# ---------------------------------------------------------------------------
# fastcore on synthetic data
# ---------------------------------------------------------------------------

def _synthetic(n=400, m=40, seed=0):
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.normal(size=(n, 4)))
    cov = U @ np.diag([5.0, 3.0, 2.0, 1.0]) @ U.T + np.eye(n)
    return rng.multivariate_normal(np.zeros(n), cov, size=m)


def test_fast_fit_matches_reference_synthetic():
    X = _synthetic()
    ref = cskl.fit_pca_signature(X, alpha=0.5, rng=np.random.default_rng(0))
    fast = fc.fast_fit_signature(X, alpha=0.5, rng=np.random.default_rng(0))
    assert len(ref.lam) == len(fast.lam)
    assert np.allclose(ref.lam, fast.lam, rtol=1e-9, atol=1e-12)
    # cskl between fast fit and reference fit of a second dataset
    X2 = _synthetic(seed=1)
    ref2 = cskl.fit_pca_signature(X2, alpha=0.5, rng=np.random.default_rng(0))
    assert abs(cskl.cskl(ref, ref2) - cskl.cskl(fast, ref2)) <= 1e-9


def test_batched_null_matches_loop_synthetic():
    sigP = cskl.fit_pca_signature(_synthetic(seed=2), alpha=0.5)
    nulls = [cskl.fit_pca_signature(_synthetic(m=30, seed=10 + i), alpha=0.5)
             for i in range(8)]
    bank = fc.SignatureBank(nulls)
    loop = np.array([cskl.cskl(sigP, r) for r in nulls])
    batched = fc.batched_null_cskl(sigP, bank)
    assert np.max(np.abs(loop - batched)) <= 1e-9


def test_signature_bank_rejects_mixed_alpha():
    with pytest.raises(ValueError, match="same alpha"):
        fc.SignatureBank([_minimal_signature(0.5), _minimal_signature(0.6)])


def test_batched_null_rejects_query_bank_alpha_mismatch():
    bank = fc.SignatureBank([_minimal_signature(0.6)])
    with pytest.raises(ValueError, match="same alpha"):
        fc.batched_null_cskl(_minimal_signature(0.5), bank)


def test_batched_many_rejects_query_bank_alpha_mismatch():
    query = fc.SignatureBank([_minimal_signature(0.5)])
    bank = fc.SignatureBank([_minimal_signature(0.6)])
    with pytest.raises(ValueError, match="same alpha"):
        fc.batched_null_cskl_many(query, bank)


def test_observed_matrix_matches_loop_synthetic():
    sigs = [cskl.fit_pca_signature(_synthetic(seed=s), alpha=0.5) for s in range(5)]
    ref = np.array([[cskl.cskl(a, b) for b in sigs] for a in sigs])
    fast = fc.observed_cskl_matrix(sigs)
    assert np.max(np.abs(ref - fast)) <= 1e-9
    assert np.allclose(fast, fast.T)  # symmetric


def test_batched_many_matches_loop_synthetic():
    qsigs = [cskl.fit_pca_signature(_synthetic(seed=s), alpha=0.5) for s in range(4)]
    nulls = [cskl.fit_pca_signature(_synthetic(m=25, seed=100 + i), alpha=0.5)
             for i in range(6)]
    query, bank = fc.SignatureBank(qsigs), fc.SignatureBank(nulls)
    ref = np.array([[cskl.cskl(q, r) for r in nulls] for q in qsigs])
    got = fc.batched_null_cskl_many(query, bank)
    assert np.max(np.abs(ref - got)) <= 1e-9
