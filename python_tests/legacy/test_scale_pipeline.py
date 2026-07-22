"""Resume + update integration tests (plan sections 3 & 5).

Verifies the two core operational promises:
  * resume: re-running ingest recomputes nothing (files are the state).
  * update: adding a dataset never recomputes existing signatures or profiles;
    only the new dataset is fit + profiled, and assemble re-sorts BH globally.

Uses the 4 CEL datasets (outreplica/work) plus one series-matrix CSV
(data/GSE245371) to also exercise the CSV path on the frozen feature space.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from cskl_pipeline.scale import (
    assemble as assemble_mod,
    ingest as ingest_mod,
    pool as pool_mod,
    store as store_mod,
)

REPO = Path(__file__).resolve().parent.parent
CEL_NAMES = ["GSE16214", "GSE33630", "GSE35570", "GSE42057"]
CEL_DIR = REPO / "outreplica" / "work"
CSV_NEW = REPO / "data" / "GSE245371_series_matrix.csv"

pytestmark = pytest.mark.skipif(
    not all((CEL_DIR / n / "expr.tsv.gz").exists() for n in CEL_NAMES)
    or not CSV_NEW.exists(),
    reason="sample data (outreplica/work + data/GSE245371) not present",
)


@pytest.fixture(scope="module")
def base_store():
    root = Path(tempfile.mkdtemp(prefix="cskl_pipe_"))
    store = store_mod.Store(root)
    for n in CEL_NAMES:
        dst = store.expr_path(n)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(CEL_DIR / n / "expr.tsv.gz", dst)
    # default (non-exact) grid so a new dataset's size is bracketed
    pool_mod.build_pool(store, "GPL570", sample_from_corpus=True,
                        probes_from=str(store.expr_path(CEL_NAMES[0])),
                        M_samples=10_000, seed=0, alpha=0.5, B=30,
                        grid_min=4, grid_max=260)
    tars = [str(REPO / "raw" / f"{n}_RAW.tar") for n in CEL_NAMES]
    ingest_mod.run_ingest(store, "GPL570", "pool_v1", tars=tars, workers=1)
    yield store
    shutil.rmtree(root, ignore_errors=True)


def _mtimes(store, names):
    return {n: (store.signature_path(n).stat().st_mtime_ns,
                store.null_profile_path(n, "pool_v1").stat().st_mtime_ns)
            for n in names}


def test_resume_recomputes_nothing(base_store):
    before = _mtimes(base_store, CEL_NAMES)
    tars = [str(REPO / "raw" / f"{n}_RAW.tar") for n in CEL_NAMES]
    summary = ingest_mod.run_ingest(base_store, "GPL570", "pool_v1", tars=tars, workers=1)
    after = _mtimes(base_store, CEL_NAMES)
    assert summary["fit_ok"] == 0            # nothing re-fit
    assert summary.get("profiles_written", 0) == 0
    assert before == after                   # no file rewritten


def test_update_adds_without_recomputing_existing(base_store):
    before = _mtimes(base_store, CEL_NAMES)

    manifest = assemble_mod.run_update(
        base_store, "GPL570", "pool_v1", "update_run",
        csvs=[str(CSV_NEW)], input_orientation="samples-by-features",
        workers=1, fdr_alpha=0.05,
    )

    # existing datasets untouched
    after = _mtimes(base_store, CEL_NAMES)
    assert before == after, "update recomputed an existing signature/profile"

    # new dataset ingested + profiled
    assert base_store.has_valid_signature("GSE245371", "GPL570")
    assert base_store.has_valid_profile("GSE245371", "GPL570", "pool_v1")

    # assemble now covers all five, BH re-sorted globally over all pairs
    assert manifest["n_datasets_kept"] == 5
    assert manifest["n_pairs"] == 10          # C(5,2)
    assert manifest["ingest"]["fit_ok"] == 1  # only the new one was fit


def test_update_run_matches_direct_assemble(base_store):
    """The observed matrix from the updated run equals a direct all-5 assemble."""
    import pandas as pd
    m = assemble_mod.run_assemble(base_store, "GPL570", "pool_v1", "direct5")
    a = pd.read_csv(base_store.run_dir("update_run") / "cskl_matrix.tsv", sep="\t", index_col=0)
    b = pd.read_csv(base_store.run_dir("direct5") / "cskl_matrix.tsv", sep="\t", index_col=0)
    ids = sorted(a.index)
    assert np.max(np.abs(a.loc[ids, ids].to_numpy() - b.loc[ids, ids].to_numpy())) <= 1e-9
