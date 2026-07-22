"""Command-line entry point for the scalable c-SKL pipeline.

Subcommands (plan section 4):

    build-pool   freeze probes.txt + <pool_version>/ for a platform
    ingest       normalize + fit + null-profile new datasets (resumable, parallel)
    assemble     shared-profile filter + observed matrix + p-values + BH + outputs
    update       ingest new inputs only, then assemble

The store root is configurable via ``--store`` or the ``CSKL_STORE`` env var and
should live on a data drive, not C:.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from . import assemble as assemble_mod
from . import ingest as ingest_mod
from . import pool as pool_mod
from . import store as store_mod


def _default_store() -> str:
    return os.environ.get("CSKL_STORE", str(Path.cwd() / "cskl_store"))


def _add_store(p: argparse.ArgumentParser) -> None:
    p.add_argument("--store", default=_default_store(),
                   help="Store root (or set CSKL_STORE). Put this on a data drive, not C:.")
    p.add_argument("--platform", default="GPL570", help="Platform id (default GPL570).")
    p.add_argument("--pool-version", default="pool_v1", help="Pool version (default pool_v1).")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="cskl-scale", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    # build-pool
    bp = sub.add_parser("build-pool", help="Freeze feature space + background pool.")
    _add_store(bp)
    src = bp.add_mutually_exclusive_group(required=True)
    src.add_argument("--from-matrix", help="Global pool matrix (features x samples).")
    src.add_argument("--sample-from-corpus", action="store_true",
                     help="Sample the pool from already-ingested expr.tsv.gz.")
    bp.add_argument("--from-matrix-orientation", default="features-by-samples",
                    choices=["features-by-samples", "samples-by-features"])
    bp.add_argument("--probes-from", default=None,
                    help="Matrix to freeze the probe order from (features x samples).")
    bp.add_argument("--M-samples", type=int, default=2000, help="Pool sample cap.")
    bp.add_argument("--seed", type=int, default=0)
    bp.add_argument("--alpha", type=float, default=0.5)
    bp.add_argument("--B", type=int, default=100, help="Bootstrap nulls per grid size.")
    bp.add_argument("--grid-min", type=int, default=None)
    bp.add_argument("--grid-max", type=int, default=None)
    bp.add_argument("--exact-size", action="store_true",
                    help="Use every distinct corpus sample size as the grid (max faithfulness).")
    bp.add_argument("--force", action="store_true", help="Rebuild even if it exists.")

    # ingest
    ig = sub.add_parser("ingest", help="Normalize + fit + null-profile new datasets.")
    _add_store(ig)
    ig.add_argument("--tars", nargs="+", default=[], help="GSE*_RAW.tar paths/globs.")
    ig.add_argument("--csvs", nargs="+", default=[], help="Preprocessed matrix paths/globs.")
    ig.add_argument("--workers", type=int, default=1, help="Parallel datasets (K~cores/2).")
    ig.add_argument("--input-orientation", default="features-by-samples",
                    choices=["features-by-samples", "samples-by-features"])
    ig.add_argument("--rscript", default="Rscript")
    ig.add_argument("--max-nonfinite-fraction", type=float, default=0.01)
    ig.add_argument("--keep-raw", action="store_true", help="Keep extracted CELs.")
    ig.add_argument("--no-profiles", action="store_true",
                    help="Skip the null-profile sweep (fit only).")

    # assemble
    asm = sub.add_parser("assemble", help="Build the network from ingested datasets.")
    _add_store(asm)
    asm.add_argument("--run-id", required=True)
    asm.add_argument("--fdr-alpha", type=float, default=0.05)
    asm.add_argument(
        "--major-overlap-coefficient", type=float, default=0.5,
        help="Shared fraction of the smaller dataset used for dotted major-overlap edges.",
    )
    asm.add_argument("--generate-html", action="store_true")
    asm.add_argument("--run-explainers", action="store_true")
    asm.add_argument("--k", type=int, default=10)
    asm.add_argument("--probe2gene", default=None)
    asm.add_argument("--fetch-geo", action="store_true")
    asm.add_argument("--output-html", default=None)

    # update
    up = sub.add_parser("update", help="Ingest new inputs only, then assemble.")
    _add_store(up)
    up.add_argument("--run-id", required=True)
    up.add_argument("--tars", nargs="+", default=[])
    up.add_argument("--csvs", nargs="+", default=[])
    up.add_argument("--workers", type=int, default=1)
    up.add_argument("--input-orientation", default="features-by-samples",
                    choices=["features-by-samples", "samples-by-features"])
    up.add_argument("--rscript", default="Rscript")
    up.add_argument("--fdr-alpha", type=float, default=0.05)
    up.add_argument("--major-overlap-coefficient", type=float, default=0.5)
    up.add_argument("--generate-html", action="store_true")
    up.add_argument("--run-explainers", action="store_true")
    up.add_argument("--k", type=int, default=10)
    up.add_argument("--probe2gene", default=None)
    return ap


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    store = store_mod.Store(args.store)
    print(f"[cskl-scale] store = {store.root}")

    if args.cmd == "build-pool":
        pool_mod.build_pool(
            store, args.platform, pool_version=args.pool_version,
            from_matrix=args.from_matrix, from_matrix_orientation=args.from_matrix_orientation,
            sample_from_corpus=args.sample_from_corpus, probes_from=args.probes_from,
            M_samples=args.M_samples, seed=args.seed, alpha=args.alpha, B=args.B,
            grid_min=args.grid_min, grid_max=args.grid_max, exact_size=args.exact_size,
            force=args.force,
        )
    elif args.cmd == "ingest":
        ingest_mod.run_ingest(
            store, args.platform, args.pool_version, tars=args.tars, csvs=args.csvs,
            workers=args.workers, input_orientation=args.input_orientation,
            rscript=args.rscript, max_nonfinite_fraction=args.max_nonfinite_fraction,
            keep_raw=args.keep_raw, compute_profiles=not args.no_profiles,
        )
    elif args.cmd == "assemble":
        assemble_mod.run_assemble(
            store, args.platform, args.pool_version, args.run_id,
            fdr_alpha=args.fdr_alpha, generate_html=args.generate_html,
            run_explainers=args.run_explainers, k=args.k, probe2gene=args.probe2gene,
            fetch_geo=args.fetch_geo, output_html=args.output_html,
            major_overlap_coefficient=args.major_overlap_coefficient,
        )
    elif args.cmd == "update":
        assemble_mod.run_update(
            store, args.platform, args.pool_version, args.run_id,
            tars=args.tars, csvs=args.csvs, workers=args.workers,
            input_orientation=args.input_orientation, rscript=args.rscript,
            fdr_alpha=args.fdr_alpha, generate_html=args.generate_html,
            run_explainers=args.run_explainers, k=args.k, probe2gene=args.probe2gene,
            major_overlap_coefficient=args.major_overlap_coefficient,
        )


if __name__ == "__main__":
    main()
