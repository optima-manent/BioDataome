"""Tabular IO with a parquet-preferred, TSV-fallback strategy.

The plan asks for ``*.parquet`` run artifacts. Parquet needs ``pyarrow``; when it
is unavailable (e.g. a bare dev box) we transparently fall back to gzip-compressed
TSV so the pipeline still runs and the validation gates still pass. The chosen
format is discoverable from the file that actually exists on disk.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

try:  # pragma: no cover - availability depends on the environment
    import pyarrow  # noqa: F401

    HAVE_PARQUET = True
except Exception:  # pragma: no cover
    HAVE_PARQUET = False


def table_path(base: Path | str, stem: str) -> Path:
    """Return the on-disk path for a table ``stem`` under ``base``.

    Prefers an existing file; otherwise uses parquet when available, else tsv.gz.
    """
    base = Path(base)
    parquet = base / f"{stem}.parquet"
    tsv = base / f"{stem}.tsv.gz"
    if parquet.exists():
        return parquet
    if tsv.exists():
        return tsv
    return parquet if HAVE_PARQUET else tsv


def write_table(df: pd.DataFrame, base: Path | str, stem: str) -> Path:
    """Atomically write ``df`` as parquet (preferred) or gzip TSV."""
    base = Path(base)
    base.mkdir(parents=True, exist_ok=True)
    if HAVE_PARQUET:
        final = base / f"{stem}.parquet"
        tmp = final.with_suffix(final.suffix + ".tmp")
        df.to_parquet(tmp, index=False)
    else:
        final = base / f"{stem}.tsv.gz"
        tmp = final.with_suffix(final.suffix + ".tmp")
        df.to_csv(tmp, sep="\t", index=False, compression="gzip")
    os.replace(tmp, final)
    return final


def read_table(base: Path | str, stem: str) -> pd.DataFrame:
    """Read a table written by :func:`write_table`."""
    path = table_path(base, stem)
    if not path.exists():
        raise FileNotFoundError(f"No table {stem!r} under {base}")
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, sep="\t", compression="gzip")
