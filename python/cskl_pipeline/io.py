from __future__ import annotations

import glob
import json
import re
import shutil
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


PIPELINE_OUTPUTS = {
    "pca_meta": "pca_meta.json",
    "cskl_matrix": "cskl_matrix.tsv",
    "cskl_similarity": "cskl_similarity_1_over_1_plus.tsv",
    "network_edges": "cskl_network_edges.tsv",
    "edge_explainers": "edge_explainers.json",
    "geo_descriptions": "geo_descriptions.json",
    "specter2_similarities": "specter2_similarities.json",
    "html": "interactive_cskl_network.html",
}


def safe_name_from_file(file_path: Path | str) -> str:
    path = Path(file_path)
    match = re.match(r"(GSE\d+)", path.name, re.IGNORECASE)
    return match.group(1).upper() if match else path.stem.replace("_RAW", "").upper()


def expand_paths(paths: Iterable[str]) -> list[str]:
    expanded: list[str] = []
    for item in paths:
        matches = glob.glob(item)
        expanded.extend(matches if matches else [item])
    return expanded


def detect_separator(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".csv") or name.endswith(".csv.gz"):
        return ","
    if (
        name.endswith(".tsv")
        or name.endswith(".tsv.gz")
        or name.endswith(".txt")
        or name.endswith(".txt.gz")
    ):
        return "\t"
    return "\t"


def load_expr(expr_path: Path | str, *, orientation: str = "features-by-samples") -> pd.DataFrame:
    """Load an expression matrix and return features x samples by default.

    The main pipeline expects rows as features and columns as samples. If a file
    was produced in samples x features form, pass orientation="samples-by-features".
    """
    path = Path(expr_path)
    if not path.exists():
        raise FileNotFoundError(f"Expression matrix not found: {path}")

    name = path.name.lower()
    compression = "gzip" if name.endswith(".gz") else None
    df = pd.read_csv(
        path,
        sep=detect_separator(path),
        compression=compression,
        index_col=0,
    )

    df.index = df.index.astype(str)
    df = df.apply(pd.to_numeric, errors="coerce")

    if orientation == "features-by-samples":
        return df
    if orientation == "samples-by-features":
        out = df.T
        out.index = out.index.astype(str)
        return out

    raise ValueError(
        "orientation must be 'features-by-samples' or 'samples-by-features'. "
        f"Got {orientation!r}."
    )


def compute_global_good_features(expr: dict[str, pd.DataFrame], common: list[str]) -> list[str]:
    """Find features that are finite across every loaded matrix."""
    good_all = None
    for df in expr.values():
        X = df.loc[common].to_numpy(dtype=np.float64)
        finite = np.isfinite(X).all(axis=1)
        good_all = finite if good_all is None else (good_all & finite)

    if good_all is None:
        return []
    return [feature for feature, ok in zip(common, good_all) if ok]


def read_json(path: Path | str, *, required: bool = True, default: Any | None = None) -> Any:
    p = Path(path)
    if not p.exists():
        if required:
            raise FileNotFoundError(f"Missing required JSON file: {p}")
        return default
    with p.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: Path | str, data: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


def backup_file(path: Path | str) -> Path | None:
    p = Path(path)
    if not p.exists():
        return None
    backup = p.with_name(p.name + ".bak")
    counter = 1
    while backup.exists():
        backup = p.with_name(f"{p.name}.bak{counter}")
        counter += 1
    shutil.copy2(p, backup)
    return backup


def edge_id(a: str, b: str) -> str:
    return f"{a}_{b}"


def edge_id_variants(a: str, b: str) -> tuple[str, str]:
    return edge_id(a, b), edge_id(b, a)


def normalized_edge(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted((str(a), str(b))))


def output_path(outdir: Path | str, key: str) -> Path:
    if key not in PIPELINE_OUTPUTS:
        raise KeyError(f"Unknown pipeline output key: {key}")
    return Path(outdir) / PIPELINE_OUTPUTS[key]
