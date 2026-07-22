from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

from .io import output_path, read_json, write_json


def fetch_geo_metadata(
    gse_ids: list[str],
    *,
    delay_seconds: float = 0.4,
    timeout_seconds: float = 20.0,
) -> dict[str, dict[str, str]]:
    """Fetch GEO titles and summaries through NCBI E-utilities."""
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    metadata: dict[str, dict[str, str]] = {}

    print(f"Fetching metadata for {len(gse_ids)} datasets from NCBI...")
    for gse in gse_ids:
        try:
            search_url = (
                f"{base_url}esearch.fcgi?db=gds&term={gse}[ACCN]+AND+gse[ETYP]"
                "&retmode=json"
            )
            search_res = requests.get(search_url, timeout=timeout_seconds).json()
            id_list = search_res.get("esearchresult", {}).get("idlist", [])
            if not id_list:
                print(f"  [Warning] Could not find {gse} in GEO DataSets.")
                continue

            internal_id = id_list[0]
            sum_url = f"{base_url}esummary.fcgi?db=gds&id={internal_id}&retmode=json"
            sum_res = requests.get(sum_url, timeout=timeout_seconds).json()
            docsum = sum_res.get("result", {}).get(internal_id, {})
            metadata[gse] = {
                "title": docsum.get("title", "No title available."),
                "summary": docsum.get("summary", "No summary available."),
            }
            print(f"  [Success] Fetched {gse}")
        except Exception as exc:
            print(f"  [Error] Failed on {gse}: {exc}")

        time.sleep(delay_seconds)

    return metadata


def collect_gse_ids(data_dir: Path | str) -> list[str]:
    root = Path(data_dir)
    gse_set: set[str] = set()

    meta_file = output_path(root, "pca_meta")
    if meta_file.exists():
        pca_meta = read_json(meta_file)
        gse_set.update(str(k) for k in pca_meta.keys())

    edges_file = output_path(root, "network_edges")
    if edges_file.exists():
        edges = pd.read_csv(edges_file, sep="\t")
        if {"Dataset_A", "Dataset_B"} <= set(edges.columns):
            gse_set.update(edges["Dataset_A"].astype(str))
            gse_set.update(edges["Dataset_B"].astype(str))

    explainers_file = output_path(root, "edge_explainers")
    if explainers_file.exists():
        explainers = read_json(explainers_file, required=False, default={})
        for edge_key in explainers.keys():
            parts = str(edge_key).split("_")
            if len(parts) == 2:
                gse_set.update(parts)

    return sorted(gse_set)


def write_geo_descriptions(data_dir: Path | str) -> dict[str, dict[str, str]]:
    root = Path(data_dir)
    out_file = output_path(root, "geo_descriptions")
    existing = read_json(out_file, required=False, default={})

    gse_ids = collect_gse_ids(root)
    missing = [gse for gse in gse_ids if gse not in existing]
    if not gse_ids:
        raise ValueError(f"No GSE IDs found in pipeline outputs under {root}")
    if missing:
        fetched = fetch_geo_metadata(missing)
        existing.update(fetched)
    else:
        print("GEO descriptions already cover all known datasets.")

    write_json(out_file, existing)
    print(f"Wrote GEO descriptions to: {out_file}")
    return existing

