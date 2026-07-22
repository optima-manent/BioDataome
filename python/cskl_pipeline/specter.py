from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .io import output_path, read_json, write_json


def make_description(gse_meta: dict) -> str:
    return " ".join(
        filter(
            None,
            [
                gse_meta.get("title", ""),
                gse_meta.get("summary", ""),
                gse_meta.get("design", ""),
            ],
        )
    )


def write_specter2_similarities(
    data_dir: Path | str,
    *,
    model_name: str = "allenai/specter2_aug2023refresh_base",
    top_n: int = 10,
    q_threshold: float = 0.05,
) -> dict[str, list[dict]]:
    root = Path(data_dir)
    geo_file = output_path(root, "geo_descriptions")
    if not geo_file.exists():
        raise FileNotFoundError(f"Missing {geo_file}. Run --fetch-geo before --run-specter2.")

    print("Loading GEO descriptions...")
    geo_meta = read_json(geo_file)
    datasets = list(geo_meta.keys())
    descriptions = [make_description(geo_meta[ds]) for ds in datasets]

    print("Loading SPECTER2 model...")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "SPECTER2 requires sentence-transformers. Install it with "
            "`pip install sentence-transformers`."
        ) from exc

    model = SentenceTransformer(model_name)
    print(f"Embedding {len(datasets)} dataset descriptions...")
    embeddings = model.encode(
        descriptions,
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=True,
    )

    sim_matrix = embeddings @ embeddings.T
    cskl_edges: set[tuple[str, str]] = set()
    edges_file = output_path(root, "network_edges")
    if edges_file.exists():
        df = pd.read_csv(edges_file, sep="\t")
        for _, row in df.iterrows():
            if float(row["q_value"]) <= q_threshold:
                a = str(row["Dataset_A"])
                b = str(row["Dataset_B"])
                cskl_edges.add((a, b))
                cskl_edges.add((b, a))

    results: dict[str, list[dict]] = {}
    for i, ds_a in enumerate(datasets):
        scores = sim_matrix[i]
        top_indices = np.argsort(scores)[::-1][1 : top_n + 1]
        matches = []
        for j in top_indices:
            ds_b = datasets[int(j)]
            matches.append(
                {
                    "dataset": ds_b,
                    "score": float(scores[int(j)]),
                    "is_cskl": (ds_a, ds_b) in cskl_edges,
                    "title": geo_meta[ds_b].get("title", "No Title Available"),
                }
            )
        results[ds_a] = matches

    out_file = output_path(root, "specter2_similarities")
    write_json(out_file, results)
    print(f"Wrote SPECTER2 similarities to: {out_file}")
    return results

