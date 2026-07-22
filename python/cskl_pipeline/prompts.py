from __future__ import annotations

import collections
import time
from pathlib import Path

import pandas as pd
import requests

from .io import edge_id, edge_id_variants, output_path, read_json


def local_rank_edges(edges_df: pd.DataFrame, *, q_threshold: float = 0.05) -> pd.DataFrame:
    sig_edges = edges_df[edges_df["q_value"] <= q_threshold].copy()
    sig_edges = sig_edges.sort_values(["q_value", "cSKL"]).reset_index(drop=True)

    node_edges: dict[str, list[tuple[str, float, float, int]]] = collections.defaultdict(list)
    for idx, row in sig_edges.iterrows():
        a = str(row["Dataset_A"])
        b = str(row["Dataset_B"])
        q = float(row["q_value"])
        c = float(row["cSKL"])
        node_edges[a].append((b, q, c, idx))
        node_edges[b].append((a, q, c, idx))

    ranks: dict[int, int] = {}
    for _, rows in node_edges.items():
        rows.sort(key=lambda item: (item[1], item[2]))
        for local_rank, item in enumerate(rows, 1):
            edge_idx = item[3]
            ranks[edge_idx] = min(local_rank, ranks.get(edge_idx, local_rank))

    sig_edges["local_rank"] = [ranks.get(i, 999999) for i in range(len(sig_edges))]
    return sig_edges


def extract_prompt_pairs(data_dir: Path | str, *, local_rank: int = 1) -> list[dict]:
    root = Path(data_dir)
    edges = pd.read_csv(output_path(root, "network_edges"), sep="\t")
    explainers = read_json(output_path(root, "edge_explainers"), required=False, default={})

    ranked = local_rank_edges(edges)
    ranked = ranked[ranked["local_rank"] == local_rank]

    pairs: list[dict] = []
    for _, row in ranked.iterrows():
        a = str(row["Dataset_A"])
        b = str(row["Dataset_B"])
        exp = explainers.get(edge_id(a, b)) or explainers.get(edge_id(b, a)) or {}
        features = exp.get("similar_features", [])
        normalized = []
        for feature in features:
            if isinstance(feature, dict):
                normalized.append(feature)
            else:
                normalized.append({"gene": str(feature), "score": 0.0})
        normalized.sort(key=lambda item: item.get("score", 0.0), reverse=True)

        seen: set[str] = set()
        genes: list[str] = []
        for feature in normalized:
            gene = str(feature.get("gene", "")).strip()
            if gene and gene not in seen:
                seen.add(gene)
                genes.append(gene)

        pairs.append({"dataset_A": a, "dataset_B": b, "top_genes": genes})
    return pairs


def fetch_geo_summary(gse_id: str) -> dict[str, str]:
    url = f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={gse_id}"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:
        raise ImportError("Prompt generation with live GEO scraping requires beautifulsoup4.") from exc

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        geo_data = {"Title": "N/A", "Summary": "N/A", "Overall_Design": "N/A"}
        for tr in soup.find_all("tr", valign="top"):
            tds = tr.find_all("td")
            if len(tds) >= 2:
                label = tds[0].get_text(strip=True).lower()
                content = tds[1].get_text(strip=True)
                if "title" in label and geo_data["Title"] == "N/A":
                    geo_data["Title"] = content
                elif "summary" in label:
                    geo_data["Summary"] = content
                elif "overall design" in label:
                    geo_data["Overall_Design"] = content
        return geo_data
    except Exception as exc:
        print(f"  [!] Error fetching GEO {gse_id}: {exc}")
        return {"Title": "Error", "Summary": "Error", "Overall_Design": "Error"}


def fetch_gene_summary(gene_symbol: str) -> str:
    if "_" in gene_symbol and "at" in gene_symbol:
        return f"{gene_symbol}: Microarray probe (no direct gene summary)."

    url = f"https://mygene.info/v3/query?q=symbol:{gene_symbol}&fields=summary,name&species=human"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        if data.get("hits"):
            hit = data["hits"][0]
            name = hit.get("name", gene_symbol)
            summary = hit.get("summary", "No biological summary available in database.")
            return f"{name}: {summary}"
        return f"{gene_symbol}: Gene not found or summary unavailable."
    except Exception as exc:
        print(f"  [!] Error fetching gene {gene_symbol}: {exc}")
        return f"{gene_symbol}: Error fetching data."


def generate_llm_prompt(pair_data: dict, *, geo_cache: dict | None = None) -> str:
    gse_a = pair_data["dataset_A"]
    gse_b = pair_data["dataset_B"]
    top_genes = pair_data["top_genes"]

    if geo_cache and gse_a in geo_cache:
        geo_a = {
            "Title": geo_cache[gse_a].get("title", "N/A"),
            "Summary": geo_cache[gse_a].get("summary", "N/A"),
            "Overall_Design": geo_cache[gse_a].get("design", "N/A"),
        }
    else:
        print(f"Fetching GEO data for {gse_a}...")
        geo_a = fetch_geo_summary(gse_a)
        time.sleep(1)

    if geo_cache and gse_b in geo_cache:
        geo_b = {
            "Title": geo_cache[gse_b].get("title", "N/A"),
            "Summary": geo_cache[gse_b].get("summary", "N/A"),
            "Overall_Design": geo_cache[gse_b].get("design", "N/A"),
        }
    else:
        print(f"Fetching GEO data for {gse_b}...")
        geo_b = fetch_geo_summary(gse_b)
        time.sleep(1)

    gene_contexts = []
    for gene in top_genes:
        gene_contexts.append(f"- **{gene}**: {fetch_gene_summary(gene)}")
        time.sleep(0.5)

    genes_text = "\n".join(gene_contexts)
    return f"""You are an expert computational biologist and bioinformatician.
We have developed a novel data-driven statistical method (C-SKL) that compares the multidimensional covariance structures of transcriptomic datasets. This method has identified a highly significant, non-trivial molecular similarity between the following two datasets:

### Dataset 1: {gse_a}
* **Title:** {geo_a['Title']}
* **Summary:** {geo_a['Summary']}
* **Design:** {geo_a['Overall_Design']}

### Dataset 2: {gse_b}
* **Title:** {geo_b['Title']}
* **Summary:** {geo_b['Summary']}
* **Design:** {geo_b['Overall_Design']}

### Molecular Drivers
Our feature attribution algorithm identified the following unique features/genes as the primary drivers of this statistical similarity, listed in descending order of correlation importance:
{genes_text}

### Task
Based on the dataset descriptions and the biological functions of the driving genes, please analyze this similarity.
1. Does this molecular similarity make biological sense? Explain the potential shared pathways, cell types, or mechanisms.
2. Classify this similarity into EXACTLY ONE of the following predefined categories:
   * [Expected Clinical Similarity]
   * [Tissue/Platform Artifact]
   * [Shared Immune/Inflammatory Response]
   * [Novel Biological Link]
   * [Unknown/Inconclusive]

Provide your analysis in a concise, structured format. End your response with the exact classification tag in brackets on a new line."""
