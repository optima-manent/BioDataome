from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from cskl_pipeline.graph_html import generate_graph_html
from cskl_pipeline.io import edge_id_variants, load_expr, normalized_edge, safe_name_from_file


def test_safe_name_from_file() -> None:
    assert safe_name_from_file(Path("GSE35570_RAW.tar")) == "GSE35570"
    assert safe_name_from_file(Path("GSE149959-GPL570_series_matrix.csv")) == "GSE149959"
    assert safe_name_from_file(Path("custom_matrix.tsv")) == "CUSTOM_MATRIX"


def test_load_expr_csv_and_tsv_features_by_samples(tmp_path: Path) -> None:
    csv_path = tmp_path / "GSE1.csv"
    tsv_path = tmp_path / "GSE2.tsv"
    csv_path.write_text("feature,s1,s2\nf1,1,2\nf2,3,4\n", encoding="utf-8")
    tsv_path.write_text("feature\ts1\ts2\nf1\t1\t2\nf2\t3\t4\n", encoding="utf-8")

    csv_df = load_expr(csv_path)
    tsv_df = load_expr(tsv_path)

    assert list(csv_df.index) == ["f1", "f2"]
    assert list(tsv_df.columns) == ["s1", "s2"]
    assert csv_df.loc["f2", "s2"] == 4


def test_load_expr_samples_by_features(tmp_path: Path) -> None:
    csv_path = tmp_path / "GSE1.csv"
    csv_path.write_text("sample_id,f1,f2\ns1,1,3\ns2,2,4\n", encoding="utf-8")

    df = load_expr(csv_path, orientation="samples-by-features")

    assert list(df.index) == ["f1", "f2"]
    assert list(df.columns) == ["s1", "s2"]
    assert df.loc["f2", "s2"] == 4


def test_edge_id_helpers() -> None:
    assert edge_id_variants("GSE1", "GSE2") == ("GSE1_GSE2", "GSE2_GSE1")
    assert normalized_edge("GSE2", "GSE1") == ("GSE1", "GSE2")


@pytest.mark.skip(
    reason="The unsafe legacy static app.py renderer is intentionally retired in C-SKL Atlas."
)
def test_graph_generator_smoke(tmp_path: Path) -> None:
    pd.DataFrame(
        [
            {
                "Dataset_A": "GSE1",
                "Dataset_B": "GSE2",
                "cSKL": 0.123,
                "p_value": 0.01,
                "q_value": 0.02,
            }
        ]
    ).to_csv(tmp_path / "cskl_network_edges.tsv", sep="\t", index=False)
    (tmp_path / "pca_meta.json").write_text(
        json.dumps(
            {
                "GSE1": {"n_samples": 2, "n_features_used": 3, "c_components": 1, "alpha": 0.5},
                "GSE2": {"n_samples": 2, "n_features_used": 3, "c_components": 1, "alpha": 0.5},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "edge_explainers.json").write_text(
        json.dumps(
            {
                "GSE1_GSE2": {
                    "similar_features": [{"gene": "GENE1", "score": 1.0}],
                    "dissimilar_features": [{"gene": "GENE2", "score": 0.5}],
                }
            }
        ),
        encoding="utf-8",
    )

    out_html = generate_graph_html(tmp_path, output_html=tmp_path / "network.html")

    assert out_html.exists()
    html = out_html.read_text(encoding="utf-8")
    assert "c-SKL Network" in html
    assert "GSE1_GSE2" in html
