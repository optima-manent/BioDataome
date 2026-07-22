from __future__ import annotations

from pathlib import Path

from .io import output_path


def generate_graph_html(
    data_dir: Path | str,
    *,
    output_html: Path | str | None = None,
    q_threshold: float = 0.05,
) -> Path:
    """Generate the static network HTML through the compatibility app module."""
    import app

    root = Path(data_dir)
    out = Path(output_html) if output_html else output_path(root, "html")
    edges_df, pca_meta, explainers, geo_meta, specter_data = app.load_data(root)
    (
        nodes_data,
        edges_data,
        ui_metadata,
        ui_explanations,
        total_edges,
        max_local,
    ) = app.build_network_data(edges_df, pca_meta, explainers, geo_meta, q_threshold=q_threshold)
    app.generate_html(
        nodes_data,
        edges_data,
        ui_metadata,
        ui_explanations,
        total_edges,
        max_local,
        specter_data,
        output_html=out,
    )
    return out

