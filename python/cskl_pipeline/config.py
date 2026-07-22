from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class PipelineConfig:
    tars: list[str]
    csvs: list[str]
    outdir: Path
    rscript: str = "Rscript"
    pool_matrix: str | None = None
    alpha: float = 0.5
    B: int = 100
    seed: int = 0
    fdr_alpha: float = 0.05
    min_common_features: int = 1000
    input_orientation: str = "features-by-samples"
    run_explainers: bool = False
    k: int = 10
    probe2gene: str | None = None
    manual_analyses: str | None = None
    fetch_geo: bool = False
    run_specter2: bool = False
    specter2_model: str = "allenai/specter2_aug2023refresh_base"
    specter2_top_n: int = 10
    generate_html: bool = False
    output_html: str | None = None

