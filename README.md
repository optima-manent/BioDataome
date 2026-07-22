# C-SKL Atlas

[![Verify](https://github.com/optima-manent/BioDataome/actions/workflows/verify.yml/badge.svg)](https://github.com/optima-manent/BioDataome/actions/workflows/verify.yml)
[![Pages](https://github.com/optima-manent/BioDataome/actions/workflows/pages.yml/badge.svg)](https://optima-manent.github.io/BioDataome/)
[![DOI](https://img.shields.io/badge/method-10.1038%2Fs41540--019--0117--0-2f6f68)](https://doi.org/10.1038/s41540-019-0117-0)

C-SKL Atlas is a reproducible pipeline and interactive research workbench for
mapping relationships among biological expression datasets. It combines the
compressed symmetric Kullback-Leibler (c-SKL) method with versioned statistical
calibration, shared-sample detection, scientific-text proximity, gene-level
relationship explanations, pathway hypotheses, and provenance-aware metadata.

The repository contains two closely related deliverables:

- the self-contained Python c-SKL library in [`python/cskl.py`](python/cskl.py),
  including PCA signatures, c-SKL scoring, bootstrap significance, network
  construction, and B(k)/W(k) feature-set explanations; and
- C-SKL Atlas, which adds incremental ingestion, immutable artifacts, recovery,
  release calibration, annotation, graph publication, APIs, and the interactive
  browser workbench around that numerical kernel.

The frozen 500-study GPL570 showcase can be opened at
[optima-manent.github.io/BioDataome](https://optima-manent.github.io/BioDataome/).
It is a completely static GitHub Pages build: graph exploration and export run
in the browser, while server-only synthesis and write operations remain disabled.

## What the Atlas shows

Each node represents one versioned GEO expression dataset. Each published edge
represents a statistically calibrated relationship between two compatible
dataset versions.

- Lower raw c-SKL indicates closer standardized covariance structure.
- Global and independent-family q-values remain separate from raw similarity.
- SPECTER2 is a separate scientific-text proximity channel, not molecular
  validation.
- Shared samples never delete a relationship. Exact or major overlap is shown
  with a dotted edge and cannot be described as independent replication.
- Node color encodes a broad anatomical system, shape encodes disease family,
  and size encodes sample count.
- Selecting an edge exposes calibration, overlap, SPECTER2, B(k)/W(k), Reactome,
  and provenance evidence when those artifacts have been computed.
- Search is a quick metadata filter. The discovery query builder applies
  explicit AND logic across molecular, statistical, semantic, anatomical,
  disease, gene, probe, pathway, and independence constraints.

The map is an evidence index, not a causal model. Similarity may reflect shared
biology, tissue, protocol, batch effects, reused samples, or other confounding.

## Frozen 500-study release

The committed showcase is bound to snapshot
`snapshot_ee201ff2e0991ea8fdb7bcad` and its checksum manifest in
[`app/data/atlas-graph.manifest.json`](app/data/atlas-graph.manifest.json).

| Measure | Released value |
| --- | ---: |
| Dataset nodes | 500 |
| Complete calibrated pair family | 124,750 |
| Published sparse graph edges | 5,344 |
| Pairs with literal shared samples | 423 |
| Independent-family pairs | 124,327 |
| Computed SPECTER2 pair scores | 124,750 |
| Attached B(k)/W(k) edge explanations | 4 |
| Severe layout collisions at the reference viewport | 0 |

The sparse graph is the union of each node's top 25 globally significant
neighbors at q <= 0.05. The complete pair family remains in the release catalog
and is not silently equated with the displayed graph.

This snapshot passes the operational release audit. Its current scientific
limitations are explicit: it uses a frozen B=100 null calibration with boundary
clamping, most semantic labels still require human review, the historical GEO
cache is Series-scoped rather than matrix-cohort-scoped, and full edge-explainer
coverage has not been precomputed. See
[`docs/RELEASE_500.md`](docs/RELEASE_500.md) before using it in confirmatory
analysis.

## Install

Python 3.11 or newer is required. The web workbench requires Node.js 22.13 or
newer and pnpm 11.9.

```bash
git clone https://github.com/optima-manent/BioDataome.git
cd BioDataome
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e ".[dev,graph]"
```

Optional SPECTER2 dependencies are installed separately because they include
PyTorch and model tooling:

```bash
python -m pip install -e ".[specter2]"
```

For the web workbench:

```bash
corepack enable
pnpm install --frozen-lockfile
pnpm dev
```

Copy [`.env.example`](.env.example) to an ignored local environment file only
when an optional external service is needed. No credential is required to use
the c-SKL library or the static showcase.

## Use the c-SKL library

`cskl` expects samples in rows and the same ordered features in columns for all
datasets being compared.

```python
from cskl import cskl, explain_topk, fit_pca_signature

# matrix_a and matrix_b are finite NumPy arrays with shape
# (samples, features); feature_names follows their shared column order.
signature_a = fit_pca_signature(
    matrix_a,
    alpha=0.5,
    feature_names=feature_names,
    rng=1729,
)
signature_b = fit_pca_signature(
    matrix_b,
    alpha=0.5,
    feature_names=feature_names,
    rng=1729,
)

distance = cskl(signature_a, signature_b)  # lower means more similar
feature_indices, details = explain_topk(
    signature_a,
    signature_b,
    k=20,
    mode="B",
    seed=1729,
    return_details=True,
)
```

The module also exports `Pool`, `pair_pvalue_vs_pool`,
`pair_significant_vs_pool`, `bh_qvalues`, `build_dataset_network`, and
`explain_set_topk`. Exact signatures and interpretation rules are documented in
[`docs/SCIENTIFIC_METHOD.md`](docs/SCIENTIFIC_METHOD.md).

## Run the incremental Atlas

Initialize the reference catalog and inspect its state:

```bash
cskl-atlas init
cskl-atlas health
cskl-atlas jobs
```

The control plane exposes explicit commands for GEO discovery and synchronization,
RAW CEL acquisition, scalable-store import, SPECTER2, annotation and ontology
audit, Reactome indexing, edge explanations, graph snapshots, static export, and
release auditing:

```bash
cskl-atlas --help
cskl-scale --help
```

An update computes only new-to-existing K x N and new-to-new K(K-1)/2 raw pairs.
It reuses valid old-to-old work, but creates a new corpus-bound calibration and
multiple-testing release when the comparison family changes. Every reuse is
gated by source, feature-order, normalization, algorithm, and configuration
fingerprints.

See [`docs/OPERATIONS.md`](docs/OPERATIONS.md) for the ordered workflow, failure
states, recovery commands, and release gates.

## Build the GitHub Pages showcase

```bash
pnpm build:pages
pnpm test:pages
```

The static artifact is written to `dist/pages/`. The
[`pages.yml`](.github/workflows/pages.yml) workflow builds and deploys the same
artifact through GitHub Pages. In the repository settings, select **GitHub
Actions** as the Pages source; no generated site files need to be committed.

## Verify

```bash
python -m pytest -ra
ruff check python/cskl_atlas python_tests --exclude python_tests/legacy
python -m compileall -q python
pnpm audit --prod --audit-level moderate
pnpm lint
pnpm typecheck
pnpm test
pnpm test:pages
```

The continuous-integration workflow runs Python 3.11, 3.12, and 3.13, builds a
wheel, checks the installed package, builds both web targets, and exercises the
scientific and release contracts.

## Repository guide

```text
python/cskl.py             Standalone c-SKL numerical library
python/cskl_atlas/         Versioned Atlas control plane, API, and workers
python/cskl_pipeline/      Scalable matrix/signature/calibration pipeline
app/                       Interactive workbench and frozen graph adapter
showcase/                  Static GitHub Pages entry point
resources/releases/gpl570 Frozen GPL570 feature and probe annotation release
config/                    Scientific release policy
python_tests/              Numerical, recovery, security, and integration tests
tests/                     Web, evidence-policy, and Pages artifact tests
docs/                      Method, architecture, data contracts, operations, and usage
```

## Citation

If using this library or the underlying c-SKL methodology, please cite:

> Lakiotaki, K., Georgakopoulos, G., Castanas, E. et al. A data driven approach
> reveals disease similarity on a molecular level. *npj Systems Biology and
> Applications* **5**, 39 (2019).
> https://doi.org/10.1038/s41540-019-0117-0

Machine-readable citation metadata is provided in [`CITATION.cff`](CITATION.cff).

## License

The software is released under the [MIT License](LICENSE). Source datasets,
metadata, ontologies, model weights, and pathway resources remain subject to
their respective providers' terms and licenses.
