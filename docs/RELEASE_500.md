# Frozen GPL570 500-study release

This document describes the graph artifact shipped with C-SKL Atlas. It records
what was computed, what passed validation, and which limits must remain visible
when results are interpreted or cited.

## Identity

| Field | Value |
| --- | --- |
| Stratum | `GPL570:global` |
| Snapshot | `snapshot_ee201ff2e0991ea8fdb7bcad` |
| Calibration | `cal_2472699c2e3443f495d8154a` |
| SPECTER2 release | `text_261c3bdde95fa1e35d254ecd` |
| Layout | `igraph-1.0.0:fr-collision-v2:1c669b96524076c8` |
| Policy SHA-256 | `1c669b96524076c891e18fe2dbd2ca481d0dff327ca49e2c187fc7981f7d5e36` |
| Static graph SHA-256 | `cfa4361e00c91eb61fd1f34665e13afafa406249fa626db233783fd5e86de804` |
| Published at | 2026-07-21 23:39:34 UTC |

The canonical counts and dependency hashes are machine-readable in
[`app/data/atlas-graph.manifest.json`](../app/data/atlas-graph.manifest.json).

## Corpus and relationships

The source corpus supplied 500 processed GPL570 matrices. One matrix,
`GSE117468`, was quarantined because 99.9% of its values were non-finite. A
separately acquired six-CEL `GSE40082` dataset completed the 500 valid dataset
versions through the RAW normalization path.

- 95,671 dataset-sample appearances resolve to 80,277 unique sample identities.
- All 124,750 unordered dataset pairs have raw c-SKL, p-values, and global
  Benjamini-Hochberg q-values.
- 124,327 pairs contain no literal shared sample and form the independent
  calibration family.
- 423 pairs share samples: 35 exact, 272 major, and 116 minor under the released
  overlap policy.
- 6,621 pairs pass global q <= 0.05; 6,178 pass the independent-family threshold.
- The displayed network contains the union of each node's top 25 globally
  significant neighbors: 5,344 edges.

Overlap is evidence, not a deletion rule. Exact and major overlap edges remain
inspectable, are drawn as dotted lines, and are excluded from independent-
replication claims.

## Additional evidence layers

- SPECTER2 contains a pinned 768-dimensional embedding and cosine score for all
  124,750 pairs. It measures source-text proximity only.
- The GPL570 probe release maps 44,647 probes to 22,167 unique Entrez genes.
- Reactome release 97 is indexed against that explicit array background.
- Four published edges contain checksum-bound B(k)/W(k) trajectories, gene
  mappings, random baselines, and Reactome over-representation results.
- The graph contains 3,759 published annotation assertions. Generated semantic
  labels remain review-required even when an ontology identifier/label pair is
  lexically concordant.

## Layout validation

The release uses seeded Fruchterman-Reingold topology followed by deterministic
collision projection. At the reference 1.5 aspect ratio:

| Measure | Value |
| --- | ---: |
| Iterations | 600 |
| Target minimum separation | 0.0447213595 |
| Observed minimum separation | 0.0392512274 |
| Severe collision pairs | 0 |
| Selected Leiden resolution | 0.5 |
| Communities | 39 |
| Mean membership NMI over 10 seeded runs | 0.993514 |

The browser recomputes view-specific packing for C-SKL topology, anatomical
system, and clinical family views while keeping the versioned release
coordinates as its source. Cluster labels can focus one group at a time, and a
global BH q-value control can narrow the 5,344 published edges without changing
the release or exposing pairs outside the sparse graph.

Color shows anatomy by default and can switch to a broad clinical-family facet;
shape keeps a coarse anatomical context. Clinical families are display-only,
derived from concordant labels, and never assign ICD codes. Unreviewed labels
remain visibly unreviewed. These browser-level views do not alter the frozen
graph checksum.

## Validation status

The operational audit passes: the static artifact, snapshot binding, calibration
family, overlap policy, worker state, and artifact checksums are internally
consistent.

The stricter confirmatory profile intentionally fails the following visible
gates:

1. The frozen calibration uses B=100 bootstraps rather than an exact-size B=500
   null release.
2. 3,472 pairs required boundary clamping to the released null grid.
3. 2,311 core semantic assertions remain unreviewed; no human acceptance record
   is present.
4. Only 4 of 5,344 displayed relationships have attached feature explainers.
5. The frozen ontology audit reports 2,379 blocking candidate findings.
6. The 500 historical GEO records are Series-scoped rather than provenanced to
   each expression matrix's exact GSM cohort.

These gates do not invalidate the software or the exploratory map. They prevent
the artifact from being represented as a confirmatory, curator-approved release.

## Appropriate use

The release supports:

- interface and workflow evaluation;
- hypothesis generation and prioritization;
- inspection of calibrated molecular relationships;
- comparison of molecular and text evidence;
- sample-overlap auditing; and
- reproducible query/export examples.

It does not, by itself, establish biological mechanism, causal direction,
clinical utility, therapeutic actionability, or independent replication.

## Integrity check

From the repository root:

```bash
python -c "import hashlib, pathlib; p=pathlib.Path('app/data/atlas-graph.json'); print(hashlib.sha256(p.read_bytes()).hexdigest())"
```

The result must equal the static graph SHA-256 recorded above and in the
manifest. The Pages build performs the same check before deployment.
