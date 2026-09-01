# Data contracts, provenance, and invalidation

This document defines the identities, immutable artifacts, relational facts, and
release boundaries used by C-SKL Atlas. Scientific interpretation is specified
in [SCIENTIFIC_METHOD.md](./SCIENTIFIC_METHOD.md).

## Contract principles

1. **Dataset identity is not dataset version identity.** A GEO accession can
   acquire a new source, normalization, signature, annotation, or policy result.
2. **Raw numerical facts are separate from release-relative facts.** A C-SKL
   score does not contain a p-value, q-value, overlap policy, or graph layout.
3. **Reuse is fingerprint-based, never file-presence-based.** Every immutable
   artifact is bound to all dependencies that can change its value.
4. **Publication is snapshot-based.** A reader sees a named, internally
   consistent calibration and policy release, not partially updated tables.
5. **Unknown is data.** Missing metadata and unresolved ontology fields are
   represented explicitly rather than guessed.
6. **Generated text never overwrites source evidence.** GEO assertions,
   computational evidence, and AI hypotheses remain separate and traceable.

## Current storage architecture

**Implemented reference architecture:**

- immutable payloads in the local content-addressed
  [`ArtifactStore`](../python/cskl_atlas/artifact_store.py);
- transactional relational state in the SQLite
  [`Catalog`](../python/cskl_atlas/catalog.py), using WAL, full synchronous
  commits, foreign keys, and disk-backed temporary operations; and
- FastAPI read/operations endpoints in [`api.py`](../python/cskl_atlas/api.py).

The schema is intentionally PostgreSQL-friendly, but PostgreSQL and remote
object-storage adapters are **planned**. The frontend does not use D1 as a
scientific catalog.

## Provenance of the numerical core

[`CORE_ORIGIN.json`](../python/cskl_atlas/CORE_ORIGIN.json) is the checked origin
record for the vendored scientific kernel and validated fast path.

The kernel has two intentionally retained byte hashes:

- vendored LF file `python/cskl.py`:
  `cae59ce12a91546d938074840c62f7a6a50fe14a9dd260c6d1a203847a0c8d7b`;
- upstream CRLF source:
  `e88193035ec77c38ac674b14904c5b290109e38268e3b6af525a5f928785023f`.

They are text-equivalent after CRLF-to-LF normalization; the record preserves
both byte identities rather than pretending the copied bytes are identical.
The validated fast-path hash is also recorded. Release code must fingerprint the
actual bytes it executes and bind that digest through the `core` component
fingerprint.

## Identity hierarchy

### Dataset and version

| Entity | Implemented identity | Meaning |
|---|---|---|
| `dataset_uid` | Stable hash of accession, platform, and cohort | Logical study/platform/cohort identity. |
| `version_id` | Stable hash including dataset UID, source revision/hash, normalized hash, signature hash, feature hash, and config hash | Immutable analysis version of that dataset. |
| `current_version_id` | Transactional pointer on `datasets` | Checksum-gated version served as current; an unpromoted ready row is only a candidate and older current versions become `superseded`. |

`dataset_versions.status` is `ready`, `invalid`, or `superseded`. A new source or
dependency creates a new version; it must not mutate the old row.

The catalog currently uses a caller-supplied `cohort` string, defaulting to
`series`. Its biological meaning must be standardized before production import.

### Comparison stratum

A scientifically valid raw pair requires the same:

- ordered feature-universe hash;
- feature count and order;
- \(\alpha\);
- C-SKL core/config fingerprint; and
- compatible normalization/QC policy.

`DatasetSignature` enforces feature-universe, count, order when available, and
\(\alpha\) before streaming a score. In persistent data, `feature_hash` and
`algorithm_hash` are the primary pair-stratum keys. A free-form release
`stratum` label is useful for serving but is not a substitute for checking these
hashes.

## Expression and signature contract

### Normalized expression matrix

Logical artifact type: `normalized-expression`.

Required semantics:

- shape `(m_samples, n_features)`;
- rows in recorded sample order;
- columns in the exact frozen feature order;
- numeric dtype and serialization recorded in the artifact manifest;
- finite values after QC/imputation;
- source, normalization, QC, ordered feature space, configuration, and seed
  included in `AnalysisDependencies`; and
- payload checksum verified before use.

The generic artifact store supports this type. The raw-CEL ingest path now
writes it through SCAN.UPC, frozen-probe alignment and QC; GSE40082 is the real
end-to-end staging proof. Calendar scheduling remains deployment-specific.

### `PCASignature`

Logical artifact type: `dataset-signature`.

| Field | Contract |
|---|---|
| `P` | Finite float matrix `(n_features, c_components)` with orthonormal columns. |
| `lam` | Finite non-negative vector `(c_components,)`, normalized to sum `alpha * n_features`. |
| `n_features` | Exact ordered feature count. |
| `m_samples` | Number of source matrix rows, at least two for fitting. |
| `alpha` | Finite value strictly between zero and one. |
| `feature_names` | Optional ordered IDs of length `n_features`; the persistent `feature_hash` remains mandatory. |

The isotropic residual \(1-\alpha\) is implicit. `PCASignature` validates
orthonormality and renormalizes `lam`; therefore loading a signature is not a
license to omit its upstream eigenvalue and code fingerprints.

The artifact metadata should include component count, numeric dtype, kernel
hash, normalization/QC versions, and any constant-feature/noise policy.

## Source, sample, and overlap contracts

### Source fingerprint

`SourceFingerprint` contains:

- exact content SHA-256 and byte size;
- stable source ID;
- optional source revision; and
- optional media type.

HTTP URL or filename alone is never sufficient for reuse.

### Sample identity

Every catalog sample requires at least one of:

- normalized uppercase GSM accession; or
- normalized lowercase aligned-expression hash.

The association table records dataset version and optional sample position.
GSM and expression hash are evidence channels, not interchangeable biological
metadata.

The implemented legacy-compatible expression identity is SHA-1 over a finite,
aligned float64 sample row rounded to ten decimals. Changing feature order,
rounding, dtype, imputation, or hashing algorithm creates a new evidence hash.

### Pair overlap evidence

Logical fields:

| Field | Contract |
|---|---|
| endpoints | Two distinct, canonically ordered dataset version IDs. |
| `evidence_hash` | Fingerprint of sample identities plus matching/classification policy. |
| shared evidence | Shared GSM IDs, expression hashes, and matched sample count; a bipartite match counts a physical sample once. |
| `fraction_a`, `fraction_b` | Matched count divided by each endpoint sample count. |
| `jaccard` | `shared / (a + b - shared)`. |
| `overlap_coefficient` | `shared / min(a, b)`. |
| `classification` | `none`, `minor`, `major`, or `exact`. |
| `discovery_excluded` | Policy result; never a dataset-deletion instruction. |

The major-overlap display threshold is 0.5. In the current scientific release,
every literal shared sample excludes the pair from an independent-replication
claim; only major and exact overlap is dotted, while minor overlap remains solid
and explicitly qualified. A published graph must bind the complete scientific
and display policy through `policy_hash`; a numeric threshold in the rendering
layer is not authoritative.

The catalog's `shared_samples_json` is a compact serving/audit field. Full
modality-specific matching evidence should also be persisted as an immutable
artifact when the production worker is implemented.

## Raw pair-score contract

Logical fact: one row in `pair_scores`.

| Field | Contract |
|---|---|
| `pair_id` | Stable ID of ordered endpoints plus `algorithm_hash`. |
| `version_a`, `version_b` | Distinct IDs with `version_a < version_b`. |
| `algorithm_hash` | Complete score-algorithm/config identity, including kernel and \(\alpha\)-relevant settings. |
| `cskl` | Finite, non-negative raw C-SKL value. |
| `created_at` | Audit timestamp, not part of numerical identity. |

There is at most one fact for an endpoint pair and algorithm hash. A raw row
must not contain a p-value, q-value, overlap classification, annotation, layout,
or generated explanation.

The incremental iterator emits `RawCSKLPair` records with endpoint IDs,
feature-universe ID, \(\alpha\), score, and pair kind (`new-existing` or
`new-new`). Persistence canonicalizes endpoint order.

## Null-profile and calibration contracts

### Null profile

Logical artifact type: `null-profile`.

`NullProfileArtifact` contains:

- dataset `version_id` and bound `signature_hash`;
- `pool_hash` and `feature_hash`;
- \(\alpha\) and bootstrap count;
- strictly increasing one-dimensional sample-size `grid`; and
- equal-length finite `mu` and non-negative `sigma` arrays.

At lookup, sigma has a numerical floor of `1e-12`. A profile from another pool,
feature space, \(\alpha\), version, or signature is rejected. Grid construction,
seed, bootstrap algorithm, normal-fit implementation, and clamp policy must be
included in the artifact's extra dependency fingerprints.

### Calibration release

| Field | Contract |
|---|---|
| `calibration_id` | Stable ID of stratum, mode, pool hash, and parameter hash. |
| `mode` | `exact` or `frozen`; semantics are defined in SCIENTIFIC_METHOD.md. |
| `pool_hash` | Exact background release identity. |
| `parameter_hash` | Grid, bootstrap count/seed, profile code, pair-family rule, and related settings. |
| `manifest_json` | Release membership and provenance; callers must make it complete. |
| `status` | `staging`, `calibrated`, `published`, `failed`, or `superseded`. |

`calibrated_edges` is keyed by `(calibration_id, pair_id)`. `p_value` is finite
in `[0,1]`; `q_value` is null until BH finalization and then finite in `[0,1]`.
The q-value has no meaning without its calibration ID.

The calibration worker freezes the algorithm/pair family, validates named
null-profile artifact checksums and pool bindings, resumes missing p-values in
bounded batches, and invokes disk-backed global BH and C-SKL percentile
ordering. It consumes imported profiles; it does not construct pools/profiles.

## Graph snapshot contract

A graph snapshot is the publication boundary, not an ad hoc query over mutable
staging state.

| Field | Contract |
|---|---|
| `snapshot_id` | Stable ID of calibration, stratum, policy, layout version, and manifest URI. |
| `calibration_id` | Finalized release providing p/q-values. |
| `policy_hash` | Overlap, q-threshold family, aggregation, and discovery rules. |
| `layout_version` | Identity of layout/community algorithm and parameters. |
| `layout_quality` | Algorithm, aspect ratio, automatic target/observed minimum separation, iteration count, and severe collision-pair count. Collision-v2 snapshots cannot publish with a nonzero severe count. |
| membership | Explicit dataset version IDs with optional x/y/community. |
| `status` | `staging`, `published`, `superseded`, or `failed`. |

Publication transactionally supersedes the previous published snapshot for the
same stratum and moves the `current_snapshot:<stratum>` pointer. The graph API
joins only nodes in that snapshot and edges from its calibration. Its
`independent_only` filter excludes overlap evidence marked
`discovery_excluded`.

Snapshot layout/community computation and manifest production are implemented
with q/overlap filtering, union-top-k sparsification, seeded Leiden stability,
deterministic collision-aware layout, content-addressed manifests, validation,
publication, and audited rollback. The policy is validated on the real
500-dataset corpus; 7,000-node acceptance still requires the planned WebGL and
community-aggregation benchmark.

## Content-addressed artifact contract

### Complete dependency identity

`AnalysisDependencies` is canonical JSON containing:

- source content fingerprint;
- normalization component name, version, code SHA-256, and config SHA-256;
- C-SKL core component name, version, code SHA-256, and config SHA-256;
- \(\alpha\) and deterministic seed;
- order-sensitive feature-space SHA-256;
- QC component name, version, code SHA-256, and config SHA-256; and
- sorted, uniquely named extra fingerprints such as pool, grid, bootstrap,
  mapping, ontology, or annotation releases.

Its SHA-256 digest is the `dependency_id`. Canonicalization normalizes strings,
sorts object keys, rejects ambiguous/non-finite JSON values, and distinguishes
ordered feature lists.

### Manifest and payload

Every bundle contains `manifest.json` and at least one file below `files/`.
The manifest records:

- schema version and validated artifact type;
- full dependency contract and matching `dependency_id`;
- sorted payload path, SHA-256, byte size, and optional media type records; and
- canonical metadata.

`artifact_id` is the SHA-256 of artifact type, dependency ID, file records,
metadata, and manifest schema. Storage is sharded as
`objects/sha256/<first-two-hex>/<artifact_id>/`.

The implemented store:

- writes privately to a staging directory and publishes with atomic rename;
- never overwrites an existing content address;
- validates manifest identity, dependency expectation, payload sizes/hashes,
  safe relative paths, and absence of unmanifested files/directories;
- rejects symlinks and path traversal; and
- cleans aborted staging bundles.

Catalog `artifacts` rows are indexes/pointers to these immutable bundles, not a
replacement for manifest validation.

Artifact type strings are syntactically validated but not currently restricted
to a central enum. The recommended initial logical types are
`source-metadata`, `normalized-expression`, `dataset-signature`,
`sample-overlap`, `null-profile`, `pair-explanation`, `set-explanation`,
`annotation-release`, and `graph-snapshot-manifest`. Production workers must
standardize this vocabulary before publication.

## Annotation and generated-explanation contracts

### GEO annotations

Implemented annotation fields are multi-valued and explicitly unknown-capable:
organism, tissue, disease, cell type, assay, intervention, experimental system,
and study design. Supported candidate ontology namespaces are NCBITaxon, EFO,
UBERON, MONDO, CL, OBI, and CHEBI.

Every known assertion carries:

- ontology namespace, candidate ID, and label;
- one or more exact evidence spans with source field and character offsets; and
- provenance: `geo_structured`, `llm_candidate`, or `human_verified`.

Generated candidates cannot overwrite a known deterministic GEO field.
For unresolved fields, the model extracts an evidence-grounded surface label;
any namespace or CURIE returned by the model is ignored. The field selects the
allowed ontology or ontologies, and the official OLS API resolves the label
from frozen, content-addressed responses. A generated assertion is retained
only when exactly one non-obsolete term has an exact normalized canonical-label
or synonym match. Ambiguous or unresolved labels remain explicitly unknown.

`llm_candidate` therefore means that the identifier has passed deterministic
lexical OLS concordance, not that the annotation is biologically correct or
curator-accepted. Generated assertions remain `unreviewed` until a curator
accepts or rejects them. Each annotation artifact records the resolver version,
decision status, canonical label, allowed namespaces, and OLS response hashes.

The workbench's **clinical family** is a separate display facet. It places
concordant disease labels into broad, versioned browsing groups and preserves an
explicit unreviewed state when that evidence is not available. It does not
assign an ICD code, replace the underlying disease assertion, or change the
scientific graph. Node shapes are likewise a coarse anatomical display aid, not
an ontology assertion.

### OpenRouter completion provenance

Every structured completion records model, endpoint, response ID, prompt
template version, schema name, request-payload hash, schema hash, response hash,
per-source hashes, timestamp, and the enforced ZDR/strict-schema flags. Models
must be explicitly allowlisted for each deployment.

### Explanation packet

An explanation request contains a selection ID, unique evidence items, source
URIs/versions where available, and warnings. Allowed evidence kinds include
C-SKL, GEO, SPECTER2, sample overlap, gene explainer, pathway, quality, and
literature.

The response separates:

- `evidence_summary`: direct statements citing packet evidence IDs;
- `hypotheses`: interpretations with citations, alternative explanations, and
  validation steps; and
- `limitations`.

Unknown evidence IDs are rejected. This is a provenance contract, not a truth
guarantee. Durable artifact/catalog wiring for production AI runs is
**implemented only as a foundation**.

## Dependency invalidation matrix

| Change | Must invalidate/rebuild | Does not inherently invalidate |
|---|---|---|
| Source expression bytes or source revision | normalized matrix, signature, sample expression hashes, incident raw pairs, null profile, downstream calibration/snapshot | unrelated dataset versions |
| Normalization code/config | normalized matrix and everything numerically downstream | original source artifact |
| QC/imputation code/config or seed | normalized/QC result, signature, expression hashes, incident pairs, profiles, release | GEO source metadata |
| Ordered feature list/order | entire affected signature/pair/profile stratum and release | other feature strata |
| Core code/config or \(\alpha\) | signatures, raw pairs, null profiles, calibration, snapshots | source and deterministic GEO assertions |
| One endpoint signature | its incident raw pairs and profile; all releases/snapshots that include them | old-old raw pairs not incident to the endpoint |
| Pool membership/content | null profiles, p-values, q-values, snapshots | signatures and raw C-SKL scores |
| Null grid, bootstrap count/seed, fit code, clamp policy | null profiles and calibration release | raw scores |
| Pair family added/removed | global BH q-values and snapshots; p-values may be reused only under the same pool/profile release | raw scores |
| GSM/hash evidence or overlap threshold | overlap artifact, policy result, affected discovery snapshot | raw C-SKL and null p-values |
| Probe-to-gene/pathway release or explainer settings | explanation/enrichment artifacts | raw pair and calibration |
| GEO metadata or ontology mapping release | annotation artifacts and label-based queries/layout | molecular signature unless expression source changed |
| AI model, prompt, schema, or evidence packet | AI run/explanation artifact | deterministic evidence and numerical facts |
| Layout/community parameters | graph snapshot layout/membership artifact as applicable | pair scores and calibration |

No worker may infer validity from an output path merely existing. It must compare
the complete expected dependency digest and validate the artifact manifest.

## Recoverable-job contract

The catalog provides idempotent jobs keyed by `(kind, job_key,
input_fingerprint)`. States are `queued`, `running`, `retry`, `succeeded`,
`dead`, and `cancelled`. Claims and transitions are transactional. Retryable
failures use bounded exponential backoff with deterministic jitter and preserve
an error code plus truncated detail. Dead/cancelled jobs require an explicit
requeue operation.

The job state machine, CLI inspection/retry, protected operations API, GEO
sync/discovery/download, normalization scale path, scoring, calibration, layout,
SPECTER2 and annotation orchestration are implemented. A single automatic DAG
scheduler across all stages remains deployment-specific.

## Implemented versus planned data flow

```text
GEO/source acquisition                         IMPLEMENTED E-UTILITIES + RESUMABLE RAW
        |
source + normalization/QC fingerprints         IMPLEMENTED CONTRACTS
        |
normalized-expression artifact                 IMPLEMENTED RAW-CEL/STORE + SCALE BRIDGE
        |
dataset-signature artifact + catalog version   IMPLEMENTED IMPORT/PROMOTION GATES
        |
K x N raw pair delta + overlap evidence        IMPLEMENTED
        |
pool/null-profile release                      PRESERVED BUILDER + VALIDATED IMPORT
        |
streamed p-values + global BH calibration      IMPLEMENTED RESUMABLE WORKER
        |
text/mapping/explainer/pathway releases        IMPLEMENTED VERSIONED RELEASE PATHS
        |
policy/layout graph snapshot                   IMPLEMENTED BUILDER/VALIDATION/ROLLBACK
        |
API/query/LOD/UI serving                       IMPLEMENTED STATIC + VERSIONED API PATHS
```
