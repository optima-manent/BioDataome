# Scientific method and interpretation

This document defines what C-SKL Atlas computes, what a graph edge means, and
which conclusions the system does and does not support. The artifact and field
schemas are specified in [DATA_CONTRACTS.md](./DATA_CONTRACTS.md).

The mathematical reference is Lakiotaki et al., "A data driven approach reveals
disease similarity on a molecular level," *npj Systems Biology and
Applications* 5, 39 (2019),
[doi:10.1038/s41540-019-0117-0](https://doi.org/10.1038/s41540-019-0117-0).
Equation and page references below refer to the journal PDF.

## Release scope

The frontend defaults to the frozen 500-dataset GPL570 release. Its snapshot,
calibration, global and independent q-value families, overlap evidence, and
computed SPECTER2 release remain explicit. The B=100 frozen calibration is an
operational research release and must not be represented as an exact-size B=500
confirmatory calibration. The complete status is recorded in
[RELEASE_500.md](./RELEASE_500.md).

## What C-SKL answers

C-SKL compares the **standardized covariance structure** of two molecular
datasets. Informally, it asks whether their measured variables vary together in
similar linear patterns. It does not compare titles, disease labels, treatment
labels, or other text, and it does not directly compare mean expression levels.

For dataset (P), let (X_P\in\mathbb{R}^{m_P\times n}), with samples in rows
and molecular variables in columns. A valid comparison requires:

1. the same ordered feature universe of length (n);
2. the same C-SKL compression parameter \(\alpha\);
3. compatible preprocessing and C-SKL code versions; and
4. valid, finite signatures.

The paper enforced this by comparing datasets only within one measurement
platform (journal PDF p. 2). Atlas makes the stronger operational rule explicit
with an order-sensitive feature-universe fingerprint. Raw scores must not be
computed across incompatible platforms or feature orders.

## Signature construction

### Standardization

Each variable is standardized within its own dataset:

\[
X'_{ri}=\frac{X_{ri}-\bar X_i}{s_i}.
\]

This removes dataset-specific means and scales while retaining linear
correlations (journal PDF p. 8). The implemented reference kernel uses sample
standard deviation (`ddof=1`). Its default `nan_policy="raise"` rejects
non-finite input. Near-constant variables receive the tiny, seeded,
R-compatible noise treatment in [`cskl.py`](../python/cskl.py); the seed and
policy therefore belong in artifact provenance.

Consequences:

- a C-SKL match is not differential expression;
- biologically meaningful mean shifts can be removed by design; and
- batch, tissue, protocol, or reused samples can align covariance without
  implying a shared disease mechanism.

### Low-rank covariance model

The standardized distribution is approximated as multivariate normal with

\[
\Sigma_P=P\Lambda_PP^\top+\sigma I_n,
\qquad \sigma=1-\alpha.
\]

(P\in\mathbb{R}^{n\times c_P}) contains orthonormal principal axes and
\(\Lambda_P\) contains non-negative retained eigenvalues. The implementation:

1. performs PCA/SVD on the standardized matrix;
2. chooses the smallest (c_P) whose raw eigenvalues reach \(\alpha n\) total
   variance;
3. rescales the retained eigenvalues so \(\sum_i\lambda_i^P=\alpha n\); and
4. stores (P), \(\lambda^P\), (n), (m_P), \(\alpha\), and optionally the
   ordered feature names in a `PCASignature`.

The paper selected \(\alpha=0.5\) after its sibling-split validation and reported
robustness around that choice (journal PDF p. 8). It is the code default, not an
immutable scientific constant. Every release must record the actual value.

## Raw C-SKL score

For compatible signatures (P) and (Q), the implemented form of paper Eq. 1 is

\[
d_{\mathrm{C\text{-}SKL}}(P,Q)=
\max\left\{0,
\frac{2\alpha n-
\sum_{i=1}^{c_P}\sum_{j=1}^{c_Q}
(\lambda_i^P+\lambda_j^Q)(P_i^\top Q_j)^2}
{2(1-\alpha)}
\right\}.
\]

The maximum only clips tiny negative numerical error. The symmetric
\(\lambda_j^Q\) index is the interpretation used by paper Eq. 2 and by the
reference implementation; Eq. 1 in the printed article appears to contain an
indexing typo in its second sum.

Computationally, the score uses the small cross-Gram matrix
\(M=P^\top Q\). Full (n\times n) covariance matrices are neither required nor
stored.

### Score semantics

- Lower means more similar standardized covariance structure.
- A value near zero means that the retained, eigenvalue-weighted PCA subspaces
  nearly coincide under this approximation.
- The score is symmetric and non-negative.
- It is **not a metric**: the triangle inequality need not hold (journal PDF
  p. 8). Metric trees, metric nearest-neighbor guarantees, and layouts that
  claim to preserve metric distance are therefore invalid assumptions.
- It is not a probability, p-value, effect size, causal claim, or percentage.
- Scores are reusable only under the same feature universe, \(\alpha\), core
  algorithm fingerprint, and signature dependencies. They should not be ranked
  across platforms merely because the numerical values look similar.

Graph layout is a visualization of relationships, not a literal geometric
embedding of the C-SKL score.

## Incremental raw-score computation

**Implemented.** [`incremental.py`](../python/cskl_atlas/incremental.py) binds a
signature to its dataset ID and frozen feature universe, validates all inputs
before emitting a result, and delegates each score to the vendored
`cskl.cskl` source of truth.

For (K) newly admitted versions and (N) existing versions in one compatible
stratum:

- the strict cross update emits exactly (K\times N) new-existing scores;
- the complete batch delta additionally emits (K(K-1)/2) new-new scores;
- it never recomputes an existing-existing score; and
- it streams individual results or bounded persistence batches instead of
  allocating an all-(N) matrix.

Raw old-old scores remain valid when a new dataset arrives. They become stale
only if an endpoint version or any signature dependency changes. Calibration is
different: its p/q-values are release-relative and can change even when the raw
score does not.

## From a raw score to a released edge

### Semi-parametric null comparison

The paper tests whether (P) and (Q) are unusually close relative to pooled
background samples (J), rather than testing exact equality of distributions
(journal PDF p. 8). The implemented reference procedure:

1. draws bootstrap signatures from (J) at the partner's sample size;
2. computes null distances (d(P,J_{m_Q})) and (d(Q,J_{m_P}));
3. fits a normal distribution to each collection of null distances;
4. evaluates the lower-tail probabilities (p_{P\rightarrow Q}) and
   (p_{Q\rightarrow P}); and
5. returns \(p=\max(p_{P\rightarrow Q},p_{Q\rightarrow P})\), requiring the
   observed pair to be unusual against both null comparisons.

Atlas can store each dataset's null mean and standard deviation across a sample
size grid in a `NullProfileArtifact`. The profile is bound to the dataset
signature hash, pool hash, ordered feature hash, \(\alpha\), bootstrap count,
and grid.

### Multiple testing and release identity

**Implemented foundation.** Pair p-values are streamed into a named calibration
release. [`Catalog.finalize_bh`](../python/cskl_atlas/catalog.py) performs exact
global Benjamini-Hochberg adjustment for the supplied release family in
disk-backed SQL; Python does not load all pair values into memory. The paper's
edge rule was (q<0.05) (journal PDF p. 8). Atlas APIs default to (q\leq0.05),
but the chosen display threshold is not part of the raw score.

A calibrated fact is always the tuple:

\[
(\text{pair ID},\text{calibration release ID},p,q).
\]

Never copy a q-value onto a raw pair without its release ID. Adding or removing
tested pairs changes the BH family and can change old q-values. Changing the
pool, grid, bootstrap count, profile algorithm, or \(\alpha\) requires a new
release, not an in-place overwrite.

### Exact and frozen update modes

Both modes preserve raw scores and compute exact BH over the p-values actually
included in that release. Their difference is null-profile range handling:

| Mode | Implemented behavior | Scientific meaning |
|---|---|---|
| `exact` | Interpolates only inside the calibrated grid and raises `OutOfCalibrationRange` outside it. | Strict-range release. An update cannot silently extrapolate beyond validated sample sizes. |
| `frozen` | Uses the named frozen pool/profile release and clamps an out-of-grid sample size to the nearest grid endpoint. | Operational approximation. The release and UI must disclose the frozen mode and clamping risk. |

The Atlas calibration worker freezes the exact raw-pair family, requires
versioned checksum-verified null-profile artifacts from the preserved scale
store, streams missing p-values, and finalizes exact BH plus C-SKL percentiles.
The separate scale ingest/profile stage generates or rebuilds those null
profiles; the calibration worker intentionally consumes rather than mutates
them. Therefore the
`mode` flag plus release manifest proves range behavior and which pool/profile
release was consumed; it must not be used alone to claim that a pool was
rebuilt.

Important boundaries:

- `exact` does **not** by itself rebuild the pool or null profiles, nor does it
  guarantee an exact-size grid. It means no out-of-range clamping in the current
  implementation.
- `frozen` does not freeze BH q-values. BH is recalculated over the pair family
  supplied to the new release.
- The exact-size grid builder remains in
  [`cskl_pipeline/scale`](../python/cskl_pipeline/scale). The scale bridge can
  import its named pool/profile artifacts and the Atlas worker requires those
  artifacts for calibration.
- Pool/profile production is implemented in the scale pipeline and was verified
  on a real raw-CEL accession. Unattended calendar scheduling and a manuscript
  B=500 release remain operational work. Published-snapshot comparison is an
  API operation, not a recalibration shortcut.

## Shared-sample overlap policy

The paper removed datasets sharing even one molecular profile before its
published graph analysis (journal PDF pp. 1-2). Atlas instead retains every
dataset and stores pair-level overlap evidence, because shared samples confound
the affected relationship rather than every relationship of either endpoint.
This is an explicit methodological extension, not a claim that partial-overlap
edges inherit the paper's validation.

**Implemented.** [`overlap.py`](../python/cskl_atlas/overlap.py) matches samples
by normalized GSM accession or by a SHA-1 hash of the aligned expression row
rounded to ten decimals. Maximum bipartite matching prevents a sample identified
by both mechanisms from being counted twice. Expression hashes are identity
evidence, not a security mechanism, and are meaningful only after the same
feature alignment and hashing policy.

For endpoint sizes (a,b) and (s) matched samples:

\[
f_A=s/a,\quad f_B=s/b,\quad
J=s/(a+b-s),\quad O=s/\min(a,b).
\]

The default versioned policy is:

- `exact`: (s=a=b>0); exclude from independent discovery;
- `major`: not exact and overlap coefficient (O\geq0.5); exclude;
- `minor`: (s>0) below the major threshold; retain by default but show the
  overlap warning; and
- `none`: no matched evidence.

The threshold and exclusion booleans are configurable and must be represented by
a `policy_hash` in a published snapshot. Reproducing a named analysis requires
its exact overlap policy; display defaults are not a substitute for that policy.

`exclude_from_discovery` means "do not count this edge as independent evidence."
It does not delete either dataset, erase the raw score, or hide the overlap-audit
view.

## Feature and group explanations

The paper's best-explaining set (B(k)) selects exactly (k) features that make
the two retained subspaces agree most strongly (paper Eq. 2, journal PDF p. 9):

\[
B(k)=\arg\max_{S\in\{0,1\}^n,\;\mathbf{1}^\top S=k}
\sum_{i,j}(\lambda_i^P+\lambda_j^Q)
(P_i^\top\operatorname{diag}(S)Q_j)^2.
\]

The implementation uses the paper's alternating bilinear relaxation (Eq. 3), a
deterministic initialization, seeded random restarts, and a finite iteration
limit. `W(k)` reverses the objective to identify features that least align. A set
explainer sums the same objective over multiple compatible pairs.

**Implemented numerical primitives:** `explain_topk` and `explain_set_topk` in
[`cskl.py`](../python/cskl.py).

Interpretation caveats:

- the optimization is non-convex and approximate; different seeds, restarts,
  iteration limits, and near-ties can change the selected set;
- (k) is user-selected, not estimated by the paper, so every result must store
  (k) and optimizer settings;
- features interact inside a set. A reported linearized score is not a unique,
  additive causal contribution;
- `B(k)` and `W(k)` describe covariance alignment, not differential expression,
  necessity, sufficiency, or therapeutic actionability;
- probe-to-gene mappings can be one-to-many or many-to-one; enrichment requires
  an explicit tested background and its own multiple-testing correction; and
- a group explanation is valid only when every pair shares the same ordered
  feature universe and \(\alpha\).

Versioned on-demand explanation artifacts, a frozen GPL570 probe/gene mapping,
and local Reactome release-97 over-representation are implemented. The tested
background is all 22,167 unique Entrez genes represented in the array mapping;
BH covers all 2,393 size-eligible pathways for the default limits. KEGG,
GeneCards, literature retrieval, and group-level persisted explanation releases
remain optional future integrations.

## AI annotations and explanations

**Implemented contract, not autonomous scientific authority.** The OpenRouter
integration requires an explicit allowlisted model, strict JSON Schema,
zero-data-retention routing, temperature zero, and hashes of the prompt payload,
schema, response, and every source packet.

Dataset annotation follows GEO-first precedence. Deterministic GEO assertions
are locked; model candidates can fill only unknown fields and must cite exact
character spans from supplied GEO fields. The model extracts surface labels but
does not assign identifiers: any model-supplied namespace or CURIE is ignored.
Each field constrains an official OLS lookup, whose responses are frozen by
content hash. Only one non-obsolete term with an exact normalized canonical
label or synonym match is retained; ambiguous and unresolved labels become
unknown. This establishes lexical ontology concordance, not biological truth.
Resolved model-derived assertions remain unreviewed candidates until accepted
or rejected by a curator.

Generated graph explanations receive a finite evidence packet. Direct
observations and hypotheses are returned in separate fields, and every claim
must cite an evidence ID from that packet. The model is instructed not to add
outside facts. These controls improve auditability; they do not make generated
hypotheses true.

The current SPECTER2 release uses the pinned official base plus proximity
adapter, 768-dimensional normalized embeddings, and the declared GEO title plus
summary fallback. Agreement with C-SKL is concordance between modalities, not
validation of molecular causality.

## Scientific limitations to preserve in the UI

1. **Covariance-only approximation.** Gaussian, linear, second-order structure
   cannot represent all biological distributional differences.
2. **Mean removal.** Standardization can remove relevant absolute-expression
   effects.
3. **Small-sample PCA.** Retained axes become unstable as lower-variance
   components are admitted; the paper explicitly notes this \(\alpha\) tradeoff.
4. **Unidentified source of similarity.** Tissue, disease, treatment, batch,
   protocol, cell line, or sample reuse can all produce an edge.
5. **Same-feature restriction.** Cross-platform/species C-SKL is not currently
   supported; the paper listed differing-variable comparison as future work.
6. **Corpus-dependent significance.** Pool composition and BH family determine
   p/q-values. A q-value from one release cannot be transplanted to another.
7. **Count-biased aggregates.** Disease-edge support counts favor diseases with
   more datasets. Always expose both support and opportunity/denominator.
8. **Association, not causation.** Neither C-SKL, pathway enrichment, SPECTER2,
   nor an LLM establishes a mechanism or clinical recommendation.

## Implementation boundary

| Capability | Status |
|---|---|
| Vendored C-SKL signature, score, bootstrap, BH helper, pair/set feature explainer | Implemented |
| Numerically validated fast PCA/null/all-pairs primitives retained for regression | Implemented |
| K-by-N streaming raw-score delta with no old-old recomputation | Implemented |
| Pair-level GSM/expression-hash overlap evidence and exclusion policy | Implemented |
| Content-addressed artifacts, dependency fingerprints, relational catalog, leased jobs, releases, snapshots, query/read API | Implemented |
| Scale-store validation/import, streamed profile p-values, disk-backed exact BH and resumable calibration worker | Implemented |
| Deterministic Leiden/layout builder, snapshot validation/publication/rollback and release diff | Implemented |
| Constrained OpenRouter annotation/explanation contracts, resumable candidate pipeline and review persistence | Implemented and live-rehearsed; candidate curation remains required |
| Identity-free GEO Series+Sample sync, E-utilities discovery, resumable RAW acquisition and recovery states | Implemented; only scheduled discovery requires NCBI_EMAIL |
| SCAN.UPC normalization, QC, signatures and null-profile builder | Implemented and real-CEL verified |
| Pinned SPECTER2 all-pairs release | Implemented on the 500-dataset corpus |
| GPL570 mapping, on-demand B(k)/W(k), and Reactome 97 enrichment | Implemented |
| Literature and licensed KEGG/GeneCards ingestion | Planned/rights-dependent |
| PostgreSQL/object-storage deployment and monitoring | Planned |
