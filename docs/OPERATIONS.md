# Operating C-SKL Atlas

This guide covers the single-host reference deployment. The same artifact and
release contracts can be implemented with PostgreSQL, object storage, and a
durable queue for a multi-host service.

## Core rule

Raw scientific facts are immutable and reusable. Calibration, annotation,
layout, and publication are versioned views over those facts. A new release is
published only after all mandatory artifacts have been staged and validated.

## Runtime requirements

- Python 3.11+
- Node.js 22.13+ and pnpm 11.9 for the web application
- R and Bioconductor SCAN.UPC for RAW Affymetrix CEL normalization
- adequate local disk for source archives, normalized matrices, signatures,
  null profiles, pair tables, and staging copies
- optional CUDA for SPECTER2; CPU is supported

Install the Python package and initialize the catalog:

```bash
python -m pip install -e ".[dev,graph]"
cskl-atlas init
cskl-atlas health
```

The default SQLite catalog uses WAL mode, full synchronous writes, foreign keys,
and file-backed temporary operations. Do not place it on a shared network
filesystem or use it as a distributed queue.

## Configuration and secrets

Use environment variables or a deployment secret manager. Never commit a real
`.env` file.

| Variable | Purpose |
| --- | --- |
| `CSKL_ATLAS_CATALOG` | Reference catalog path |
| `CSKL_ATLAS_OPS_TOKEN` | Protected write/operations endpoints |
| `CSKL_ATLAS_ALLOWED_ORIGINS` | API CORS allowlist |
| `NCBI_EMAIL` | Required contact for scheduled E-utilities discovery |
| `NCBI_API_KEY` | Optional higher NCBI request allowance |
| `OPENROUTER_API_KEY` | Optional semantic labeling/synthesis credential |
| `OPENROUTER_MODEL` | Explicit allowlisted model identifier |
| `CSKL_ATLAS_EXPLAIN_ENABLED` | Server-side synthesis feature gate |
| `CSKL_ATLAS_EXPLAIN_ACCESS_TOKEN` | Authenticated synthesis caller token |

Known-accession GEO Series and Sample SOFT synchronization does not require an
NCBI email or API key. Scheduled E-utilities discovery must use a real operator
contact; the software does not invent one.

## Ingestion sequence

### 1. Discover or provide accessions

Use `discover-geo` for a bounded date window or supply an explicit accession
file to `sync-geo`.

```bash
cskl-atlas discover-geo --help
cskl-atlas sync-geo --help
```

Discovery is bounded, rate-limited, retried with backoff, and written atomically.
SOFT synchronization filters Sample metadata to the catalog expression-matrix
cohort. If that cohort cannot be established, the record becomes operator-
required rather than silently substituting every sample in a GEO Series.

### 2. Acquire RAW CEL archives

```bash
cskl-atlas download-geo-raw --help
```

Resumable downloads bind the partial file to ETag/Last-Modified and validate
Content-Range, advertised length, configured caps, and an optional authoritative
SHA-256. Changed upstream validators restart the transfer rather than appending
incompatible bytes.

Archive extraction rejects traversal, links, devices, duplicate/reserved names,
excess member counts, per-file limits, aggregate expansion limits, and inadequate
disk reserve. Files are extracted to private staging and published atomically
only after validation.

### 3. Normalize and import

The RAW path invokes SCAN.UPC, validates CEL membership and matrix finiteness,
checks that the source did not change during work, writes the normalized matrix
atomically, and records its SHA-256 provenance. Revised RAW input invalidates
derived normalization, signatures, null profiles, and incident pair scores.

Existing scalable stores can be imported without recomputation:

```bash
cskl-atlas import-scale-store --help
cskl-atlas import-scale-release --help
```

Import validates platform, source revision, ordered feature hash, sample count,
alpha, algorithm fingerprints, artifact checksums, and the complete pair family.

## Incremental scoring and calibration

For K new current dataset versions and N unchanged current versions, raw scoring
computes K x N new-to-existing pairs and K(K-1)/2 new-to-new pairs. It does not
recompute unaffected N(N-1)/2 old-to-old pairs.

Raw c-SKL is pair-local. P-values, percentiles, and Benjamini-Hochberg q-values
belong to a named corpus and null release. Adding or removing a dataset therefore
requires a new calibration family even when old raw scores are reused.

- **Exact mode** builds the corpus-bound reference calibration and rejects sample
  sizes outside its validated grid.
- **Frozen mode** reuses an explicitly named operational calibration and records
  every boundary clamp. It is a distinct methodological variant.

Incomplete K x N or K(K-1)/2 families fail closed. The catalog never promotes a
partial calibration.

## Metadata, ontology, text, and pathways

```bash
cskl-atlas run-specter2 --help
cskl-atlas label-geo --help
cskl-atlas build-ontology-index --help
cskl-atlas audit-annotations --help
cskl-atlas build-reactome-index --help
```

SPECTER2 releases bind source text, base/adaptor revisions, tokenizer, device
policy, and normalized embeddings. Generated label candidates must quote source
spans. A deterministic resolver accepts only an unambiguous exact canonical
label or synonym from a frozen official ontology response; model-proposed CURIEs
are not trusted. Unresolved, obsolete, or ambiguous terms remain unknown.

Pathway enrichment binds the Reactome release, probe-to-gene mapping, GPL570
background, size policy, test, and correction family.

## Relationship explanations

```bash
cskl-atlas explain-edge --help
cskl-atlas explain-snapshot --help
```

`explain-snapshot` is the safe bulk path. It resumes checksum-valid per-edge
artifacts, enforces edge and wall-clock budgets, and writes an atomic checkpoint
report after each attempted relationship. A running optimizer may finish after
the soft time budget; its artifact is still committed atomically.

Public GET requests replay existing explanation artifacts. Starting computation
through the API requires an operations token. A static deployment never starts
scientific work.

## Build and publish a graph release

```bash
cskl-atlas build-snapshot --help
cskl-atlas validate-snapshot SNAPSHOT_ID
cskl-atlas publish-snapshot SNAPSHOT_ID --operator OPERATOR --reason REASON
cskl-atlas export-static-graph --help
cskl-atlas audit-release --help
```

The builder freezes exact dataset versions, pair IDs, overlap evidence,
calibration, text release, policy, communities, and layout. Publication changes
the current-snapshot pointer in one transaction, so readers see either the old
complete release or the new complete release.

Run the operational audit for every public snapshot. Run the stricter
confirmatory profile before using a release for a confirmatory analysis. A
failed gate is a release result, not a reason to edit the report.

Rollback repoints the stratum to an earlier validated snapshot and records the
operator and reason; it does not delete newer artifacts.

## Jobs and recovery

```bash
cskl-atlas jobs
cskl-atlas reap
cskl-atlas retry JOB_ID
cskl-atlas health
```

Jobs are idempotent over `(kind, job_key, input_fingerprint)` and move through
`queued`, `running`, `retry`, `succeeded`, `dead`, or `cancelled`. Workers own
leases and heartbeats. Expired leases return retryable work to the queue. Errors
retain a stable code, bounded detail, attempt count, and backoff deadline.

Recovery procedure:

1. Stop the affected worker from claiming more work.
2. Inspect the structured error and the job's input fingerprint.
3. Verify source/artifact checksums and available disk before changing state.
4. Correct configuration or upstream input without editing immutable artifacts.
5. Reap an expired lease or explicitly requeue dead work.
6. Resume and rerun validation before publication.

Never mark work complete because a process exited successfully or an output path
exists. Validate its checksum, manifest, dependency fingerprint, and expected
record counts.

## Static showcase

```bash
pnpm build:pages
pnpm test:pages
```

The Pages build verifies the frozen graph checksum and emits `dist/pages/`.
GitHub Actions uploads that directory as the Pages artifact. The static site has
no server secret, catalog, write route, or live synthesis endpoint.

## Verification and backups

Before a release:

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

Back up the catalog and content-addressed artifact store as one consistency set.
Test restoration into a new location, run SQLite integrity/foreign-key checks,
verify artifact hashes, and validate a snapshot before considering the backup
usable.

## Multi-host boundary

The reference implementation deliberately does not pretend that SQLite and
process-local rate limits are distributed infrastructure. A multi-host service
requires PostgreSQL migrations and transactional claims, object storage,
durable queueing, distributed rate limiting, authentication/authorization,
metrics and alerts, backup/restore automation, and an explicit privacy policy
for user workspaces or unpublished uploads.
