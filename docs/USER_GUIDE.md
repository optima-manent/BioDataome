# Atlas workbench guide

The workbench is designed around one question: which datasets appear related,
what evidence produced the relationship, what could confound it, and what is
worth investigating next?

## Read the map

- **Node:** one immutable dataset version.
- **Node color:** broad anatomical system by default, with an optional clinical
  family view.
- **Node shape:** coarse anatomical or tissue context, so the map remains
  readable without relying on color alone.
- **Node size:** sample count.
- **Solid edge:** published relationship without major/exact sample overlap.
- **Dotted edge:** major or exact shared-sample relationship; inspectable but not
  independent replication.
- **Edge visibility:** strongest relationships form the overview; progressive
  detail appears as the view is enlarged. The global BH q-value slider also
  lets you keep fewer, stronger links or restore more of the published graph.

Clinical families are broad display groups derived from concordant disease
labels. They are useful for browsing, but they are not diagnoses and do not
assign ICD codes. Unreviewed labels remain in an explicit unreviewed group.

The raw c-SKL direction is easy to misread: lower values indicate closer
standardized covariance structure. Percentiles in the interface reverse that
direction for display, so a higher c-SKL similarity percentile is stronger.

## Navigate

- On the first visit, use the short visual primer while the published snapshot
  prepares, then choose **Open the atlas**. The **?** button reopens the primer.
- Drag the background to pan.
- Use the wheel or zoom controls to change scale.
- Select a cluster or group label to focus that part of the map. Choose **Show
  all** or press Escape to return to the full view.
- Select a node to open its GEO metadata, semantic-label provenance, and nearest
  published relationships.
- Select an edge to inspect c-SKL, p/q-values, SPECTER2, shared samples, and a
  feature/pathway explanation when one has been computed.
- Shift-select nodes to create a temporary group.
- Switch between C-SKL topology, anatomical system, and clinical family
  layouts. Group labels remain in screen space so they do not shrink while
  zooming. Node color can be switched separately between anatomy and clinical
  family; shape continues to show anatomical context.

The maximum global BH q-value control only filters relationships already in the
published sparse graph. It does not recalculate q-values or reveal unpublished
pairs from the complete calibrated family.

## Search versus discovery query

The header search is a fast node filter over accession, title, tissue, anatomical
system, and disease text. It is not the scientific query engine.

The **Discovery query** panel builds an explicit AND expression. It supports:

- independent relationships only;
- maximum q-value;
- minimum c-SKL similarity percentile;
- same or different validated anatomical system;
- same or different validated disease label;
- minimum SPECTER2 percentile; and
- a gene, probe, or pathway mechanism term when an explainer is attached.

Anatomy/disease equality filters only use ontology-label-concordant fields.
Unknown or unreviewed values are not coerced into a match. Mechanism queries are
limited to edges with computed explanation artifacts; the interface reports that
coverage directly.

The static showcase queries the 5,344 edges in the published sparse graph. A
complete-pair backend is required to query relationships omitted by graph policy
or to perform corpus-wide text-only searches.

## Evidence lenses

- **All supported links** shows the published graph.
- **Cross-modal agreement** requires both molecular evidence and high computed
  SPECTER2 proximity.
- **Molecular-only signal** highlights strong c-SKL relationships without strong
  text agreement.
- **Same tissue, different disease** is a starting point for cross-diagnosis
  hypotheses, not proof of a shared mechanism.
- **Sample-overlap audit** isolates overlap-qualified relationships.

SPECTER2 is based on scientific text. It is useful for comparison and triage but
is not an independent molecular validation of c-SKL.

## Feature and pathway evidence

For an explained relationship, B(k) is the set of k features that best aligns
the retained covariance subspaces. W(k) is the set that least aligns them. The
trajectory compares their optimization objective with a seeded random-feature
baseline across k.

Feature scores are not expression fold changes, additive causal effects, or
percent contributions. Reactome entries are over-representation hypotheses with
their own multiple-testing correction and the frozen GPL570 gene universe.

## Research synthesis

The server deployment can send a bounded, structured evidence packet to an
allowlisted OpenRouter model. The request separates observations, hypotheses,
alternatives, limitations, and follow-up questions. Generated text remains a
hypothesis and never modifies GEO facts or accepted annotations.

The GitHub Pages showcase has no server credential and does not offer live
synthesis. Cached deterministic evidence remains fully explorable.

## Export

Export uses the selected edge, selected group, or visible graph as its scope.
JSON preserves structured evidence and provenance. CSV is intended for tabular
analysis and neutralizes spreadsheet-formula cells. Each export identifies the
snapshot, calibration, evidence mode, filters, and omitted fields.

## Interpretation checklist

Before reporting a relationship:

1. Record the snapshot and calibration identifiers.
2. Distinguish raw c-SKL from p/q-values.
3. Check shared-sample evidence.
4. Check tissue, disease, organism, platform, and preprocessing confounders.
5. Treat SPECTER2 as text proximity.
6. Verify annotation review and ontology status.
7. Treat B(k)/W(k), pathways, and generated narratives as hypotheses.
8. Follow the source links and review the underlying studies.
