import type {
  AnnotationCandidate,
  EdgeExplainer,
  GraphDataset,
  GraphEdge,
  GraphNode,
  Tissue,
} from "./graph-data";

export type ResearchExportView = {
  evidenceMode: "cskl" | "specter2" | "agreement";
  clusterMode: "topology" | "tissue" | "disease";
  lens: "all" | "agreement" | "cskl-only" | "cross-disease" | "overlap";
  search: string;
  activeTissueSystems: Tissue[];
  minSamples: number;
};

export type ResearchExportInput = {
  graph: GraphDataset;
  visibleNodes: GraphNode[];
  visibleEdges: GraphEdge[];
  selectedNodeIds: ReadonlySet<string>;
  selectedEdgeId: string | null;
  view: ResearchExportView;
};

const MAX_EXPORT_NODES = 25_000;
const MAX_EXPORT_EDGES = 100_000;
const MAX_EXPORT_BYTES = 64 * 1024 * 1024;

const compareText = (left: string, right: string) =>
  left < right ? -1 : left > right ? 1 : 0;

function optional<Key extends string, Value>(key: Key, value: Value | undefined) {
  return (value === undefined ? {} : { [key]: value }) as Partial<Record<Key, Value>>;
}

function normalizeAnnotationCandidates(
  candidates: Record<string, AnnotationCandidate[]> | undefined,
) {
  if (!candidates) return undefined;
  const entries = Object.entries(candidates)
    .filter(([, values]) => values.length > 0)
    .sort(([left], [right]) => compareText(left, right))
    .map(([field, values]) => [
      field,
      [...values]
        .sort(
          (left, right) =>
            compareText(left.label, right.label) ||
            compareText(left.ontologyId ?? "", right.ontologyId ?? "") ||
            compareText(left.sourceKind, right.sourceKind),
        )
        .map((candidate) => ({
          label: candidate.label,
          ...optional("ontologyId", candidate.ontologyId),
          sourceKind: candidate.sourceKind,
          reviewState: candidate.reviewState,
          extractorVersion: candidate.extractorVersion,
          ...optional("ontologyValidation", candidate.ontologyValidation),
        })),
    ] as const);
  return entries.length ? Object.fromEntries(entries) : undefined;
}

function requireFinite(value: number, field: string) {
  if (!Number.isFinite(value)) {
    throw new Error(`Cannot export non-finite numeric field: ${field}.`);
  }
  return value;
}

function optionalFinite(value: number | undefined, field: string) {
  return value === undefined ? undefined : requireFinite(value, field);
}

function normalizeNode(node: GraphNode) {
  return {
    id: node.id,
    ...optional("datasetUid", node.datasetUid),
    ...optional("versionId", node.versionId),
    title: node.title,
    tissue: node.tissue,
    ...optional("tissueSystem", node.tissueSystem),
    ...optional("tissueSystemSource", node.tissueSystemSource),
    disease: node.disease,
    ...optional("diseaseLabelSource", node.diseaseLabelSource),
    diseaseFamily: node.diseaseFamily,
    samples: requireFinite(node.samples, `${node.id}.samples`),
    platform: node.platform,
    organism: node.organism,
    layout: {
      x: requireFinite(node.x, `${node.id}.x`),
      y: requireFinite(node.y, `${node.id}.y`),
      ...optional("community", node.community),
    },
    annotation: {
      confidence: requireFinite(
        node.annotationConfidence,
        `${node.id}.annotationConfidence`,
      ),
      source: node.annotationSource,
      ...optional("state", node.annotationState),
      ...optional("candidates", normalizeAnnotationCandidates(node.annotationCandidates)),
    },
    summary: node.summary,
    ...optional("featureHash", node.featureHash),
    ...optional("configHash", node.configHash),
  };
}

function normalizeExplainer(edgeId: string, explainer: EdgeExplainer) {
  return {
    provenance: explainer.provenance,
    bSet: explainer.bSet.map((item) => ({
      feature: item.feature,
      ...optional("gene", item.gene),
    })),
    wSet: explainer.wSet.map((item) => ({
      feature: item.feature,
      ...optional("gene", item.gene),
    })),
    trajectory: explainer.trajectory.map((point, index) => ({
      k: requireFinite(point.k, `${edgeId}.explainer.trajectory[${index}].k`),
      bestObjective: requireFinite(
        point.bestObjective,
        `${edgeId}.explainer.trajectory[${index}].bestObjective`,
      ),
      worstObjective: requireFinite(
        point.worstObjective,
        `${edgeId}.explainer.trajectory[${index}].worstObjective`,
      ),
      randomObjective: requireFinite(
        point.randomObjective,
        `${edgeId}.explainer.trajectory[${index}].randomObjective`,
      ),
    })),
    ...optional("bestPathways", explainer.bestPathways),
    ...optional("worstPathways", explainer.worstPathways),
    ...optional("reactomeRelease", explainer.reactomeRelease),
    ...optional("interpretation", explainer.interpretation),
  };
}

function normalizeEdge(edge: GraphEdge) {
  const textEvidence =
    edge.specter2Provenance === "computed" && edge.specter2 !== undefined
      ? {
          provenance: "computed" as const,
          specter2: requireFinite(edge.specter2, `${edge.id}.specter2`),
          ...optional("textReleaseId", edge.textReleaseId),
        }
      : undefined;

  return {
    id: edge.id,
    source: edge.source,
    target: edge.target,
    molecularEvidence: {
      cskl: requireFinite(edge.cskl, `${edge.id}.cskl`),
      ...optional(
        "csklPercentile",
        optionalFinite(edge.csklPercentile, `${edge.id}.csklPercentile`),
      ),
      ...optional("pValue", optionalFinite(edge.pValue, `${edge.id}.pValue`)),
      qValue: requireFinite(edge.qValue, `${edge.id}.qValue`),
      ...optional(
        "independentQValue",
        optionalFinite(edge.independentQValue, `${edge.id}.independentQValue`),
      ),
      ...optional("algorithmHash", edge.algorithmHash),
    },
    ...optional("textEvidence", textEvidence),
    sampleOverlapEvidence: {
      sharedSamples: requireFinite(edge.sharedSamples, `${edge.id}.sharedSamples`),
      overlapFraction: requireFinite(edge.overlapFraction, `${edge.id}.overlapFraction`),
      ...optional("classification", edge.overlapClassification),
      ...optional("discoveryExcluded", edge.discoveryExcluded),
      ...optional(
        "overlapCoefficient",
        optionalFinite(edge.overlapCoefficient, `${edge.id}.overlapCoefficient`),
      ),
      ...optional("jaccard", optionalFinite(edge.jaccard, `${edge.id}.jaccard`)),
      ...optional(
        "fractionSource",
        optionalFinite(edge.overlapFractionSource, `${edge.id}.overlapFractionSource`),
      ),
      ...optional(
        "fractionTarget",
        optionalFinite(edge.overlapFractionTarget, `${edge.id}.overlapFractionTarget`),
      ),
      ...optional("evidenceId", edge.overlapEvidenceId),
    },
    ...optional("explainer", edge.explainer ? normalizeExplainer(edge.id, edge.explainer) : undefined),
  };
}

export function buildResearchExport(input: ResearchExportInput) {
  const { graph, selectedEdgeId, selectedNodeIds } = input;
  const graphNodeMap = new Map(graph.nodes.map((node) => [node.id, node]));
  const selectedEdge = selectedEdgeId
    ? graph.edges.find((edge) => edge.id === selectedEdgeId)
    : undefined;

  if (selectedEdgeId && !selectedEdge) {
    throw new Error(`The selected relationship ${selectedEdgeId} is not present in this graph.`);
  }

  let scope: "selected-edge" | "manual-selection" | "visible-graph";
  let nodes: GraphNode[];
  let edges: GraphEdge[];

  if (selectedEdge) {
    scope = "selected-edge";
    nodes = [graphNodeMap.get(selectedEdge.source), graphNodeMap.get(selectedEdge.target)].filter(
      (node): node is GraphNode => node !== undefined,
    );
    edges = [selectedEdge];
  } else if (selectedNodeIds.size > 0) {
    scope = "manual-selection";
    nodes = [...selectedNodeIds]
      .map((id) => graphNodeMap.get(id))
      .filter((node): node is GraphNode => node !== undefined);
    if (nodes.length !== selectedNodeIds.size) {
      throw new Error("At least one selected dataset is not present in this graph.");
    }
    const selected = new Set(nodes.map((node) => node.id));
    edges = input.visibleEdges.filter(
      (edge) => selected.has(edge.source) && selected.has(edge.target),
    );
  } else {
    scope = "visible-graph";
    nodes = input.visibleNodes;
    const visible = new Set(nodes.map((node) => node.id));
    edges = input.visibleEdges.filter(
      (edge) => visible.has(edge.source) && visible.has(edge.target),
    );
  }

  const uniqueNodes = [...new Map(nodes.map((node) => [node.id, node])).values()].sort((a, b) =>
    compareText(a.id, b.id),
  );
  const uniqueEdges = [...new Map(edges.map((edge) => [edge.id, edge])).values()].sort(
    (a, b) =>
      compareText(a.id, b.id) ||
      compareText(a.source, b.source) ||
      compareText(a.target, b.target),
  );

  if (uniqueNodes.length > MAX_EXPORT_NODES || uniqueEdges.length > MAX_EXPORT_EDGES) {
    throw new Error(
      `This export contains ${uniqueNodes.length.toLocaleString()} datasets and ${uniqueEdges.length.toLocaleString()} relationships. Narrow the visible graph before exporting.`,
    );
  }

  if (
    selectedEdge &&
    (!graphNodeMap.has(selectedEdge.source) || !graphNodeMap.has(selectedEdge.target))
  ) {
    throw new Error("The selected relationship references a dataset missing from this graph.");
  }

  const exportedSelectionIds =
    scope === "selected-edge"
      ? uniqueNodes.map((node) => node.id)
      : scope === "manual-selection"
        ? [...selectedNodeIds].sort(compareText)
        : [];

  return {
    schemaVersion: "cskl-atlas.research-export.v1" as const,
    provenance: {
      ...optional("snapshotId", graph.snapshotId),
      release: graph.release,
      platform: graph.platform,
      ...optional("releaseStatus", graph.releaseStatus),
      ...optional("publishedAt", graph.publishedAt),
      ...optional("calibrationId", graph.calibrationId),
      ...optional("textReleaseId", graph.textReleaseId),
      ...optional("policyHash", graph.policyHash),
      note: graph.note,
    },
    view: {
      evidenceMode: input.view.evidenceMode,
      clusterMode: input.view.clusterMode,
      lens: input.view.lens,
      filters: {
        search: input.view.search,
        activeTissueSystems: [...input.view.activeTissueSystems].sort(compareText),
        minSamples: requireFinite(input.view.minSamples, "view.minSamples"),
      },
    },
    scope: {
      type: scope,
      selectedNodeIds: exportedSelectionIds,
      ...optional("selectedEdgeId", selectedEdge?.id),
    },
    nodes: uniqueNodes.map(normalizeNode),
    edges: uniqueEdges.map(normalizeEdge),
  };
}

export type ResearchExport = ReturnType<typeof buildResearchExport>;

function assertSerializedSize(content: string) {
  const bytes = new TextEncoder().encode(content).byteLength;
  if (bytes > MAX_EXPORT_BYTES) {
    throw new Error("This export exceeds the 64 MiB browser limit. Narrow the visible graph first.");
  }
  return content;
}

export function serializeResearchExportJson(researchExport: ResearchExport) {
  return assertSerializedSize(`${JSON.stringify(researchExport, null, 2)}\n`);
}

const csvColumns = [
  "schema_version",
  "record_type",
  "scope_type",
  "selected_node_ids",
  "selected_edge_id",
  "snapshot_id",
  "release",
  "release_note",
  "release_status",
  "published_at",
  "calibration_id",
  "release_text_id",
  "policy_hash",
  "platform",
  "evidence_mode",
  "cluster_mode",
  "lens",
  "filter_search",
  "filter_active_tissue_systems",
  "filter_min_samples",
  "node_id",
  "dataset_uid",
  "version_id",
  "title",
  "tissue",
  "tissue_system",
  "tissue_system_source",
  "disease",
  "disease_label_source",
  "disease_family",
  "samples",
  "node_platform",
  "organism",
  "layout_x",
  "layout_y",
  "community",
  "annotation_confidence",
  "annotation_source",
  "annotation_state",
  "annotation_candidates_json",
  "summary",
  "feature_hash",
  "config_hash",
  "edge_id",
  "source",
  "target",
  "molecular_cskl",
  "molecular_cskl_percentile",
  "molecular_p_value",
  "molecular_q_value",
  "molecular_independent_q_value",
  "molecular_algorithm_hash",
  "text_specter2",
  "text_provenance",
  "edge_text_release_id",
  "overlap_shared_samples",
  "overlap_fraction",
  "overlap_classification",
  "overlap_discovery_excluded",
  "overlap_coefficient",
  "overlap_jaccard",
  "overlap_fraction_source",
  "overlap_fraction_target",
  "overlap_evidence_id",
  "explainer_provenance",
  "explainer_b_set_json",
  "explainer_w_set_json",
  "explainer_trajectory_json",
  "explainer_best_pathways_json",
  "explainer_worst_pathways_json",
  "explainer_reactome_release",
] as const;

type CsvRow = Partial<Record<(typeof csvColumns)[number], string | number | boolean>>;

export function sanitizeCsvCell(value: unknown) {
  if (value === undefined || value === null) return '""';
  let rendered = typeof value === "string" ? value : String(value);
  if (
    typeof value === "string" &&
    (/^[\t\r\n]/.test(rendered) || /^\s*[=+\-@]/.test(rendered))
  ) {
    rendered = `'${rendered}`;
  }
  return `"${rendered.replaceAll('"', '""')}"`;
}

function provenanceColumns(researchExport: ResearchExport): CsvRow {
  return {
    schema_version: researchExport.schemaVersion,
    scope_type: researchExport.scope.type,
    selected_node_ids: JSON.stringify(researchExport.scope.selectedNodeIds),
    selected_edge_id: researchExport.scope.selectedEdgeId,
    snapshot_id: researchExport.provenance.snapshotId,
    release: researchExport.provenance.release,
    release_note: researchExport.provenance.note,
    release_status: researchExport.provenance.releaseStatus,
    published_at: researchExport.provenance.publishedAt,
    calibration_id: researchExport.provenance.calibrationId,
    release_text_id: researchExport.provenance.textReleaseId,
    policy_hash: researchExport.provenance.policyHash,
    platform: researchExport.provenance.platform,
    evidence_mode: researchExport.view.evidenceMode,
    cluster_mode: researchExport.view.clusterMode,
    lens: researchExport.view.lens,
    filter_search: researchExport.view.filters.search,
    filter_active_tissue_systems: JSON.stringify(researchExport.view.filters.activeTissueSystems),
    filter_min_samples: researchExport.view.filters.minSamples,
  };
}

export function serializeResearchExportCsv(researchExport: ResearchExport) {
  const context = provenanceColumns(researchExport);
  const rows: CsvRow[] = [{ record_type: "manifest", ...context }];

  for (const node of researchExport.nodes) {
    rows.push({
      record_type: "node",
      ...context,
      node_id: node.id,
      dataset_uid: node.datasetUid,
      version_id: node.versionId,
      title: node.title,
      tissue: node.tissue,
      tissue_system: node.tissueSystem,
      tissue_system_source: node.tissueSystemSource,
      disease: node.disease,
      disease_label_source: node.diseaseLabelSource,
      disease_family: node.diseaseFamily,
      samples: node.samples,
      node_platform: node.platform,
      organism: node.organism,
      layout_x: node.layout.x,
      layout_y: node.layout.y,
      community: node.layout.community,
      annotation_confidence: node.annotation.confidence,
      annotation_source: node.annotation.source,
      annotation_state: node.annotation.state,
      annotation_candidates_json: node.annotation.candidates
        ? JSON.stringify(node.annotation.candidates)
        : undefined,
      summary: node.summary,
      feature_hash: node.featureHash,
      config_hash: node.configHash,
    });
  }

  for (const edge of researchExport.edges) {
    rows.push({
      record_type: "edge",
      ...context,
      edge_id: edge.id,
      source: edge.source,
      target: edge.target,
      molecular_cskl: edge.molecularEvidence.cskl,
      molecular_cskl_percentile: edge.molecularEvidence.csklPercentile,
      molecular_p_value: edge.molecularEvidence.pValue,
      molecular_q_value: edge.molecularEvidence.qValue,
      molecular_independent_q_value: edge.molecularEvidence.independentQValue,
      molecular_algorithm_hash: edge.molecularEvidence.algorithmHash,
      text_specter2:
        edge.textEvidence?.provenance === "computed" ? edge.textEvidence.specter2 : undefined,
      text_provenance: edge.textEvidence?.provenance,
      edge_text_release_id:
        edge.textEvidence?.provenance === "computed"
          ? edge.textEvidence.textReleaseId
          : undefined,
      overlap_shared_samples: edge.sampleOverlapEvidence.sharedSamples,
      overlap_fraction: edge.sampleOverlapEvidence.overlapFraction,
      overlap_classification: edge.sampleOverlapEvidence.classification,
      overlap_discovery_excluded: edge.sampleOverlapEvidence.discoveryExcluded,
      overlap_coefficient: edge.sampleOverlapEvidence.overlapCoefficient,
      overlap_jaccard: edge.sampleOverlapEvidence.jaccard,
      overlap_fraction_source: edge.sampleOverlapEvidence.fractionSource,
      overlap_fraction_target: edge.sampleOverlapEvidence.fractionTarget,
      overlap_evidence_id: edge.sampleOverlapEvidence.evidenceId,
      explainer_provenance: edge.explainer?.provenance,
      explainer_b_set_json: edge.explainer ? JSON.stringify(edge.explainer.bSet) : undefined,
      explainer_w_set_json: edge.explainer ? JSON.stringify(edge.explainer.wSet) : undefined,
      explainer_trajectory_json: edge.explainer
        ? JSON.stringify(edge.explainer.trajectory)
        : undefined,
      explainer_best_pathways_json: edge.explainer
        ? JSON.stringify(edge.explainer.bestPathways ?? [])
        : undefined,
      explainer_worst_pathways_json: edge.explainer
        ? JSON.stringify(edge.explainer.worstPathways ?? [])
        : undefined,
      explainer_reactome_release: edge.explainer?.reactomeRelease,
    });
  }

  const lines = [
    csvColumns.map(sanitizeCsvCell).join(","),
    ...rows.map((row) => csvColumns.map((column) => sanitizeCsvCell(row[column])).join(",")),
  ];
  return assertSerializedSize(`${lines.join("\r\n")}\r\n`);
}

export function researchExportFilename(researchExport: ResearchExport, extension: "json" | "csv") {
  const source = researchExport.provenance.snapshotId ?? researchExport.provenance.release;
  const releaseSlug = source
    .normalize("NFKD")
    .replace(/[^a-zA-Z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .toLowerCase()
    .slice(0, 64) || "release";
  return `cskl-atlas-${releaseSlug}-${researchExport.scope.type}.${extension}`;
}

export function downloadResearchExport(content: string, filename: string, mediaType: string) {
  const blob = new Blob([content], { type: `${mediaType};charset=utf-8` });
  if (blob.size > MAX_EXPORT_BYTES) {
    throw new Error("This export exceeds the 64 MiB browser limit. Narrow the visible graph first.");
  }
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.rel = "noopener";
  anchor.hidden = true;
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 0);
}
