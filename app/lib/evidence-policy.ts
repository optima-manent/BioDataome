import type { GraphEdge, GraphNode } from "./graph-data";

export type EvidenceLens = "all" | "agreement" | "cskl-only" | "cross-disease" | "overlap";

export function formatProbability(value: number): string {
  if (value === 0) return "0";
  return value < 0.001 ? value.toExponential(2) : value.toFixed(3);
}

export function publishedOverlapClassification({
  classification,
  evidenceId,
  sharedCount,
  fractionSource,
  fractionTarget,
  overlapCoefficient,
  jaccard,
  discoveryExcluded,
}: {
  classification?: NonNullable<GraphEdge["overlapClassification"]> | null;
  evidenceId?: unknown;
  sharedCount?: unknown;
  fractionSource?: unknown;
  fractionTarget?: unknown;
  overlapCoefficient?: unknown;
  jaccard?: unknown;
  discoveryExcluded?: unknown;
}): NonNullable<GraphEdge["overlapClassification"]> {
  if (classification) return classification;
  const positive = (value: unknown) =>
    typeof value === "number" && Number.isFinite(value) && value > 0;
  const hasOverlapEvidence =
    (typeof evidenceId === "string" && Boolean(evidenceId.trim())) ||
    positive(sharedCount) ||
    positive(fractionSource) ||
    positive(fractionTarget) ||
    positive(overlapCoefficient) ||
    positive(jaccard) ||
    discoveryExcluded === true ||
    discoveryExcluded === 1;
  return hasOverlapEvidence ? "unknown" : "none";
}

export function computedSpecter2(edge: GraphEdge): number | undefined {
  return edge.specter2Provenance === "computed" &&
    typeof edge.specter2 === "number" &&
    Number.isFinite(edge.specter2)
    ? edge.specter2
    : undefined;
}

export function hasComputedSpecter2(edges: readonly GraphEdge[]): boolean {
  return edges.some((edge) => computedSpecter2(edge) !== undefined);
}

export function isOverlapQualified(edge: GraphEdge): boolean {
  return (
    edge.discoveryExcluded === true ||
    (edge.overlapClassification !== undefined && edge.overlapClassification !== "none")
  );
}

export function usesDottedOverlapStyle(edge: GraphEdge): boolean {
  return edge.overlapClassification === "major" || edge.overlapClassification === "exact";
}

export function edgeMatchesLens({
  edge,
  lens,
  csklMedian,
  source,
  target,
}: {
  edge: GraphEdge;
  lens: EvidenceLens;
  csklMedian: number;
  source: GraphNode;
  target: GraphNode;
}): boolean {
  if (lens === "agreement") {
    const score = computedSpecter2(edge);
    return score !== undefined && score >= 0.75;
  }
  if (lens === "cskl-only") {
    const score = computedSpecter2(edge);
    return score !== undefined && edge.cskl <= csklMedian && score < 0.65;
  }
  if (lens === "cross-disease") {
    return source.tissue === target.tissue && source.disease !== target.disease;
  }
  if (lens === "overlap") return isOverlapQualified(edge);
  return true;
}

export function edgeForScientificEvidence(edge: GraphEdge): GraphEdge {
  if (computedSpecter2(edge) !== undefined) return { ...edge };
  const scientificEdge = { ...edge };
  delete scientificEdge.specter2;
  delete scientificEdge.specter2Provenance;
  return scientificEdge;
}

const AI_GROUP_NODE_LIMIT = 50;
const AI_GROUP_EDGE_LIMIT = 100;
// Leave headroom under the 80 kB API request ceiling for headers and encoding.
// Omit complete records deterministically here; never truncate serialized JSON.
const AI_PACKET_BYTE_LIMIT = 60_000;

function utf8Bytes(value: unknown): number {
  return new TextEncoder().encode(JSON.stringify(value)).byteLength;
}

function compactGroupNode(node: GraphNode) {
  const annotationCandidates = node.annotationCandidates;
  const reviewedState = new Set(["accepted", "human_verified"]).has(
    node.annotationState ?? "",
  );
  const semanticFactsTrusted =
    node.annotationSource === "geo_structured" ||
    node.annotationSource === "human_verified" ||
    (node.annotationSource === "deterministic_ontology" && reviewedState);
  const semanticFields = new Set([
    "tissue",
    "tissueSystem",
    "tissueSystemSource",
    "disease",
    "diseaseFamily",
    "diseaseLabelSource",
  ]);
  const scientific = Object.fromEntries(
    Object.entries(node).filter(
      ([key]) =>
        key !== "x" &&
        key !== "y" &&
        key !== "annotationCandidates" &&
        (semanticFactsTrusted || !semanticFields.has(key)),
    ),
  );
  const reviewedCandidates = Object.fromEntries(
    Object.entries(annotationCandidates ?? {})
      .map(([field, candidates]) => [
        field,
        candidates
          .filter(
            (candidate) => {
              const reviewed =
                candidate.reviewState === "accepted" ||
                candidate.reviewState === "human_verified";
              const ontologyCompatible =
                !candidate.ontologyId ||
                candidate.ontologyValidation === "canonical_or_synonym";
              return reviewed && ontologyCompatible;
            },
          )
          .slice(0, 3),
      ])
      .filter(([, candidates]) => candidates.length > 0),
  );
  return {
    ...scientific,
    ...(!semanticFactsTrusted
      ? { semantic_annotation_policy: "withheld_until_reviewed" }
      : {}),
    ...(Object.keys(reviewedCandidates).length > 0
      ? { annotationCandidates: reviewedCandidates }
      : {}),
  };
}

function compactGroupEdge(edge: GraphEdge) {
  const scientific = edgeForScientificEvidence(edge);
  const explainer = scientific.explainer;
  return {
    ...scientific,
    explainer: explainer
      ? {
          provenance: explainer.provenance,
          bSet: explainer.bSet.slice(0, 5),
          wSet: explainer.wSet.slice(0, 5),
          trajectory: explainer.trajectory,
          bestPathways: explainer.bestPathways?.slice(0, 2),
          worstPathways: explainer.worstPathways?.slice(0, 2),
          reactomeRelease: explainer.reactomeRelease,
          interpretation: explainer.interpretation,
        }
      : undefined,
  };
}

export function buildAiEvidencePacket({
  selectedEdge,
  selectedNodes,
  visibleEdges,
  selectedNodeIds,
  nodeMap,
}: {
  selectedEdge: GraphEdge | null;
  selectedNodes: GraphNode[];
  visibleEdges: GraphEdge[];
  selectedNodeIds: Set<string>;
  nodeMap: Map<string, GraphNode>;
}) {
  if (selectedEdge) {
    return {
      type: "edge" as const,
      packet_policy: {
        byte_limit: AI_PACKET_BYTE_LIMIT,
        unreviewed_semantic_candidates_excluded: true,
      },
      edge: compactGroupEdge(selectedEdge),
      source: nodeMap.get(selectedEdge.source)
        ? compactGroupNode(nodeMap.get(selectedEdge.source)!)
        : undefined,
      target: nodeMap.get(selectedEdge.target)
        ? compactGroupNode(nodeMap.get(selectedEdge.target)!)
        : undefined,
    };
  }
  const inducedEdges = visibleEdges
    .filter((edge) => selectedNodeIds.has(edge.source) && selectedNodeIds.has(edge.target))
    .sort(
      (left, right) =>
        left.qValue - right.qValue || left.cskl - right.cskl || left.id.localeCompare(right.id),
    );
  const orderedNodes = [...selectedNodes].sort((left, right) => left.id.localeCompare(right.id));
  const packet = {
    type: "selection" as const,
    selection_summary: {
      dataset_count: orderedNodes.length,
      edge_count: inducedEdges.length,
      significant_edge_count: inducedEdges.filter((edge) => edge.qValue <= 0.05).length,
      overlap_qualified_edge_count: inducedEdges.filter(isOverlapQualified).length,
      computed_text_edge_count: inducedEdges.filter(
        (edge) => computedSpecter2(edge) !== undefined,
      ).length,
    },
    packet_policy: {
      dataset_limit: AI_GROUP_NODE_LIMIT,
      edge_limit: AI_GROUP_EDGE_LIMIT,
      byte_limit: AI_PACKET_BYTE_LIMIT,
      unreviewed_semantic_candidates_excluded: true,
      dataset_order: "accession ascending",
      edge_order: "global q-value, C-SKL, edge id",
      omitted_dataset_count: orderedNodes.length,
      omitted_edge_count: inducedEdges.length,
    },
    datasets: [] as ReturnType<typeof compactGroupNode>[],
    edges: [] as ReturnType<typeof compactGroupEdge>[],
  };

  for (const node of orderedNodes.slice(0, AI_GROUP_NODE_LIMIT)) {
    const compact = compactGroupNode(node);
    const candidate = { ...packet, datasets: [...packet.datasets, compact] };
    if (utf8Bytes(candidate) > AI_PACKET_BYTE_LIMIT) break;
    packet.datasets.push(compact);
  }
  for (const edge of inducedEdges.slice(0, AI_GROUP_EDGE_LIMIT)) {
    const compact = compactGroupEdge(edge);
    const candidate = { ...packet, edges: [...packet.edges, compact] };
    if (utf8Bytes(candidate) > AI_PACKET_BYTE_LIMIT) break;
    packet.edges.push(compact);
  }
  packet.packet_policy.omitted_dataset_count = orderedNodes.length - packet.datasets.length;
  packet.packet_policy.omitted_edge_count = inducedEdges.length - packet.edges.length;
  return packet;
}
