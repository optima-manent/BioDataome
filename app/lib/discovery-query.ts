import {
  computedSpecter2,
  isOverlapQualified,
} from "./evidence-policy.ts";
import { nodeTissueSystem, type GraphEdge, type GraphNode } from "./graph-data.ts";

export type RelationOperator = "any" | "same" | "different";
export type IndependenceOperator = "any" | "independent" | "overlap-qualified";
export type ThresholdOperator = "any" | "gte" | "lte";

export type DiscoveryQuery = {
  independence: IndependenceOperator;
  qMax: number | null;
  csklPercentileMin: number | null;
  tissueRelation: RelationOperator;
  diseaseRelation: RelationOperator;
  specter2Operator: ThresholdOperator;
  specter2Percentile: number | null;
  mechanismTerm: string;
};

export const DEFAULT_DISCOVERY_QUERY: DiscoveryQuery = {
  independence: "independent",
  qMax: 0.05,
  csklPercentileMin: 0.95,
  tissueRelation: "any",
  diseaseRelation: "any",
  specter2Operator: "any",
  specter2Percentile: null,
  mechanismTerm: "",
};

export const EMPTY_DISCOVERY_QUERY: DiscoveryQuery = {
  independence: "any",
  qMax: null,
  csklPercentileMin: null,
  tissueRelation: "any",
  diseaseRelation: "any",
  specter2Operator: "any",
  specter2Percentile: null,
  mechanismTerm: "",
};

function known(value: string, unknownValues: readonly string[]) {
  const normalized = value.trim().toLocaleLowerCase();
  return Boolean(normalized) && !unknownValues.includes(normalized);
}

function relationMatches(
  left: string,
  right: string,
  relation: RelationOperator,
  unknownValues: readonly string[],
) {
  if (relation === "any") return true;
  if (!known(left, unknownValues) || !known(right, unknownValues)) return false;
  const same = left.trim().toLocaleLowerCase() === right.trim().toLocaleLowerCase();
  return relation === "same" ? same : !same;
}

function mechanismValues(edge: GraphEdge) {
  const explainer = edge.explainer;
  if (!explainer) return [];
  return [
    ...explainer.bSet.flatMap((item) => [item.feature, item.gene ?? ""]),
    ...explainer.wSet.flatMap((item) => [item.feature, item.gene ?? ""]),
    ...(explainer.bestPathways ?? []).flatMap((item) => [item.pathway_id, item.pathway_name]),
    ...(explainer.worstPathways ?? []).flatMap((item) => [item.pathway_id, item.pathway_name]),
  ];
}

export function edgeMatchesDiscoveryQuery({
  edge,
  source,
  target,
  query,
}: {
  edge: GraphEdge;
  source: GraphNode;
  target: GraphNode;
  query: DiscoveryQuery;
}) {
  const overlapQualified = isOverlapQualified(edge);
  if (query.independence === "independent" && overlapQualified) return false;
  if (query.independence === "overlap-qualified" && !overlapQualified) return false;
  if (query.qMax !== null && edge.qValue > query.qMax) return false;
  if (
    query.csklPercentileMin !== null &&
    (edge.csklPercentile === undefined || edge.csklPercentile < query.csklPercentileMin)
  ) {
    return false;
  }
  if (
    query.tissueRelation !== "any" &&
    [source, target].some(
      (node) => node.tissueSystemSource !== "ontology_label_concordant",
    )
  ) {
    return false;
  }
  if (
    query.diseaseRelation !== "any" &&
    [source, target].some(
      (node) => node.diseaseLabelSource !== "ontology_label_concordant",
    )
  ) {
    return false;
  }
  if (
    !relationMatches(
      nodeTissueSystem(source),
      nodeTissueSystem(target),
      query.tissueRelation,
      [
        "mixed anatomical systems",
        "mixed / unknown",
        "mixed / unspecified",
        "other anatomy",
        "unknown",
        "unknown / not reviewed",
      ],
    )
  ) {
    return false;
  }
  if (
    !relationMatches(
      source.disease,
      target.disease,
      query.diseaseRelation,
      ["unknown", "unknown / not reviewed"],
    )
  ) {
    return false;
  }
  if (query.specter2Operator !== "any") {
    const score = computedSpecter2(edge);
    if (score === undefined || query.specter2Percentile === null) return false;
    if (query.specter2Operator === "gte" && score < query.specter2Percentile) return false;
    if (query.specter2Operator === "lte" && score > query.specter2Percentile) return false;
  }
  const mechanism = query.mechanismTerm.trim().toLocaleLowerCase();
  if (
    mechanism &&
    !mechanismValues(edge).some((value) => value.toLocaleLowerCase().includes(mechanism))
  ) {
    return false;
  }
  return true;
}

export function discoveryQueryAst(query: DiscoveryQuery) {
  const predicates: Record<string, unknown>[] = [];
  if (query.independence !== "any") {
    predicates.push({ "edge.independent": { eq: query.independence === "independent" } });
  }
  if (query.qMax !== null) predicates.push({ "edge.q_value": { lte: query.qMax } });
  if (query.csklPercentileMin !== null) {
    predicates.push({ "edge.cskl_percentile": { gte: query.csklPercentileMin } });
  }
  if (query.tissueRelation !== "any") {
    predicates.push({ "node.tissue_system": { [query.tissueRelation]: true } });
  }
  if (query.diseaseRelation !== "any") {
    predicates.push({ "node.disease": { [query.diseaseRelation]: true } });
  }
  if (query.specter2Operator !== "any" && query.specter2Percentile !== null) {
    predicates.push({
      "edge.specter2_percentile": {
        [query.specter2Operator]: query.specter2Percentile,
      },
    });
  }
  if (query.mechanismTerm.trim()) {
    predicates.push({
      "explanation.feature_or_pathway": { contains: query.mechanismTerm.trim() },
    });
  }
  if (!predicates.length) return { all: true };
  return predicates.length === 1 ? predicates[0] : { and: predicates };
}

export function queryIsValid(query: DiscoveryQuery) {
  const probability = (value: number | null) =>
    value === null || (Number.isFinite(value) && value >= 0 && value <= 1);
  return (
    probability(query.qMax) &&
    probability(query.csklPercentileMin) &&
    probability(query.specter2Percentile) &&
    (query.specter2Operator === "any" || query.specter2Percentile !== null)
  );
}
