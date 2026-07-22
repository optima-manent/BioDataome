export type Tissue = string;

export type DiseaseFamily =
  | "Oncology"
  | "Exposure"
  | "Metabolic"
  | "Reference"
  | "Other";

export type AnnotationProvenance =
  | "geo_structured"
  | "deterministic_ontology"
  | "llm_candidate"
  | "human_verified"
  | "unknown";

export type AnnotationCandidate = {
  label: string;
  ontologyId?: string;
  sourceKind: string;
  reviewState: string;
  extractorVersion: string;
  ontologyValidation?: string;
};

export type OverlapClassification = "none" | "minor" | "major" | "exact" | "unknown";

export type GraphNode = {
  id: string;
  datasetUid?: string;
  versionId?: string;
  title: string;
  tissue: Tissue;
  tissueSystem?: Tissue;
  tissueSystemSource?: string;
  disease: string;
  diseaseLabelSource?: string;
  diseaseFamily: DiseaseFamily;
  samples: number;
  platform: string;
  organism: string;
  x: number;
  y: number;
  annotationConfidence: number;
  annotationSource: AnnotationProvenance;
  annotationState?: string;
  annotationCandidates?: Record<string, AnnotationCandidate[]>;
  summary: string;
  community?: string;
  featureHash?: string;
  configHash?: string;
};

export type ExplainerFeature = {
  feature: string;
  gene?: string;
};

export type ExplainerTrajectoryPoint = {
  k: number;
  bestObjective: number;
  worstObjective: number;
  randomObjective: number;
};

export type PathwayEnrichment = {
  pathway_id: string;
  pathway_name: string;
  url: string;
  overlap_count: number;
  fold_enrichment: number;
  q_value: number;
};

export type EdgeExplainer = {
  provenance: "computed";
  bSet: ExplainerFeature[];
  wSet: ExplainerFeature[];
  trajectory: ExplainerTrajectoryPoint[];
  bestPathways?: PathwayEnrichment[];
  worstPathways?: PathwayEnrichment[];
  reactomeRelease?: string;
  interpretation?: string;
};

export type GraphEdge = {
  id: string;
  source: string;
  target: string;
  cskl: number;
  csklPercentile?: number;
  pValue?: number;
  qValue: number;
  independentQValue?: number;
  specter2?: number;
  specter2Provenance?: "computed";
  sharedSamples: number;
  overlapFraction: number;
  overlapClassification?: OverlapClassification;
  discoveryExcluded?: boolean;
  overlapCoefficient?: number;
  jaccard?: number;
  overlapFractionSource?: number;
  overlapFractionTarget?: number;
  algorithmHash?: string;
  overlapEvidenceId?: string;
  textReleaseId?: string;
  explainer?: EdgeExplainer;
};

export type GraphDataset = {
  snapshotId?: string;
  release: string;
  platform: string;
  note: string;
  releaseStatus?: "published" | "draft";
  publishedAt?: string;
  calibrationId?: string;
  textReleaseId?: string;
  policyHash?: string;
  nodes: GraphNode[];
  edges: GraphEdge[];
};

export const tissueColors: Record<string, string> = {
  "Blood & immune": "#cf7de5",
  Respiratory: "#4ad6c3",
  "Digestive & hepatobiliary": "#ff9f5b",
  "Nervous system": "#68afff",
  Skin: "#f2c15d",
  Breast: "#ff7fa7",
  Reproductive: "#e996d7",
  "Kidney & urinary": "#74c6f2",
  Musculoskeletal: "#bddb68",
  Cardiovascular: "#f07878",
  "Endocrine & metabolic": "#9fd069",
  Ocular: "#74b7ff",
  "Cell culture / in vitro": "#aab3c5",
  "Mixed anatomical systems": "#c6a77a",
  "Mixed / unspecified": "#8995a8",
  "Other anatomy": "#9aa7b7",
};

export function nodeTissueSystem(node: Pick<GraphNode, "tissue" | "tissueSystem">): Tissue {
  return node.tissueSystem?.trim() || node.tissue;
}

export function tissueColor(tissue: string): string {
  const known = tissueColors[tissue];
  if (known) return known;
  let hash = 2166136261;
  for (const character of tissue) {
    hash ^= character.codePointAt(0) ?? 0;
    hash = Math.imul(hash, 16777619);
  }
  const hue = Math.abs(hash) % 360;
  return `hsl(${hue} 58% 62%)`;
}

export const diseaseShapeLabel: Record<DiseaseFamily, string> = {
  Oncology: "hexagon",
  Exposure: "triangle",
  Metabolic: "diamond",
  Reference: "square",
  Other: "circle",
};
