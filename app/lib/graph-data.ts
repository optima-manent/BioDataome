export type Tissue = string;

export type DiseaseFamily =
  | "Neoplastic"
  | "Infectious"
  | "Immune & inflammatory"
  | "Endocrine & metabolic"
  | "Neurological & mental health"
  | "Cardiovascular"
  | "Respiratory"
  | "Digestive & hepatic"
  | "Renal & reproductive"
  | "Injury & exposure"
  | "Reference / no disease"
  | "Mixed disease families"
  | "Other / unclassified"
  | "Unreviewed";

export type TissueShape = "circle" | "triangle" | "diamond" | "square" | "hexagon" | "cross";
export type NodeColorMode = "tissue" | "disease";

export const DISEASE_FAMILY_VERSION = "atlas-clinical-family-v2";

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
  diseaseFamilyVersion?: string;
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

export const diseaseFamilyColors: Record<DiseaseFamily, string> = {
  Neoplastic: "#f07ba2",
  Infectious: "#f2b84b",
  "Immune & inflammatory": "#b78aef",
  "Endocrine & metabolic": "#91ce68",
  "Neurological & mental health": "#70adf5",
  Cardiovascular: "#ef746c",
  Respiratory: "#48cfc0",
  "Digestive & hepatic": "#ee9558",
  "Renal & reproductive": "#dc91cf",
  "Injury & exposure": "#c5a173",
  "Reference / no disease": "#a8b5c5",
  "Mixed disease families": "#d7a7e6",
  "Other / unclassified": "#8996a8",
  Unreviewed: "#647386",
};

const diseaseFamilyValues = new Set<DiseaseFamily>(
  Object.keys(diseaseFamilyColors) as DiseaseFamily[],
);

const diseaseFamilyPatterns: Array<[DiseaseFamily, RegExp[]]> = [
  [
    "Neoplastic",
    [
      /\bcancer\b/i,
      /\bcarcinoma\b/i,
      /\badenocarcinoma\b/i,
      /\btumou?r\b/i,
      /\bneoplasm\b/i,
      /\bleuk(?:a?emia)\b/i,
      /\blymphoma\b/i,
      /\bmelanoma\b/i,
      /\bmyeloma\b/i,
      /\bsarcoma\b/i,
      /\bblastoma\b/i,
      /\bglioma\b/i,
      /\bglioblastoma\b/i,
      /\bmesothelioma\b/i,
      /\bmalignan/i,
      /\bmyelodysplastic\b/i,
    ],
  ],
  [
    "Infectious",
    [
      /\binfect(?:ion|ious|ed)\b/i,
      /\bviral\b/i,
      /\bvirus\b/i,
      /\bhiv\b/i,
      /\bhepatitis\b/i,
      /\binfluenza\b/i,
      /\btuberculosis\b/i,
      /\bsepsis\b/i,
      /\bseptic\b/i,
      /\bbacter(?:ia|ial)\b/i,
    ],
  ],
  [
    "Immune & inflammatory",
    [
      /\bautoimmune\b/i,
      /\barthritis\b/i,
      /\bdermatitis\b/i,
      /\bpsoriasis\b/i,
      /\binflamm/i,
      /\bmultiple sclerosis\b/i,
      /\bsarcoidosis\b/i,
      /\bvasculitis\b/i,
      /\ballerg/i,
      /\bcrohn/i,
      /\bcolitis\b/i,
      /\bco?eliac\b/i,
      /\blupus\b/i,
      /\bscleroderma\b/i,
    ],
  ],
  [
    "Endocrine & metabolic",
    [
      /\bdiabet/i,
      /\bobes/i,
      /\bmetabolic\b/i,
      /\binsulin\b/i,
      /\bthyroid\b/i,
      /\badrenal\b/i,
      /\bhyperglyc/i,
      /\bdyslipid/i,
    ],
  ],
  [
    "Neurological & mental health",
    [
      /\balzheimer/i,
      /\bparkinson/i,
      /\bhuntington/i,
      /\bdementia\b/i,
      /\bschizophren/i,
      /\bdepress/i,
      /\banxiety\b/i,
      /\bbipolar\b/i,
      /\bpsychot/i,
      /\bautis/i,
      /\bneurolog/i,
      /\bneuropath/i,
      /\bepilep/i,
      /\bmood disorder\b/i,
      /\bpost-traumatic stress\b/i,
    ],
  ],
  [
    "Cardiovascular",
    [
      /\bcardiomyopath/i,
      /\bcardiac\b/i,
      /\bcardiovascular\b/i,
      /\bcoronary\b/i,
      /\bmyocard/i,
      /\batheroscler/i,
      /\bhypertension\b/i,
      /\bheart disease\b/i,
      /\bstroke\b/i,
    ],
  ],
  [
    "Respiratory",
    [
      /\bcopd\b/i,
      /\bobstructive pulmonary\b/i,
      /\basthma\b/i,
      /\bcystic fibrosis\b/i,
      /\brespiratory\b/i,
      /\bpulmonary\b/i,
      /\bairway\b/i,
      /\bemphysema\b/i,
    ],
  ],
  [
    "Digestive & hepatic",
    [
      /\bcirrhosis\b/i,
      /\bliver disease\b/i,
      /\bhepatic disease\b/i,
      /\bbowel disease\b/i,
      /\bgastric disease\b/i,
      /\bpancreatitis\b/i,
      /\bintestinal disease\b/i,
    ],
  ],
  [
    "Renal & reproductive",
    [
      /\bkidney disease\b/i,
      /\brenal disease\b/i,
      /\bnephro/i,
      /\bendometriosis\b/i,
      /\binfertil/i,
      /\bpolycystic ovar/i,
      /\bpreeclampsia\b/i,
    ],
  ],
  [
    "Injury & exposure",
    [
      /\bburn\b/i,
      /\binjur/i,
      /\btrauma\b/i,
      /\btobacco\b/i,
      /\bsmoking\b/i,
      /\birradiat/i,
      /\bradiation exposure\b/i,
      /\btoxic/i,
      /\bexpos(?:ure|ed)\b/i,
      /\bpoison/i,
    ],
  ],
  [
    "Reference / no disease",
    [/\bhealthy\b/i, /\bnormal\b/i, /\bcontrol\b/i, /\breference\b/i, /\bnon-diabetic\b/i],
  ],
];

function classifyDiseaseSegment(value: string): DiseaseFamily {
  const normalized = value.trim().replace(/\s+/g, " ");
  if (!normalized || /^(?:unknown(?: \/ not reviewed)?|not reviewed)$/i.test(normalized)) {
    return "Unreviewed";
  }
  for (const [family, patterns] of diseaseFamilyPatterns) {
    if (patterns.some((pattern) => pattern.test(normalized))) return family;
  }
  return "Other / unclassified";
}

/**
 * Derive a broad, display-only clinical family. This does not assign an ICD-10
 * code, and unvalidated source labels stay visibly unreviewed.
 */
export function deriveDiseaseFamily(
  disease: string,
  {
    labelSource,
    declaredFamily,
    declaredVersion,
  }: { labelSource?: string; declaredFamily?: unknown; declaredVersion?: unknown } = {},
): DiseaseFamily {
  if (labelSource !== "ontology_label_concordant") return "Unreviewed";
  if (
    declaredVersion === DISEASE_FAMILY_VERSION &&
    typeof declaredFamily === "string" &&
    diseaseFamilyValues.has(declaredFamily as DiseaseFamily)
  ) {
    return declaredFamily as DiseaseFamily;
  }

  const families = new Set(
    disease
      .split(/\s*;\s*/)
      .filter(Boolean)
      .map(classifyDiseaseSegment),
  );
  families.delete("Unreviewed");
  families.delete("Other / unclassified");
  const hasReference = families.delete("Reference / no disease");
  if (families.size > 1) return "Mixed disease families";
  if (families.size === 1) return [...families][0];
  if (hasReference) return "Reference / no disease";
  return disease.trim() ? "Other / unclassified" : "Unreviewed";
}

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

export function diseaseFamilyColor(family: DiseaseFamily): string {
  return diseaseFamilyColors[family] ?? diseaseFamilyColors["Other / unclassified"];
}

export function nodeColor(
  node: Pick<GraphNode, "diseaseFamily" | "tissue" | "tissueSystem">,
  mode: NodeColorMode,
): string {
  return mode === "disease"
    ? diseaseFamilyColor(node.diseaseFamily)
    : tissueColor(nodeTissueSystem(node));
}

export function tissueShape(tissue: string): TissueShape {
  if (tissue === "Blood & immune" || tissue === "Cardiovascular") return "circle";
  if (tissue === "Respiratory") return "triangle";
  if (tissue === "Nervous system" || tissue === "Ocular") return "diamond";
  if (tissue === "Skin" || tissue === "Breast" || tissue === "Musculoskeletal") {
    return "square";
  }
  if (
    tissue === "Digestive & hepatobiliary" ||
    tissue === "Endocrine & metabolic" ||
    tissue === "Kidney & urinary" ||
    tissue === "Reproductive"
  ) {
    return "hexagon";
  }
  return "cross";
}

export const tissueShapeLabel: Record<TissueShape, string> = {
  circle: "blood / circulatory",
  triangle: "respiratory",
  diamond: "neural / sensory",
  square: "surface / structural",
  hexagon: "organ tissue",
  cross: "mixed / in vitro",
};
