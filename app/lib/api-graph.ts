import type {
  AnnotationCandidate,
  AnnotationProvenance,
  DiseaseFamily,
  GraphDataset,
  GraphEdge,
  GraphNode,
  Tissue,
} from "./graph-data";
import { publishedOverlapClassification } from "./evidence-policy.ts";

type UnknownRecord = Record<string, unknown>;

type ApiNode = {
  version_id: string;
  dataset_uid: string;
  accession: string;
  platform: string;
  sample_count: number;
  x?: number | null;
  y?: number | null;
  community?: string | null;
  feature_hash?: string | null;
  config_hash?: string | null;
  metadata?: UnknownRecord;
};

type ApiEdge = {
  pair_id: string;
  version_a: string;
  version_b: string;
  cskl: number;
  algorithm_hash?: string | null;
  p_value?: number | null;
  q_value: number;
  independent_q_value?: number | null;
  cskl_similarity_percentile?: number | null;
  shared_count?: number | null;
  fraction_a?: number | null;
  fraction_b?: number | null;
  jaccard?: number | null;
  overlap_coefficient?: number | null;
  classification?: "none" | "minor" | "major" | "exact" | null;
  discovery_excluded?: number | boolean | null;
  overlap_id?: string | null;
  specter2_cosine?: number | null;
  specter2_percentile?: number | null;
  text_release_id?: string | null;
  explainer?: {
    provenance: "computed";
    bSet: Array<{ feature: string; gene?: string | null }>;
    wSet: Array<{ feature: string; gene?: string | null }>;
    trajectory: Array<{
      k: number;
      best_objective: number;
      worst_objective: number;
      random_objective: number;
    }>;
    bestPathways?: Array<{
      pathway_id: string;
      pathway_name: string;
      url: string;
      overlap_count: number;
      fold_enrichment: number;
      q_value: number;
    }>;
    worstPathways?: Array<{
      pathway_id: string;
      pathway_name: string;
      url: string;
      overlap_count: number;
      fold_enrichment: number;
      q_value: number;
    }>;
    reactomeRelease?: string;
    interpretation?: string;
  } | null;
};

type ApiGraph = {
  snapshot: {
    snapshot_id: string;
    stratum: string;
    published_at?: string | null;
    calibration_id?: string;
    text_release_id?: string | null;
    policy_hash?: string;
  };
  nodes: ApiNode[];
  edges: ApiEdge[];
};

const diseaseFamilies = new Set<DiseaseFamily>([
  "Oncology",
  "Exposure",
  "Metabolic",
  "Reference",
  "Other",
]);

function text(value: unknown, fallback: string) {
  return typeof value === "string" && value.trim() ? value.trim() : fallback;
}

function finite(value: unknown, fallback: number) {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function tissue(value: unknown): Tissue {
  return text(value, "Mixed / unknown");
}

function diseaseFamily(value: unknown): DiseaseFamily {
  return typeof value === "string" && diseaseFamilies.has(value as DiseaseFamily)
    ? (value as DiseaseFamily)
    : "Other";
}

const annotationProvenance = new Set<AnnotationProvenance>([
  "geo_structured",
  "deterministic_ontology",
  "llm_candidate",
  "human_verified",
  "unknown",
]);

function annotationSource(value: unknown): AnnotationProvenance {
  return typeof value === "string" && annotationProvenance.has(value as AnnotationProvenance)
    ? (value as AnnotationProvenance)
    : "unknown";
}

function annotationCandidates(value: unknown): Record<string, AnnotationCandidate[]> {
  if (!value || typeof value !== "object" || Array.isArray(value)) return {};
  const result: Record<string, AnnotationCandidate[]> = {};
  for (const [field, rawItems] of Object.entries(value)) {
    if (!Array.isArray(rawItems)) continue;
    const items = rawItems.flatMap((rawItem) => {
      if (!rawItem || typeof rawItem !== "object" || Array.isArray(rawItem)) return [];
      const item = rawItem as UnknownRecord;
      const label = text(item.label, "");
      if (!label) return [];
      return [
        {
          label,
          ontologyId: text(item.ontology_id, "") || undefined,
          sourceKind: text(item.source_kind, "unknown"),
          reviewState: text(item.review_state, "unknown"),
          extractorVersion: text(item.extractor_version, "unknown"),
          ontologyValidation: text(item.ontology_validation, "not_audited"),
        },
      ];
    });
    if (items.length) result[field] = items;
  }
  return result;
}

export function adaptPublishedGraph(payload: ApiGraph): GraphDataset {
  if (!payload || !Array.isArray(payload.nodes) || !Array.isArray(payload.edges)) {
    throw new Error("Atlas API returned an invalid graph payload.");
  }

  const versionToAccession = new Map<string, string>();
  const nodeCount = Math.max(payload.nodes.length, 1);
  const nodes: GraphNode[] = payload.nodes.map((node, index) => {
    const metadata = node.metadata && typeof node.metadata === "object" ? node.metadata : {};
    const accession = text(node.accession, node.version_id);
    versionToAccession.set(node.version_id, accession);
    const phase = (index / nodeCount) * Math.PI * 2;
    return {
      id: accession,
      datasetUid: node.dataset_uid,
      versionId: node.version_id,
      title: text(metadata.title, accession),
      tissue: tissue(metadata.tissue),
      tissueSystem: tissue(metadata.tissue_system ?? metadata.tissue),
      tissueSystemSource: text(metadata.tissue_system_source, "") || undefined,
      disease: text(metadata.disease, "Unknown / not reviewed"),
      diseaseLabelSource: text(metadata.disease_label_source, "") || undefined,
      diseaseFamily: diseaseFamily(metadata.disease_family),
      samples: Math.max(0, Math.trunc(finite(node.sample_count, 0))),
      platform: text(node.platform, payload.snapshot.stratum),
      organism: text(metadata.organism, "Unknown / not reviewed"),
      x: finite(node.x, 0.5 + Math.cos(phase) * 0.34),
      y: finite(node.y, 0.5 + Math.sin(phase) * 0.34),
      annotationConfidence: Math.max(
        0,
        Math.min(1, finite(metadata.annotation_confidence, 0)),
      ),
      annotationSource: annotationSource(metadata.annotation_source),
      annotationState: text(metadata.annotation_state, "") || undefined,
      annotationCandidates: annotationCandidates(metadata.annotation_candidates),
      summary: text(metadata.summary, "No reviewed summary is available for this release."),
      community: text(node.community, "") || undefined,
      featureHash: text(node.feature_hash, "") || undefined,
      configHash: text(node.config_hash, "") || undefined,
    };
  });

  const edges: GraphEdge[] = payload.edges.flatMap((edge) => {
    const source = versionToAccession.get(edge.version_a);
    const target = versionToAccession.get(edge.version_b);
    if (!source || !target) return [];
    return [
      {
        id: edge.pair_id,
        source,
        target,
        cskl: finite(edge.cskl, Number.POSITIVE_INFINITY),
        csklPercentile: finite(edge.cskl_similarity_percentile, Number.NaN),
        pValue: finite(edge.p_value, Number.NaN),
        qValue: Math.max(0, Math.min(1, finite(edge.q_value, 1))),
        independentQValue:
          typeof edge.independent_q_value === "number" && Number.isFinite(edge.independent_q_value)
            ? Math.max(0, Math.min(1, edge.independent_q_value))
            : undefined,
        specter2: finite(edge.specter2_percentile, Number.NaN),
        specter2Provenance:
          Number.isFinite(edge.specter2_percentile) && edge.text_release_id
            ? ("computed" as const)
            : undefined,
        sharedSamples: Math.max(0, Math.trunc(finite(edge.shared_count, 0))),
        overlapFraction: Math.max(
          0,
          Math.min(1, Math.max(finite(edge.fraction_a, 0), finite(edge.fraction_b, 0))),
        ),
        overlapFractionSource: Math.max(0, Math.min(1, finite(edge.fraction_a, 0))),
        overlapFractionTarget: Math.max(0, Math.min(1, finite(edge.fraction_b, 0))),
        overlapCoefficient: Math.max(
          0,
          Math.min(1, finite(edge.overlap_coefficient, 0)),
        ),
        jaccard: Math.max(0, Math.min(1, finite(edge.jaccard, 0))),
        overlapClassification: publishedOverlapClassification({
          classification: edge.classification,
          evidenceId: edge.overlap_id,
          sharedCount: edge.shared_count,
          fractionSource: edge.fraction_a,
          fractionTarget: edge.fraction_b,
          overlapCoefficient: edge.overlap_coefficient,
          jaccard: edge.jaccard,
          discoveryExcluded: edge.discovery_excluded,
        }),
        discoveryExcluded:
          edge.discovery_excluded === true || edge.discovery_excluded === 1,
        algorithmHash: text(edge.algorithm_hash, "") || undefined,
        overlapEvidenceId: text(edge.overlap_id, "") || undefined,
        textReleaseId: text(edge.text_release_id, "") || undefined,
        explainer: edge.explainer
          ? {
              provenance: "computed" as const,
              bSet: edge.explainer.bSet.map((item) => ({
                feature: item.feature,
                gene: item.gene ?? undefined,
              })),
              wSet: edge.explainer.wSet.map((item) => ({
                feature: item.feature,
                gene: item.gene ?? undefined,
              })),
              trajectory: edge.explainer.trajectory.map((item) => ({
                k: item.k,
                bestObjective: item.best_objective,
                worstObjective: item.worst_objective,
                randomObjective: item.random_objective,
              })),
              bestPathways: edge.explainer.bestPathways ?? [],
              worstPathways: edge.explainer.worstPathways ?? [],
              reactomeRelease: edge.explainer.reactomeRelease,
              interpretation: edge.explainer.interpretation,
            }
          : undefined,
      },
    ];
  }).filter((edge) => Number.isFinite(edge.cskl)).map((edge) => ({
    ...edge,
    csklPercentile: Number.isFinite(edge.csklPercentile) ? edge.csklPercentile : undefined,
    pValue: Number.isFinite(edge.pValue) ? edge.pValue : undefined,
    specter2: Number.isFinite(edge.specter2) ? edge.specter2 : undefined,
  }));

  return {
    snapshotId: payload.snapshot.snapshot_id,
    release: `${payload.snapshot.stratum} · ${payload.snapshot.snapshot_id}`,
    platform: payload.snapshot.stratum,
    note: "Published Atlas API snapshot. Evidence values are bound to the named immutable release.",
    releaseStatus: "published",
    publishedAt: payload.snapshot.published_at ?? undefined,
    calibrationId: payload.snapshot.calibration_id,
    textReleaseId: payload.snapshot.text_release_id ?? undefined,
    policyHash: payload.snapshot.policy_hash,
    nodes,
    edges,
  };
}

export async function loadPublishedGraph(): Promise<GraphDataset> {
  const configured = process.env.CSKL_ATLAS_API_URL?.trim();
  const stratum = process.env.CSKL_ATLAS_STRATUM?.trim();
  if (!configured || !stratum) {
    const { default: staticGraphPayload } = await import("../data/atlas-graph.json");
    return adaptPublishedGraph(staticGraphPayload as unknown as ApiGraph);
  }

  const base = new URL(configured);
  if (!["http:", "https:"].includes(base.protocol)) {
    throw new Error("CSKL_ATLAS_API_URL must use HTTP or HTTPS.");
  }
  const currentUrl = new URL("/v1/snapshots/current", base);
  currentUrl.searchParams.set("stratum", stratum);
  const currentResponse = await fetch(currentUrl, { cache: "no-store" });
  if (!currentResponse.ok) {
    throw new Error(`Atlas API has no current ${stratum} snapshot (${currentResponse.status}).`);
  }
  const current = (await currentResponse.json()) as { snapshot_id?: string };
  if (!current.snapshot_id) throw new Error("Atlas API omitted snapshot_id.");

  const graphUrl = new URL("/v1/graph", base);
  graphUrl.searchParams.set("snapshot_id", current.snapshot_id);
  graphUrl.searchParams.set("independent_only", "false");
  graphUrl.searchParams.set("edge_limit", "50000");
  const graphResponse = await fetch(graphUrl, { cache: "no-store" });
  if (!graphResponse.ok) {
    throw new Error(`Atlas API graph request failed (${graphResponse.status}).`);
  }
  return adaptPublishedGraph((await graphResponse.json()) as ApiGraph);
}
