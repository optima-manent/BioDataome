"use client";

import { useCallback, useEffect, useMemo, useRef, useState, type RefObject } from "react";
import { AtlasIntro } from "./AtlasIntro";
import {
  GraphCanvas,
  type ClusterMode,
  type GraphGroupSelection,
  type GraphMode,
} from "./GraphCanvas";
import {
  buildAiEvidencePacket,
  computedSpecter2,
  edgeMatchesLens,
  formatProbability,
  hasComputedSpecter2,
  isOverlapQualified,
  type EvidenceLens,
} from "../lib/evidence-policy";
import {
  diseaseFamilyColor,
  nodeColor,
  nodeTissueSystem,
  tissueColor,
  tissueShape,
  type DiseaseFamily,
  type GraphDataset,
  type GraphEdge,
  type GraphNode,
  type NodeColorMode,
  type Tissue,
} from "../lib/graph-data";
import {
  DEFAULT_DISCOVERY_QUERY,
  EMPTY_DISCOVERY_QUERY,
  discoveryQueryAst,
  edgeMatchesDiscoveryQuery,
  queryIsValid,
  type DiscoveryQuery,
  type IndependenceOperator,
  type RelationOperator,
  type ThresholdOperator,
} from "../lib/discovery-query";
import {
  buildResearchExport,
  downloadResearchExport,
  researchExportFilename,
  serializeResearchExportCsv,
  serializeResearchExportJson,
} from "../lib/research-export";
import { graphGroupIdentity } from "../lib/graph-layout";

type Lens = EvidenceLens;
type AiState = { status: "idle" | "loading" | "ready" | "error"; content?: string };
type FocusedGroup = GraphGroupSelection & { mode: ClusterMode };

const EDGE_Q_THRESHOLDS = [1e-12, 1e-9, 1e-6, 1e-4, 1e-3, 1e-2, 0.05] as const;
const SAMPLE_THRESHOLDS = [2, 10, 25, 50, 100, 250, 500, 1_000, 2_000] as const;
const INTRO_STORAGE_KEY = "cskl-atlas:intro:v2";

const lensCopy: Record<Lens, { label: string; detail: string }> = {
  all: { label: "All supported links", detail: "The published graph evidence network." },
  agreement: {
    label: "Cross-modal agreement",
    detail: "C-SKL links also supported by high semantic proximity.",
  },
  "cskl-only": {
    label: "Molecular-only signal",
    detail: "Strong expression similarity with weak text agreement.",
  },
  "cross-disease": {
    label: "Same tissue, different disease",
    detail: "Potential shared biology across diagnoses in one tissue context.",
  },
  overlap: {
    label: "Sample-overlap audit",
    detail: "Relationships qualified by shared molecular profiles.",
  },
};

function formatCskl(value: number) {
  return value < 0.001 ? value.toExponential(2) : value.toFixed(4);
}

function formatPublishedDate(value: string) {
  return new Intl.DateTimeFormat("en-GB", {
    day: "2-digit",
    month: "short",
    year: "numeric",
    timeZone: "UTC",
  }).format(new Date(value));
}

function endpoint(edge: GraphEdge, nodeId: string) {
  return edge.source === nodeId ? edge.target : edge.source;
}

function NodeGlyph({ node, colorMode }: { node: GraphNode; colorMode: NodeColorMode }) {
  return (
    <i
      aria-hidden="true"
      className={`node-glyph shape-${tissueShape(nodeTissueSystem(node))}`}
      style={{ background: nodeColor(node, colorMode) }}
    />
  );
}

function SourceBadge({ node }: { node: GraphNode }) {
  const provenance = {
    geo_structured: { label: "GEO structured", className: "source" },
    deterministic_ontology: { label: "Ontology mapped", className: "mapped" },
    llm_candidate: { label: "AI candidate", className: "draft" },
    human_verified: { label: "Human verified", className: "verified" },
    unknown: { label: "Provenance unknown", className: "unknown" },
  }[node.annotationSource];
  return (
    <span className={`source-badge ${provenance.className}`}>
      {provenance.label}
    </span>
  );
}

function readableStatus(value: string) {
  return value.replaceAll("_", " ").replace(/^./, (character) => character.toUpperCase());
}

function ontologyValidationPresentation(value?: string) {
  if (value === "accepted") return { label: "Human accepted", className: "verified" };
  if (value === "canonical_or_synonym") {
    return { label: "Ontology match", className: "verified" };
  }
  if (value === "label_mismatch") return { label: "Label mismatch", className: "warning" };
  if (value === "obsolete") return { label: "Obsolete ID", className: "warning" };
  if (value === "missing") return { label: "ID not found", className: "warning" };
  if (value === "not_in_release") return { label: "Not in audit", className: "pending" };
  if (!value || value === "not_audited") return { label: "Not audited", className: "pending" };
  return { label: readableStatus(value), className: "pending" };
}

function semanticLabelPresentation(source?: string) {
  if (source === "ontology_label_concordant") {
    return { label: "Ontology label matched · not curated", className: "pending" };
  }
  if (source === "unvalidated_or_missing") {
    return { label: "Unvalidated or missing", className: "warning" };
  }
  if (!source) {
    return { label: "Validation source unavailable", className: "pending" };
  }
  return { label: readableStatus(source), className: "pending" };
}

function tissueSystemPresentation(node: GraphNode) {
  return semanticLabelPresentation(node.tissueSystemSource);
}

export function AtlasExplorer({
  graph,
  synthesisEndpoint = "/api/explain",
}: {
  graph: GraphDataset;
  synthesisEndpoint?: string | null;
}) {
  const allTissueSystems = useMemo(
    () => [...new Set(graph.nodes.map((node) => nodeTissueSystem(node)))].sort(),
    [graph.nodes],
  );
  const [mode, setMode] = useState<GraphMode>("cskl");
  const [clusterMode, setClusterMode] = useState<ClusterMode>("topology");
  const [nodeColorMode, setNodeColorMode] = useState<NodeColorMode>("tissue");
  const [lens, setLens] = useState<Lens>("all");
  const [search, setSearch] = useState("");
  const [activeTissueSystems, setActiveTissueSystems] = useState<Set<Tissue>>(
    () => new Set(graph.nodes.map((node) => nodeTissueSystem(node))),
  );
  const [sampleThresholdIndex, setSampleThresholdIndex] = useState(0);
  const [edgeQStep, setEdgeQStep] = useState(EDGE_Q_THRESHOLDS.length - 1);
  const [focusedGroup, setFocusedGroup] = useState<FocusedGroup | null>(null);
  const [selectedNodeIds, setSelectedNodeIds] = useState<Set<string>>(new Set());
  const [selectedEdgeId, setSelectedEdgeId] = useState<string | null>(null);
  const [inspectorOpen, setInspectorOpen] = useState(false);
  const [controlsOpen, setControlsOpen] = useState(false);
  const [introOpen, setIntroOpen] = useState(true);
  const [aiState, setAiState] = useState<AiState>({ status: "idle" });
  const [queryOpen, setQueryOpen] = useState(false);
  const searchInputRef = useRef<HTMLInputElement>(null);
  const [draftQuery, setDraftQuery] = useState<DiscoveryQuery>(() => ({
    ...DEFAULT_DISCOVERY_QUERY,
  }));
  const [activeQuery, setActiveQuery] = useState<DiscoveryQuery | null>(null);
  const [exportOpen, setExportOpen] = useState(false);
  const [exportError, setExportError] = useState<string | null>(null);
  const introReturnFocusRef = useRef<HTMLElement | null>(null);
  const aiAbortRef = useRef<AbortController | null>(null);
  const aiRequestRef = useRef(0);
  const aiRequestedPacketFingerprintRef = useRef<string | null>(null);
  const showAllButtonRef = useRef<HTMLButtonElement>(null);
  const exportTriggerRef = useRef<HTMLButtonElement>(null);
  const controlsTriggerRef = useRef<HTMLButtonElement>(null);
  const queryTriggerRef = useRef<HTMLButtonElement>(null);
  const queryPanelRef = useRef<HTMLElement>(null);
  const hasTextEvidence = useMemo(() => hasComputedSpecter2(graph.edges), [graph.edges]);
  const explainerCoverage = useMemo(
    () => graph.edges.filter((edge) => edge.explainer).length,
    [graph.edges],
  );
  const minSamples = SAMPLE_THRESHOLDS[sampleThresholdIndex];
  const maxQValue = EDGE_Q_THRESHOLDS[edgeQStep];

  const diseaseFamilyCounts = useMemo(() => {
    const counts = new Map<DiseaseFamily, number>();
    for (const node of graph.nodes) {
      counts.set(node.diseaseFamily, (counts.get(node.diseaseFamily) ?? 0) + 1);
    }
    return [...counts.entries()].sort(
      ([leftFamily, leftCount], [rightFamily, rightCount]) =>
        rightCount - leftCount || leftFamily.localeCompare(rightFamily),
    );
  }, [graph.nodes]);

  const resetAi = useCallback(() => {
    aiRequestRef.current += 1;
    aiAbortRef.current?.abort();
    aiAbortRef.current = null;
    aiRequestedPacketFingerprintRef.current = null;
    setAiState({ status: "idle" });
  }, []);

  const closeExport = useCallback(() => {
    setExportOpen(false);
    window.requestAnimationFrame(() => exportTriggerRef.current?.focus());
  }, []);

  const closeControls = useCallback(() => {
    setControlsOpen(false);
    window.requestAnimationFrame(() => controlsTriggerRef.current?.focus());
  }, []);

  const leaveFocusedGroup = useCallback(() => {
    setFocusedGroup(null);
    window.requestAnimationFrame(() => {
      document
        .querySelector<HTMLCanvasElement>('[data-testid="graph-stage"] canvas')
        ?.focus();
    });
  }, []);

  const closeQuery = useCallback(() => {
    setQueryOpen(false);
    window.requestAnimationFrame(() => queryTriggerRef.current?.focus());
  }, []);

  useEffect(() => () => aiAbortRef.current?.abort(), []);

  useEffect(() => {
    if (!focusedGroup) return;
    window.requestAnimationFrame(() => showAllButtonRef.current?.focus());
  }, [focusedGroup]);

  useEffect(() => {
    if (!queryOpen) return;
    window.requestAnimationFrame(() => queryPanelRef.current?.focus());
  }, [queryOpen]);

  useEffect(() => {
    const focusSearch = (event: KeyboardEvent) => {
      if (!(event.ctrlKey || event.metaKey) || event.key.toLowerCase() !== "k") return;
      event.preventDefault();
      searchInputRef.current?.focus();
    };
    window.addEventListener("keydown", focusSearch);
    return () => window.removeEventListener("keydown", focusSearch);
  }, []);

  useEffect(() => {
    let dismissed = false;
    try {
      dismissed = window.sessionStorage.getItem(INTRO_STORAGE_KEY) === "dismissed";
    } catch {
      // Storage may be unavailable in a privacy-restricted browser. The primer
      // still works for this visit without persisting its dismissal.
    }
    if (!dismissed) return;
    const timer = window.setTimeout(() => setIntroOpen(false), 0);
    return () => window.clearTimeout(timer);
  }, []);

  useEffect(() => {
    const closeTopLayer = (event: KeyboardEvent) => {
      if (event.key !== "Escape" || introOpen) return;
      if (controlsOpen) {
        event.preventDefault();
        closeControls();
        return;
      }
      if (queryOpen) {
        event.preventDefault();
        closeQuery();
        return;
      }
      if (exportOpen) {
        event.preventDefault();
        closeExport();
        return;
      }
      if (!focusedGroup) return;
      event.preventDefault();
      leaveFocusedGroup();
    };
    window.addEventListener("keydown", closeTopLayer);
    return () => window.removeEventListener("keydown", closeTopLayer);
  }, [
    closeControls,
    closeExport,
    closeQuery,
    controlsOpen,
    exportOpen,
    focusedGroup,
    introOpen,
    leaveFocusedGroup,
    queryOpen,
  ]);

  const nodeMap = useMemo(
    () => new Map(graph.nodes.map((node) => [node.id, node])),
    [graph.nodes],
  );

  const filteredNodes = useMemo(() => {
    const needle = search.trim().toLowerCase();
    return graph.nodes.filter((node) => {
      const matchesSearch =
        !needle ||
        [node.id, node.title, node.tissue, nodeTissueSystem(node), node.disease]
          .join(" ")
          .toLowerCase()
          .includes(needle);
      return matchesSearch && activeTissueSystems.has(nodeTissueSystem(node)) && node.samples >= minSamples;
    });
  }, [activeTissueSystems, graph.nodes, minSamples, search]);

  const baseNodes = useMemo(() => {
    if (!focusedGroup || focusedGroup.mode !== clusterMode) return filteredNodes;
    return filteredNodes.filter(
      (node) => graphGroupIdentity(node, focusedGroup.mode).id === focusedGroup.id,
    );
  }, [clusterMode, filteredNodes, focusedGroup]);

  const visibleNodeIds = useMemo(() => new Set(baseNodes.map((node) => node.id)), [baseNodes]);
  const csklMedian = useMemo(() => {
    const values = [...graph.edges].map((edge) => edge.cskl).sort((a, b) => a - b);
    return values.length ? values[Math.floor(values.length / 2)] : Number.POSITIVE_INFINITY;
  }, [graph.edges]);

  const allCandidateEdges = useMemo(
    () =>
      graph.edges.filter((edge) => {
        if (!visibleNodeIds.has(edge.source) || !visibleNodeIds.has(edge.target)) return false;
        return nodeMap.has(edge.source) && nodeMap.has(edge.target);
      }),
    [graph.edges, nodeMap, visibleNodeIds],
  );

  const candidateEdges = useMemo(
    () => allCandidateEdges.filter((edge) => edge.qValue <= maxQValue),
    [allCandidateEdges, maxQValue],
  );

  const evidenceEdges = useMemo(
    () =>
      mode === "cskl"
        ? candidateEdges
        : candidateEdges.filter((edge) => computedSpecter2(edge) !== undefined),
    [candidateEdges, mode],
  );

  const visibleEdges = useMemo(
    () =>
      evidenceEdges.filter((edge) => {
        const source = nodeMap.get(edge.source);
        const target = nodeMap.get(edge.target);
        if (!source || !target) return false;
        if (!edgeMatchesLens({ edge, lens, csklMedian, source, target })) return false;
        return activeQuery
          ? edgeMatchesDiscoveryQuery({ edge, source, target, query: activeQuery })
          : true;
      }),
    [activeQuery, csklMedian, evidenceEdges, lens, nodeMap],
  );

  const draftQueryValid = queryIsValid(draftQuery);
  const draftQueryResultCount = useMemo(
    () =>
      draftQueryValid
        ? allCandidateEdges.filter((edge) => {
            const source = nodeMap.get(edge.source);
            const target = nodeMap.get(edge.target);
            return Boolean(
              source &&
                target &&
                edgeMatchesDiscoveryQuery({ edge, source, target, query: draftQuery }),
            );
          }).length
        : 0,
    [allCandidateEdges, draftQuery, draftQueryValid, nodeMap],
  );

  const graphNodes = useMemo(() => {
    if (lens === "all" && !activeQuery) return baseNodes;
    const connected = new Set(visibleEdges.flatMap((edge) => [edge.source, edge.target]));
    return baseNodes.filter((node) => connected.has(node.id));
  }, [activeQuery, baseNodes, lens, visibleEdges]);

  const selectedEdge = selectedEdgeId
    ? graph.edges.find((edge) => edge.id === selectedEdgeId) ?? null
    : null;
  const selectedNodes = useMemo(
    () =>
      [...selectedNodeIds]
        .map((id) => nodeMap.get(id))
        .filter((node): node is GraphNode => Boolean(node)),
    [nodeMap, selectedNodeIds],
  );
  const primaryNode = selectedNodes.length === 1 ? selectedNodes[0] : null;
  const aiEvidencePacket = useMemo(
    () =>
      buildAiEvidencePacket({
        selectedEdge,
        selectedNodes,
        visibleEdges,
        selectedNodeIds,
        nodeMap,
      }),
    [nodeMap, selectedEdge, selectedNodeIds, selectedNodes, visibleEdges],
  );
  const aiEvidencePacketFingerprint = useMemo(
    () => JSON.stringify(aiEvidencePacket),
    [aiEvidencePacket],
  );

  useEffect(() => {
    const requestedFingerprint = aiRequestedPacketFingerprintRef.current;
    if (
      requestedFingerprint !== null &&
      requestedFingerprint !== aiEvidencePacketFingerprint
    ) {
      resetAi();
    }
  }, [aiEvidencePacketFingerprint, resetAi]);

  const setSingleNode = (id: string, additive: boolean) => {
    setSelectedEdgeId(null);
    resetAi();
    setSelectedNodeIds((current) => {
      if (!additive) return new Set([id]);
      const next = new Set(current);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
    setInspectorOpen(true);
  };

  const selectEdge = (id: string) => {
    setSelectedEdgeId(id);
    setSelectedNodeIds(new Set());
    resetAi();
    setInspectorOpen(true);
  };

  const clearSelection = () => {
    setSelectedNodeIds(new Set());
    setSelectedEdgeId(null);
    resetAi();
  };

  const changeEdgeQStep = (nextStep: number) => {
    const nextMaxQValue = EDGE_Q_THRESHOLDS[nextStep];
    setEdgeQStep(nextStep);
    if (selectedEdge && selectedEdge.qValue > nextMaxQValue) {
      setSelectedEdgeId(null);
      resetAi();
    }
  };

  const changeEvidenceMode = (nextMode: GraphMode) => {
    setMode(nextMode);
    if (nextMode !== "cskl" && selectedEdge && computedSpecter2(selectedEdge) === undefined) {
      setSelectedEdgeId(null);
      resetAi();
    }
  };

  const changeClusterMode = (nextMode: ClusterMode) => {
    setClusterMode(nextMode);
    setFocusedGroup(null);
    if (nextMode === "tissue") setNodeColorMode("tissue");
    if (nextMode === "disease") setNodeColorMode("disease");
    clearSelection();
  };

  const focusGroup = (group: GraphGroupSelection) => {
    setFocusedGroup({ ...group, mode: clusterMode });
    setControlsOpen(false);
    clearSelection();
  };

  const dismissIntro = () => {
    try {
      window.sessionStorage.setItem(INTRO_STORAGE_KEY, "dismissed");
    } catch {
      // The primer can close normally even when storage is unavailable.
    }
    setIntroOpen(false);
    const returnTarget = introReturnFocusRef.current;
    introReturnFocusRef.current = null;
    window.requestAnimationFrame(() => returnTarget?.focus());
  };

  const openIntro = () => {
    introReturnFocusRef.current =
      document.activeElement instanceof HTMLElement ? document.activeElement : null;
    setIntroOpen(true);
  };

  const askAi = async () => {
    if (!synthesisEndpoint) return;
    aiAbortRef.current?.abort();
    const controller = new AbortController();
    aiAbortRef.current = controller;
    const requestId = ++aiRequestRef.current;
    const evidence = aiEvidencePacket;
    aiRequestedPacketFingerprintRef.current = aiEvidencePacketFingerprint;
    setAiState({ status: "loading" });
    try {
      const response = await fetch(synthesisEndpoint, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(evidence),
        signal: controller.signal,
      });
      const data = (await response.json()) as { explanation?: string; message?: string };
      if (!response.ok) throw new Error(data.message || "The explanation service is not configured.");
      if (requestId !== aiRequestRef.current) return;
      setAiState({ status: "ready", content: data.explanation });
    } catch (error) {
      if (controller.signal.aborted || requestId !== aiRequestRef.current) return;
      setAiState({
        status: "error",
        content:
          error instanceof Error
            ? error.message
            : "The selected evidence packet is ready, but the explanation service is unavailable.",
      });
    } finally {
      if (requestId === aiRequestRef.current) aiAbortRef.current = null;
    }
  };

  const applyLens = (nextLens: Lens) => {
    if ((nextLens === "agreement" || nextLens === "cskl-only") && !hasTextEvidence) return;
    setLens(nextLens);
    setActiveQuery(null);
    if (nextLens === "agreement") setMode("agreement");
    if (nextLens === "cskl-only") setMode("agreement");
    clearSelection();
    if (queryOpen) closeQuery();
  };

  const nodeConnections = primaryNode
    ? visibleEdges
        .filter((edge) => edge.source === primaryNode.id || edge.target === primaryNode.id)
        .sort((a, b) => a.cskl - b.cskl)
        .slice(0, 5)
    : [];

  const exportScopeLabel = selectedEdge
    ? "Selected relationship · 2 datasets"
    : selectedNodeIds.size > 0
      ? `${selectedNodeIds.size} selected dataset${selectedNodeIds.size === 1 ? "" : "s"}`
      : `Visible graph · ${graphNodes.length} datasets`;

  const handleResearchExport = (format: "json" | "csv") => {
    try {
      const researchExport = buildResearchExport({
        graph,
        visibleNodes: graphNodes,
        visibleEdges,
        selectedNodeIds,
        selectedEdgeId,
        view: {
          evidenceMode: mode,
          clusterMode,
          lens,
          search,
          activeTissueSystems: [...activeTissueSystems],
          minSamples,
          maxQValue,
          nodeColorMode,
          focusedCluster: focusedGroup
            ? `${focusedGroup.mode}:${focusedGroup.label}`
            : undefined,
          discoveryQuery: activeQuery,
        },
      });
      const content =
        format === "json"
          ? serializeResearchExportJson(researchExport)
          : serializeResearchExportCsv(researchExport);
      downloadResearchExport(
        content,
        researchExportFilename(researchExport, format),
        format === "json" ? "application/json" : "text/csv",
      );
      setExportError(null);
      closeExport();
    } catch (error) {
      setExportError(error instanceof Error ? error.message : "The research export could not be prepared.");
    }
  };

  return (
    <main className={`atlas-shell ${inspectorOpen ? "inspector-visible" : ""}`}>
      <div
        className="atlas-workspace"
        aria-hidden={introOpen}
        inert={introOpen ? true : undefined}
      >
      <header className="atlas-header">
        <div className="brand-lockup" aria-label="C-SKL Atlas">
          <span className="brand-mark" aria-hidden="true">
            <i />
            <i />
            <i />
          </span>
          <span>
            <strong>C-SKL</strong>
            <em>ATLAS</em>
          </span>
        </div>
        <div className="header-search">
          <span aria-hidden="true">⌕</span>
          <label htmlFor="atlas-search">
            <b>Find datasets</b>
            <input
              ref={searchInputRef}
              id="atlas-search"
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              placeholder="GSE, title, tissue, disease…"
            />
          </label>
          <kbd>Ctrl/⌘ K</kbd>
        </div>
        <div className="header-actions">
          <span className="release-pill">
            <i />
            {graph.releaseStatus === "published" ? "Published snapshot" : "Snapshot loaded"}
          </span>
          <div
            className="export-control"
            onKeyDown={(event) => {
              if (event.key !== "Escape") return;
              event.preventDefault();
              event.stopPropagation();
              closeExport();
            }}
          >
            <button
              ref={exportTriggerRef}
              type="button"
              className="export-button"
              aria-expanded={exportOpen}
              aria-haspopup="dialog"
              onClick={() => {
                if (exportOpen) closeExport();
                else {
                  setExportOpen(true);
                  setQueryOpen(false);
                  setControlsOpen(false);
                }
                setExportError(null);
              }}
            >
              <span aria-hidden="true">↓</span>
              <span className="export-button-label">Export</span>
            </button>
            {exportOpen && (
              <div className="export-popover" role="dialog" aria-label="Research export">
                <div className="export-popover-heading">
                  <strong>Research export</strong>
                  <small>{exportScopeLabel}</small>
                </div>
                <button type="button" onClick={() => handleResearchExport("json")}>
                  <span><strong>JSON</strong><small>Structured evidence and provenance</small></span>
                  <b aria-hidden="true">↓</b>
                </button>
                <button type="button" onClick={() => handleResearchExport("csv")}>
                  <span><strong>CSV</strong><small>Flat, analysis-ready research table</small></span>
                  <b aria-hidden="true">↓</b>
                </button>
                <p>Exports the selection, or the visible graph when nothing is selected.</p>
                {exportError && <p className="export-error" role="alert">{exportError}</p>}
              </div>
            )}
          </div>
          <button
            ref={queryTriggerRef}
            type="button"
            className={`query-button ${activeQuery ? "active" : ""}`}
            aria-pressed={Boolean(activeQuery)}
            aria-expanded={queryOpen}
            aria-haspopup="dialog"
            onClick={() => {
              if (queryOpen) closeQuery();
              else {
                setQueryOpen(true);
                setExportOpen(false);
                setControlsOpen(false);
              }
            }}
          >
            <span aria-hidden="true">⌘</span> {activeQuery ? "Query active" : "Discovery query"}
          </button>
          <button
            ref={controlsTriggerRef}
            type="button"
            className={`mobile-controls-button ${controlsOpen ? "active" : ""}`}
            aria-label={controlsOpen ? "Close graph controls" : "Open graph controls"}
            aria-expanded={controlsOpen}
            onClick={() => {
              if (controlsOpen) closeControls();
              else {
                setControlsOpen(true);
                setExportOpen(false);
                setQueryOpen(false);
              }
            }}
          >
            <span aria-hidden="true">≡</span>
          </button>
          <button
            type="button"
            className="icon-button"
            aria-label="How to read the atlas"
            onClick={openIntro}
          >
            ?
          </button>
        </div>
      </header>

      <aside className={`lens-panel ${controlsOpen ? "mobile-open" : ""}`} aria-label="Graph controls">
        <div className="panel-scroll">
          <div className="mobile-panel-heading">
            <strong>Graph controls</strong>
            <span>
              <button type="button" onClick={openIntro}>How to read</button>
              <button type="button" aria-label="Close graph controls" onClick={closeControls}>×</button>
            </span>
          </div>
          <div className="panel-eyebrow">Evidence layer</div>
          <div className="mode-switcher" role="group" aria-label="Evidence layer">
            {(["cskl", "specter2", "agreement"] as GraphMode[]).map((item) => (
              <button
                key={item}
                type="button"
                aria-pressed={mode === item}
                className={mode === item ? "active" : ""}
                disabled={item !== "cskl" && !hasTextEvidence}
                title={
                  item !== "cskl" && !hasTextEvidence
                    ? "Unavailable until this release contains computed SPECTER2 evidence."
                    : undefined
                }
                onClick={() => changeEvidenceMode(item)}
              >
                {item === "cskl" ? "C-SKL" : item === "specter2" ? "SPECTER2" : "Agreement"}
              </button>
            ))}
          </div>
          <p className="control-note">
            {!hasTextEvidence
              ? "SPECTER2 modes are unavailable: illustrative preview values are excluded from analysis."
              : mode === "cskl"
              ? "Molecular similarity; lower raw distance is stronger."
              : mode === "specter2"
                ? "Scientific-text proximity; not molecular validation."
                : "Compare where molecular and text evidence agree or diverge."}
          </p>

          <div className="control-section">
            <div className="section-heading">
              <span>Map organization</span>
            </div>
            <label className="select-field">
              <span>Group datasets by</span>
              <select value={clusterMode} onChange={(event) => changeClusterMode(event.target.value as ClusterMode)}>
                <option value="topology">C-SKL topology</option>
                <option value="tissue">Anatomical system</option>
                <option value="disease">Clinical family</option>
              </select>
            </label>
            <div className="encoding-field">
              <span>Node color</span>
              <div className="encoding-switcher" role="group" aria-label="Node color">
                <button
                  type="button"
                  className={nodeColorMode === "tissue" ? "active" : ""}
                  aria-pressed={nodeColorMode === "tissue"}
                  onClick={() => setNodeColorMode("tissue")}
                >
                  Anatomy
                </button>
                <button
                  type="button"
                  className={nodeColorMode === "disease" ? "active" : ""}
                  aria-pressed={nodeColorMode === "disease"}
                  onClick={() => setNodeColorMode("disease")}
                >
                  Clinical family
                </button>
              </div>
              <small>Shape always keeps a broad anatomical context for color-independent reading.</small>
            </div>
            <details className="family-key">
              <summary>Clinical family key <span>{diseaseFamilyCounts.length}</span></summary>
              <div>
                {diseaseFamilyCounts.map(([family, count]) => (
                  <span key={family}>
                    <i style={{ background: diseaseFamilyColor(family) }} />
                    <b>{family}</b>
                    <small>{count}</small>
                  </span>
                ))}
              </div>
              <p>Broad display groups from concordant labels; no ICD code is assigned.</p>
            </details>
            <label className="range-field edge-q-field">
              <span>
                Maximum global BH q-value <strong>≤ {formatProbability(maxQValue)}</strong>
              </span>
              <input
                type="range"
                min="0"
                max={EDGE_Q_THRESHOLDS.length - 1}
                step="1"
                value={edgeQStep}
                aria-label="Maximum global BH q-value"
                aria-valuetext={`q-value at most ${formatProbability(maxQValue)}, ${candidateEdges.length} relationships retained`}
                onChange={(event) => changeEdgeQStep(Number(event.target.value))}
              />
              <small className="range-ends"><span>Fewer, strongest</span><span>More links</span></small>
              <em>{candidateEdges.length.toLocaleString()} of {allCandidateEdges.length.toLocaleString()} published links under current dataset filters</em>
            </label>
            <p className="facet-note">
              This changes the visible sparse graph only; it does not reveal unpublished pairs.
            </p>
          </div>

          <div className="control-section">
            <div className="section-heading">
              <span>Discovery lens</span>
              <button type="button" onClick={() => applyLens("all")}>Reset</button>
            </div>
            <div className="lens-list">
              {(Object.keys(lensCopy) as Lens[]).map((item) => (
                <button
                  key={item}
                  type="button"
                  className={lens === item ? "active" : ""}
                  disabled={(item === "agreement" || item === "cskl-only") && !hasTextEvidence}
                  title={
                    (item === "agreement" || item === "cskl-only") && !hasTextEvidence
                      ? "Unavailable without computed SPECTER2 evidence."
                      : undefined
                  }
                  onClick={() => applyLens(item)}
                >
                  <span>{lensCopy[item].label}</span>
                  <small>{lensCopy[item].detail}</small>
                </button>
              ))}
            </div>
          </div>

          <div className="control-section">
            <div className="section-heading">
              <span>Anatomical system</span>
              <button type="button" onClick={() => setActiveTissueSystems(new Set(allTissueSystems))}>All</button>
            </div>
            <div className="tissue-filter">
              {allTissueSystems.map((tissueSystem) => (
                <label key={tissueSystem}>
                  <input
                    type="checkbox"
                    checked={activeTissueSystems.has(tissueSystem)}
                    onChange={() =>
                      setActiveTissueSystems((current) => {
                        const next = new Set(current);
                        if (next.has(tissueSystem)) next.delete(tissueSystem);
                        else next.add(tissueSystem);
                        return next;
                      })
                    }
                  />
                  <i
                    className={`node-glyph shape-${tissueShape(tissueSystem)}`}
                    style={{ background: tissueColor(tissueSystem) }}
                  />
                  <span>{tissueSystem}</span>
                </label>
              ))}
            </div>
          </div>

          <div className="control-section">
            <label className="range-field">
              <span>
                Minimum sample count <strong>{minSamples}</strong>
              </span>
              <input
                type="range"
                min="0"
                max={SAMPLE_THRESHOLDS.length - 1}
                step="1"
                value={sampleThresholdIndex}
                aria-valuetext={`At least ${minSamples} samples`}
                onChange={(event) => setSampleThresholdIndex(Number(event.target.value))}
              />
              <small className="range-ends"><span>2</span><span>2,000</span></small>
            </label>
          </div>

          <div className="control-section dataset-results">
            <div className="section-heading">
              <span>Datasets in view</span>
              <b>{graphNodes.length}</b>
            </div>
            <div className="dataset-list" aria-label="Datasets in view">
              {graphNodes.map((node) => (
                <button
                  type="button"
                  key={node.id}
                  className={selectedNodeIds.has(node.id) ? "selected" : ""}
                  onClick={(event) => setSingleNode(node.id, event.shiftKey || event.ctrlKey || event.metaKey)}
                >
                  <NodeGlyph node={node} colorMode={nodeColorMode} />
                  <span><strong>{node.id}</strong><small>{node.disease}</small></span>
                  <em>{node.samples}</em>
                </button>
              ))}
              {graphNodes.length === 0 && <p className="empty-state">No datasets match these filters.</p>}
            </div>
          </div>
        </div>
      </aside>
      {controlsOpen && (
        <button
          type="button"
          className="mobile-panel-backdrop"
          aria-label="Close graph controls"
          onClick={closeControls}
        />
      )}

      <section className="map-panel" aria-label="Dataset similarity map">
        <div className="map-toolbar">
          <div>
            <span className="map-kicker">{graph.release}</span>
            <h1>{lensCopy[lens].label}</h1>
          </div>
          <div className="map-toolbar-stats">
            <span><strong>{graphNodes.length}</strong> datasets</span>
            <span><strong>{visibleEdges.length}</strong> relationships</span>
            <span className="freshness">
              <i />
              {graph.publishedAt
                ? `published ${formatPublishedDate(graph.publishedAt)}`
                : "publication date unavailable"}
            </span>
          </div>
        </div>
        {focusedGroup && (
          <section
            className="cluster-focus-card"
            aria-label={`Focused cluster: ${focusedGroup.label}`}
            aria-live="polite"
          >
            <div>
              <span>Focused {focusedGroup.mode === "topology" ? "cluster" : "group"}</span>
              <strong>{focusedGroup.label}</strong>
              <small>
                {baseNodes.length.toLocaleString()} dataset{baseNodes.length === 1 ? "" : "s"} · {visibleEdges.length.toLocaleString()} visible relationships
              </small>
            </div>
            <button ref={showAllButtonRef} type="button" onClick={leaveFocusedGroup}>
              Show all <span aria-hidden="true">×</span>
            </button>
            <label className="cluster-mobile-threshold">
              <span>q ≤ {formatProbability(maxQValue)}</span>
              <input
                type="range"
                min="0"
                max={EDGE_Q_THRESHOLDS.length - 1}
                step="1"
                value={edgeQStep}
                aria-label="Maximum global BH q-value in focused cluster"
                aria-valuetext={`q-value at most ${formatProbability(maxQValue)}`}
                onChange={(event) => changeEdgeQStep(Number(event.target.value))}
              />
            </label>
          </section>
        )}
        <GraphCanvas
          key={`${clusterMode}:${focusedGroup?.id ?? "overview"}`}
          nodes={graphNodes}
          edges={visibleEdges}
          mode={mode}
          clusterMode={clusterMode}
          nodeColorMode={nodeColorMode}
          focusedGroupId={focusedGroup?.id}
          topOverlayInset={focusedGroup ? 148 : 66}
          selectedNodeIds={selectedNodeIds}
          selectedEdgeId={selectedEdgeId}
          onSelectNode={setSingleNode}
          onSelectEdge={selectEdge}
          onFocusGroup={focusGroup}
          onClear={clearSelection}
        />
        <div className="map-legend" aria-label="Graph legend">
          <div className="legend-block">
            <span className="legend-title">Node</span>
            <span><i className="legend-size small" /> sample count</span>
            <span><i className="legend-color" /> {nodeColorMode === "tissue" ? "anatomical system" : "clinical family"}</span>
            <span><i className="legend-shape" /> anatomical context</span>
          </div>
          <div className="legend-block">
            <span className="legend-title">Relationship</span>
            <span><i className="legend-line strong" /> stronger evidence</span>
            <span><i className="legend-line dotted" /> major / exact overlap</span>
            <span><i className="legend-dot" /> computed SPECTER2 available</span>
          </div>
          <span className="preview-disclosure" title={graph.note}>
            Published immutable snapshot · hover for provenance
          </span>
        </div>
      </section>

      {queryOpen && (
        <DiscoveryQueryPanel
          draft={draftQuery}
          setDraft={setDraftQuery}
          resultCount={draftQueryResultCount}
          candidateCount={allCandidateEdges.length}
          valid={draftQueryValid}
          activeState={
            !activeQuery
              ? "none"
              : JSON.stringify(discoveryQueryAst(activeQuery)) ===
                  JSON.stringify(discoveryQueryAst(draftQuery))
                ? "current"
                : "different"
          }
          hasTextEvidence={hasTextEvidence}
          explainerCoverage={explainerCoverage}
          edgeCount={graph.edges.length}
          onApply={() => {
            if (!draftQueryValid) return;
            setActiveQuery({ ...draftQuery });
            setLens("all");
            clearSelection();
          }}
          onReset={() => {
            setDraftQuery({ ...EMPTY_DISCOVERY_QUERY });
            setActiveQuery(null);
            setLens("all");
            clearSelection();
          }}
          onQuickLens={applyLens}
          onClose={closeQuery}
          panelRef={queryPanelRef}
        />
      )}

      <aside
        className={`inspector ${inspectorOpen ? "open" : ""}`}
        aria-label="Evidence inspector"
        aria-hidden={!inspectorOpen}
        inert={inspectorOpen ? undefined : true}
      >
        <button type="button" className="inspector-close" aria-label="Close evidence inspector" onClick={() => setInspectorOpen(false)}>
          ×
        </button>
        {selectedEdge ? (
          <EdgeInspector
            edge={selectedEdge}
            nodeMap={nodeMap}
            aiState={aiState}
            onAskAi={askAi}
            synthesisAvailable={Boolean(synthesisEndpoint)}
            colorMode={nodeColorMode}
          />
        ) : primaryNode ? (
          <NodeInspector
            node={primaryNode}
            connections={nodeConnections}
            nodeMap={nodeMap}
            onSelectEdge={selectEdge}
            colorMode={nodeColorMode}
          />
        ) : selectedNodes.length > 1 ? (
          <SelectionInspector
            nodes={selectedNodes}
            aiState={aiState}
            onAskAi={askAi}
            synthesisAvailable={Boolean(synthesisEndpoint)}
            colorMode={nodeColorMode}
          />
        ) : (
          <WelcomeInspector />
        )}
      </aside>
      {!inspectorOpen && (
        <button type="button" className="reopen-inspector" onClick={() => setInspectorOpen(true)}>
          Evidence panel
        </button>
      )}
      </div>
      <AtlasIntro open={introOpen} onDismiss={dismissIntro} />
    </main>
  );
}

function optionalProbability(value: string) {
  return value.trim() ? Number(value) : null;
}

function DiscoveryQueryPanel({
  draft,
  setDraft,
  resultCount,
  candidateCount,
  valid,
  activeState,
  hasTextEvidence,
  explainerCoverage,
  edgeCount,
  onApply,
  onReset,
  onQuickLens,
  onClose,
  panelRef,
}: {
  draft: DiscoveryQuery;
  setDraft: (query: DiscoveryQuery) => void;
  resultCount: number;
  candidateCount: number;
  valid: boolean;
  activeState: "none" | "current" | "different";
  hasTextEvidence: boolean;
  explainerCoverage: number;
  edgeCount: number;
  onApply: () => void;
  onReset: () => void;
  onQuickLens: (lens: Lens) => void;
  onClose: () => void;
  panelRef: RefObject<HTMLElement | null>;
}) {
  const expression = JSON.stringify(discoveryQueryAst(draft), null, 2);
  return (
    <section
      ref={panelRef}
      className="query-popover"
      role="dialog"
      aria-label="Discovery query builder"
      tabIndex={-1}
      onKeyDown={(event) => {
        if (event.key !== "Escape") return;
        event.preventDefault();
        event.stopPropagation();
        onClose();
      }}
    >
      <div className="query-header">
        <div>
          <span className="panel-eyebrow">Published-graph query</span>
          <h2>Find relationships with reproducible AND rules</h2>
        </div>
        <button type="button" aria-label="Close discovery query" onClick={onClose}>×</button>
      </div>
      <p>
        Every filled rule must match. Results are evaluated locally against this immutable
        snapshot and the active dataset filters. The map&apos;s separate q-value slider can narrow
        what is drawn without changing this query.
      </p>

      <form
        className="query-builder"
        onSubmit={(event) => {
          event.preventDefault();
          onApply();
        }}
      >
        <label>
          <span>Sample independence</span>
          <select
            value={draft.independence}
            onChange={(event) => setDraft({
              ...draft,
              independence: event.target.value as IndependenceOperator,
            })}
          >
            <option value="any">Any overlap state</option>
            <option value="independent">Independent only</option>
            <option value="overlap-qualified">Overlap-qualified only</option>
          </select>
        </label>
        <label>
          <span>Global BH q-value ≤</span>
          <input
            type="number"
            min="0"
            max="1"
            step="0.001"
            value={draft.qMax ?? ""}
            placeholder="Any"
            onChange={(event) => setDraft({ ...draft, qMax: optionalProbability(event.target.value) })}
          />
        </label>
        <label>
          <span>C-SKL similarity percentile ≥</span>
          <input
            type="number"
            min="0"
            max="1"
            step="0.01"
            value={draft.csklPercentileMin ?? ""}
            placeholder="Any"
            onChange={(event) => setDraft({
              ...draft,
              csklPercentileMin: optionalProbability(event.target.value),
            })}
          />
        </label>
        <label>
          <span>Anatomical-system relation</span>
          <select
            value={draft.tissueRelation}
            onChange={(event) => setDraft({
              ...draft,
              tissueRelation: event.target.value as RelationOperator,
            })}
          >
            <option value="any">Any</option>
            <option value="same">Same known system</option>
            <option value="different">Different known systems</option>
          </select>
        </label>
        <label>
          <span>Disease-label relation</span>
          <select
            value={draft.diseaseRelation}
            onChange={(event) => setDraft({
              ...draft,
              diseaseRelation: event.target.value as RelationOperator,
            })}
          >
            <option value="any">Any</option>
            <option value="same">Same known label</option>
            <option value="different">Different known labels</option>
          </select>
        </label>
        <label>
          <span>SPECTER2 percentile</span>
          <div className="query-inline-control">
            <select
              value={draft.specter2Operator}
              disabled={!hasTextEvidence}
              onChange={(event) => {
                const operator = event.target.value as ThresholdOperator;
                setDraft({
                  ...draft,
                  specter2Operator: operator,
                  specter2Percentile:
                    operator === "any" ? null : (draft.specter2Percentile ?? 0.9),
                });
              }}
            >
              <option value="any">Any</option>
              <option value="gte">At least</option>
              <option value="lte">At most</option>
            </select>
            <input
              type="number"
              min="0"
              max="1"
              step="0.01"
              aria-label="SPECTER2 percentile threshold"
              disabled={!hasTextEvidence || draft.specter2Operator === "any"}
              value={draft.specter2Percentile ?? ""}
              onChange={(event) => setDraft({
                ...draft,
                specter2Percentile: optionalProbability(event.target.value),
              })}
            />
          </div>
        </label>
        <label className="query-wide-field">
          <span>Gene, probe, or Reactome mechanism contains</span>
          <input
            type="search"
            value={draft.mechanismTerm}
            disabled={explainerCoverage === 0}
            placeholder={explainerCoverage ? "e.g. TP53 or R-HSA-…" : "No computed explainers in this release"}
            onChange={(event) => setDraft({ ...draft, mechanismTerm: event.target.value })}
          />
          <small>
            Computed explainer coverage: {explainerCoverage.toLocaleString()} of {edgeCount.toLocaleString()} published links.
            Uncomputed links cannot satisfy a mechanism rule.
          </small>
        </label>

        <div className="query-semantic-note">
          <strong>Semantic safeguards</strong>
          <span>
            Anatomy and disease relations use only exact ontology-label-concordant values.
            Curator acceptance is shown separately and is still required for manuscript claims.
          </span>
        </div>

        <div className={`query-result-summary ${valid ? "" : "invalid"}`} role="status">
          <strong>{valid ? resultCount.toLocaleString() : "Invalid"}</strong>
          <span>
            {valid
              ? `matching relationships among ${candidateCount.toLocaleString()} links under current dataset filters`
              : "Thresholds must be finite values between 0 and 1."}
          </span>
          {activeState === "current" && <em>This exact query is applied to the map</em>}
          {activeState === "different" && (
            <em>The map still uses the previous query; apply to use these edits</em>
          )}
        </div>
        <div className="query-actions">
          <button type="button" onClick={onReset}>Clear query</button>
          <button type="submit" disabled={!valid}>Apply to map</button>
        </div>
      </form>

      <details className="query-expression">
        <summary>Exact query expression</summary>
        <pre>{expression}</pre>
      </details>

      <div className="query-scope-note">
        <strong>Scope boundary</strong>
        <span>
          This static workbench searches the {edgeCount.toLocaleString()} links already published in the graph.
          Finding text-only or molecularly non-significant pairs requires the backend complete-pair query family.
        </span>
      </div>

      <div className="query-quick-heading">
        <strong>Quick visual lenses</strong>
        <span>These are convenient map views, not saved structured queries.</span>
      </div>
      <div className="query-presets compact">
        {(["cross-disease", "agreement", "cskl-only", "overlap"] as Lens[]).map((item) => (
          <button
            type="button"
            key={item}
            disabled={(item === "agreement" || item === "cskl-only") && !hasTextEvidence}
            onClick={() => onQuickLens(item)}
          >
            <span>{lensCopy[item].label}</span>
            <small>{lensCopy[item].detail}</small>
            <b>Open lens →</b>
          </button>
        ))}
      </div>
    </section>
  );
}

function EdgeInspector({
  edge,
  nodeMap,
  aiState,
  onAskAi,
  synthesisAvailable,
  colorMode,
}: {
  edge: GraphEdge;
  nodeMap: Map<string, GraphNode>;
  aiState: AiState;
  onAskAi: () => void;
  synthesisAvailable: boolean;
  colorMode: NodeColorMode;
}) {
  const source = nodeMap.get(edge.source);
  const target = nodeMap.get(edge.target);
  const overlapQualified = isOverlapQualified(edge);
  const textScore = computedSpecter2(edge);
  const agreement = textScore !== undefined && textScore >= 0.75;
  const trajectoryMaximum = Math.max(
    1e-12,
    ...(edge.explainer?.trajectory.flatMap((point) => [
      point.bestObjective,
      point.randomObjective,
    ]) ?? []),
  );
  const textStatus = textScore !== undefined ? "computed evidence" : "not computed";
  return (
    <div className="inspector-content">
      <span className="panel-eyebrow">Relationship evidence</span>
      <h2>{edge.source} <span>↔</span> {edge.target}</h2>
      <p className="inspector-subtitle">{source?.disease} · {target?.disease}</p>

      {overlapQualified && (
        <div className="warning-card">
          <strong>Shared-sample qualification</strong>
          <p>
            {edge.sharedSamples} molecular profiles overlap ({Math.round(edge.overlapFraction * 100)}% containment;
            {` ${edge.overlapClassification ?? "unknown"} classification`}). {edge.discoveryExcluded
              ? "The published policy excludes this relationship from independent discovery; do not treat it as independent replication."
              : "The published policy retains it as overlap-qualified evidence."}
          </p>
        </div>
      )}

      <div className="evidence-grid">
        <div><span>C-SKL distance</span><strong>{formatCskl(edge.cskl)}</strong><small>lower is closer</small></div>
        <div>
          <span>Global BH q-value</span>
          <strong>{formatProbability(edge.qValue)}</strong>
          <small>
            {edge.independentQValue === undefined
              ? "overlap-qualified; no independent q"
              : `independent q ${formatProbability(edge.independentQValue)}`}
          </small>
        </div>
        <div><span>SPECTER2</span><strong>{textScore?.toFixed(2) ?? "—"}</strong><small>{textStatus}</small></div>
        <div><span>Cross-modal</span><strong className={agreement ? "positive" : "neutral"}>{textScore === undefined ? "Unavailable" : agreement ? "High agreement" : "Mixed evidence"}</strong><small>not validation</small></div>
      </div>

      <section className="inspector-section">
        <div className="inspector-section-title"><h3>B(k) / W(k) explainer trajectory</h3><span>{edge.explainer ? "computed" : "not computed"}</span></div>
        {edge.explainer ? (
          <>
            <div className="driver-summary">
              <span><strong>B(k)</strong> retains similarity; <strong>W(k)</strong> identifies differentiating feature sets.</span>
              <small>{edge.explainer.trajectory.length} alignment-objective points. This is not an additive gene effect or percentage contribution.</small>
            </div>
            {edge.explainer.trajectory.length > 0 && (
              <div className="trajectory-chart" aria-label="Explainer objective trajectory">
                {edge.explainer.trajectory.map((point) => (
                  <div key={point.k}>
                    <b>k={point.k}</b>
                    <span>
                      <i className="best" style={{ width: `${Math.max(1, (point.bestObjective / trajectoryMaximum) * 100)}%` }} />
                      <i className="random" style={{ width: `${Math.max(1, (point.randomObjective / trajectoryMaximum) * 100)}%` }} />
                    </span>
                    <small>B(k) / random</small>
                  </div>
                ))}
              </div>
            )}
            <div className="driver-list">
              {[...edge.explainer.bSet.map((feature) => ({ ...feature, set: "B(k)" })), ...edge.explainer.wSet.map((feature) => ({ ...feature, set: "W(k)" }))].map((feature) => (
                <div key={`${feature.set}:${feature.feature}`}>
                  <span><b>{feature.gene ?? feature.feature}</b><small>{feature.gene ? feature.feature : "probe · gene mapping pending"}</small></span>
                  <em>{feature.set}</em>
                </div>
              ))}
            </div>
            {(edge.explainer.bestPathways?.length ?? 0) > 0 && (
              <div className="pathway-list">
                <div className="inspector-section-title">
                  <h3>Reactome hypotheses</h3>
                  <span>release {edge.explainer.reactomeRelease}</span>
                </div>
                {edge.explainer.bestPathways?.slice(0, 3).map((pathway) => (
                  <a key={pathway.pathway_id} href={pathway.url} target="_blank" rel="noreferrer">
                    <span>
                      <strong>{pathway.pathway_name}</strong>
                      <small>{pathway.overlap_count} genes · {pathway.fold_enrichment.toFixed(1)}× enrichment</small>
                    </span>
                    <em>q={pathway.q_value < 0.001 ? pathway.q_value.toExponential(1) : pathway.q_value.toFixed(3)}</em>
                  </a>
                ))}
                <small>{edge.explainer.interpretation}</small>
              </div>
            )}
          </>
        ) : (
          <div className="empty-evidence">
            <span aria-hidden="true">∿</span>
            <p><strong>Explainer trajectory not computed</strong>Use the local Atlas API to compute ordered B(k)/W(k) sets, their alignment-objective trajectory, a random-feature baseline, and Reactome hypotheses.</p>
          </div>
        )}
      </section>

      <section className="inspector-section">
        <div className="inspector-section-title"><h3>Dataset context</h3></div>
        <div className="pair-context">
          {[source, target].map((node) => node && (
            <a key={node.id} href={`https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=${node.id}`} target="_blank" rel="noreferrer">
              <NodeGlyph node={node} colorMode={colorMode} />
              <span><strong>{node.id}</strong><small>{node.tissue} · {node.samples} samples</small></span>
              <b>↗</b>
            </a>
          ))}
        </div>
      </section>

      <AiCard
        state={aiState}
        onAsk={onAskAi}
        label="Explain this relationship"
        available={synthesisAvailable}
      />
    </div>
  );
}

function NodeInspector({
  node,
  connections,
  nodeMap,
  onSelectEdge,
  colorMode,
}: {
  node: GraphNode;
  connections: GraphEdge[];
  nodeMap: Map<string, GraphNode>;
  onSelectEdge: (id: string) => void;
  colorMode: NodeColorMode;
}) {
  const systemValidation = tissueSystemPresentation(node);
  const diseaseValidation = semanticLabelPresentation(node.diseaseLabelSource);
  return (
    <div className="inspector-content">
      <span className="panel-eyebrow">Dataset</span>
      <div className="dataset-title-row">
        <NodeGlyph node={node} colorMode={colorMode} />
        <div><h2>{node.id}</h2><p>{node.title}</p></div>
      </div>
      <SourceBadge node={node} />
      <p className="dataset-summary">{node.summary}</p>

      <div className="metadata-list">
        <div><span>Anatomical system</span><strong>{nodeTissueSystem(node)}</strong></div>
        <div>
          <span>System validation</span>
          <strong className={`metadata-status ${systemValidation.className}`}>
            {systemValidation.label}
          </strong>
        </div>
        <div><span>Detailed tissue candidate</span><strong>{node.tissue}</strong></div>
        <div><span>Disease</span><strong>{node.disease}</strong></div>
        <div>
          <span>Disease-label validation</span>
          <strong className={`metadata-status ${diseaseValidation.className}`}>
            {diseaseValidation.label}
          </strong>
        </div>
        <div>
          <span>Clinical family</span>
          <strong title="Broad display grouping; not an assigned diagnosis code">{node.diseaseFamily}</strong>
        </div>
        <div><span>Family facet</span><strong>{node.diseaseFamilyVersion ?? "display version unavailable"}</strong></div>
        <div><span>Samples</span><strong>{node.samples}</strong></div>
        <div><span>Platform</span><strong>{node.platform}</strong></div>
        <div><span>Organism</span><strong>{node.organism}</strong></div>
      </div>

      {node.annotationCandidates && Object.keys(node.annotationCandidates).length > 0 && (
        <section className="inspector-section">
          <div className="inspector-section-title">
            <h3>Annotation provenance</h3>
            <span>{node.annotationState === "review_required" ? "review required" : "released"}</span>
          </div>
          <div className="driver-list annotation-provenance-list">
            {Object.entries(node.annotationCandidates).flatMap(([field, candidates]) => {
              const rows = candidates.slice(0, 8).map((candidate, index) => {
                const validation = ontologyValidationPresentation(candidate.ontologyValidation);
                return (
                  <div key={`${field}:${candidate.ontologyId ?? "none"}:${candidate.label}:${index}`}>
                    <span>
                      <b>{candidate.label}</b>
                      <small>
                        {field.replaceAll("_", " ")} · {candidate.ontologyId ?? "ontology ID missing"}
                        {` · review ${candidate.reviewState.replaceAll("_", " ")}`}
                      </small>
                    </span>
                    <em className={`ontology-status ${validation.className}`}>{validation.label}</em>
                  </div>
                );
              });
              if (candidates.length > 8) {
                rows.push(
                  <div key={`${field}:remainder`}>
                    <span><b>+{candidates.length - 8} more</b><small>{field.replaceAll("_", " ")} candidates in export</small></span>
                    <em>bounded view</em>
                  </div>,
                );
              }
              return rows;
            })}
          </div>
          <small>
            An ontology match validates the submitted ID/label pair, not the biological truth of
            the annotation. Unreviewed generated candidates remain hypotheses until reviewed.
          </small>
        </section>
      )}

      <section className="inspector-section">
        <div className="inspector-section-title"><h3>Nearest relationships</h3><span>{connections.length} shown</span></div>
        <div className="connection-list">
          {connections.map((edge) => {
            const other = nodeMap.get(endpoint(edge, node.id));
            return (
              <button type="button" key={edge.id} onClick={() => onSelectEdge(edge.id)}>
                {other
                  ? <NodeGlyph node={other} colorMode={colorMode} />
                  : <i className="node-glyph shape-cross" style={{ background: "#9ba7ba" }} />}
                <span><strong>{other?.id}</strong><small>{other?.disease}</small></span>
                <em>{formatCskl(edge.cskl)}</em>
              </button>
            );
          })}
        </div>
      </section>
      <a className="primary-link" href={`https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=${node.id}`} target="_blank" rel="noreferrer">
        Open GEO record <span>↗</span>
      </a>
    </div>
  );
}

function SelectionInspector({
  nodes,
  aiState,
  onAskAi,
  synthesisAvailable,
  colorMode,
}: {
  nodes: GraphNode[];
  aiState: AiState;
  onAskAi: () => void;
  synthesisAvailable: boolean;
  colorMode: NodeColorMode;
}) {
  const tissueSystems = [...new Set(nodes.map((node) => nodeTissueSystem(node)))];
  const diseases = [...new Set(nodes.map((node) => node.disease))];
  return (
    <div className="inspector-content">
      <span className="panel-eyebrow">Hand-picked workspace</span>
      <h2>{nodes.length} datasets selected</h2>
      <p className="inspector-subtitle">Shift-click nodes or dataset rows to refine this group.</p>
      <div className="selection-stack">
        {nodes.map((node) => (
          <div key={node.id}><NodeGlyph node={node} colorMode={colorMode} /><span><strong>{node.id}</strong><small>{node.disease}</small></span></div>
        ))}
      </div>
      <div className="selection-summary">
        <div><span>Anatomical systems</span><strong>{tissueSystems.join(", ")}</strong></div>
        <div><span>Diseases</span><strong>{diseases.length}</strong></div>
        <div><span>Total samples</span><strong>{nodes.reduce((sum, node) => sum + node.samples, 0)}</strong></div>
      </div>
      <AiCard
        state={aiState}
        onAsk={onAskAi}
        label="Investigate this group"
        available={synthesisAvailable}
      />
    </div>
  );
}

function AiCard({
  state,
  onAsk,
  label,
  available,
}: {
  state: AiState;
  onAsk: () => void;
  label: string;
  available: boolean;
}) {
  return (
    <section className="ai-card">
      <div className="ai-card-title"><span aria-hidden="true">✦</span><div><strong>Research synthesis</strong><small>OpenRouter · evidence-bounded</small></div></div>
      <p>Creates hypotheses, alternatives, limitations, and follow-up questions from the selected evidence packet.</p>
      {state.status === "ready" && <div className="ai-response">{state.content}</div>}
      {state.status === "error" && <div className="ai-response setup"><strong>Preview is not connected.</strong>{state.content}</div>}
      {!available && (
        <div className="ai-response setup">
          <strong>Static showcase</strong>
          Live synthesis is available only in an authenticated server deployment.
        </div>
      )}
      <button type="button" onClick={onAsk} disabled={!available || state.status === "loading"}>
        {!available ? "Server synthesis unavailable" : state.status === "loading" ? "Preparing evidence…" : label} <span>→</span>
      </button>
      <small className="transmission-note">
        {available
          ? "Only this structured evidence packet is sent when a server credential is configured."
          : "No evidence leaves the browser in this static deployment."}
      </small>
    </section>
  );
}

function WelcomeInspector() {
  return (
    <div className="inspector-content welcome-inspector">
      <span className="panel-eyebrow">Evidence inspector</span>
      <h2>Follow the biology, not the plumbing.</h2>
      <p>Select a dataset, a relationship, or shift-click a group to inspect the evidence behind the map.</p>
      <div className="how-to-read">
        <div><b>1</b><span><strong>Scan biological neighborhoods</strong><small>Color shows the chosen facet; shape keeps anatomical context.</small></span></div>
        <div><b>2</b><span><strong>Open a relationship</strong><small>See significance, gene-driver concentration, semantic agreement, and overlap warnings.</small></span></div>
        <div><b>3</b><span><strong>Ask a discovery question</strong><small>Query across metadata, topology, genes, pathways, and provenance.</small></span></div>
      </div>
      <div className="scientific-note">
        <strong>Interpretation rule</strong>
        <p>C-SKL is dataset similarity, SPECTER2 is text proximity, and neither alone establishes mechanism or causality.</p>
      </div>
    </div>
  );
}
