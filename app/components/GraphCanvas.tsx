"use client";

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  diseaseShapeLabel,
  nodeTissueSystem,
  tissueColor,
  type GraphEdge,
  type GraphNode,
} from "../lib/graph-data";
import {
  computedSpecter2,
  isOverlapQualified,
} from "../lib/evidence-policy";
import {
  computeGraphLayout,
  placeGroupLabels,
  selectRenderedEdges,
} from "../lib/graph-layout";

export type GraphMode = "cskl" | "specter2" | "agreement";
export type ClusterMode = "topology" | "tissue" | "disease";

type Props = {
  nodes: GraphNode[];
  edges: GraphEdge[];
  mode: GraphMode;
  clusterMode: ClusterMode;
  selectedNodeIds: Set<string>;
  selectedEdgeId: string | null;
  onSelectNode: (id: string, additive: boolean) => void;
  onSelectEdge: (id: string) => void;
  onClear: () => void;
};

type Viewport = { width: number; height: number };
type Point = { x: number; y: number };
type Hover =
  | { kind: "node"; id: string; x: number; y: number }
  | { kind: "edge"; id: string; x: number; y: number }
  | null;

function drawNodeShape(
  context: CanvasRenderingContext2D,
  node: GraphNode,
  x: number,
  y: number,
  radius: number,
) {
  context.beginPath();
  const shape = diseaseShapeLabel[node.diseaseFamily];
  if (shape === "circle") {
    context.arc(x, y, radius, 0, Math.PI * 2);
    return;
  }
  if (shape === "triangle") {
    context.moveTo(x, y - radius * 1.15);
    context.lineTo(x + radius, y + radius * 0.85);
    context.lineTo(x - radius, y + radius * 0.85);
    context.closePath();
    return;
  }
  if (shape === "diamond") {
    context.moveTo(x, y - radius * 1.15);
    context.lineTo(x + radius, y);
    context.lineTo(x, y + radius * 1.15);
    context.lineTo(x - radius, y);
    context.closePath();
    return;
  }
  if (shape === "square") {
    context.rect(x - radius * 0.86, y - radius * 0.86, radius * 1.72, radius * 1.72);
    return;
  }
  for (let index = 0; index < 6; index += 1) {
    const angle = Math.PI / 6 + (index * Math.PI) / 3;
    const px = x + Math.cos(angle) * radius;
    const py = y + Math.sin(angle) * radius;
    if (index === 0) context.moveTo(px, py);
    else context.lineTo(px, py);
  }
  context.closePath();
}

function distanceToSegment(point: Point, a: Point, b: Point) {
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  if (dx === 0 && dy === 0) return Math.hypot(point.x - a.x, point.y - a.y);
  const t = Math.max(
    0,
    Math.min(1, ((point.x - a.x) * dx + (point.y - a.y) * dy) / (dx * dx + dy * dy)),
  );
  return Math.hypot(point.x - (a.x + t * dx), point.y - (a.y + t * dy));
}

function nodeRadius(node: GraphNode, zoom: number, nodeCount: number, viewport: Viewport) {
  const base = Math.max(4, Math.min(10, 3 + Math.log2(node.samples + 1) * 0.65));
  const pixelsPerNode = (viewport.width * viewport.height) / Math.max(nodeCount, 1);
  const densityScale = Math.max(0.3, Math.min(1, Math.sqrt(pixelsPerNode / 900)));
  return base * densityScale * Math.min(Math.sqrt(zoom), 2.4);
}

function truncateCanvasLabel(
  context: CanvasRenderingContext2D,
  value: string,
  maximumWidth: number,
) {
  if (context.measureText(value).width <= maximumWidth) return value;
  let end = value.length;
  while (end > 1 && context.measureText(`${value.slice(0, end)}…`).width > maximumWidth) {
    end -= 1;
  }
  return `${value.slice(0, Math.max(1, end)).trimEnd()}…`;
}

export function GraphCanvas({
  nodes,
  edges,
  mode,
  clusterMode,
  selectedNodeIds,
  selectedEdgeId,
  onSelectNode,
  onSelectEdge,
  onClear,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapperRef = useRef<HTMLDivElement>(null);
  const dragRef = useRef<{
    pointerId: number;
    start: Point;
    origin: Point;
    moved: boolean;
  } | null>(null);
  const [viewport, setViewport] = useState<Viewport>({ width: 900, height: 640 });
  const [pan, setPan] = useState<Point>({ x: 0, y: 0 });
  const [zoom, setZoom] = useState(1);
  const [hover, setHover] = useState<Hover>(null);

  const nodeMap = useMemo(() => new Map(nodes.map((node) => [node.id, node])), [nodes]);
  const visibleEdges = useMemo(() => {
    if (mode === "specter2" || mode === "agreement") {
      return edges.filter((item) => computedSpecter2(item) !== undefined);
    }
    return edges;
  }, [edges, mode]);
  const csklExtent = useMemo(() => {
    const values = edges.map((item) => Math.log10(Math.max(item.cskl, 1e-12)));
    if (!values.length) return { min: 0, max: 1 };
    return { min: Math.min(...values), max: Math.max(...values) };
  }, [edges]);

  const layout = useMemo(() => computeGraphLayout(nodes, clusterMode), [nodes, clusterMode]);
  const positions = layout.positions;
  const renderedEdges = useMemo(
    () =>
      selectRenderedEdges({
        edges: visibleEdges,
        mode,
        zoom,
        selectedNodeIds,
        selectedEdgeId,
      }),
    [mode, selectedEdgeId, selectedNodeIds, visibleEdges, zoom],
  );

  useEffect(() => {
    const wrapper = wrapperRef.current;
    if (!wrapper) return;
    const observer = new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect;
      setViewport({ width: Math.max(width, 320), height: Math.max(height, 360) });
    });
    observer.observe(wrapper);
    return () => observer.disconnect();
  }, []);

  const worldToScreen = useCallback(
    (point: Point) => {
      const margin = Math.min(80, viewport.width * 0.09);
      const usableWidth = Math.max(viewport.width - margin * 2, 180);
      const usableHeight = Math.max(viewport.height - margin * 2, 180);
      return {
        x: viewport.width / 2 + (point.x - 0.5) * usableWidth * zoom + pan.x,
        y: viewport.height / 2 + (point.y - 0.5) * usableHeight * zoom + pan.y,
      };
    },
    [pan, viewport, zoom],
  );

  const hitTest = useCallback(
    (point: Point): Hover => {
      for (let index = nodes.length - 1; index >= 0; index -= 1) {
        const node = nodes[index];
        const position = worldToScreen(positions.get(node.id) ?? node);
        if (
          Math.hypot(point.x - position.x, point.y - position.y) <=
          nodeRadius(node, zoom, nodes.length, viewport) + 5
        ) {
          return { kind: "node", id: node.id, x: point.x, y: point.y };
        }
      }
      let nearest: { edge: GraphEdge; distance: number } | null = null;
      for (const item of renderedEdges) {
        const source = positions.get(item.source);
        const target = positions.get(item.target);
        if (!source || !target) continue;
        const distance = distanceToSegment(point, worldToScreen(source), worldToScreen(target));
        if (distance <= 6 && (!nearest || distance < nearest.distance)) {
          nearest = { edge: item, distance };
        }
      }
      return nearest
        ? { kind: "edge", id: nearest.edge.id, x: point.x, y: point.y }
        : null;
    },
    [nodes, positions, renderedEdges, viewport, worldToScreen, zoom],
  );

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.round(viewport.width * ratio);
    canvas.height = Math.round(viewport.height * ratio);
    canvas.style.width = `${viewport.width}px`;
    canvas.style.height = `${viewport.height}px`;
    const context = canvas.getContext("2d");
    if (!context) return;
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, viewport.width, viewport.height);

    const background = context.createRadialGradient(
      viewport.width * 0.5,
      viewport.height * 0.45,
      20,
      viewport.width * 0.5,
      viewport.height * 0.45,
      Math.max(viewport.width, viewport.height) * 0.72,
    );
    background.addColorStop(0, "#14233a");
    background.addColorStop(1, "#091321");
    context.fillStyle = background;
    context.fillRect(0, 0, viewport.width, viewport.height);

    context.save();
    context.globalAlpha = 0.12;
    context.strokeStyle = "#99aec9";
    context.lineWidth = 1;
    const grid = 44;
    for (let x = ((pan.x % grid) + grid) % grid; x < viewport.width; x += grid) {
      context.beginPath();
      context.moveTo(x, 0);
      context.lineTo(x, viewport.height);
      context.stroke();
    }
    for (let y = ((pan.y % grid) + grid) % grid; y < viewport.height; y += grid) {
      context.beginPath();
      context.moveTo(0, y);
      context.lineTo(viewport.width, y);
      context.stroke();
    }
    context.restore();

    if (clusterMode !== "topology") {
      for (const group of layout.groups) {
        const topLeft = worldToScreen({ x: group.bounds.x, y: group.bounds.y });
        const bottomRight = worldToScreen({
          x: group.bounds.x + group.bounds.width,
          y: group.bounds.y + group.bounds.height,
        });
        const width = bottomRight.x - topLeft.x;
        const height = bottomRight.y - topLeft.y;
        if (width < 2 || height < 2) continue;
        context.save();
        context.fillStyle = clusterMode === "tissue" ? tissueColor(group.label) : "#91a9c4";
        context.strokeStyle = clusterMode === "tissue" ? tissueColor(group.label) : "#aec2da";
        context.globalAlpha = 0.055;
        context.beginPath();
        context.roundRect(topLeft.x, topLeft.y, width, height, Math.min(10, width / 5, height / 5));
        context.fill();
        context.globalAlpha = 0.2;
        context.lineWidth = 1;
        context.stroke();
        context.restore();
      }
    }

    const overview = renderedEdges.length < visibleEdges.length;
    const orderedEdges = [...renderedEdges].sort(
      (left, right) => Number(left.id === selectedEdgeId) - Number(right.id === selectedEdgeId),
    );
    for (const item of orderedEdges) {
      const source = positions.get(item.source);
      const target = positions.get(item.target);
      if (!source || !target) continue;
      const a = worldToScreen(source);
      const b = worldToScreen(target);
      const logValue = Math.log10(Math.max(item.cskl, 1e-12));
      const fallbackStrength =
        1 - (logValue - csklExtent.min) / Math.max(csklExtent.max - csklExtent.min, 0.0001);
      const csklStrength = Math.max(
        0,
        Math.min(1, item.csklPercentile ?? fallbackStrength),
      );
      const textStrength = computedSpecter2(item) ?? 0;
      const qOpacity = 0.14 + 0.46 * (1 - Math.sqrt(Math.max(0, Math.min(1, item.qValue))));
      const selected = selectedEdgeId === item.id;
      const connectedToSelection =
        selectedNodeIds.has(item.source) || selectedNodeIds.has(item.target);
      let strength = mode === "specter2" ? textStrength : csklStrength;
      let color = "#5d8fb9";
      if (mode === "specter2") color = "#bc8cff";
      if (mode === "agreement") {
        const molecularHigh = csklStrength >= 0.55;
        const textHigh = textStrength >= 0.75;
        color = molecularHigh && textHigh
          ? "#4ad6c3"
          : textHigh
            ? "#bc8cff"
            : molecularHigh
              ? "#68afff"
              : "#6f7f92";
        strength = csklStrength;
      }
      context.save();
      context.strokeStyle = selected ? "#fff2bf" : color;
      context.globalAlpha = selected
        ? 1
        : connectedToSelection
          ? Math.max(0.84, qOpacity)
          : overview
            ? 0.08 + strength * 0.22
            : qOpacity;
      context.lineWidth = selected
        ? 4
        : overview
          ? 0.45 + strength * 1.2
          : 0.65 + strength * 2;
      if (isOverlapQualified(item)) context.setLineDash([2, 7]);
      context.beginPath();
      context.moveTo(a.x, a.y);
      context.lineTo(b.x, b.y);
      context.stroke();
      context.restore();

      const computedTextScore = computedSpecter2(item);
      if (mode === "cskl" && computedTextScore !== undefined) {
        context.save();
        context.fillStyle = computedTextScore >= 0.75 ? "#4ad6c3" : "#bc8cff";
        context.globalAlpha = 0.9;
        context.beginPath();
        context.arc((a.x + b.x) / 2, (a.y + b.y) / 2, selected ? 4.5 : 3, 0, Math.PI * 2);
        context.fill();
        context.restore();
      }
    }

    for (const node of nodes) {
      const position = worldToScreen(positions.get(node.id) ?? node);
      const radius = nodeRadius(node, zoom, nodes.length, viewport);
      const selected = selectedNodeIds.has(node.id);
      const hovered = hover?.kind === "node" && hover.id === node.id;
      if (selected || hovered) {
        context.save();
        context.shadowColor = tissueColor(nodeTissueSystem(node));
        context.shadowBlur = selected ? 22 : 14;
        context.strokeStyle = selected ? "#fff3c4" : "#dbe9f7";
        context.lineWidth = selected ? 2.8 : 1.5;
        context.beginPath();
        context.arc(position.x, position.y, radius + 6, 0, Math.PI * 2);
        context.stroke();
        context.restore();
      }
      context.save();
      drawNodeShape(context, node, position.x, position.y, radius);
      context.fillStyle = tissueColor(nodeTissueSystem(node));
      context.globalAlpha = 0.96;
      context.fill();
      context.strokeStyle = "rgba(255,255,255,.72)";
      context.lineWidth = 1.35;
      context.stroke();
      if (node.annotationSource === "llm_candidate") {
        context.fillStyle = "#f6c85f";
        context.beginPath();
        context.arc(position.x + radius * 0.72, position.y - radius * 0.72, 2.7, 0, Math.PI * 2);
        context.fill();
      }
      context.restore();

      const labelZoom = 2.2 * Math.max(1, Math.pow(nodes.length / 500, 0.35));
      if (nodes.length <= 80 || zoom >= labelZoom || selected || hovered) {
        context.save();
        context.font = `${selected ? 650 : 550} ${selected ? 12 : 11}px Inter, Arial, sans-serif`;
        context.textAlign = "center";
        context.textBaseline = "top";
        context.strokeStyle = "rgba(7, 16, 29, .9)";
        context.lineWidth = 4;
        context.strokeText(node.id, position.x, position.y + radius + 7);
        context.fillStyle = selected ? "#fff4ca" : "#d8e4f2";
        context.fillText(node.id, position.x, position.y + radius + 7);
        context.restore();
      }
    }

    const topologyMinimum = zoom < 1.2 ? 6 : zoom < 2.1 ? 2 : 1;
    const labelCandidates = layout.groups
      .filter((group) => clusterMode !== "topology" || group.nodeIds.length >= topologyMinimum)
      .map((group) => {
        const topLeft = worldToScreen({ x: group.bounds.x, y: group.bounds.y });
        const bottomRight = worldToScreen({
          x: group.bounds.x + group.bounds.width,
          y: group.bounds.y + group.bounds.height,
        });
        const selected = group.nodeIds.some((id) => selectedNodeIds.has(id));
        return {
          id: group.id,
          label: group.label,
          nodeCount: group.nodeIds.length,
          priority: selected ? 10_000 : 0,
          bounds: {
            x: Math.min(topLeft.x, bottomRight.x),
            y: Math.min(topLeft.y, bottomRight.y),
            width: Math.abs(bottomRight.x - topLeft.x),
            height: Math.abs(bottomRight.y - topLeft.y),
          },
        };
      });
    const groupLabels = placeGroupLabels({
      candidates: labelCandidates,
      viewport,
      maximum: clusterMode === "topology" ? (zoom < 1.2 ? 10 : zoom < 2.1 ? 18 : 32) : 40,
    });
    for (const label of groupLabels) {
      const raw = `${label.label} · ${label.nodeCount}`;
      context.save();
      context.font = "650 13px Inter, Arial, sans-serif";
      const display = truncateCanvasLabel(context, raw, label.width - 22);
      context.fillStyle = "rgba(7, 17, 30, .93)";
      context.strokeStyle = "rgba(192, 213, 235, .58)";
      context.lineWidth = 1;
      context.beginPath();
      context.roundRect(label.x, label.y, label.width, label.height, 7);
      context.fill();
      context.stroke();
      context.fillStyle =
        clusterMode === "tissue" ? tissueColor(label.label) : clusterMode === "disease" ? "#f1b94d" : "#4ad6c3";
      context.fillRect(label.x + 6, label.y + 6, 3, label.height - 12);
      context.fillStyle = "#f0f6fc";
      context.textAlign = "left";
      context.textBaseline = "middle";
      context.fillText(display, label.x + 14, label.y + label.height / 2);
      context.restore();
    }

    if (hover) {
      const label =
        hover.kind === "node"
          ? `${hover.id} · ${nodeMap.get(hover.id)?.title ?? "Dataset"}`
          : (() => {
              const item = edges.find((candidate) => candidate.id === hover.id);
              return item ? `${item.source} ↔ ${item.target}` : "Relationship";
            })();
      context.save();
      context.font = "500 12px Inter, Arial, sans-serif";
      const width = Math.min(context.measureText(label).width + 22, viewport.width - 24);
      const x = Math.min(Math.max(12, hover.x + 14), viewport.width - width - 12);
      const y = Math.min(Math.max(12, hover.y + 14), viewport.height - 42);
      context.fillStyle = "rgba(8, 17, 30, .94)";
      context.strokeStyle = "rgba(163, 187, 216, .4)";
      context.lineWidth = 1;
      context.beginPath();
      context.roundRect(x, y, width, 30, 7);
      context.fill();
      context.stroke();
      context.fillStyle = "#eef6ff";
      context.textBaseline = "middle";
      context.fillText(label, x + 11, y + 15, width - 18);
      context.restore();
    }
  }, [
    csklExtent,
    clusterMode,
    edges,
    hover,
    layout,
    mode,
    nodeMap,
    nodes,
    pan,
    positions,
    renderedEdges,
    selectedEdgeId,
    selectedNodeIds,
    viewport,
    visibleEdges,
    worldToScreen,
    zoom,
  ]);

  const eventPoint = (event: React.MouseEvent<HTMLCanvasElement>): Point => {
    const rect = event.currentTarget.getBoundingClientRect();
    return { x: event.clientX - rect.left, y: event.clientY - rect.top };
  };

  return (
    <div className="graph-stage" ref={wrapperRef} data-testid="graph-stage">
      <canvas
        ref={canvasRef}
        className="graph-canvas"
        role="img"
        tabIndex={0}
        aria-label={`Interactive dataset graph with ${nodes.length} datasets and ${visibleEdges.length} visible relationships. Use the accessible dataset list to inspect individual studies.`}
        onContextMenu={(event) => event.preventDefault()}
        onPointerDown={(event) => {
          const point = eventPoint(event);
          event.currentTarget.setPointerCapture(event.pointerId);
          dragRef.current = { pointerId: event.pointerId, start: point, origin: pan, moved: false };
        }}
        onPointerMove={(event) => {
          const point = eventPoint(event);
          const drag = dragRef.current;
          if (drag?.pointerId === event.pointerId) {
            const dx = point.x - drag.start.x;
            const dy = point.y - drag.start.y;
            if (Math.hypot(dx, dy) > 3) drag.moved = true;
            if (drag.moved) setPan({ x: drag.origin.x + dx, y: drag.origin.y + dy });
            return;
          }
          setHover(hitTest(point));
        }}
        onPointerUp={(event) => {
          const point = eventPoint(event);
          const drag = dragRef.current;
          dragRef.current = null;
          if (drag?.moved) return;
          const hit = hitTest(point);
          if (hit?.kind === "node") onSelectNode(hit.id, event.shiftKey || event.ctrlKey || event.metaKey);
          else if (hit?.kind === "edge") onSelectEdge(hit.id);
          else onClear();
        }}
        onPointerLeave={() => {
          if (!dragRef.current) setHover(null);
        }}
        onWheel={(event) => {
          event.preventDefault();
          const point = eventPoint(event);
          const next = Math.min(8, Math.max(0.5, zoom * Math.exp(-event.deltaY * 0.001)));
          const ratio = next / zoom;
          setPan({
            x: point.x - viewport.width / 2 - (point.x - viewport.width / 2 - pan.x) * ratio,
            y: point.y - viewport.height / 2 - (point.y - viewport.height / 2 - pan.y) * ratio,
          });
          setZoom(next);
        }}
        onDoubleClick={() => {
          setPan({ x: 0, y: 0 });
          setZoom(1);
        }}
      />
      <div className="map-controls" aria-label="Graph zoom controls">
        <button type="button" aria-label="Zoom in" onClick={() => setZoom((value) => Math.min(value * 1.25, 8))}>
          +
        </button>
        <button type="button" aria-label="Zoom out" onClick={() => setZoom((value) => Math.max(value / 1.25, 0.5))}>
          −
        </button>
        <button
          type="button"
          aria-label="Fit graph to view"
          onClick={() => {
            setZoom(1);
            setPan({ x: 0, y: 0 });
          }}
        >
          Fit
        </button>
      </div>
      <div className="graph-status" aria-hidden="true">
        <span>{nodes.length} datasets</span>
        <span>
          {renderedEdges.length === visibleEdges.length
            ? `${visibleEdges.length} links`
            : `${renderedEdges.length} of ${visibleEdges.length} links · zoom for detail`}
        </span>
        <span>{Math.round(zoom * 100)}%</span>
      </div>
    </div>
  );
}
