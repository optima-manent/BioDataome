import { nodeTissueSystem, type GraphEdge, type GraphNode } from "./graph-data.ts";

export type GraphLayoutMode = "topology" | "tissue" | "disease";
export type EvidenceMode = "cskl" | "specter2" | "agreement";
export type LayoutPoint = { x: number; y: number };

type Rect = { x: number; y: number; width: number; height: number };

export type LayoutGroup = {
  id: string;
  label: string;
  nodeIds: string[];
  bounds: Rect;
  center: LayoutPoint;
};

export type GraphLayout = {
  positions: Map<string, LayoutPoint>;
  groups: LayoutGroup[];
};

export type ScreenRect = { x: number; y: number; width: number; height: number };

export type GroupLabelCandidate = {
  id: string;
  label: string;
  nodeCount: number;
  bounds: ScreenRect;
  priority?: number;
};

export type GroupLabelPlacement = GroupLabelCandidate & {
  x: number;
  y: number;
  width: number;
  height: number;
};

const OUTER_MARGIN = 0.032;
const LAYOUT_ASPECT_RATIO = 1.5;

function rectanglesOverlap(left: ScreenRect, right: ScreenRect, gap = 4) {
  return !(
    left.x + left.width + gap <= right.x ||
    right.x + right.width + gap <= left.x ||
    left.y + left.height + gap <= right.y ||
    right.y + right.height + gap <= left.y
  );
}

/** Place readable group callouts near their bounds without allowing collisions. */
export function placeGroupLabels({
  candidates,
  viewport,
  topInset = 66,
  bottomInset = 42,
  rightInset = 64,
  maximum = 40,
}: {
  candidates: GroupLabelCandidate[];
  viewport: { width: number; height: number };
  topInset?: number;
  bottomInset?: number;
  rightInset?: number;
  maximum?: number;
}): GroupLabelPlacement[] {
  const safe = {
    x: 8,
    y: topInset,
    // Reserve the right-side zoom controls so callouts never sit underneath
    // interface chrome at higher zoom levels.
    width: Math.max(0, viewport.width - 8 - rightInset),
    height: Math.max(0, viewport.height - topInset - bottomInset),
  };
  const placements: GroupLabelPlacement[] = [];
  const ordered = [...candidates].sort(
    (left, right) =>
      (right.priority ?? 0) - (left.priority ?? 0) ||
      right.nodeCount - left.nodeCount ||
      left.label.localeCompare(right.label),
  );
  for (const candidate of ordered) {
    if (placements.length >= maximum) break;
    const left = Math.max(safe.x, candidate.bounds.x);
    const top = Math.max(safe.y, candidate.bounds.y);
    const right = Math.min(safe.x + safe.width, candidate.bounds.x + candidate.bounds.width);
    const bottom = Math.min(safe.y + safe.height, candidate.bounds.y + candidate.bounds.height);
    // Tiny groups still deserve a readable title. Requiring a node-sized
    // rectangle here used to drop two-study groups at overview zoom.
    if (right < left || bottom < top) continue;
    const estimatedTextWidth =
      (candidate.label.length + String(candidate.nodeCount).length + 3) * 7.35 + 22;
    // A small biological group must not force its title into an unreadable sliver.
    // Labels are callouts, so they may extend beyond the group while remaining
    // inside the viewport and collision-free.
    const width = Math.min(safe.width, Math.min(260, Math.max(112, estimatedTextWidth)));
    const height = 24;
    const clampX = (value: number) =>
      Math.min(Math.max(value, safe.x), safe.x + safe.width - width);
    const clampY = (value: number) =>
      Math.min(Math.max(value, safe.y), safe.y + safe.height - height);
    const centerX = (left + right) / 2 - width / 2;
    const xCandidates = [
      centerX,
      left + 4,
      right - width - 4,
      centerX - width - 8,
      centerX + width + 8,
    ];
    const yCandidates = [
      top + 4,
      bottom - height - 4,
      top - height - 6,
      bottom + 6,
      top + height + 10,
      bottom - height * 2 - 10,
      top - height * 2 - 12,
      bottom + height + 12,
      top - height * 3 - 18,
      bottom + height * 2 + 18,
    ];
    const localPositions = yCandidates.flatMap((y) =>
      xCandidates.map((x) => ({ x: clampX(x), y: clampY(y) })),
    );
    const target = { x: (left + right) / 2, y: (top + bottom) / 2 };
    const fallbackX = [
      ...xCandidates,
      safe.x,
      safe.x + safe.width / 2 - width / 2,
      safe.x + safe.width - width,
    ].map(clampX);
    const fallbackY: number[] = [];
    for (let y = safe.y; y <= safe.y + safe.height - height; y += height + 6) {
      fallbackY.push(y);
    }
    fallbackY.push(safe.y + safe.height - height);
    const fallbackPositions = fallbackY
      .flatMap((y) => fallbackX.map((x) => ({ x, y })))
      .sort(
        (first, second) =>
          Math.hypot(first.x + width / 2 - target.x, first.y + height / 2 - target.y) -
            Math.hypot(second.x + width / 2 - target.x, second.y + height / 2 - target.y) ||
          first.y - second.y ||
          first.x - second.x,
      );
    const positions = [...localPositions, ...fallbackPositions].filter(
      (point, index, values) =>
        values.findIndex((other) => other.x === point.x && other.y === point.y) === index,
    );
    const selected = positions
      .map((point) => ({ ...candidate, ...point, width, height }))
      .find(
        (placement) =>
          placement.x >= safe.x &&
          placement.y >= safe.y &&
          placement.x + placement.width <= safe.x + safe.width &&
          placement.y + placement.height <= safe.y + safe.height &&
          !placements.some((other) => rectanglesOverlap(placement, other)),
      );
    if (selected) placements.push(selected);
  }
  return placements;
}

function finite(value: number, fallback: number) {
  return Number.isFinite(value) ? value : fallback;
}

function hash(value: string) {
  let result = 2166136261;
  for (let index = 0; index < value.length; index += 1) {
    result ^= value.charCodeAt(index);
    result = Math.imul(result, 16777619);
  }
  return result >>> 0;
}

function layoutTarget(nodeCount: number) {
  return nodeCount <= 1 ? 0 : Math.min(0.09, 1 / Math.sqrt(nodeCount));
}

function hasSevereCollision(positions: Map<string, LayoutPoint>, nodeIds: string[]) {
  const target = layoutTarget(nodeIds.length) * 0.8;
  if (!target) return false;
  const grid = new Map<string, number[]>();
  const values = nodeIds.map((id) => positions.get(id)!);
  values.forEach((point, index) => {
    const key = `${Math.floor((point.x * LAYOUT_ASPECT_RATIO) / target)}:${Math.floor(
      point.y / target,
    )}`;
    const entries = grid.get(key) ?? [];
    entries.push(index);
    grid.set(key, entries);
  });
  for (const [index, point] of values.entries()) {
    const cellX = Math.floor((point.x * LAYOUT_ASPECT_RATIO) / target);
    const cellY = Math.floor(point.y / target);
    for (let x = cellX - 1; x <= cellX + 1; x += 1) {
      for (let y = cellY - 1; y <= cellY + 1; y += 1) {
        for (const other of grid.get(`${x}:${y}`) ?? []) {
          if (other <= index) continue;
          const candidate = values[other];
          if (
            Math.hypot(
              (candidate.x - point.x) * LAYOUT_ASPECT_RATIO,
              candidate.y - point.y,
            ) < target
          ) {
            return true;
          }
        }
      }
    }
  }
  return false;
}

function separatePositions(
  initial: Map<string, LayoutPoint>,
  nodeIds: string[],
): Map<string, LayoutPoint> {
  const ordered = [...nodeIds].sort((left, right) => left.localeCompare(right));
  if (!hasSevereCollision(initial, ordered)) return initial;
  const target = layoutTarget(ordered.length);
  const margin = 0.025;
  const positions = ordered.map((id) => ({ ...initial.get(id)! }));
  const anchors = positions.map((point) => ({ ...point }));
  const iterations = ordered.length > 2_000 ? 180 : 420;
  for (let iteration = 0; iteration < iterations; iteration += 1) {
    const grid = new Map<string, number[]>();
    positions.forEach((point, index) => {
      const key = `${Math.floor((point.x * LAYOUT_ASPECT_RATIO) / target)}:${Math.floor(
        point.y / target,
      )}`;
      const entries = grid.get(key) ?? [];
      entries.push(index);
      grid.set(key, entries);
    });
    const deltas = positions.map(() => ({ x: 0, y: 0 }));
    let collisions = 0;
    positions.forEach((point, index) => {
      const metricX = point.x * LAYOUT_ASPECT_RATIO;
      const cellX = Math.floor(metricX / target);
      const cellY = Math.floor(point.y / target);
      for (let x = cellX - 1; x <= cellX + 1; x += 1) {
        for (let y = cellY - 1; y <= cellY + 1; y += 1) {
          for (const other of grid.get(`${x}:${y}`) ?? []) {
            if (other <= index) continue;
            const candidate = positions[other];
            const deltaX = candidate.x * LAYOUT_ASPECT_RATIO - metricX;
            const deltaY = candidate.y - point.y;
            const distance = Math.hypot(deltaX, deltaY);
            if (distance >= target) continue;
            collisions += 1;
            let unitX: number;
            let unitY: number;
            if (distance < 1e-12) {
              const angle =
                (hash(`${ordered[index]}:${ordered[other]}`) / 0xffffffff) * Math.PI * 2;
              unitX = Math.cos(angle);
              unitY = Math.sin(angle);
            } else {
              unitX = deltaX / distance;
              unitY = deltaY / distance;
            }
            const push = (target - distance) * 0.58;
            deltas[index].x -= (unitX * push) / LAYOUT_ASPECT_RATIO;
            deltas[index].y -= unitY * push;
            deltas[other].x += (unitX * push) / LAYOUT_ASPECT_RATIO;
            deltas[other].y += unitY * push;
          }
        }
      }
    });
    if (!collisions) break;
    const progress = iteration / Math.max(iterations - 1, 1);
    const temperature = target * (0.5 - 0.3 * progress);
    positions.forEach((point, index) => {
      const moveX = deltas[index].x + (anchors[index].x - point.x) * 0.0005;
      const moveY = deltas[index].y + (anchors[index].y - point.y) * 0.0005;
      const magnitude = Math.hypot(moveX * LAYOUT_ASPECT_RATIO, moveY);
      const scale = Math.min(1, temperature / Math.max(magnitude, 1e-12));
      point.x = Math.min(1 - margin, Math.max(margin, point.x + moveX * scale));
      point.y = Math.min(1 - margin, Math.max(margin, point.y + moveY * scale));
    });
  }
  return new Map(ordered.map((id, index) => [id, positions[index]]));
}

function boundsFor(nodeIds: string[], positions: Map<string, LayoutPoint>): Rect {
  const points = nodeIds.map((id) => positions.get(id)!);
  const minX = Math.min(...points.map((point) => point.x));
  const maxX = Math.max(...points.map((point) => point.x));
  const minY = Math.min(...points.map((point) => point.y));
  const maxY = Math.max(...points.map((point) => point.y));
  const padding = Math.min(0.012, layoutTarget(positions.size) * 0.28);
  return {
    x: Math.max(0.01, minX - padding),
    y: Math.max(0.01, minY - padding),
    width: Math.min(0.99, maxX + padding) - Math.max(0.01, minX - padding),
    height: Math.min(0.99, maxY + padding) - Math.max(0.01, minY - padding),
  };
}

function partition(
  groups: Array<{ id: string; label: string; nodes: GraphNode[] }>,
  bounds: Rect,
  output: Map<string, Rect>,
) {
  if (!groups.length) return;
  if (groups.length === 1) {
    output.set(groups[0].id, bounds);
    return;
  }
  const total = groups.reduce((sum, group) => sum + group.nodes.length, 0);
  let splitIndex = 1;
  let before = groups[0].nodes.length;
  let bestDifference = Math.abs(total / 2 - before);
  for (let index = 1; index < groups.length; index += 1) {
    const difference = Math.abs(total / 2 - before);
    if (difference < bestDifference) {
      bestDifference = difference;
      splitIndex = index;
    }
    before += groups[index].nodes.length;
  }
  const left = groups.slice(0, splitIndex);
  const right = groups.slice(splitIndex);
  const leftWeight = left.reduce((sum, group) => sum + group.nodes.length, 0);
  const ratio = leftWeight / total;
  if (bounds.width >= bounds.height) {
    const leftWidth = bounds.width * ratio;
    partition(left, { ...bounds, width: leftWidth }, output);
    partition(
      right,
      {
        x: bounds.x + leftWidth,
        y: bounds.y,
        width: bounds.width - leftWidth,
        height: bounds.height,
      },
      output,
    );
  } else {
    const topHeight = bounds.height * ratio;
    partition(left, { ...bounds, height: topHeight }, output);
    partition(
      right,
      {
        x: bounds.x,
        y: bounds.y + topHeight,
        width: bounds.width,
        height: bounds.height - topHeight,
      },
      output,
    );
  }
}

function inset(bounds: Rect, amount: number): Rect {
  const horizontal = Math.min(amount, bounds.width * 0.18);
  const vertical = Math.min(amount, bounds.height * 0.18);
  return {
    x: bounds.x + horizontal,
    y: bounds.y + vertical,
    width: Math.max(0.0001, bounds.width - horizontal * 2),
    height: Math.max(0.0001, bounds.height - vertical * 2),
  };
}

function placeGrid(nodes: GraphNode[], bounds: Rect, positions: Map<string, LayoutPoint>) {
  if (nodes.length === 1) {
    positions.set(nodes[0].id, {
      x: bounds.x + bounds.width / 2,
      y: bounds.y + bounds.height / 2,
    });
    return;
  }
  const aspect = Math.max(bounds.width / Math.max(bounds.height, 0.0001), 0.05);
  let columns = Math.max(1, Math.ceil(Math.sqrt(nodes.length * aspect)));
  let rows = Math.ceil(nodes.length / columns);
  while (columns > 1 && (columns - 1) * rows >= nodes.length) columns -= 1;
  rows = Math.ceil(nodes.length / columns);
  const cellWidth = bounds.width / columns;
  const cellHeight = bounds.height / rows;
  nodes.forEach((node, index) => {
    const row = Math.floor(index / columns);
    const column = index % columns;
    const valuesInRow = Math.min(columns, nodes.length - row * columns);
    const rowOffset = (columns - valuesInRow) * cellWidth * 0.5;
    positions.set(node.id, {
      x: bounds.x + rowOffset + (column + 0.5) * cellWidth,
      y: bounds.y + (row + 0.5) * cellHeight,
    });
  });
}

function groupedLayout(
  nodes: GraphNode[],
  mode: Exclude<GraphLayoutMode, "topology">,
): GraphLayout {
  const byGroup = new Map<string, GraphNode[]>();
  for (const node of nodes) {
    const label = mode === "tissue" ? nodeTissueSystem(node) : node.diseaseFamily;
    const group = label.trim() || "Unknown";
    const values = byGroup.get(group) ?? [];
    values.push(node);
    byGroup.set(group, values);
  }
  const groups = [...byGroup.entries()]
    .map(([label, values]) => ({
      id: `${mode}:${label}`,
      label,
      nodes: [...values].sort((left, right) => left.id.localeCompare(right.id)),
    }))
    .sort(
      (left, right) =>
        right.nodes.length - left.nodes.length || left.label.localeCompare(right.label),
    );
  const rectangles = new Map<string, Rect>();
  partition(
    groups,
    {
      x: OUTER_MARGIN,
      y: OUTER_MARGIN,
      width: 1 - OUTER_MARGIN * 2,
      height: 1 - OUTER_MARGIN * 2,
    },
    rectangles,
  );
  const initialPositions = new Map<string, LayoutPoint>();
  const gap = Math.max(
    0.0025,
    Math.min(0.012, 0.09 / Math.sqrt(Math.max(groups.length, 1))),
  );
  const layoutGroups = groups.map((group) => {
    const bounds = inset(rectangles.get(group.id)!, gap);
    placeGrid(group.nodes, inset(bounds, Math.min(gap, 0.006)), initialPositions);
    return {
      id: group.id,
      label: group.label,
      nodeIds: group.nodes.map((node) => node.id),
      bounds,
      center: { x: bounds.x + bounds.width / 2, y: bounds.y + bounds.height / 2 },
    };
  });
  const positions = separatePositions(
    initialPositions,
    nodes.map((node) => node.id),
  );
  return {
    positions,
    groups: layoutGroups.map((group) => {
      const bounds = boundsFor(group.nodeIds, positions);
      return {
        ...group,
        bounds,
        center: { x: bounds.x + bounds.width / 2, y: bounds.y + bounds.height / 2 },
      };
    }),
  };
}

function topologyLayout(nodes: GraphNode[]): GraphLayout {
  const positions = new Map<string, LayoutPoint>();
  for (const [index, node] of nodes.entries()) {
    const phase = (index / Math.max(nodes.length, 1)) * Math.PI * 2;
    positions.set(node.id, {
      x: finite(node.x, 0.5 + Math.cos(phase) * 0.34),
      y: finite(node.y, 0.5 + Math.sin(phase) * 0.34),
    });
  }
  const grouped = new Map<string, GraphNode[]>();
  for (const node of nodes) {
    const label = node.community || "Unassigned";
    const values = grouped.get(label) ?? [];
    values.push(node);
    grouped.set(label, values);
  }
  const separated = separatePositions(
    positions,
    nodes.map((node) => node.id),
  );
  const groups = [...grouped.entries()].map(([label, values]) => {
    const bounds = boundsFor(
      values.map((node) => node.id),
      separated,
    );
    return {
      id: `topology:${label}`,
      label: label.replace(/^community-0*/, "Cluster "),
      nodeIds: values.map((node) => node.id),
      bounds,
      center: {
        x: bounds.x + bounds.width / 2,
        y: bounds.y + bounds.height / 2,
      },
    };
  });
  return { positions: separated, groups };
}

export function computeGraphLayout(nodes: GraphNode[], mode: GraphLayoutMode): GraphLayout {
  return mode === "topology" ? topologyLayout(nodes) : groupedLayout(nodes, mode);
}

function edgeStrength(edge: GraphEdge, mode: EvidenceMode) {
  const molecular = Math.max(0, Math.min(1, edge.csklPercentile ?? 1 - edge.qValue));
  const text =
    edge.specter2Provenance === "computed" && typeof edge.specter2 === "number"
      ? Math.max(0, Math.min(1, edge.specter2))
      : 0;
  if (mode === "specter2") return text;
  if (mode === "agreement") return Math.min(molecular, text);
  return molecular;
}

export function selectRenderedEdges({
  edges,
  mode,
  zoom,
  selectedNodeIds,
  selectedEdgeId,
}: {
  edges: GraphEdge[];
  mode: EvidenceMode;
  zoom: number;
  selectedNodeIds: ReadonlySet<string>;
  selectedEdgeId: string | null;
}): GraphEdge[] {
  if (zoom >= 2.1) return edges;
  const topK = zoom >= 1.65 ? 12 : zoom >= 1.25 ? 5 : 2;
  const adjacency = new Map<string, GraphEdge[]>();
  for (const edge of edges) {
    for (const id of [edge.source, edge.target]) {
      const values = adjacency.get(id) ?? [];
      values.push(edge);
      adjacency.set(id, values);
    }
  }
  const selected = new Set<string>();
  for (const values of adjacency.values()) {
    values
      .sort(
        (left, right) =>
          edgeStrength(right, mode) - edgeStrength(left, mode) ||
          left.qValue - right.qValue ||
          left.id.localeCompare(right.id),
      )
      .slice(0, topK)
      .forEach((edge) => selected.add(edge.id));
  }
  for (const edge of edges) {
    if (
      edge.id === selectedEdgeId ||
      selectedNodeIds.has(edge.source) ||
      selectedNodeIds.has(edge.target)
    ) {
      selected.add(edge.id);
    }
  }
  return edges.filter((edge) => selected.has(edge.id));
}
