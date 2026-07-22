"""Deterministic, versioned Leiden community and layout snapshot builder."""

from __future__ import annotations

import hashlib
import math
import os
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from .catalog import Catalog, canonical_json


class GraphDependencyError(RuntimeError):
    pass


LAYOUT_ALGORITHM = "igraph-fr-collision-v2"
LAYOUT_ASPECT_RATIO = 1.5
LAYOUT_MARGIN = 0.025
LAYOUT_ITERATIONS = 600


def _igraph():
    try:
        import igraph as ig
    except ImportError as exc:
        raise GraphDependencyError(
            "Graph building requires the optional 'graph' dependency: pip install 'cskl-atlas[graph]'."
        ) from exc
    return ig


def _canonical_membership(node_ids: list[str], membership: list[int]) -> list[str]:
    groups: dict[int, list[str]] = defaultdict(list)
    for node_id, group in zip(node_ids, membership, strict=True):
        groups[group].append(node_id)
    ordered = sorted(groups, key=lambda group: min(groups[group]))
    labels = {group: f"community-{index + 1:04d}" for index, group in enumerate(ordered)}
    return [labels[group] for group in membership]


def _normalise_layout(coordinates: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not coordinates:
        return []
    xs = [float(value[0]) for value in coordinates]
    ys = [float(value[1]) for value in coordinates]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    def scale(value: float, lower: float, upper: float) -> float:
        return 0.5 if math.isclose(lower, upper) else 0.05 + 0.9 * (value - lower) / (upper - lower)

    return [
        (scale(x, min_x, max_x), scale(y, min_y, max_y))
        for x, y in zip(xs, ys, strict=True)
    ]


def _stable_unit_vector(left: str, right: str) -> tuple[float, float]:
    digest = hashlib.sha256(f"{left}:{right}".encode()).digest()
    angle = int.from_bytes(digest[:8], "big") / ((1 << 64) - 1) * math.tau
    return math.cos(angle), math.sin(angle)


def _layout_target(node_count: int) -> float:
    if node_count <= 1:
        return 0.0
    # The aspect-aware packing area remains feasible while giving 500 nodes
    # roughly 20 screen pixels between centres in the reference viewport.
    return min(0.09, 1.0 / math.sqrt(node_count))


def _separation_quality(
    coordinates: list[tuple[float, float]],
    *,
    target: float,
    aspect_ratio: float,
) -> dict[str, float | int]:
    if len(coordinates) <= 1 or target <= 0:
        return {
            "target_minimum_separation": target,
            "observed_minimum_separation": target,
            "severe_collision_pair_count": 0,
        }
    grid: dict[tuple[int, int], list[int]] = defaultdict(list)
    metric = [(x * aspect_ratio, y) for x, y in coordinates]
    for index, (x, y) in enumerate(metric):
        grid[(math.floor(x / target), math.floor(y / target))].append(index)
    observed = target
    severe = 0
    for index, (x, y) in enumerate(metric):
        cell_x, cell_y = math.floor(x / target), math.floor(y / target)
        for nearby_x in range(cell_x - 1, cell_x + 2):
            for nearby_y in range(cell_y - 1, cell_y + 2):
                for other in grid.get((nearby_x, nearby_y), []):
                    if other <= index:
                        continue
                    distance = math.hypot(metric[other][0] - x, metric[other][1] - y)
                    observed = min(observed, distance)
                    if distance < target * 0.8:
                        severe += 1
    return {
        "target_minimum_separation": target,
        # This value is intentionally capped at the target: it is the lower
        # bound relevant to the publication gate, not a decorative statistic.
        "observed_minimum_separation": observed,
        "severe_collision_pair_count": severe,
    }


def _collision_aware_layout(
    coordinates: list[tuple[float, float]],
    node_ids: list[str],
    *,
    aspect_ratio: float = LAYOUT_ASPECT_RATIO,
    margin: float = LAYOUT_MARGIN,
    iterations: int = LAYOUT_ITERATIONS,
) -> tuple[list[tuple[float, float]], dict[str, float | int | str]]:
    """Project a deterministic force layout onto a readable non-overlap packing.

    A spatial hash keeps each iteration close to linear for sparse local
    neighborhoods. The weak anchor spring preserves the topology found by
    Fruchterman-Reingold while the collision force accounts for the wider
    browser viewport. This is release-time work, never an animated client-side
    simulation, so unchanged inputs produce byte-identical coordinates.
    """

    if len(coordinates) != len(node_ids):
        raise ValueError("layout coordinates and node identifiers must have equal length")
    if aspect_ratio <= 0 or not math.isfinite(aspect_ratio):
        raise ValueError("layout aspect ratio must be finite and positive")
    if not 0 <= margin < 0.5:
        raise ValueError("layout margin must be in [0, 0.5)")
    if iterations < 1:
        raise ValueError("layout iterations must be positive")
    if len(coordinates) <= 1:
        quality = _separation_quality(
            coordinates, target=0.0, aspect_ratio=aspect_ratio
        )
        return coordinates, {
            "algorithm": LAYOUT_ALGORITHM,
            "aspect_ratio": aspect_ratio,
            "iterations": 0,
            **quality,
        }

    positions = [
        [
            min(1.0 - margin, max(margin, float(x))),
            min(1.0 - margin, max(margin, float(y))),
        ]
        for x, y in coordinates
    ]
    anchors = [value.copy() for value in positions]
    target = _layout_target(len(positions))
    completed_iterations = iterations

    for iteration in range(iterations):
        metric = [(x * aspect_ratio, y) for x, y in positions]
        grid: dict[tuple[int, int], list[int]] = defaultdict(list)
        for index, (x, y) in enumerate(metric):
            grid[(math.floor(x / target), math.floor(y / target))].append(index)
        deltas = [[0.0, 0.0] for _ in positions]
        collision_count = 0
        for index, (x, y) in enumerate(metric):
            cell_x, cell_y = math.floor(x / target), math.floor(y / target)
            for nearby_x in range(cell_x - 1, cell_x + 2):
                for nearby_y in range(cell_y - 1, cell_y + 2):
                    for other in grid.get((nearby_x, nearby_y), []):
                        if other <= index:
                            continue
                        delta_x = metric[other][0] - x
                        delta_y = metric[other][1] - y
                        distance = math.hypot(delta_x, delta_y)
                        if distance >= target:
                            continue
                        collision_count += 1
                        if distance < 1e-12:
                            unit_x, unit_y = _stable_unit_vector(
                                node_ids[index], node_ids[other]
                            )
                        else:
                            unit_x, unit_y = delta_x / distance, delta_y / distance
                        push = (target - distance) * 0.58
                        deltas[index][0] -= unit_x * push / aspect_ratio
                        deltas[index][1] -= unit_y * push
                        deltas[other][0] += unit_x * push / aspect_ratio
                        deltas[other][1] += unit_y * push
        if collision_count == 0:
            completed_iterations = iteration
            break

        progress = iteration / max(iterations - 1, 1)
        temperature = target * (0.5 - 0.3 * progress)
        anchor_strength = 0.0005
        for index, (x, y) in enumerate(positions):
            move_x = deltas[index][0] + (anchors[index][0] - x) * anchor_strength
            move_y = deltas[index][1] + (anchors[index][1] - y) * anchor_strength
            magnitude = math.hypot(move_x * aspect_ratio, move_y)
            scale = min(1.0, temperature / max(magnitude, 1e-12))
            positions[index][0] = min(
                1.0 - margin, max(margin, x + move_x * scale)
            )
            positions[index][1] = min(
                1.0 - margin, max(margin, y + move_y * scale)
            )

    output = [(float(x), float(y)) for x, y in positions]
    quality = _separation_quality(
        output, target=target, aspect_ratio=aspect_ratio
    )
    if quality["severe_collision_pair_count"]:
        raise RuntimeError(
            "Collision-aware graph layout failed its minimum-separation publication gate: "
            f"{quality['severe_collision_pair_count']} severe collisions remain."
        )
    return output, {
        "algorithm": LAYOUT_ALGORITHM,
        "aspect_ratio": aspect_ratio,
        "iterations": completed_iterations,
        **quality,
    }


def build_graph_snapshot(
    catalog: Catalog,
    *,
    calibration_id: str,
    manifest_directory: str | Path,
    q_max: float,
    independent_only: bool,
    top_k_per_node: int,
    resolution: float,
    seed: int,
    stability_runs: int = 5,
    text_release_id: str | None = None,
) -> dict[str, Any]:
    """Build and stage (but never auto-publish) one auditable graph snapshot."""

    if not 0 <= q_max <= 1:
        raise ValueError("q_max must be in [0, 1]")
    if top_k_per_node < 1:
        raise ValueError("top_k_per_node must be positive")
    if not math.isfinite(resolution) or resolution <= 0:
        raise ValueError("resolution must be finite and positive")
    if stability_runs < 1 or stability_runs > 50:
        raise ValueError("stability_runs must be between 1 and 50")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")

    ig = _igraph()
    with catalog.reader() as connection:
        release = connection.execute(
            "SELECT * FROM calibration_releases WHERE calibration_id=?", (calibration_id,)
        ).fetchone()
        if not release or release["status"] not in {"calibrated", "published"}:
            raise ValueError("Graph building requires a finalized calibration release.")
        bound_member_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM calibration_release_members WHERE calibration_id=?",
                (calibration_id,),
            ).fetchone()[0]
        )
        if bound_member_count:
            endpoint_scope = "frozen_calibration_members"
            node_rows = connection.execute(
                """SELECT version_id FROM calibration_release_members
                   WHERE calibration_id=? ORDER BY version_id""",
                (calibration_id,),
            ).fetchall()
            edges = connection.execute(
                """SELECT p.pair_id,p.version_a,p.version_b,p.cskl,c.q_value,
                          c.cskl_similarity_percentile,
                          COALESCE(o.discovery_excluded,0) AS discovery_excluded
                   FROM calibrated_edges c
                   JOIN calibration_release_pairs family
                     ON family.calibration_id=c.calibration_id AND family.pair_id=c.pair_id
                   JOIN pair_scores p ON p.pair_id=c.pair_id
                   JOIN calibration_release_members ma
                     ON ma.calibration_id=c.calibration_id AND ma.version_id=p.version_a
                   JOIN calibration_release_members mb
                     ON mb.calibration_id=c.calibration_id AND mb.version_id=p.version_b
                   LEFT JOIN overlap_evidence o ON o.overlap_id=family.overlap_id
                   WHERE c.calibration_id=? AND c.q_value<=?
                   ORDER BY c.q_value,p.cskl,p.pair_id""",
                (calibration_id, q_max),
            ).fetchall()
        else:
            # Compatibility for imported schema-v3 releases. Both edge endpoints
            # are joined to the current pointer so superseded score facts cannot
            # enter adjacency or trigger an index failure after a dataset update.
            endpoint_scope = "legacy_current_versions"
            node_rows = connection.execute(
                """SELECT DISTINCT v.version_id FROM dataset_versions v
                   JOIN datasets d ON d.current_version_id=v.version_id
                   WHERE v.version_id IN (
                     SELECT p.version_a FROM calibrated_edges c JOIN pair_scores p USING(pair_id)
                     WHERE c.calibration_id=?
                     UNION
                     SELECT p.version_b FROM calibrated_edges c JOIN pair_scores p USING(pair_id)
                     WHERE c.calibration_id=?
                   ) ORDER BY v.version_id""",
                (calibration_id, calibration_id),
            ).fetchall()
            edges = connection.execute(
                """SELECT p.pair_id,p.version_a,p.version_b,p.cskl,c.q_value,
                          c.cskl_similarity_percentile,
                          COALESCE(o.discovery_excluded,0) AS discovery_excluded
                   FROM calibrated_edges c
                   JOIN pair_scores p ON p.pair_id=c.pair_id
                   JOIN datasets da ON da.current_version_id=p.version_a
                   JOIN datasets db ON db.current_version_id=p.version_b
                   LEFT JOIN overlap_evidence o ON o.overlap_id=(
                     SELECT latest.overlap_id FROM overlap_evidence latest
                     WHERE latest.version_a=p.version_a AND latest.version_b=p.version_b
                     ORDER BY latest.created_at DESC,latest.overlap_id DESC LIMIT 1)
                   WHERE c.calibration_id=? AND c.q_value<=?
                   ORDER BY c.q_value,p.cskl,p.pair_id""",
                (calibration_id, q_max),
            ).fetchall()
    node_ids = [row["version_id"] for row in node_rows]
    if not node_ids:
        raise ValueError("Calibration release contains no eligible dataset endpoints.")
    node_id_set = set(node_ids)
    if any(
        edge["version_a"] not in node_id_set or edge["version_b"] not in node_id_set
        for edge in edges
    ):
        raise ValueError("Calibration edge endpoints differ from the graph member family.")
    edge_rows = [row for row in edges if not independent_only or not row["discovery_excluded"]]

    adjacency: dict[str, list[Any]] = defaultdict(list)
    for edge in edge_rows:
        adjacency[edge["version_a"]].append(edge)
        adjacency[edge["version_b"]].append(edge)
    selected_pair_ids: set[str] = set()
    for node_id in node_ids:
        ranked = sorted(
            adjacency[node_id],
            key=lambda row: (
                -(row["cskl_similarity_percentile"] or 0.0),
                row["q_value"],
                row["cskl"],
                row["pair_id"],
            ),
        )
        selected_pair_ids.update(row["pair_id"] for row in ranked[:top_k_per_node])
    selected = [row for row in edge_rows if row["pair_id"] in selected_pair_ids]

    index = {node_id: position for position, node_id in enumerate(node_ids)}
    graph = ig.Graph(
        n=len(node_ids),
        edges=[(index[row["version_a"]], index[row["version_b"]]) for row in selected],
        directed=False,
    )
    weights = [max(float(row["cskl_similarity_percentile"] or 0.0), 1e-9) for row in selected]
    memberships: list[list[int]] = []
    modularities: list[float] = []
    for run in range(stability_runs):
        ig.set_random_number_generator(random.Random(seed + run))
        if graph.ecount():
            clustering = graph.community_leiden(
                objective_function="modularity",
                weights=weights,
                resolution=resolution,
                n_iterations=-1,
            )
            memberships.append(list(clustering.membership))
            modularities.append(float(clustering.modularity))
        else:
            memberships.append(list(range(len(node_ids))))
            modularities.append(0.0)
    best_index = max(range(len(memberships)), key=lambda value: (modularities[value], -value))
    membership = memberships[best_index]
    stability_values = [
        float(ig.compare_communities(membership, candidate, method="nmi"))
        for candidate in memberships
    ]
    communities = _canonical_membership(node_ids, membership)

    ig.set_random_number_generator(random.Random(seed))
    if graph.vcount() == 1:
        coordinates = [(0.5, 0.5)]
    elif graph.ecount():
        layout = graph.layout_fruchterman_reingold(weights=weights, niter=max(500, graph.vcount() * 5))
        coordinates = _normalise_layout([(float(row[0]), float(row[1])) for row in layout])
    else:
        layout = graph.layout_circle()
        coordinates = _normalise_layout([(float(row[0]), float(row[1])) for row in layout])
    coordinates, layout_quality = _collision_aware_layout(coordinates, node_ids)

    policy = {
        "q_max": q_max,
        "independent_only": independent_only,
        "top_k_per_node": top_k_per_node,
        "top_k_policy": "union",
        "community_algorithm": "igraph.community_leiden",
        "objective": "modularity",
        "resolution": resolution,
        "seed": seed,
        "stability_runs": stability_runs,
        "layout_algorithm": LAYOUT_ALGORITHM,
        "endpoint_scope": endpoint_scope,
        "layout_aspect_ratio": LAYOUT_ASPECT_RATIO,
        "layout_minimum_separation": "min(0.09, 1/sqrt(node_count))",
    }
    policy_hash = hashlib.sha256(canonical_json(policy).encode()).hexdigest()
    layout_version = f"igraph-{ig.__version__}:fr-collision-v2:{policy_hash[:16]}"
    manifest = {
        "schema_version": 2,
        "calibration_id": calibration_id,
        "stratum": release["stratum"],
        "policy": policy,
        "policy_hash": policy_hash,
        "layout_version": layout_version,
        "node_count": len(node_ids),
        "eligible_edge_count": len(edge_rows),
        "sparsified_edge_count": len(selected),
        "community_count": len(set(communities)),
        "mean_membership_nmi": mean(stability_values),
        "run_modularities": modularities,
        "layout_quality": layout_quality,
        "text_release_id": text_release_id,
        "members": [
            {"version_id": node_id, "x": x, "y": y, "community": community}
            for node_id, (x, y), community in zip(node_ids, coordinates, communities, strict=True)
        ],
    }
    encoded = canonical_json(manifest).encode("utf-8")
    checksum = hashlib.sha256(encoded).hexdigest()
    directory = Path(manifest_directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / f"graph-manifest-{checksum}.json"
    if not manifest_path.exists():
        temporary = manifest_path.with_suffix(".json.tmp")
        with temporary.open("wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, manifest_path)
    catalog.record_artifact(
        artifact_id=hashlib.sha256(f"graph_manifest:{checksum}".encode()).hexdigest(),
        kind="graph_manifest",
        uri=str(manifest_path),
        checksum=checksum,
        dependency_hash=hashlib.sha256(
            f"{calibration_id}:{policy_hash}:{text_release_id or ''}".encode()
        ).hexdigest(),
        manifest={
            "calibration_id": calibration_id,
            "layout_version": layout_version,
            "text_release_id": text_release_id,
        },
    )
    snapshot_id = catalog.stage_snapshot(
        calibration_id=calibration_id,
        stratum=release["stratum"],
        policy_hash=policy_hash,
        layout_version=layout_version,
        manifest_uri=str(manifest_path),
        manifest_checksum=checksum,
        text_release_id=text_release_id,
        datasets=[
            (node_id, x, y, community)
            for node_id, (x, y), community in zip(node_ids, coordinates, communities, strict=True)
        ],
        pair_ids=sorted(selected_pair_ids),
    )
    return {
        "snapshot_id": snapshot_id,
        "manifest_uri": str(manifest_path),
        "manifest_checksum": checksum,
        "validation": catalog.validate_snapshot(snapshot_id),
        "manifest": manifest,
    }
