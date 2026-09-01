"""Strict, snapshot-bound discovery query compiler.

The DSL deliberately supports only named biological graph fields. It never
accepts SQL fragments, identifiers, functions, or user-selected table names.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .catalog import Catalog, canonical_json, stable_id
from .ontology_validation import OLS_LABEL_RESOLVER_VERSION


class QueryContractError(ValueError):
    """The query is malformed, over-budget, or uses an unknown predicate."""


class UnsupportedQueryError(QueryContractError):
    """The predicate is known but its evidence release is unavailable."""


_NUMERIC_FIELDS = {
    "edge.q_value": "c.q_value",
    "edge.p_value": "c.p_value",
    "edge.cskl": "p.cskl",
    "edge.cskl_percentile": "c.cskl_similarity_percentile",
    "edge.specter2_percentile": "t.similarity_percentile",
    "edge.overlap_coefficient": "COALESCE(o.overlap_coefficient,0)",
    "edge.shared_count": "COALESCE(o.shared_count,0)",
}
_NODE_ANNOTATION_FIELDS = {
    "node.tissue": "tissue",
    "node.disease": "disease",
    "node.organism": "organism",
}
_KNOWN_UNAVAILABLE = {
    "explanation.feature_or_pathway": "Gene/pathway explanation artifacts are not present in this snapshot.",
    "snapshot.diff": (
        "Snapshot comparison is not an edge predicate; use the bounded "
        "/v1/snapshots/diff endpoint."
    ),
}
_COMPARISONS = {"eq": "=", "gt": ">", "gte": ">=", "lt": "<", "lte": "<="}


@dataclass(slots=True)
class _Compiler:
    parameters: list[Any]
    leaf_count: int = 0
    uses_specter2: bool = False
    uses_independence: bool = False

    def compile(self, value: Any, *, depth: int = 0) -> str:
        if depth > 12:
            raise QueryContractError("Query nesting is too deep (maximum 12).")
        if not isinstance(value, Mapping) or len(value) != 1:
            raise QueryContractError("Every query expression must be a one-key object.")
        key, operand = next(iter(value.items()))
        if key in {"and", "or"}:
            if not isinstance(operand, Sequence) or isinstance(operand, (str, bytes)):
                raise QueryContractError(f"{key} requires an array of expressions.")
            if not 1 <= len(operand) <= 32:
                raise QueryContractError(f"{key} requires between 1 and 32 expressions.")
            compiled = [self.compile(child, depth=depth + 1) for child in operand]
            joiner = f" {key.upper()} "
            return "(" + joiner.join(compiled) + ")"
        if key == "not":
            return f"(NOT {self.compile(operand, depth=depth + 1)})"

        self.leaf_count += 1
        if self.leaf_count > 64:
            raise QueryContractError("Query exceeds the 64-predicate cost limit.")
        if key in _KNOWN_UNAVAILABLE:
            raise UnsupportedQueryError(_KNOWN_UNAVAILABLE[key])
        if not isinstance(operand, Mapping) or len(operand) != 1:
            raise QueryContractError(f"Predicate {key!r} must contain exactly one operator.")
        operator, expected = next(iter(operand.items()))

        if key in _NUMERIC_FIELDS:
            if operator not in _COMPARISONS:
                raise QueryContractError(f"Numeric predicate {key!r} does not support {operator!r}.")
            if isinstance(expected, bool) or not isinstance(expected, (int, float)) or not math.isfinite(expected):
                raise QueryContractError(f"Numeric predicate {key!r} requires a finite number.")
            if key in {"edge.q_value", "edge.p_value", "edge.cskl_percentile", "edge.specter2_percentile", "edge.overlap_coefficient"} and not 0 <= expected <= 1:
                raise QueryContractError(f"{key!r} must be compared with a value in [0, 1].")
            if key == "edge.specter2_percentile":
                self.uses_specter2 = True
            self.parameters.append(float(expected))
            return f"({_NUMERIC_FIELDS[key]} {_COMPARISONS[operator]} ?)"

        if key == "edge.independent":
            if operator != "eq" or not isinstance(expected, bool):
                raise QueryContractError("edge.independent supports only {eq: boolean}.")
            self.uses_independence = True
            independent = """(
                (o.overlap_id IS NOT NULL AND o.discovery_excluded=0)
                OR EXISTS (
                  SELECT 1 FROM calibrated_edges independent
                  WHERE independent.pair_id=p.pair_id
                    AND independent.calibration_id=s.independent_calibration_id
                )
            )"""
            return independent if expected else f"(NOT {independent})"

        if key == "edge.overlap_classification":
            if operator != "eq" or expected not in {"none", "minor", "major", "exact"}:
                raise QueryContractError("edge.overlap_classification requires {eq: none|minor|major|exact}.")
            self.parameters.append(expected)
            return "(COALESCE(o.classification,'none')=?)"

        if key in _NODE_ANNOTATION_FIELDS:
            return self._paired_annotation_predicate(
                key, _NODE_ANNOTATION_FIELDS[key], operator, expected
            )

        if key == "node.platform":
            return self._paired_text_predicate(key, operator, expected, "da.platform", "db.platform")

        if key == "node.sample_count":
            if operator not in _COMPARISONS or isinstance(expected, bool) or not isinstance(expected, int) or expected < 0:
                raise QueryContractError("node.sample_count requires a non-negative integer comparison.")
            self.parameters.extend([expected, expected])
            symbol = _COMPARISONS[operator]
            return f"((va.sample_count {symbol} ?) OR (vb.sample_count {symbol} ?))"

        if key == "topology.community":
            return self._paired_text_predicate(
                key, operator, expected, "COALESCE(ga.community,'unknown')", "COALESCE(gb.community,'unknown')"
            )

        raise QueryContractError(f"Unsupported query field: {key}")

    def _paired_text_predicate(
        self,
        field: str,
        operator: str,
        expected: Any,
        left: str,
        right: str,
    ) -> str:
        if operator in {"same", "different"}:
            if expected is not True:
                raise QueryContractError(f"{field}.{operator} requires the literal true.")
            symbol = "=" if operator == "same" else "<>"
            return f"({left} {symbol} {right})"
        if operator == "eq" and isinstance(expected, str) and expected.strip():
            self.parameters.extend([expected.strip(), expected.strip()])
            return f"(({left}=?) OR ({right}=?))"
        raise QueryContractError(f"{field} supports eq:string, same:true, or different:true.")

    def _paired_annotation_predicate(
        self,
        field: str,
        annotation_field: str,
        operator: str,
        expected: Any,
    ) -> str:
        def eligible(alias: str) -> str:
            return f"""{alias}.field='{annotation_field}' AND (
                {alias}.review_state='accepted' OR (
                  {alias}.review_state='unreviewed'
                  AND {alias}.extractor_version='{OLS_LABEL_RESOLVER_VERSION}'
                  AND NOT EXISTS (
                    SELECT 1 FROM annotation_assertions accepted
                    WHERE accepted.version_id={alias}.version_id
                      AND accepted.field={alias}.field
                      AND accepted.review_state='accepted'
                  )
                )
            )"""

        left_exists = (
            "EXISTS (SELECT 1 FROM annotation_assertions left_value "
            f"WHERE left_value.version_id=p.version_a AND {eligible('left_value')})"
        )
        right_exists = (
            "EXISTS (SELECT 1 FROM annotation_assertions right_value "
            f"WHERE right_value.version_id=p.version_b AND {eligible('right_value')})"
        )
        shared = f"""EXISTS (
            SELECT 1 FROM annotation_assertions left_value
            JOIN annotation_assertions right_value
              ON right_value.version_id=p.version_b
             AND COALESCE(NULLIF(right_value.ontology_id,''),LOWER(right_value.value))=
                 COALESCE(NULLIF(left_value.ontology_id,''),LOWER(left_value.value))
            WHERE left_value.version_id=p.version_a
              AND {eligible('left_value')} AND {eligible('right_value')}
        )"""
        if operator in {"same", "different"}:
            if expected is not True:
                raise QueryContractError(f"{field}.{operator} requires the literal true.")
            return (
                f"({left_exists} AND {right_exists} AND {shared})"
                if operator == "same"
                else f"({left_exists} AND {right_exists} AND NOT {shared})"
            )
        if operator == "eq" and isinstance(expected, str) and expected.strip():
            self.parameters.extend([expected.strip()] * 4)
            return f"""(
              EXISTS (SELECT 1 FROM annotation_assertions left_value
                      WHERE left_value.version_id=p.version_a AND {eligible('left_value')}
                        AND (LOWER(left_value.value)=LOWER(?) OR LOWER(COALESCE(left_value.ontology_id,''))=LOWER(?)))
              OR EXISTS (SELECT 1 FROM annotation_assertions right_value
                         WHERE right_value.version_id=p.version_b AND {eligible('right_value')}
                           AND (LOWER(right_value.value)=LOWER(?) OR LOWER(COALESCE(right_value.ontology_id,''))=LOWER(?)))
            )"""
        raise QueryContractError(f"{field} supports eq:string, same:true, or different:true.")


def validate_query_ast(query: Mapping[str, Any]) -> dict[str, Any]:
    serialized = canonical_json(query)
    if len(serialized.encode("utf-8")) > 20_000:
        raise QueryContractError("Query is too large (maximum 20 KB).")
    compiler = _Compiler(parameters=[])
    sql = compiler.compile(query)
    return {
        "canonical": serialized,
        "predicate_sql": sql,
        "parameters": compiler.parameters,
        "uses_specter2": compiler.uses_specter2,
        "uses_independence": compiler.uses_independence,
        "predicate_count": compiler.leaf_count,
    }


def execute_query(
    catalog: Catalog,
    *,
    snapshot_id: str,
    query: Mapping[str, Any],
    label: str | None = None,
    limit: int = 500,
    offset: int = 0,
    scope: str = "published_graph",
) -> dict[str, Any]:
    if not 1 <= int(limit) <= 5_000:
        raise QueryContractError("limit must be between 1 and 5000.")
    if not 0 <= int(offset) <= 100_000:
        raise QueryContractError("offset must be between 0 and 100000.")
    if scope not in {"published_graph", "complete_pair_family"}:
        raise QueryContractError("scope must be published_graph or complete_pair_family.")
    compiled = validate_query_ast(query)
    with catalog.reader() as connection:
        snapshot = connection.execute(
            "SELECT * FROM graph_snapshots WHERE snapshot_id=?", (snapshot_id,)
        ).fetchone()
        if not snapshot or snapshot["published_at"] is None:
            raise KeyError(snapshot_id)
        if compiled["uses_specter2"] and not snapshot["text_release_id"]:
            raise UnsupportedQueryError(
                "This snapshot has no finalized SPECTER2 release; the predicate cannot be executed."
            )
        if compiled["uses_independence"]:
            edge_count, overlap_count = connection.execute(
                """SELECT COUNT(*),COUNT(overlap_id) FROM graph_snapshot_edges
                   WHERE snapshot_id=?""",
                (snapshot_id,),
            ).fetchone()
            if overlap_count != edge_count and not snapshot["independent_calibration_id"]:
                raise UnsupportedQueryError(
                    "Independence cannot be inferred: this snapshot has neither complete "
                    "overlap bindings nor a bound independent calibration family."
                )
        parameters = [snapshot["text_release_id"], snapshot["calibration_id"]]
        parameters.extend(compiled["parameters"])
        parameters.extend([int(limit) + 1, int(offset)])
        if scope == "published_graph":
            source = """graph_snapshot_edges se
                JOIN graph_snapshots s ON s.snapshot_id=se.snapshot_id
                JOIN calibrated_edges c ON c.calibration_id=s.calibration_id AND c.pair_id=se.pair_id"""
            scope_clause = "se.snapshot_id=?"
            parameters.insert(2, snapshot_id)
            overlap_join = "LEFT JOIN overlap_evidence o ON o.overlap_id=se.overlap_id"
            pair_reference = "se.pair_id"
            snapshot_reference = "se.snapshot_id"
        else:
            source = """graph_snapshots s
                JOIN calibrated_edges c ON c.calibration_id=s.calibration_id"""
            scope_clause = "s.snapshot_id=?"
            parameters.insert(2, snapshot_id)
            overlap_join = """LEFT JOIN overlap_evidence o ON o.overlap_id=(
                SELECT latest.overlap_id FROM overlap_evidence latest
                WHERE latest.version_a=p.version_a AND latest.version_b=p.version_b
                ORDER BY latest.created_at DESC,latest.overlap_id DESC LIMIT 1)"""
            pair_reference = "c.pair_id"
            snapshot_reference = "s.snapshot_id"
        rows = connection.execute(
            f"""SELECT p.pair_id,p.version_a,p.version_b,p.algorithm_hash,p.cskl,
                       c.p_value,c.q_value,c.cskl_similarity_percentile,
                       o.shared_count,o.fraction_a,o.fraction_b,o.jaccard,o.overlap_coefficient,
                       COALESCE(o.classification,'none') AS overlap_classification,
                       COALESCE(o.discovery_excluded,0) AS discovery_excluded,
                       t.cosine_similarity AS specter2_cosine,
                       t.similarity_percentile AS specter2_percentile,
                       da.dataset_uid AS dataset_a_uid,da.accession AS accession_a,
                       db.dataset_uid AS dataset_b_uid,db.accession AS accession_b,
                       ga.community AS community_a,gb.community AS community_b
                FROM {source}
                JOIN pair_scores p ON p.pair_id={pair_reference}
                JOIN dataset_versions va ON va.version_id=p.version_a
                JOIN dataset_versions vb ON vb.version_id=p.version_b
                JOIN datasets da ON da.dataset_uid=va.dataset_uid
                JOIN datasets db ON db.dataset_uid=vb.dataset_uid
                JOIN graph_snapshot_datasets ga ON ga.snapshot_id={snapshot_reference} AND ga.version_id=p.version_a
                JOIN graph_snapshot_datasets gb ON gb.snapshot_id={snapshot_reference} AND gb.version_id=p.version_b
                {overlap_join}
                LEFT JOIN text_pair_scores t
                  ON t.text_release_id=? AND t.version_a=p.version_a AND t.version_b=p.version_b
                WHERE c.calibration_id=? AND {scope_clause} AND {compiled['predicate_sql']}
                ORDER BY c.q_value ASC,p.cskl ASC,p.pair_id ASC LIMIT ? OFFSET ?""",
            parameters,
        ).fetchall()
        assertion_cursor = connection.execute(
            """SELECT a.assertion_id FROM annotation_assertions a
               JOIN graph_snapshot_datasets g ON g.version_id=a.version_id
               WHERE g.snapshot_id=? AND (
                 a.review_state='accepted' OR (
                   a.review_state='unreviewed' AND a.extractor_version=?
                 )
               )
               ORDER BY a.assertion_id""",
            (snapshot_id, OLS_LABEL_RESOLVER_VERSION),
        )
        annotation_hash = hashlib.sha256(
            "".join(f"{row['assertion_id']}\0" for row in assertion_cursor).encode()
        ).hexdigest()
    has_more = len(rows) > int(limit)
    rows = rows[: int(limit)]
    query_id = stable_id("query", snapshot_id, scope, compiled["canonical"])
    return {
        "query_id": query_id,
        "snapshot_id": snapshot_id,
        "label": label,
        "scope": scope,
        "ast": json.loads(compiled["canonical"]),
        "predicate_count": compiled["predicate_count"],
        "offset": int(offset),
        "limit": int(limit),
        "has_more": has_more,
        "edges": [dict(row) for row in rows],
        "provenance": {
            "calibration_id": snapshot["calibration_id"],
            "independent_calibration_id": snapshot["independent_calibration_id"],
            "text_release_id": snapshot["text_release_id"],
            "policy_hash": snapshot["policy_hash"],
            "annotation_policy": "accepted_or_exact_ols_resolved_candidate",
            "annotation_assertion_hash": annotation_hash,
            "annotation_snapshot_bound": False,
        },
        "warnings": [
            "Annotation assertions are hashed at query time but are not yet frozen into the graph snapshot."
        ],
    }
