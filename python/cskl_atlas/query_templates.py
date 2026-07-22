from __future__ import annotations

from typing import Any

QUERY_TEMPLATES: list[dict[str, Any]] = [
    {
        "id": "unexpected-analogues",
        "title": "Unexpected molecular analogues",
        "question": "Which independent datasets are molecularly close despite different disease or tissue labels?",
        "execution": "available",
        "ast": {
            "and": [
                {"edge.q_value": {"lte": 0.05}},
                {"edge.independent": {"eq": True}},
                {"edge.cskl_percentile": {"gte": 0.95}},
                {"or": [{"node.tissue": {"different": True}}, {"node.disease": {"different": True}}]},
            ]
        },
    },
    {
        "id": "replication-candidates",
        "title": "Independent replication candidates",
        "question": "Which studies share phenotype, molecular structure, and topic without sharing samples?",
        "execution": "requires_text_release",
        "ast": {
            "and": [
                {"edge.q_value": {"lte": 0.05}},
                {"edge.independent": {"eq": True}},
                {"node.disease": {"same": True}},
                {"edge.specter2_percentile": {"gte": 0.9}},
            ]
        },
    },
    {
        "id": "molecular-only",
        "title": "Molecular-only signal",
        "question": "Where does C-SKL find a strong relationship that scientific text does not suggest?",
        "execution": "requires_text_release",
        "ast": {
            "and": [
                {"edge.q_value": {"lte": 0.05}},
                {"edge.independent": {"eq": True}},
                {"edge.cskl_percentile": {"gte": 0.95}},
                {"edge.specter2_percentile": {"lte": 0.5}},
            ]
        },
    },
    {
        "id": "text-molecular-discordance",
        "title": "Textual concordance, molecular discordance",
        "question": "Which topically similar studies have weak or non-significant molecular similarity?",
        "execution": "requires_text_release",
        "scope": "complete_pair_family",
        "ast": {
            "and": [
                {"edge.specter2_percentile": {"gte": 0.9}},
                {"edge.cskl_percentile": {"lte": 0.5}},
                {"edge.independent": {"eq": True}},
            ]
        },
    },
    {
        "id": "sample-overlap-audit",
        "title": "Sample and duplicate audit",
        "question": "Which links are partly or fully explained by shared samples?",
        "execution": "available",
        "ast": {"edge.overlap_coefficient": {"gt": 0}},
    },
    {
        "id": "mechanism-search",
        "title": "Gene or pathway mechanism search",
        "question": "Which independent edges or communities are explained by a named gene or Reactome pathway?",
        "execution": "planned",
        "unsupported_reason": "Versioned gene/pathway explainer artifacts are not available yet.",
        "ast": {
            "and": [
                {"edge.independent": {"eq": True}},
                {"explanation.feature_or_pathway": {"parameter": "query"}},
            ]
        },
    },
    {
        "id": "release-changes",
        "title": "What changed?",
        "question": "Which nodes or calibrated edges appeared, disappeared, or materially changed since another snapshot?",
        "execution": "dedicated_endpoint",
        "endpoint": "GET /v1/snapshots/diff",
        "ast": {"snapshot.diff": {"parameter": "baseline_snapshot_id"}},
    },
]


def get_query_templates() -> list[dict[str, Any]]:
    return QUERY_TEMPLATES
