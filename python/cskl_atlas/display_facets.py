"""Deterministic, non-evidentiary facets used to organize the Atlas UI.

The ontology assertions remain the scientific record.  These deliberately
coarser labels are presentation/query facets: they keep synonymous specimen
wording from producing hundreds of tiny graph regions while preserving the
original tissue candidate for inspection and curation.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

TISSUE_SYSTEM_VERSION = "atlas-tissue-system-v1"

_UNKNOWN = "Mixed / unspecified"
_OTHER = "Other anatomy"

# Order is intentional.  For example, bone marrow is haematological rather
# than musculoskeletal, and an explicitly named cell line is an in-vitro
# source even when its lineage is recognizable.
_SYSTEM_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "Cell culture / in vitro",
        (
            r"\bcell lines?\b",
            r"\bcell culture\b",
            r"\bcultured cells?\b",
            r"\blymphoblastoid\b",
            r"\borganoid\b",
        ),
    ),
    (
        "Blood & immune",
        (
            r"\bblood\b",
            r"\bbone marrow\b",
            r"\bmarrow\b",
            r"\blymph(?: node|oid)?\b",
            r"\bspleen\b",
            r"\bthym(?:us|ic)\b",
            r"\btonsil\b",
            r"\bleuk(?:ocyte|emia|aemia)\b",
            r"\bmononuclear\b",
            r"\bpbmc\b",
            r"\bmonocyte\b",
            r"\bmacrophage\b",
            r"\bneutrophil\b",
            r"\bplasma cells?\b",
            r"\bb cells?\b",
            r"\bt cells?\b",
            r"\blymphocyte\b",
            r"\blymphoblast\b",
            r"\blymphoma\b",
            r"\bpmn\b",
            r"\bhematopoietic\b",
            r"\bhaematopoietic\b",
        ),
    ),
    (
        "Respiratory",
        (
            r"\blung\b",
            r"\bairway\b",
            r"\bbronch",
            r"\balveol",
            r"\bpulmon",
            r"\btrache",
            r"\bpleur",
            r"\bnasal\b",
            r"\bsputum\b",
            r"\bnsclc\b",
        ),
    ),
    (
        "Digestive & hepatobiliary",
        (
            r"\bcolon(?:ic)?\b",
            r"\bcolorectal\b",
            r"\brect(?:um|al)\b",
            r"\bintestin",
            r"\bbowel\b",
            r"\bileum\b",
            r"\bjejunu",
            r"\bduoden",
            r"\bstomach\b",
            r"\bgastr",
            r"\besophag",
            r"\boesophag",
            r"\bliver\b",
            r"\bhepat",
            r"\bbile\b",
            r"\bbiliary\b",
            r"\bgallbladder\b",
            r"\bpancrea",
            r"\boral\b",
            r"\bgingiv",
            r"\bsalivary\b",
            r"\bsaliva\b",
        ),
    ),
    (
        "Nervous system",
        (
            r"\bbrain\b",
            r"\bcerebr",
            r"\bcortex\b",
            r"\bcortical\b",
            r"\bcerebell",
            r"\bhippocamp",
            r"\bspinal\b",
            r"\bneural\b",
            r"\bneuron",
            r"\bglia",
            r"\bneuroblast",
            r"\bglioblast",
            r"\bcentral nervous system\b",
        ),
    ),
    (
        "Skin",
        (
            r"\bskin\b",
            r"\bderm",
            r"\bepiderm",
            r"\bkeratinocyte\b",
            r"\bmelanoma\b",
        ),
    ),
    ("Breast", (r"\bbreast\b", r"\bmammary\b")),
    (
        "Reproductive",
        (
            r"\bovar",
            r"\buterus\b",
            r"\buterine\b",
            r"\bendometri",
            r"\bcervix\b",
            r"\bectocervix\b",
            r"\bvagin",
            r"\bvulva",
            r"\bplacent",
            r"\bprostate\b",
            r"\btest(?:is|es|icular)\b",
            r"\bsemin",
        ),
    ),
    (
        "Kidney & urinary",
        (
            r"\bkidney\b",
            r"\brenal\b",
            r"\bbladder\b",
            r"\burinary\b",
            r"\bureter\b",
            r"\burine\b",
        ),
    ),
    (
        "Musculoskeletal",
        (
            r"\bskeletal muscle\b",
            r"\bmuscle\b",
            r"\bvastus lateralis\b",
            r"\bmyocyte\b",
            r"\bmyoblast\b",
            r"\bbone\b",
            r"\bcartilage\b",
            r"\bchondro",
            r"\btendon\b",
            r"\bsynovi",
            r"\bjoint\b",
        ),
    ),
    (
        "Cardiovascular",
        (
            r"\bheart\b",
            r"\bcardiac\b",
            r"\bmyocard",
            r"\baorta\b",
            r"\barter",
            r"\bvascular\b",
            r"\bvein\b",
            r"\bendotheli",
        ),
    ),
    (
        "Endocrine & metabolic",
        (
            r"\bthyroid\b",
            r"\badrenal\b",
            r"\bpituitar",
            r"\bislets?\b",
            r"\badipose\b",
            r"\badipocyte\b",
        ),
    ),
    (
        "Ocular",
        (
            r"\bretina\b",
            r"\bretinal\b",
            r"\bcornea\b",
            r"\bocular\b",
            r"\beye\b",
        ),
    ),
)

_COMPILED = tuple(
    (system, tuple(re.compile(pattern, re.IGNORECASE) for pattern in patterns))
    for system, patterns in _SYSTEM_PATTERNS
)
_UNINFORMATIVE = re.compile(
    r"^(?:mixed|mixed\s*/\s*unknown|unknown|not reviewed|tissue|organ|anatomical entity|"
    r"tumou?r(?: biopsy| specimen| tissue)?|frozen tumou?r|cancer,?\s*lcm|"
    r"various tissues?|multiple tissues?|"
    r"uberon:[a-z0-9_.-]+(?:\s*/\s*uberon:[a-z0-9_.-]+)*)$",
    re.IGNORECASE,
)


def _classify_segment(value: str) -> str:
    text = " ".join(value.strip().split())
    if not text or _UNINFORMATIVE.fullmatch(text):
        return _UNKNOWN
    for system, patterns in _COMPILED:
        if any(pattern.search(text) for pattern in patterns):
            return system
    return _OTHER


def derive_tissue_system(labels: Iterable[str]) -> str:
    """Return a stable broad display facet without altering source assertions."""

    segments: list[str] = []
    for label in labels:
        # Slash and semicolon are how the export represents multi-valued fields.
        # Commas are retained because they commonly occur inside anatomical names.
        segments.extend(part for part in re.split(r"\s*(?:/|;)\s*", str(label)) if part)
    systems = {_classify_segment(segment) for segment in segments}
    informative = systems - {_UNKNOWN, _OTHER}
    if len(informative) == 1:
        return next(iter(informative))
    if len(informative) > 1:
        return "Mixed anatomical systems"
    if _OTHER in systems:
        return _OTHER
    return _UNKNOWN
