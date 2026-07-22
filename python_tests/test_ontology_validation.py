import json

import httpx
from cskl_atlas.ontology_validation import (
    OLSLabelResolver,
    audit_annotation_candidates,
    build_ols_candidate_index,
)


def _candidate(path, *, label: str, curie: str) -> None:
    path.write_text(
        json.dumps(
            {
                "annotations": {
                    "tissue": {
                        "unknown": False,
                        "values": [
                            {
                                "ontology": "UBERON",
                                "ontology_id": curie,
                                "label": label,
                            }
                        ],
                    }
                }
            }
        ),
        encoding="utf-8",
    )


def test_frozen_ols_audit_rejects_semantically_wrong_curie(tmp_path) -> None:
    candidates = tmp_path / "candidates"
    candidates.mkdir()
    _candidate(candidates / "GSE1.json", label="breast", curie="UBERON:0000314")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["obo_id"] == "UBERON:0000314"
        return httpx.Response(
            200,
            request=request,
            json={
                "_embedded": {
                    "terms": [
                        {
                            "obo_id": "UBERON:0000314",
                            "label": "cecum mucosa",
                            "synonyms": ["caecum mucosa"],
                            "is_obsolete": False,
                            "iri": "http://purl.obolibrary.org/obo/UBERON_0000314",
                        }
                    ]
                }
            },
        )

    manifest = build_ols_candidate_index(
        annotation_directory=candidates,
        output_directory=tmp_path / "ontology",
        workers=1,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    report = audit_annotation_candidates(
        annotation_directory=candidates,
        ontology_index=manifest["index_path"],
        output_path=tmp_path / "audit.json",
    )
    assert report["paper_gate"] == "fail"
    assert report["results"][0]["status"] == "label_mismatch"


def test_frozen_ols_audit_accepts_exact_synonym(tmp_path) -> None:
    candidates = tmp_path / "candidates"
    candidates.mkdir()
    _candidate(candidates / "GSE2.json", label="mammary region", curie="UBERON:0000310")

    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                request=request,
                json={
                    "_embedded": {
                        "terms": [
                            {
                                "obo_id": "UBERON:0000310",
                                "label": "breast",
                                "synonyms": ["mammary region"],
                                "is_obsolete": False,
                            }
                        ]
                    }
                },
            )
        )
    )
    manifest = build_ols_candidate_index(
        annotation_directory=candidates,
        output_directory=tmp_path / "ontology",
        workers=1,
        http_client=client,
    )
    report = audit_annotation_candidates(
        annotation_directory=candidates,
        ontology_index=manifest["index_path"],
        output_path=tmp_path / "audit.json",
    )
    assert report["paper_gate"] == "pass"
    assert report["results"][0]["status"] == "canonical_or_synonym"


def test_label_resolver_accepts_unique_exact_match_and_replays_frozen_cache(tmp_path) -> None:
    requests = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        assert request.url.path.endswith("/search")
        assert request.url.params["ontology"] == "uberon"
        assert request.url.params["exact"] == "true"
        return httpx.Response(
            200,
            request=request,
            json={
                "response": {
                    "docs": [
                        {
                            "ontology_prefix": "UBERON",
                            "obo_id": "UBERON:0000310",
                            "label": "breast",
                            "synonym": ["mammary region"],
                            "is_obsolete": False,
                        }
                    ]
                }
            },
        )

    cache = tmp_path / "resolver"
    client = httpx.Client(transport=httpx.MockTransport(handler))
    resolver = OLSLabelResolver(cache, http_client=client)
    decision = resolver.resolve(label="Mammary region", allowed_ontologies=("UBERON",))
    assert decision.status == "resolved"
    assert decision.curie == "UBERON:0000310"
    assert decision.match_kind == "synonym"
    assert requests == 1

    def unexpected_request(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"frozen resolver unexpectedly requested {request.url}")

    replay = OLSLabelResolver(
        cache,
        http_client=httpx.Client(transport=httpx.MockTransport(unexpected_request)),
    ).resolve(label="mammary region", allowed_ontologies=("UBERON",))
    assert replay.status == "resolved"
    assert replay.curie == decision.curie
    assert replay.query_response_sha256 == decision.query_response_sha256
    assert requests == 1


def test_label_resolver_fails_closed_for_ambiguous_and_unresolved_labels(tmp_path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        label = request.url.params["q"]
        documents = (
            [
                {
                    "ontology_prefix": "EFO",
                    "obo_id": "EFO:1",
                    "label": "culture",
                    "is_obsolete": False,
                },
                {
                    "ontology_prefix": "EFO",
                    "obo_id": "EFO:2",
                    "label": "culture",
                    "is_obsolete": False,
                },
            ]
            if label == "culture"
            else []
        )
        return httpx.Response(200, request=request, json={"response": {"docs": documents}})

    resolver = OLSLabelResolver(
        tmp_path / "resolver",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    ambiguous = resolver.resolve(label="culture", allowed_ontologies=("EFO",))
    unresolved = resolver.resolve(label="not a concept", allowed_ontologies=("EFO",))
    assert ambiguous.status == "ambiguous"
    assert ambiguous.candidate_count == 2
    assert ambiguous.curie is None
    assert unresolved.status == "unresolved"
    assert unresolved.curie is None
