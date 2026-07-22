from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from . import __version__
from .catalog import Catalog
from .query_engine import (
    QueryContractError,
    UnsupportedQueryError,
    execute_query,
    validate_query_ast,
)
from .query_templates import get_query_templates


class RetryResponse(BaseModel):
    job_id: str
    status: str


class QueryValidationRequest(BaseModel):
    snapshot_id: str
    query: dict[str, Any]
    label: str | None = Field(default=None, max_length=200)
    scope: str = Field(default="published_graph", pattern="^(published_graph|complete_pair_family)$")


class QueryExecutionRequest(QueryValidationRequest):
    limit: int = Field(default=500, ge=1, le=5_000)
    offset: int = Field(default=0, ge=0, le=100_000)


class HeartbeatRequest(BaseModel):
    worker_id: str = Field(min_length=1, max_length=200)
    lease_seconds: int = Field(default=300, ge=5, le=86_400)


class RollbackRequest(BaseModel):
    operator: str = Field(min_length=1, max_length=200)
    reason: str = Field(min_length=1, max_length=2_000)


class AnnotationReviewRequest(BaseModel):
    reviewer: str = Field(min_length=1, max_length=300)
    decision: str = Field(pattern="^(accepted|rejected)$")
    note: str = Field(default="", max_length=4_000)


@lru_cache(maxsize=1)
def get_catalog() -> Catalog:
    path = Path(os.getenv("CSKL_ATLAS_CATALOG", "var/catalog/atlas.sqlite"))
    catalog = Catalog(path)
    catalog.initialize()
    return catalog


def require_ops_token(
    authorization: Annotated[str | None, Header(alias="X-Atlas-Ops-Token")] = None,
) -> None:
    configured = os.getenv("CSKL_ATLAS_OPS_TOKEN")
    if not configured:
        raise HTTPException(status_code=503, detail="Operations API is disabled until CSKL_ATLAS_OPS_TOKEN is set.")
    if authorization != configured:
        raise HTTPException(status_code=401, detail="Invalid operations token.")


app = FastAPI(
    title="C-SKL Atlas API",
    version=__version__,
    description=(
        "Versioned graph serving and operations API. Raw C-SKL scores, corpus-dependent "
        "calibration, annotations, overlap evidence, and generated narratives remain separate."
    ),
)

origins = [item.strip() for item in os.getenv("CSKL_ATLAS_ALLOWED_ORIGINS", "").split(",") if item.strip()]
if origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-Atlas-Ops-Token"],
    )


@app.get("/health")
def health(catalog: Annotated[Catalog, Depends(get_catalog)]) -> dict[str, Any]:
    result = catalog.health()
    result.pop("database", None)
    result["service_version"] = __version__
    return result


@app.get("/v1/snapshots/current")
def current_snapshot(
    stratum: str = Query(..., min_length=3, max_length=300),
    catalog: Catalog = Depends(get_catalog),
) -> dict[str, Any]:
    snapshot = catalog.current_snapshot(stratum)
    if not snapshot:
        raise HTTPException(status_code=404, detail="No published snapshot exists for this stratum.")
    return snapshot


@app.get("/v1/snapshots/diff")
def snapshot_diff(
    from_snapshot_id: str = Query(..., min_length=8, max_length=100),
    to_snapshot_id: str = Query(..., min_length=8, max_length=100),
    detail_limit: int = Query(500, ge=1, le=5_000),
    q_change_limit: int = Query(100, ge=1, le=1_000),
    catalog: Catalog = Depends(get_catalog),
) -> dict[str, Any]:
    try:
        return catalog.snapshot_diff(
            from_snapshot_id=from_snapshot_id,
            to_snapshot_id=to_snapshot_id,
            detail_limit=detail_limit,
            q_change_limit=q_change_limit,
        )
    except KeyError as exc:
        detail = exc.args[0] if exc.args else "Published snapshot not found."
        raise HTTPException(status_code=404, detail=detail) from None
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/v1/graph")
def graph(
    snapshot_id: str = Query(..., min_length=8, max_length=100),
    q_max: float = Query(0.05, ge=0, le=1),
    independent_only: bool = Query(True),
    edge_limit: int = Query(50_000, ge=1, le=100_000),
    catalog: Catalog = Depends(get_catalog),
) -> dict[str, Any]:
    try:
        return catalog.graph_payload(
            snapshot_id=snapshot_id,
            q_max=q_max,
            independent_only=independent_only,
            edge_limit=edge_limit,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="Snapshot not found.") from None
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/v1/datasets/{dataset_uid}")
def dataset_detail(
    dataset_uid: str,
    snapshot_id: str | None = Query(default=None, min_length=8, max_length=100),
    catalog: Catalog = Depends(get_catalog),
) -> dict[str, Any]:
    with catalog.reader() as connection:
        if snapshot_id:
            row = connection.execute(
                """SELECT d.*,v.*,g.x,g.y,g.community,? AS snapshot_id
                   FROM datasets d JOIN dataset_versions v ON v.dataset_uid=d.dataset_uid
                   JOIN graph_snapshot_datasets g ON g.version_id=v.version_id
                   JOIN graph_snapshots s ON s.snapshot_id=g.snapshot_id
                   WHERE d.dataset_uid=? AND g.snapshot_id=? AND s.published_at IS NOT NULL""",
                (snapshot_id, dataset_uid, snapshot_id),
            ).fetchone()
        else:
            row = connection.execute(
                """SELECT d.*,v.* FROM datasets d
                   JOIN dataset_versions v ON v.version_id=d.current_version_id
                   WHERE d.dataset_uid=?""",
                (dataset_uid,),
            ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Dataset not found.")
        assertions = connection.execute(
            """SELECT * FROM annotation_assertions WHERE version_id=?
               ORDER BY field,created_at DESC""",
            (row["version_id"],),
        ).fetchall()
        sample_count = connection.execute(
            "SELECT COUNT(*) FROM dataset_samples WHERE version_id=?", (row["version_id"],)
        ).fetchone()[0]
    payload = dict(row)
    payload["metadata"] = json.loads(payload.pop("metadata_json"))
    payload["sample_records"] = int(sample_count)
    payload["annotations"] = [dict(item) for item in assertions]
    return payload


@app.get("/v1/graph/overview")
def graph_overview(
    snapshot_id: str = Query(..., min_length=8, max_length=100),
    q_max: float = Query(0.05, ge=0, le=1),
    independent_only: bool = Query(True),
    catalog: Catalog = Depends(get_catalog),
) -> dict[str, Any]:
    try:
        return catalog.graph_overview(
            snapshot_id=snapshot_id,
            q_max=q_max,
            independent_only=independent_only,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="Published snapshot not found.") from None


@app.get("/v1/graph/neighborhood")
def graph_neighborhood(
    snapshot_id: str = Query(..., min_length=8, max_length=100),
    version_id: str = Query(..., min_length=8, max_length=100),
    q_max: float = Query(0.05, ge=0, le=1),
    independent_only: bool = Query(True),
    limit: int = Query(250, ge=1, le=5_000),
    catalog: Catalog = Depends(get_catalog),
) -> dict[str, Any]:
    try:
        return catalog.graph_neighborhood(
            snapshot_id=snapshot_id,
            version_id=version_id,
            q_max=q_max,
            independent_only=independent_only,
            limit=limit,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="Snapshot or center dataset not found.") from None


@app.get("/v1/edges/{pair_id}")
def edge_detail(
    pair_id: str,
    snapshot_id: str | None = Query(default=None, min_length=8, max_length=100),
    catalog: Catalog = Depends(get_catalog),
) -> dict[str, Any]:
    with catalog.reader() as connection:
        pair = connection.execute(
            "SELECT * FROM pair_scores WHERE pair_id=?", (pair_id,)
        ).fetchone()
        if not pair:
            raise HTTPException(status_code=404, detail="Relationship not found.")
        if snapshot_id:
            bound = connection.execute(
                """SELECT s.*,se.overlap_id FROM graph_snapshot_edges se
                   JOIN graph_snapshots s ON s.snapshot_id=se.snapshot_id
                   WHERE se.snapshot_id=? AND se.pair_id=? AND s.published_at IS NOT NULL""",
                (snapshot_id, pair_id),
            ).fetchone()
            if not bound:
                raise HTTPException(status_code=404, detail="Relationship is not in this snapshot.")
            overlap = connection.execute(
                "SELECT * FROM overlap_evidence WHERE overlap_id=?", (bound["overlap_id"],)
            ).fetchone() if bound["overlap_id"] else None
            calibrations = connection.execute(
                """SELECT e.*,r.mode,r.stratum,r.pool_hash,r.parameter_hash,r.algorithm_hash,
                          r.family_hash,r.status
                   FROM calibrated_edges e JOIN calibration_releases r USING(calibration_id)
                   WHERE e.pair_id=? AND e.calibration_id=?""",
                (pair_id, bound["calibration_id"]),
            ).fetchall()
            semantic = connection.execute(
                """SELECT * FROM text_pair_scores WHERE text_release_id=?
                   AND version_a=? AND version_b=?""",
                (bound["text_release_id"], pair["version_a"], pair["version_b"]),
            ).fetchone() if bound["text_release_id"] else None
        else:
            overlap = connection.execute(
                """SELECT * FROM overlap_evidence WHERE version_a=? AND version_b=?
                   ORDER BY created_at DESC,overlap_id DESC LIMIT 1""",
                (pair["version_a"], pair["version_b"]),
            ).fetchone()
            calibrations = connection.execute(
                """SELECT e.*,r.mode,r.stratum,r.pool_hash,r.parameter_hash,r.algorithm_hash,
                          r.family_hash,r.status
                   FROM calibrated_edges e JOIN calibration_releases r USING(calibration_id)
                   WHERE e.pair_id=? ORDER BY r.created_at DESC""",
                (pair_id,),
            ).fetchall()
            semantic = None
    return {
        "pair_score": dict(pair),
        "overlap": dict(overlap) if overlap else None,
        "calibrations": [dict(row) for row in calibrations],
        "text_similarity": dict(semantic) if semantic else None,
        "snapshot_id": snapshot_id,
        "interpretation": {
            "cskl": "Lower raw distance means more similar standardized covariance structure.",
            "significance": "p/q-values are specific to the named calibration release.",
            "overlap": "Overlap-confounded pairs do not count as independent replication by default.",
        },
    }


@app.get("/v1/edges/{pair_id}/explanation")
def edge_explanation(
    pair_id: str,
    k: int = Query(default=20, ge=1, le=500),
    catalog: Catalog = Depends(get_catalog),
) -> dict[str, Any]:
    """Replay a versioned explainer without allowing a public GET to start work."""

    from .edge_explanation import ExplanationNotCachedError, replay_edge_explanation

    try:
        return replay_edge_explanation(catalog, pair_id=pair_id, k=k)
    except ExplanationNotCachedError:
        raise HTTPException(
            status_code=404,
            detail="No cataloged explanation is available for this relationship and k.",
        ) from None
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post(
    "/v1/edges/{pair_id}/explanation",
    dependencies=[Depends(require_ops_token)],
)
def compute_edge_explanation(
    pair_id: str,
    k: int = Query(default=20, ge=1, le=500),
    catalog: Catalog = Depends(get_catalog),
) -> dict[str, Any]:
    """Compute or replay one explainer through the authenticated operations surface."""

    from .edge_explanation import compute_edge_explanation as compute

    try:
        return compute(
            catalog,
            pair_id=pair_id,
            probes_path=os.getenv(
                "CSKL_ATLAS_PROBES", "resources/releases/gpl570/probes.txt"
            ),
            annotation_path=os.getenv(
                "CSKL_ATLAS_PROBE_ANNOTATION",
                "resources/releases/gpl570/probe-annotation.tsv",
            ),
            reactome_database_path=os.getenv(
                "CSKL_ATLAS_REACTOME_INDEX", "var/reactome/reactome-gpl570.sqlite"
            ),
            cache_directory=os.getenv(
                "CSKL_ATLAS_EXPLANATION_CACHE", "var/explanations"
            ),
            k=k,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="Relationship not found.") from None
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Explanation dependency is not installed: {exc}",
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/v1/query/templates")
def query_templates() -> list[dict[str, Any]]:
    return get_query_templates()


@app.post("/v1/query/validate")
def validate_query(request: QueryValidationRequest) -> dict[str, Any]:
    """Validate and freeze the exact whitelisted AST used by execution."""

    try:
        compiled = validate_query_ast(request.query)
    except UnsupportedQueryError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except QueryContractError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    from .catalog import stable_id

    return {
        "query_id": stable_id("query", request.snapshot_id, request.scope, compiled["canonical"]),
        "snapshot_id": request.snapshot_id,
        "label": request.label,
        "scope": request.scope,
        "ast": json.loads(compiled["canonical"]),
        "predicate_count": compiled["predicate_count"],
        "status": "valid",
    }


@app.post("/v1/query/execute")
def run_query(
    request: QueryExecutionRequest,
    catalog: Catalog = Depends(get_catalog),
) -> dict[str, Any]:
    try:
        return execute_query(
            catalog,
            snapshot_id=request.snapshot_id,
            query=request.query,
            label=request.label,
            limit=request.limit,
            offset=request.offset,
            scope=request.scope,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="Published snapshot not found.") from None
    except UnsupportedQueryError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except QueryContractError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/v1/ops/jobs", dependencies=[Depends(require_ops_token)])
def list_jobs(
    status: str | None = Query(default=None),
    limit: int = Query(200, ge=1, le=1_000),
    catalog: Catalog = Depends(get_catalog),
) -> list[dict[str, Any]]:
    return catalog.list_jobs(status=status, limit=limit)


@app.post(
    "/v1/ops/jobs/{job_id}/retry",
    response_model=RetryResponse,
    dependencies=[Depends(require_ops_token)],
)
def retry_job(job_id: str, catalog: Catalog = Depends(get_catalog)) -> RetryResponse:
    try:
        catalog.requeue_job(job_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return RetryResponse(job_id=job_id, status="queued")


@app.post("/v1/ops/jobs/{job_id}/heartbeat", dependencies=[Depends(require_ops_token)])
def heartbeat_job(
    job_id: str,
    request: HeartbeatRequest,
    catalog: Catalog = Depends(get_catalog),
) -> dict[str, Any]:
    try:
        expires_at = catalog.heartbeat_job(
            job_id, worker_id=request.worker_id, lease_seconds=request.lease_seconds
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"job_id": job_id, "status": "running", "lease_expires_at": expires_at}


@app.post("/v1/ops/jobs/reap", dependencies=[Depends(require_ops_token)])
def reap_jobs(catalog: Catalog = Depends(get_catalog)) -> dict[str, int]:
    return catalog.reap_expired_jobs()


@app.post("/v1/ops/snapshots/{snapshot_id}/rollback", dependencies=[Depends(require_ops_token)])
def rollback_snapshot(
    snapshot_id: str,
    request: RollbackRequest,
    catalog: Catalog = Depends(get_catalog),
) -> dict[str, str]:
    try:
        catalog.rollback_snapshot(
            snapshot_id, operator=request.operator, reason=request.reason
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"snapshot_id": snapshot_id, "status": "published"}


@app.get("/v1/ops/annotations", dependencies=[Depends(require_ops_token)])
def list_annotation_assertions(
    review_state: str = Query(default="unreviewed", pattern="^(unreviewed|accepted|rejected|superseded)$"),
    limit: int = Query(default=200, ge=1, le=1_000),
    catalog: Catalog = Depends(get_catalog),
) -> list[dict[str, Any]]:
    with catalog.reader() as connection:
        rows = connection.execute(
            """SELECT a.*,d.accession,d.platform FROM annotation_assertions a
               JOIN dataset_versions v ON v.version_id=a.version_id
               JOIN datasets d ON d.dataset_uid=v.dataset_uid
               WHERE a.review_state=? ORDER BY a.created_at ASC LIMIT ?""",
            (review_state, limit),
        ).fetchall()
    return [dict(row) for row in rows]


@app.post(
    "/v1/ops/annotations/{assertion_id}/review",
    dependencies=[Depends(require_ops_token)],
)
def review_annotation(
    assertion_id: str,
    request: AnnotationReviewRequest,
    catalog: Catalog = Depends(get_catalog),
) -> dict[str, str]:
    try:
        review_id = catalog.review_annotation(
            assertion_id,
            reviewer=request.reviewer,
            decision=request.decision,
            note=request.note,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"assertion_id": assertion_id, "review_id": review_id, "status": request.decision}
