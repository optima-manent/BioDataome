"""Pinned SPECTER2 proximity releases for the Atlas graph."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from cskl_pipeline.scale.store import read_json

from .catalog import Catalog, canonical_json

BASE_MODEL_ID = "allenai/specter2_base"
BASE_REVISION = "3447645e1def9117997203454fa4495937bfbd83"
ADAPTER_MODEL_ID = "allenai/specter2"
ADAPTER_REVISION = "2081559630a80fc5851d8f798a05ba81e9468089"

Embedder = Callable[[list[str], list[str], str, int], tuple[np.ndarray, dict[str, Any]]]


def _default_embedder(
    titles: list[str],
    summaries: list[str],
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        import torch
        from adapters import AutoAdapterModel
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - exercised by installation smoke
        raise RuntimeError(
            "SPECTER2 dependencies are missing; install `cskl-atlas[specter2]`."
        ) from exc

    resolved_device = device
    if device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    if resolved_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but this PyTorch build cannot use it")
    resolved_batch = batch_size or (32 if resolved_device == "cuda" else 8)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, revision=BASE_REVISION)
    model = AutoAdapterModel.from_pretrained(BASE_MODEL_ID, revision=BASE_REVISION)
    model.load_adapter(
        ADAPTER_MODEL_ID,
        source="hf",
        revision=ADAPTER_REVISION,
        set_active=True,
    )
    model.to(resolved_device)
    model.eval()
    texts = [
        f"{title}{tokenizer.sep_token}{summary}"
        for title, summary in zip(titles, summaries, strict=True)
    ]
    batches: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(texts), resolved_batch):
            encoded = tokenizer(
                texts[start : start + resolved_batch],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            encoded = {key: value.to(resolved_device) for key, value in encoded.items()}
            output = model(**encoded)
            cls = output.last_hidden_state[:, 0, :]
            cls = torch.nn.functional.normalize(cls, p=2, dim=1)
            batches.append(cls.detach().cpu().numpy().astype(np.float32, copy=False))
    embeddings = np.concatenate(batches, axis=0)
    return embeddings, {
        "device": resolved_device,
        "batch_size": resolved_batch,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "embedding_dimension": int(embeddings.shape[1]),
    }


def _text_release_status(catalog: Catalog, release_id: str) -> str:
    with catalog.reader() as connection:
        row = connection.execute(
            "SELECT status FROM text_releases WHERE text_release_id=?", (release_id,)
        ).fetchone()
    if not row:
        raise KeyError(release_id)
    return str(row["status"])


def build_specter2_release(
    catalog: Catalog,
    *,
    metadata_directory: str | Path,
    output_path: str | Path,
    device: str = "auto",
    batch_size: int = 0,
    embedder: Embedder | None = None,
) -> dict[str, Any]:
    """Embed the current GEO corpus, persist all pairs, and finalize percentiles."""

    records_dir = Path(metadata_directory).resolve() / "records"
    with catalog.reader() as connection:
        current = {
            row["accession"]: row["current_version_id"]
            for row in connection.execute(
                """SELECT accession,current_version_id FROM datasets
                   WHERE current_version_id IS NOT NULL ORDER BY accession"""
            )
        }
    records: list[dict[str, Any]] = []
    missing: list[str] = []
    for accession, version_id in current.items():
        path = records_dir / f"{accession}.json"
        if not path.is_file():
            missing.append(accession)
            continue
        record = read_json(path)
        title = str(record.get("title") or "").strip()
        summary = str(record.get("summary") or "").strip()
        if not title and not summary:
            missing.append(accession)
            continue
        records.append(
            {
                "accession": accession,
                "version_id": version_id,
                "title": title,
                "summary": summary,
                "metadata_hash": record.get("content_sha256"),
            }
        )
    if missing:
        raise ValueError(
            f"SPECTER2 requires metadata for every current dataset; missing {len(missing)} records"
        )
    if len(records) < 2:
        raise ValueError("SPECTER2 requires at least two metadata records")

    corpus_hash = hashlib.sha256(canonical_json(records).encode()).hexdigest()
    parameters = {
        "base_model": BASE_MODEL_ID,
        "base_revision": BASE_REVISION,
        "adapter_model": ADAPTER_MODEL_ID,
        "adapter_revision": ADAPTER_REVISION,
        "input_kind": "geo_title_summary_fallback",
        "input_format": "title[SEP]summary",
        "max_length": 512,
        "pooling": "cls",
        "l2_normalized": True,
    }
    parameter_hash = hashlib.sha256(canonical_json(parameters).encode()).hexdigest()
    release_id = catalog.stage_text_release(
        model_id=f"{BASE_MODEL_ID}+{ADAPTER_MODEL_ID}",
        model_revision=f"{BASE_REVISION}+{ADAPTER_REVISION}",
        input_fields=("title", "summary"),
        corpus_hash=corpus_hash,
        parameter_hash=parameter_hash,
        manifest=parameters,
    )
    status = _text_release_status(catalog, release_id)
    if status == "finalized":
        return {
            "text_release_id": release_id,
            "status": "finalized",
            "reused": True,
            "dataset_count": len(records),
            "pair_count": len(records) * (len(records) - 1) // 2,
        }
    if status != "staging":
        raise ValueError(f"Text release cannot resume from status {status}")

    runner = embedder or _default_embedder
    embeddings, runtime = runner(
        [record["title"] for record in records],
        [record["summary"] for record in records],
        device,
        batch_size,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(records):
        raise ValueError("SPECTER2 embedder returned an invalid matrix shape")
    if not np.all(np.isfinite(embeddings)):
        raise ValueError("SPECTER2 embeddings contain non-finite values")
    norms = np.linalg.norm(embeddings, axis=1)
    if np.any(norms <= 0):
        raise ValueError("SPECTER2 returned a zero embedding")
    embeddings = embeddings / norms[:, None]

    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            accessions=np.asarray([record["accession"] for record in records]),
            version_ids=np.asarray([record["version_id"] for record in records]),
            embeddings=embeddings,
            corpus_hash=np.asarray(corpus_hash),
            parameter_hash=np.asarray(parameter_hash),
        )
    os.replace(temporary, destination)
    checksum = hashlib.sha256(destination.read_bytes()).hexdigest()
    catalog.record_artifact(
        artifact_id=hashlib.sha256(f"specter2:{release_id}:{checksum}".encode()).hexdigest(),
        kind="specter2_embeddings",
        uri=str(destination),
        checksum=checksum,
        dependency_hash=hashlib.sha256(f"{corpus_hash}\0{parameter_hash}".encode()).hexdigest(),
        manifest={**parameters, **runtime, "text_release_id": release_id},
    )

    similarities = embeddings @ embeddings.T
    rows: list[tuple[str, str, float]] = []
    for left in range(len(records)):
        for right in range(left + 1, len(records)):
            rows.append(
                (
                    records[left]["version_id"],
                    records[right]["version_id"],
                    float(np.clip(similarities[left, right], -1.0, 1.0)),
                )
            )
    catalog.record_text_pair_scores(release_id, rows)
    pair_count = catalog.finalize_text_release(release_id)
    expected_pairs = len(records) * (len(records) - 1) // 2
    if pair_count != expected_pairs:
        raise ValueError(f"Incomplete SPECTER2 family: expected {expected_pairs}, got {pair_count}")
    return {
        "text_release_id": release_id,
        "status": "finalized",
        "reused": False,
        "dataset_count": len(records),
        "pair_count": pair_count,
        "corpus_hash": corpus_hash,
        "parameter_hash": parameter_hash,
        "artifact": str(destination),
        "runtime": runtime,
    }
