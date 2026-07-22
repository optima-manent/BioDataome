from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORIGIN = ROOT / "python" / "cskl_atlas" / "CORE_ORIGIN.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_vendored_kernel_hashes_match_origin_manifest() -> None:
    origin = json.loads(ORIGIN.read_text(encoding="utf-8"))
    for key in ("scientific_kernel", "validated_fast_path"):
        record = origin[key]
        assert sha256(ROOT / record["path"]) == record["sha256"]
