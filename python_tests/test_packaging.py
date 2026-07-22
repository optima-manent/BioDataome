from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import zipfile
from email.parser import BytesParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_wheel_contains_core_origin_manifest(tmp_path: Path) -> None:
    """The installed distribution must retain its numerical-core provenance."""

    source = tmp_path / "source"
    wheelhouse = tmp_path / "wheelhouse"
    source.mkdir()
    wheelhouse.mkdir()
    shutil.copy2(ROOT / "pyproject.toml", source / "pyproject.toml")
    shutil.copy2(ROOT / "README.md", source / "README.md")
    shutil.copytree(
        ROOT / "python",
        source / "python",
        ignore=shutil.ignore_patterns("__pycache__", "*.py[cod]", "*.egg-info"),
    )

    environment = os.environ.copy()
    environment["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            ".",
            "--no-deps",
            "--wheel-dir",
            str(wheelhouse),
        ],
        cwd=source,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    wheels = list(wheelhouse.glob("cskl_atlas-*.whl"))
    assert len(wheels) == 1
    with zipfile.ZipFile(wheels[0]) as archive:
        member = "cskl_atlas/CORE_ORIGIN.json"
        assert member in archive.namelist()
        origin = json.loads(archive.read(member))
        metadata_name = next(
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        )
        metadata = BytesParser().parsebytes(archive.read(metadata_name))

    assert origin["scientific_kernel"]["path"] == "python/cskl.py"
    assert len(origin["scientific_kernel"]["sha256"]) == 64
    assert len(origin["validated_fast_path"]["sha256"]) == 64
    requirements = metadata.get_all("Requires-Dist", [])
    assert any(
        item.startswith("pandas") and ">=2.2" in item and "<4" in item
        for item in requirements
    )
    assert any(
        item.startswith("requests") and ">=2.32" in item and "<3" in item
        for item in requirements
    )
