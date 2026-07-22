"""Fail when the publication tree contains machine-private or credential material."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()


def candidate_paths() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return [ROOT / value.decode() for value in result.stdout.split(b"\0") if value]


def main() -> None:
    forbidden_text = (
        "chat" + "gpt",
        "co" + "dex",
        "Py" + "charmProjects",
        "C:" + "\\Users\\",
    )
    secret_patterns = (
        re.compile(r"sk-or-" + r"v1-[A-Za-z0-9_-]{20,}"),
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
        re.compile(r"AKIA[0-9A-Z]{16}"),
        re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}"),
    )
    failures: list[str] = []

    for path in candidate_paths():
        if path.resolve() == SELF or not path.is_file():
            continue
        payload = path.read_bytes()
        if b"\0" in payload:
            continue
        text = payload.decode("utf-8", errors="replace")
        lowered = text.casefold()
        for marker in forbidden_text:
            if marker.casefold() in lowered:
                failures.append(f"{path.relative_to(ROOT)}: forbidden publication marker")
        for pattern in secret_patterns:
            if pattern.search(text):
                failures.append(f"{path.relative_to(ROOT)}: credential-like value")

    if failures:
        raise SystemExit("Publication scan failed:\n" + "\n".join(sorted(set(failures))))
    print(f"Publication scan passed for {len(candidate_paths())} files.")


if __name__ == "__main__":
    main()
