$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot

Push-Location $projectRoot
try {
    $env:PYTHONPATH = "python"
    $env:PYTHONDONTWRITEBYTECODE = "1"
    python -m pytest -p no:cacheprovider python_tests -q
    python scripts/check_publication.py
    python -m compileall -q python
    pnpm install --frozen-lockfile
    pnpm run lint
    pnpm test
    pnpm test:pages
}
finally {
    Pop-Location
}
