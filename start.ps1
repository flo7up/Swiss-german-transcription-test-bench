[CmdletBinding()]
param(
    [ValidateRange(1, 65535)]
    [int]$Port = 8001
)

$ErrorActionPreference = "Stop"
$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $python -PathType Leaf)) {
    throw "Python environment not found. Run 'py -m venv .venv' and './.venv/Scripts/python.exe -m pip install -r requirements.txt' first."
}

Push-Location $PSScriptRoot
try {
    Write-Host "Starting Swiss German Test Bench at http://127.0.0.1:$Port"
    & $python -m uvicorn backend.app.api:app --reload --host 127.0.0.1 --port $Port
}
finally {
    Pop-Location
}