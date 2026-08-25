# Bootstraps and verifies the Day 0A environment on a clean checkout.
# Usage: powershell -File setup.ps1  (or: pwsh -File setup.ps1)

$ErrorActionPreference = "Stop"

if (-not (Test-Path ".venv")) {
    py -m venv .venv
}

.venv\Scripts\pip.exe install -r requirements.txt
.venv\Scripts\pytest.exe -q
