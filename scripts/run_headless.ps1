$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

$env:HEADLESS_DATA_COLLECTION = "true"
$env:LLM_PROVIDER = "none"
$env:ENABLE_GEMINI = "false"

python main.py @args
