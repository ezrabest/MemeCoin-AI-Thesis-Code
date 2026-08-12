$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

$env:HEADLESS_DATA_COLLECTION = "false"
$env:LLM_PROVIDER = "gemini"
$env:ENABLE_GEMINI = "true"

python main.py @args
