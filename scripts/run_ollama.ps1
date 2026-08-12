$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

$env:HEADLESS_DATA_COLLECTION = "false"
$env:LLM_PROVIDER = "ollama"
$env:ENABLE_GEMINI = "false"
$env:OLLAMA_BASE_URL = "http://localhost:11434/v1"
$env:OLLAMA_MODEL = "qwen3:8b"
$env:OLLAMA_MAX_CALLS_PER_SCAN = "5"
$env:OLLAMA_TIMEOUT_SECONDS = "60"

python main.py @args
