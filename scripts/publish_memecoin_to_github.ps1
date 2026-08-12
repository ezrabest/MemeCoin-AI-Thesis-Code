$ErrorActionPreference = "Stop"

# =========================
# USER SETTINGS
# =========================

$ProjectRoot = "E:\Projects\Final Project\memecoin_trader"
$StagingRoot = "E:\Projects\Final Project\memecoin_trader_github_submission"

$GitHubUser = "ezrabest"
$RepoName = "MemeCoin-AI-Thesis-Code"

# Use "private" first. You can change to "public" after manual inspection.
$RepoVisibility = "private"

$CommitMessage = "Initial thesis code submission package"

# Set to $true only after manually reviewing potential secret warnings.
$AllowPotentialSecrets = $false

# =========================
# PRE-FLIGHT
# =========================

Write-Host ""
Write-Host "=== MemeCoin thesis GitHub publisher ===" -ForegroundColor Cyan
Write-Host "Project root:  $ProjectRoot"
Write-Host "Staging root:  $StagingRoot"
Write-Host "GitHub repo:   https://github.com/$GitHubUser/$RepoName"
Write-Host "Visibility:    $RepoVisibility"
Write-Host ""

if (!(Test-Path $ProjectRoot)) {
    throw "Project root not found: $ProjectRoot"
}

if (!(Get-Command git -ErrorAction SilentlyContinue)) {
    throw "git is not installed or not available in PATH."
}

if (!(Get-Command gh -ErrorAction SilentlyContinue)) {
    throw @"
GitHub CLI 'gh' is not installed or not available in PATH.

Install GitHub CLI, then run:
  gh auth login

Download:
  https://cli.github.com/

After that, rerun this script.
"@
}

# Authenticate if needed
Write-Host "Checking GitHub CLI authentication..." -ForegroundColor Cyan
$ghAuthOk = $true
try {
    gh auth status | Out-Null
} catch {
    $ghAuthOk = $false
}

if (-not $ghAuthOk) {
    Write-Host "GitHub CLI is not authenticated. Starting browser login..." -ForegroundColor Yellow
    gh auth login --hostname github.com --web
}

# =========================
# CREATE CLEAN STAGING FOLDER
# =========================

if (Test-Path $StagingRoot) {
    Write-Host ""
    Write-Host "Staging folder already exists:" -ForegroundColor Yellow
    Write-Host $StagingRoot
    $answer = Read-Host "Delete and rebuild it? Type YES to continue"
    if ($answer -ne "YES") {
        throw "Aborted by user."
    }
    Remove-Item $StagingRoot -Recurse -Force
}

New-Item -ItemType Directory -Path $StagingRoot | Out-Null

Write-Host ""
Write-Host "Copying code to clean staging folder..." -ForegroundColor Cyan

# Excluded directories: heavy data, caches, virtual envs, local outputs, previous git history
$ExcludeDirs = @(
    ".git",
    ".venv",
    ".venv-tabicl",
    ".venv*",
    "venv",
    "venv*",
    "env",
    "*site-packages*",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".cache",
    "node_modules",
    "dist",
    "build",
    "data",
    "logs",
    "outputs",
    "runs",
    "artifacts",
    ".cursor",
    ".idea",
    ".vscode",
    "CHATGPT"
)

# Excluded files: secrets, DBs, large binary/model/data artifacts
$ExcludeFiles = @(
    ".env",
    ".env.*",
    "*.db",
    "*.sqlite",
    "*.sqlite3",
    "*.duckdb",
    "*.parquet",
    "*.feather",
    "*.pkl",
    "*.pickle",
    "*.joblib",
    "*.onnx",
    "*.pt",
    "*.pth",
    "*.bin",
    "*.zip",
    "*.7z",
    "*.rar",
    "*.tar",
    "*.gz",
    "*.log",
    "*.key",
    "*.pem",
    "*.crt",
    "*.p12",
    "credentials*.json",
    "secret*.json",
    "token*.json",
    "*api_key*",
    "*private_key*",
    "*mnemonic*",
    "*seed_phrase*"
)

$RoboArgs = @(
    $ProjectRoot,
    $StagingRoot,
    "/E",
    "/NFL",
    "/NDL",
    "/NJH",
    "/NJS",
    "/NC",
    "/NS",
    "/NP",
    "/XD"
) + $ExcludeDirs + @("/XF") + $ExcludeFiles

robocopy @RoboArgs
$RoboCode = $LASTEXITCODE

# Robocopy codes 0-7 are success/warnings. 8+ is failure.
if ($RoboCode -ge 8) {
    throw "Robocopy failed with exit code $RoboCode"
}

# Defensive cleanup: remove any virtual environments or copied site-packages that slipped through.
Write-Host "Removing virtual environments / site-packages from staging if present..." -ForegroundColor Cyan

$ForbiddenDirs = Get-ChildItem $StagingRoot -Recurse -Directory -Force -ErrorAction SilentlyContinue |
    Where-Object {
        $_.Name -like ".venv*" -or
        $_.Name -like "venv*" -or
        $_.Name -eq "env" -or
        $_.Name -eq "site-packages" -or
        $_.FullName -match "\\Lib\\site-packages($|\\)"
    } |
    Sort-Object { $_.FullName.Length } -Descending

foreach ($d in $ForbiddenDirs) {
    Write-Host ("Removing forbidden staged directory: " + $d.FullName) -ForegroundColor Yellow
    Remove-Item $d.FullName -Recurse -Force -ErrorAction SilentlyContinue
}

# =========================
# WRITE GITHUB-SAFE FILES
# =========================

Write-Host "Writing .gitignore and submission notes..." -ForegroundColor Cyan

@"
# Environment / secrets
.env
.env.*
*.key
*.pem
*.crt
*.p12
credentials*.json
secret*.json
token*.json
*api_key*
*private_key*
*mnemonic*
*seed_phrase*

# Python
__pycache__/
*.py[cod]
*.pyo
.venv/
.venv*/
.venv-tabicl/
venv/
venv*/
env/
site-packages/
.pytest_cache/
.mypy_cache/
.ruff_cache/

# Local databases and large data
data/
logs/
outputs/
runs/
artifacts/
*.db
*.sqlite
*.sqlite3
*.duckdb
*.parquet
*.feather
*.csv
*.jsonl

# Model binaries / archives
*.pkl
*.pickle
*.joblib
*.onnx
*.pt
*.pth
*.bin
*.zip
*.7z
*.rar
*.tar
*.gz

# IDE / OS
.vscode/
.idea/
.cursor/
.DS_Store
Thumbs.db
"@ | Set-Content -Path (Join-Path $StagingRoot ".gitignore") -Encoding UTF8

@"
# MemeCoin AI Thesis Code

This repository contains the source-code submission package for the MSc thesis:

**A Multi-Layer AI Framework for Meme-Coin Market Monitoring, Risk Explanation, and Paper-Trading Validation**

Author: Dr. Ezra Ella  
Supervisor: Dr. Nir Andelman  
Advisor: Dr. Sharon Yalov-Handzel

## Scope

This repository is intended to preserve the implementation code used for the thesis.

The thesis experiments were conducted in demo / paper-trading mode only.

No live trading, wallet connection, or real transaction execution is included in this repository.

## Excluded from repository

The following are intentionally excluded:

- local databases
- raw provider payloads
- cached API responses
- audit packages
- generated thesis figures
- API keys and `.env` files
- model binaries and large data artifacts

The validated thesis evidence packages are documented separately in the thesis appendices.

## Safety boundary

This code package should not be interpreted as a production trading system or as proof of live profitability.
"@ | Set-Content -Path (Join-Path $StagingRoot "README.md") -Encoding UTF8

@"
# Example environment file

# Do not commit real keys.
# Copy this file to .env locally if needed.

APP_MODE=DEMO
LIVE_TRADING_ENABLED=false
WALLET_CONNECTION_ENABLED=false

# Optional providers
HELIUS_API_KEY=
GEMINI_API_KEY=
"@ | Set-Content -Path (Join-Path $StagingRoot ".env.example") -Encoding UTF8

@"
# GitHub Submission Notes

This package was generated from:

$ProjectRoot

Generated staging folder:

$StagingRoot

Repository target:

https://github.com/$GitHubUser/$RepoName

Important:

- This package excludes local data, databases, raw provider payloads, caches, generated audit packages, and secrets.
- The thesis evidence packages should remain documented in the thesis appendices.
- Do not enable live trading from this code without a separate production safety review.
"@ | Set-Content -Path (Join-Path $StagingRoot "GITHUB_SUBMISSION_NOTES.md") -Encoding UTF8

# =========================
# SANITIZE STAGED PLACEHOLDERS / TEST KEYS
# =========================

Write-Host ""
Write-Host "Sanitizing staged placeholder credentials before secret scan..." -ForegroundColor Cyan

$SanitizeExtensions = @(
    ".py", ".ps1", ".md", ".txt", ".toml", ".yaml", ".yml",
    ".json", ".ini", ".cfg", ".html", ".css", ".js", ".ts"
)

$SanitizeFiles = Get-ChildItem $StagingRoot -Recurse -File -Force -ErrorAction SilentlyContinue |
    Where-Object {
        $_.Length -lt 5MB -and
        $SanitizeExtensions -contains $_.Extension.ToLower()
    }

foreach ($f in $SanitizeFiles) {
    $content = Get-Content -Path $f.FullName -Raw -Encoding UTF8 -ErrorAction SilentlyContinue
    if ($null -eq $content) {
        continue
    }

    $newContent = $content

    # Remove real-looking Google/Gemini API keys from staged files.
    # Use a short dummy value so the lightweight secret regex will not match it.
    $newContent = $newContent -replace 'AIzaSy[A-Za-z0-9_\-]{20,}', 'DUMMY'

    # Remove real-looking OpenAI-style test literals from staged files.
    $newContent = $newContent -replace 'sk-[A-Za-z0-9_\-]{20,}', 'DUMMY'

    # Normalize obvious local test placeholders.
    $newContent = $newContent -replace 'api-key=test-key-[0-9]+', 'api-key=test-key'
    $newContent = $newContent -replace 'DUMMY', 'DUMMY'

    if ($newContent -ne $content) {
        Set-Content -Path $f.FullName -Value $newContent -Encoding UTF8
        Write-Host ("Sanitized staged file: " + $f.FullName) -ForegroundColor Yellow
    }
}

# Remove obsolete backup audit script from GitHub staging if present.
# It is not needed for thesis source publication and duplicates provider-key examples.
$ObsoleteStagedFiles = @(
    "scripts\run_thesis_wallet_flow_coverage_expansion_audit_v1_before_windows_cache_filename_fix.py"
)

foreach ($rel in $ObsoleteStagedFiles) {
    $p = Join-Path $StagingRoot $rel
    if (Test-Path $p) {
        Remove-Item $p -Force
        Write-Host ("Removed obsolete staged file: " + $p) -ForegroundColor Yellow
    }
}

# =========================
# SECRET SCAN
# =========================

Write-Host ""
Write-Host "Running lightweight secret scan on staged text files..." -ForegroundColor Cyan

$TextExtensions = @(
    ".py", ".ps1", ".md", ".txt", ".toml", ".yaml", ".yml",
    ".json", ".ini", ".cfg", ".html", ".css", ".js", ".ts"
)

$FilesToScan = Get-ChildItem $StagingRoot -Recurse -File |
    Where-Object {
        $_.Length -lt 2MB -and
        $TextExtensions -contains $_.Extension.ToLower()
    }

$SecretPattern = '(?i)(api[_-]?key|secret|private[_-]?key|access[_-]?token|refresh[_-]?token|mnemonic|seed[_-]?phrase)\s*[:=]\s*["'']?[A-Za-z0-9_\-\.]{12,}'

$PotentialSecrets = @()
foreach ($f in $FilesToScan) {
    $matches = Select-String -Path $f.FullName -Pattern $SecretPattern -ErrorAction SilentlyContinue
    if ($matches) {
        $PotentialSecrets += $matches
    }
}

# Filter obvious non-secret placeholders/status constants.
$FalsePositiveSecretPatterns = @(
    '(?i)api_key\s*=\s*read_dotenv_key\s*\(',
    '(?i)api_key\s*=\s*read_key\s*\(',
    '(?i)DUMMY',
    '(?i)DUMMY',
    '(?i)api-key=test-key',
    '(?i)test-key-[0-9]+',
    '(?i)api-key=test-key-[0-9]+',
    '(?i)api[_-]?key=test-key-[0-9]+',
    '(?i)NO_API_KEY',
    '(?i)SKIPPED_NO_API_KEY',
    '(?i)os\.environ\.get\(',
    '(?i)your-[a-z0-9_-]*api-key',
    '(?i)fake-client-secret',
    '(?i)my_secret_key',
    '(?i)api[_-]?key\s*=\s*$',
    '(?i)HELIUS_SKIPPED_NO_API_KEY'
)

$PotentialSecrets = @(
    $PotentialSecrets | Where-Object {
        $line = $_.Line.Trim()
        $isFalsePositive = $false
        foreach ($pat in $FalsePositiveSecretPatterns) {
            if ($line -match $pat) {
                $isFalsePositive = $true
                break
            }
        }
        -not $isFalsePositive
    }
)

# Final hard guard: never allow virtualenv/site-packages to be committed.
$StillForbiddenDirs = Get-ChildItem $StagingRoot -Recurse -Directory -Force -ErrorAction SilentlyContinue |
    Where-Object {
        $_.Name -like ".venv*" -or
        $_.Name -like "venv*" -or
        $_.Name -eq "env" -or
        $_.Name -eq "site-packages" -or
        $_.FullName -match "\\Lib\\site-packages($|\\)"
    }

if ($StillForbiddenDirs.Count -gt 0) {
    $StillForbiddenDirs | Select-Object -First 20 | ForEach-Object {
        Write-Host ("Forbidden directory still staged: " + $_.FullName) -ForegroundColor Red
    }
    throw "Aborting because virtualenv/site-packages content is still staged."
}

if ($PotentialSecrets.Count -gt 0) {
    Write-Host ""
    Write-Host "Potential secrets were detected in the staging folder:" -ForegroundColor Red
    $PotentialSecrets |
        Select-Object -First 30 |
        ForEach-Object {
            Write-Host ("{0}:{1}: {2}" -f $_.Path, $_.LineNumber, $_.Line.Trim()) -ForegroundColor Yellow
        }

    Write-Host ""
    Write-Host "Review these before pushing." -ForegroundColor Red

    if (-not $AllowPotentialSecrets) {
        throw "Aborting push because potential secrets were detected. After manual cleanup, rerun the script. Only set `$AllowPotentialSecrets = `$true if these are confirmed placeholders."
    }
}

# =========================
# INIT GIT REPO
# =========================

Set-Location $StagingRoot

Write-Host ""
Write-Host "Initializing git repository..." -ForegroundColor Cyan

git init
git branch -M main
git add .
git status --short

$HasChanges = git status --porcelain
if (-not $HasChanges) {
    throw "No files staged for commit. Check exclusions and project folder."
}

git commit -m $CommitMessage

# =========================
# CREATE / CONNECT GITHUB REPO
# =========================

Write-Host ""
Write-Host "Checking GitHub repository..." -ForegroundColor Cyan

$RepoFullName = "$GitHubUser/$RepoName"

$RepoExists = $true
try {
    gh repo view $RepoFullName | Out-Null
} catch {
    $RepoExists = $false
}

if (-not $RepoExists) {
    Write-Host "Creating GitHub repository: $RepoFullName" -ForegroundColor Cyan

    if ($RepoVisibility -eq "public") {
        gh repo create $RepoFullName --public --description "MSc thesis code package for meme-coin AI paper/demo trading framework"
    } else {
        gh repo create $RepoFullName --private --description "MSc thesis code package for meme-coin AI paper/demo trading framework"
    }
} else {
    Write-Host "Repository already exists: $RepoFullName" -ForegroundColor Yellow
    Write-Host "The script will push without force. If the remote has existing commits, push may be rejected." -ForegroundColor Yellow
}

$RemoteUrl = "https://github.com/$RepoFullName.git"

$ExistingRemote = git remote
if ($ExistingRemote -contains "origin") {
    git remote set-url origin $RemoteUrl
} else {
    git remote add origin $RemoteUrl
}

Write-Host ""
Write-Host "Pushing to GitHub..." -ForegroundColor Cyan

git push -u origin main

Write-Host ""
Write-Host "DONE." -ForegroundColor Green
Write-Host "Repository URL:" -ForegroundColor Green
Write-Host "https://github.com/$RepoFullName"
Write-Host ""
Write-Host "Staged local copy:" -ForegroundColor Green
Write-Host $StagingRoot
