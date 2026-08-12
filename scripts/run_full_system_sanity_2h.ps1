param(
    [int]$DurationMinutes = 120,
    [int]$SampleEveryMinutes = 10,
    [switch]$UseHelius,
    [switch]$SkipDatasetBuild,
    [switch]$SkipTabICLSmoke,
    [switch]$SkipNoOpPatchTest
)

$ErrorActionPreference = "Continue"

# ==============================================================================
# MemeCoin AI Trader - Full System Sanity Audit
# ASCII-only PowerShell script
# Output folder:
# E:\Projects\Final Project\memecoin_trader\tests\Results
# ==============================================================================

$ProjectRoot = "E:\Projects\Final Project\memecoin_trader"
$ResultsRoot = Join-Path $ProjectRoot "tests\Results"
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$RunDir = Join-Path $ResultsRoot ("full_sanity_" + $Timestamp)
$ZipPath = Join-Path $ResultsRoot ("full_sanity_" + $Timestamp + ".zip")

$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$TabPython = Join-Path $ProjectRoot ".venv-tabicl\Scripts\python.exe"
$ApiBase = "http://localhost:8080"
$HealthUrl = $ApiBase + "/api/healthz"

New-Item -ItemType Directory -Force -Path $RunDir | Out-Null

$Global:Steps = New-Object System.Collections.Generic.List[object]
$Global:Issues = New-Object System.Collections.Generic.List[object]
$Global:ServerProcess = $null
$Global:ServerStartedByScript = $false

$TranscriptPath = Join-Path $RunDir "transcript.txt"
try {
    Start-Transcript -Path $TranscriptPath -Force | Out-Null
} catch {
    Write-Host ("Could not start transcript: " + $_.Exception.Message)
}

function Add-Step {
    param(
        [string]$Name,
        [string]$Status,
        [string]$Message = "",
        [string]$Path = "",
        [object]$Extra = $null
    )

    $obj = [pscustomobject]@{
        timestamp = (Get-Date).ToString("o")
        name = $Name
        status = $Status
        message = $Message
        path = $Path
        extra = $Extra
    }

    $Global:Steps.Add($obj) | Out-Null
    Write-Host ("[{0}] {1} - {2}" -f $Status.ToUpper(), $Name, $Message)
}

function Add-Issue {
    param(
        [string]$Severity,
        [string]$Area,
        [string]$Message,
        [object]$Extra = $null
    )

    $obj = [pscustomobject]@{
        timestamp = (Get-Date).ToString("o")
        severity = $Severity
        area = $Area
        message = $Message
        extra = $Extra
    }

    $Global:Issues.Add($obj) | Out-Null
    Write-Host ("ISSUE [{0}] {1}: {2}" -f $Severity, $Area, $Message) -ForegroundColor Yellow
}

function Safe-Name {
    param([string]$Name)
    return ($Name -replace '[^\w\-.]+', '_')
}

function Save-Json {
    param(
        [object]$Object,
        [string]$Path,
        [int]$Depth = 30
    )

    try {
        $Object | ConvertTo-Json -Depth $Depth | Out-File -Encoding UTF8 -FilePath $Path
    } catch {
        Add-Issue "ERROR" "json" ("Failed to save JSON: " + $Path + " :: " + $_.Exception.Message)
    }
}

function Save-Text {
    param(
        [string]$Text,
        [string]$Path
    )

    try {
        $Text | Out-File -Encoding UTF8 -FilePath $Path
    } catch {
        Add-Issue "ERROR" "text" ("Failed to save text: " + $Path + " :: " + $_.Exception.Message)
    }
}

function Invoke-Api {
    param(
        [string]$Name,
        [string]$Uri,
        [string]$Method = "GET",
        [object]$Body = $null,
        [int]$Depth = 40
    )

    $safe = Safe-Name $Name
    $outPath = Join-Path $RunDir ($safe + ".json")
    $errPath = Join-Path $RunDir ($safe + ".error.txt")

    try {
        if ($null -ne $Body) {
            $bodyJson = $Body | ConvertTo-Json -Depth $Depth
            $resp = Invoke-RestMethod -Method $Method -Uri $Uri -ContentType "application/json" -Body $bodyJson -TimeoutSec 45
        } else {
            $resp = Invoke-RestMethod -Method $Method -Uri $Uri -TimeoutSec 45
        }

        Save-Json $resp $outPath $Depth
        Add-Step $Name "ok" ($Method + " " + $Uri) $outPath
        return $resp
    } catch {
        $msg = $_.Exception.Message
        Save-Text $msg $errPath
        Add-Step $Name "failed" ($Method + " " + $Uri + " :: " + $msg) $errPath
        Add-Issue "ERROR" "api" ($Name + " failed: " + $msg) @{ uri = $Uri; method = $Method }
        return $null
    }
}

function Quote-ProcessArg {
    param([string]$Value)

    if ($null -eq $Value) {
        return '""'
    }

    $v = [string]$Value
    $v = $v.Replace('"', '\"')

    if ($v -match '\s') {
        return '"' + $v + '"'
    }

    return $v
}

function Run-Cmd {
    param(
        [string]$Name,
        [string]$Exe,
        [string[]]$CmdArgs = @(),
        [int]$TimeoutSec = 900,
        [switch]$Critical
    )

    $safe = Safe-Name $Name
    $stdout = Join-Path $RunDir ($safe + ".stdout.txt")
    $stderr = Join-Path $RunDir ($safe + ".stderr.txt")
    $meta = Join-Path $RunDir ($safe + ".meta.json")

    if ($null -eq $CmdArgs) {
        $CmdArgs = @()
    }

    $CmdArgs = @($CmdArgs | Where-Object { $null -ne $_ })
    $argLine = (($CmdArgs | ForEach-Object { Quote-ProcessArg $_ }) -join " ")

    Write-Host ""
    Write-Host ("=== RUN: " + $Name + " ===") -ForegroundColor Cyan
    Write-Host ($Exe + " " + $argLine)

    $knownCommands = @("git", "git.exe", "powershell", "powershell.exe", "cmd", "cmd.exe")

    if ((-not (Test-Path $Exe)) -and ($knownCommands -notcontains $Exe)) {
        Add-Step $Name "skipped" ("Executable not found: " + $Exe)
        Add-Issue "WARN" "command" ("Executable not found: " + $Exe)
        return $null
    }

    try {
        $psi = New-Object System.Diagnostics.ProcessStartInfo
        $psi.FileName = $Exe
        $psi.Arguments = $argLine
        $psi.WorkingDirectory = $ProjectRoot
        $psi.UseShellExecute = $false
        $psi.RedirectStandardOutput = $true
        $psi.RedirectStandardError = $true
        $psi.CreateNoWindow = $true

        $p = New-Object System.Diagnostics.Process
        $p.StartInfo = $psi

        $started = $p.Start()

        if (-not $started) {
            throw "Process did not start"
        }

        $stdoutTask = $p.StandardOutput.ReadToEndAsync()
        $stderrTask = $p.StandardError.ReadToEndAsync()

        $finished = $p.WaitForExit($TimeoutSec * 1000)

        if (-not $finished) {
            try { $p.Kill() } catch {}

            $outText = ""
            $errText = "Timed out after " + $TimeoutSec + " seconds"

            try { $outText = $stdoutTask.Result } catch {}
            try { $errText = $errText + [Environment]::NewLine + $stderrTask.Result } catch {}

            Save-Text $outText $stdout
            Save-Text $errText $stderr

            $result = [pscustomobject]@{
                name = $Name
                status = "timeout"
                exit_code = $null
                timeout_sec = $TimeoutSec
                command = ($Exe + " " + $argLine)
                stdout = $stdout
                stderr = $stderr
            }

            Save-Json $result $meta 20
            Add-Step $Name "timeout" ("Timed out after " + $TimeoutSec + " seconds") $meta
            Add-Issue "ERROR" "command" ($Name + " timed out") $result
            return $result
        }

        $p.WaitForExit()

        $outText = ""
        $errText = ""

        try { $outText = $stdoutTask.Result } catch {}
        try { $errText = $stderrTask.Result } catch {}

        Save-Text $outText $stdout
        Save-Text $errText $stderr

        $exit = [int]$p.ExitCode

        if ($exit -eq 0) {
            $status = "ok"
        } else {
            $status = "failed"
        }

        $result = [pscustomobject]@{
            name = $Name
            status = $status
            exit_code = $exit
            timeout_sec = $TimeoutSec
            command = ($Exe + " " + $argLine)
            stdout = $stdout
            stderr = $stderr
        }

        Save-Json $result $meta 20
        Add-Step $Name $status ("exit_code=" + $exit) $meta

        if ($exit -ne 0) {
            if ($Critical) {
                $sev = "ERROR"
            } else {
                $sev = "WARN"
            }

            Add-Issue $sev "command" ($Name + " failed with exit code " + $exit) $result
        }

        return $result
    } catch {
        $msg = $_.Exception.Message
        Save-Text $msg $stderr

        $result = [pscustomobject]@{
            name = $Name
            status = "exception"
            command = ($Exe + " " + $argLine)
            error = $msg
            stdout = $stdout
            stderr = $stderr
        }

        Save-Json $result $meta 20
        Add-Step $Name "exception" $msg $meta
        Add-Issue "ERROR" "command" ($Name + " exception: " + $msg) $result
        return $result
    }
}

function Test-ApiUp {
    try {
        Invoke-RestMethod -Uri $HealthUrl -TimeoutSec 3 | Out-Null
        return $true
    } catch {
        return $false
    }
}

function Start-Backend-IfNeeded {
    if (Test-ApiUp) {
        Add-Step "backend_existing" "ok" ("Backend already responds at " + $HealthUrl)
        return
    }

    Add-Issue "WARN" "backend" "Backend is not running. Starting with safe diagnostic env."

    $env:LLM_PROVIDER = "none"
    $env:ENABLE_GEMINI = "false"
    $env:HEADLESS_DATA_COLLECTION = "false"

    $stdout = Join-Path $RunDir "backend_started_by_sanity.stdout.txt"
    $stderr = Join-Path $RunDir "backend_started_by_sanity.stderr.txt"

    try {
        $Global:ServerProcess = Start-Process `
            -FilePath $Python `
            -ArgumentList @("main.py") `
            -WorkingDirectory $ProjectRoot `
            -RedirectStandardOutput $stdout `
            -RedirectStandardError $stderr `
            -PassThru `
            -NoNewWindow

        $Global:ServerStartedByScript = $true

        $deadline = (Get-Date).AddSeconds(90)

        while ((Get-Date) -lt $deadline) {
            Start-Sleep -Seconds 3
            if (Test-ApiUp) {
                Add-Step "backend_start" "ok" "Backend started by sanity script" $stdout
                return
            }
        }

        Add-Step "backend_start" "failed" "Backend did not respond within 90 seconds" $stderr
        Add-Issue "ERROR" "backend" "Backend did not respond after startup." @{ stdout = $stdout; stderr = $stderr }
    } catch {
        Add-Step "backend_start" "exception" $_.Exception.Message
        Add-Issue "ERROR" "backend" ("Failed to start backend: " + $_.Exception.Message)
    }
}

function Get-StorageRows {
    param([object]$Storage, [string]$Table)

    try {
        if ($null -eq $Storage) { return 0 }
        if ($null -eq $Storage.$Table) { return 0 }
        return [int]$Storage.$Table.rows
    } catch {
        return 0
    }
}

function Check-Files {
    $files = @(
        "data\trader.db",
        "data\settings.json",
        "data\paper_state.json",
        "data\training\models\label_profitable_after_fees_4h__random_forest.joblib",
        "data\training\models\baseline_metrics.json",
        "data\training\models\predictions_validation.parquet",
        "data\training\models\predictions_test.parquet",
        "data\training\model_ready_dataset.parquet",
        "scripts\dry_run_economic_gate_recent.py",
        "scripts\probe_solana_pool_activity.py",
        "scripts\reconcile_storage.py",
        "scripts\build_training_dataset.py",
        "static\index.html",
        "static\system_config.js",
        "app\observability\settings_patch.py",
        "app\observability\model_runtime_inference.py",
        "app\observability\economic_gate.py",
        "app\providers\solana_rpc.py",
        "app\providers\helius.py",
        "app\parsers\solana_pool_activity.py",
        "app\parsers\solana_wallet_behavior.py"
    )

    $rows = @()

    foreach ($rel in $files) {
        $p = Join-Path $ProjectRoot $rel
        $exists = Test-Path $p

        if ($exists) {
            $item = Get-Item $p
            $size = $item.Length
            $last = $item.LastWriteTime.ToString("o")
        } else {
            $size = $null
            $last = $null
            Add-Issue "WARN" "files" ("Missing expected file: " + $rel)
        }

        $rows += [pscustomobject]@{
            path = $rel
            exists = $exists
            size = $size
            last_write = $last
        }
    }

    $out = Join-Path $RunDir "file_existence_check.json"
    Save-Json $rows $out 10
    Add-Step "file_existence_check" "ok" "Checked expected files" $out
}

function Snapshot-Directory {
    param([string]$RelPath, [string]$Name)

    $path = Join-Path $ProjectRoot $RelPath
    $out = Join-Path $RunDir ($Name + ".json")

    if (-not (Test-Path $path)) {
        Save-Json @{ exists = $false; path = $RelPath } $out 10
        Add-Step ("snapshot_" + $Name) "skipped" ("Directory missing: " + $RelPath) $out
        return
    }

    $items = Get-ChildItem $path -Recurse -File -ErrorAction SilentlyContinue |
        Select-Object FullName, Length, LastWriteTime |
        ForEach-Object {
            [pscustomobject]@{
                path = $_.FullName.Replace($ProjectRoot + "\", "")
                length = $_.Length
                last_write = $_.LastWriteTime.ToString("o")
            }
        }

    Save-Json @{ exists = $true; path = $RelPath; files = $items } $out 20
    Add-Step ("snapshot_" + $Name) "ok" ("Directory snapshot: " + $RelPath) $out
}

function Check-StaticUI {
    $index = Join-Path $ProjectRoot "static\index.html"
    $js = Join-Path $ProjectRoot "static\system_config.js"

    $uiReport = [ordered]@{
        index_exists = Test-Path $index
        system_config_js_exists = Test-Path $js
        index_contains_system_configuration = $false
        js_fetches_effective_settings = $false
        js_uses_patch_settings = $false
        js_contains_dirty_payload = $false
        js_contains_discard = $false
        js_contains_percent_conversion = $false
        js_contains_legacy_aliases = $false
    }

    if (Test-Path $index) {
        $txt = Get-Content $index -Raw -ErrorAction SilentlyContinue
        $uiReport.index_contains_system_configuration = ($txt -match "System Configuration")
    }

    if (Test-Path $js) {
        $txt = Get-Content $js -Raw -ErrorAction SilentlyContinue
        $uiReport.js_fetches_effective_settings = ($txt -match "/api/settings/effective")
        $uiReport.js_uses_patch_settings = (($txt -match "PATCH") -and ($txt -match "/api/settings"))
        $uiReport.js_contains_dirty_payload = (($txt -match "dirty") -or ($txt -match "modified"))
        $uiReport.js_contains_discard = (($txt -match "Discard") -or ($txt -match "Refresh from Server"))
        $uiReport.js_contains_percent_conversion = (($txt -match "percent") -or ($txt -match "pct"))
        $uiReport.js_contains_legacy_aliases = ($txt -match "minLiquidity|stopLossPct|takeProfitPct|positionSizePct")
    }

    $path = Join-Path $RunDir "static_ui_check.json"
    Save-Json $uiReport $path 10
    Add-Step "static_ui_check" "ok" "Static UI check completed" $path

    if (-not $uiReport.index_contains_system_configuration) {
        Add-Issue "WARN" "ui" "static/index.html does not appear to contain System Configuration."
    }
    if (-not $uiReport.js_fetches_effective_settings) {
        Add-Issue "WARN" "ui" "static/system_config.js does not appear to fetch /api/settings/effective."
    }
    if (-not $uiReport.js_uses_patch_settings) {
        Add-Issue "WARN" "ui" "static/system_config.js does not appear to PATCH /api/settings."
    }
    if ($uiReport.js_contains_legacy_aliases) {
        Add-Issue "WARN" "ui" "system_config.js contains legacy alias names. Verify they are not sent in payload."
    }
}

function Analyze-EffectiveSettings {
    param([object]$Effective)

    if ($null -eq $Effective) {
        Add-Issue "ERROR" "settings" "No effective settings returned."
        return
    }

    $c = $Effective.canonical

    if ($null -eq $c) {
        Add-Issue "ERROR" "settings" "Effective settings missing canonical object."
        return
    }

    $required = @(
        "economic_gate_enabled",
        "demo_aggressive_enabled",
        "paper_trading_enabled",
        "rf_gate_enabled",
        "rf_probability_threshold",
        "tab_confidence_boost_enabled",
        "tab_confidence_boost_enabled_demo",
        "tab_confidence_boost_enabled_live",
        "llm_enabled_for_demo",
        "live_trading_enabled",
        "max_slippage_pct",
        "baseline_slippage_pct",
        "round_trip_fee_pct",
        "max_price_drift_from_model_pct",
        "required_margin_after_costs_pct"
    )

    foreach ($k in $required) {
        if (-not ($c.PSObject.Properties.Name -contains $k)) {
            Add-Issue "WARN" "settings" ("Missing canonical setting: " + $k)
        }
    }

    try {
        if ($c.economic_gate_enabled -ne $true) {
            Add-Issue "WARN" "settings" "economic_gate_enabled is OFF. RF/economic gate will not promote candidates."
        }

        if ($c.paper_trading_enabled -ne $true) {
            Add-Issue "WARN" "settings" "paper_trading_enabled is OFF. No paper trades will be created."
        }

        if ($c.live_trading_enabled -eq $true) {
            Add-Issue "ERROR" "settings" "live_trading_enabled is TRUE. This sanity run expects LIVE disabled."
        }

        if (($c.tab_confidence_boost_enabled -ne $true) -and (($c.tab_confidence_boost_enabled_demo -eq $true) -or ($c.tab_confidence_boost_enabled_live -eq $true))) {
            Add-Issue "WARN" "settings" "TAB demo/live boost is ON but master tab_confidence_boost_enabled is OFF."
        }

        $pctKeys = @(
            "max_slippage_pct",
            "baseline_slippage_pct",
            "round_trip_fee_pct",
            "max_price_drift_from_model_pct",
            "required_margin_after_costs_pct",
            "max_position_size_pct",
            "stop_loss_pct",
            "take_profit_pct"
        )

        foreach ($k in $pctKeys) {
            if ($c.PSObject.Properties.Name -contains $k) {
                $v = $c.$k

                if ($v -is [string]) {
                    Add-Issue "WARN" "settings" ($k + " is a string. Canonical settings should be numeric.") @{ key = $k; value = $v }
                }

                $num = $null
                try { $num = [double]$v } catch {}

                if ($null -ne $num) {
                    if ($num -gt 1.0) {
                        Add-Issue "WARN" "settings" ($k + " is greater than 1.0. Check percent-vs-decimal normalization.") @{ key = $k; value = $num }
                    }
                }
            }
        }
    } catch {
        Add-Issue "WARN" "settings" ("Settings semantic analysis failed: " + $_.Exception.Message)
    }
}

function Run-NoOpPatchTest {
    param([object]$Effective)

    if ($SkipNoOpPatchTest) {
        Add-Step "settings_noop_patch" "skipped" "SkipNoOpPatchTest was set"
        return
    }

    if (($null -eq $Effective) -or ($null -eq $Effective.canonical)) {
        Add-Step "settings_noop_patch" "skipped" "No effective settings available"
        return
    }

    $c = $Effective.canonical
    $payload = @{}

    if ($c.PSObject.Properties.Name -contains "paper_trading_enabled") {
        $payload.paper_trading_enabled = [bool]$c.paper_trading_enabled
    } elseif ($c.PSObject.Properties.Name -contains "rf_gate_enabled") {
        $payload.rf_gate_enabled = [bool]$c.rf_gate_enabled
    } else {
        Add-Step "settings_noop_patch" "skipped" "No safe no-op key found"
        return
    }

    $resp = Invoke-Api "settings_noop_patch" ($ApiBase + "/api/settings") "PATCH" $payload 40

    if ($null -eq $resp) {
        Add-Issue "ERROR" "settings" "No-op PATCH /api/settings failed."
    }
}

function Monitor-Runtime {
    $monitorPath = Join-Path $RunDir "runtime_monitor_samples.jsonl"
    $summaryPath = Join-Path $RunDir "runtime_monitor_summary.json"

    $samples = New-Object System.Collections.Generic.List[object]
    $start = Get-Date
    $end = $start.AddMinutes($DurationMinutes)
    $i = 0

    Write-Host ""
    Write-Host ("=== Runtime monitoring for " + $DurationMinutes + " minutes, every " + $SampleEveryMinutes + " minutes ===") -ForegroundColor Cyan

    while ((Get-Date) -lt $end) {
        $sampleTime = Get-Date
        $storage = $null
        $collection = $null
        $audit = $null

        try { $storage = Invoke-RestMethod ($ApiBase + "/api/debug/storage") -TimeoutSec 45 } catch {}
        try { $collection = Invoke-RestMethod ($ApiBase + "/api/debug/collection") -TimeoutSec 45 } catch {}
        try { $audit = Invoke-RestMethod ($ApiBase + "/api/pipeline/audit/recent?minutes=" + $SampleEveryMinutes) -TimeoutSec 45 } catch {}

        $sample = [pscustomobject]@{
            index = $i
            timestamp = $sampleTime.ToString("o")
            storage = $storage
            collection = $collection
            recent_audit = $audit
        }

        $samples.Add($sample) | Out-Null

        try {
            ($sample | ConvertTo-Json -Depth 50 -Compress) | Add-Content -Encoding UTF8 -Path $monitorPath
        } catch {
            Add-Issue "WARN" "runtime" ("Failed writing monitor sample: " + $_.Exception.Message)
        }

        Write-Host ("Sample {0}: {1}" -f $i, $sampleTime.ToString("HH:mm:ss"))
        $i += 1

        $remaining = ($end - (Get-Date)).TotalSeconds
        if ($remaining -le 0) { break }

        $sleep = [Math]::Min($SampleEveryMinutes * 60, [int]$remaining)
        Start-Sleep -Seconds $sleep
    }

    if ($samples.Count -ge 2) {
        $first = $samples[0].storage
        $last = $samples[$samples.Count - 1].storage

        $delta = [ordered]@{
            duration_minutes = $DurationMinutes
            samples = $samples.Count
            market_snapshots_delta = (Get-StorageRows $last "market_snapshots") - (Get-StorageRows $first "market_snapshots")
            signals_delta = (Get-StorageRows $last "signals") - (Get-StorageRows $first "signals")
            whale_alerts_delta = (Get-StorageRows $last "whale_alerts") - (Get-StorageRows $first "whale_alerts")
            paper_trades_delta = (Get-StorageRows $last "paper_trades") - (Get-StorageRows $first "paper_trades")
            sentiment_records_delta = (Get-StorageRows $last "sentiment_records") - (Get-StorageRows $first "sentiment_records")
            raw_provider_payloads_delta = (Get-StorageRows $last "raw_provider_payloads") - (Get-StorageRows $first "raw_provider_payloads")
            coins_delta = (Get-StorageRows $last "coins") - (Get-StorageRows $first "coins")
        }

        Save-Json $delta $summaryPath 10
        Add-Step "runtime_monitor" "ok" "Runtime monitoring complete" $summaryPath $delta

        if ($delta.market_snapshots_delta -le 0) {
            Add-Issue "ERROR" "runtime" "No market_snapshots growth during monitoring." $delta
        }

        if ($delta.signals_delta -le 0) {
            Add-Issue "WARN" "runtime" "No signals growth during monitoring." $delta
        }

        if ($delta.paper_trades_delta -le 0) {
            Add-Issue "WARN" "runtime" "No paper trades during monitoring. Check economic gate, RF threshold, blockers, and audit reasons." $delta
        }

        if ($delta.sentiment_records_delta -le 0) {
            Add-Issue "WARN" "runtime" "No sentiment_records growth during monitoring." $delta
        }
    } else {
        Add-Issue "ERROR" "runtime" "Not enough runtime samples collected."
    }
}

function Build-FinalReport {
    $summaryJson = Join-Path $RunDir "sanity_summary.json"
    $reportMd = Join-Path $RunDir "sanity_report.md"

    $finalStorage = Invoke-Api "final_storage" ($ApiBase + "/api/debug/storage")
    $finalSettings = Invoke-Api "final_settings_effective" ($ApiBase + "/api/settings/effective")
    $finalAudit = Invoke-Api "final_pipeline_audit_recent_120m" ($ApiBase + "/api/pipeline/audit/recent?minutes=120")

    $summary = [ordered]@{
        run_timestamp = $Timestamp
        project_root = $ProjectRoot
        run_dir = $RunDir
        duration_minutes = $DurationMinutes
        sample_every_minutes = $SampleEveryMinutes
        use_helius = [bool]$UseHelius
        skip_dataset_build = [bool]$SkipDatasetBuild
        skip_tabicl_smoke = [bool]$SkipTabICLSmoke
        steps = $Global:Steps
        issues = $Global:Issues
        final_storage = $finalStorage
        final_settings = $finalSettings
        final_audit = $finalAudit
    }

    Save-Json $summary $summaryJson 60

    $errorCount = ($Global:Issues | Where-Object { $_.severity -eq "ERROR" }).Count
    $warnCount = ($Global:Issues | Where-Object { $_.severity -eq "WARN" }).Count

    $lines = New-Object System.Collections.Generic.List[string]
    $lines.Add("# Full System Sanity Report")
    $lines.Add("")
    $lines.Add(("- Timestamp: {0}" -f $Timestamp))
    $lines.Add(("- Project: {0}" -f $ProjectRoot))
    $lines.Add(("- Duration minutes: {0}" -f $DurationMinutes))
    $lines.Add(("- Errors: {0}" -f $errorCount))
    $lines.Add(("- Warnings: {0}" -f $warnCount))
    $lines.Add("")
    $lines.Add("## Top Issues")
    $lines.Add("")

    if ($Global:Issues.Count -eq 0) {
        $lines.Add("No issues recorded.")
    } else {
        foreach ($iss in $Global:Issues) {
            $lines.Add(("- **{0}** [{1}] {2}" -f $iss.severity, $iss.area, $iss.message))
        }
    }

    $lines.Add("")
    $lines.Add("## Steps")
    $lines.Add("")

    foreach ($s in $Global:Steps) {
        $lines.Add(("- **{0}** - {1}: {2}" -f $s.status, $s.name, $s.message))
    }

    $lines.Add("")
    $lines.Add("## Important output files")
    $lines.Add("")
    $lines.Add("- sanity_summary.json")
    $lines.Add("- sanity_report.md")
    $lines.Add("- runtime_monitor_samples.jsonl")
    $lines.Add("- runtime_monitor_summary.json")
    $lines.Add("- final_storage.json")
    $lines.Add("- final_settings_effective.json")
    $lines.Add("- final_pipeline_audit_recent_120m.json")
    $lines.Add("- transcript.txt")

    $reportText = $lines -join [Environment]::NewLine
    Save-Text $reportText $reportMd

    Add-Step "final_report" "ok" "Final report generated" $reportMd
    return $summaryJson
}

# ==============================================================================
# MAIN
# ==============================================================================

Write-Host "FULL SYSTEM SANITY AUDIT" -ForegroundColor Green
Write-Host ("Project: " + $ProjectRoot)
Write-Host ("Results: " + $RunDir)
Write-Host ("Duration: " + $DurationMinutes + " minutes")
Write-Host ""

try {
    if (-not (Test-Path $ProjectRoot)) {
        throw ("Project root not found: " + $ProjectRoot)
    }

    if (-not (Test-Path $Python)) {
        throw ("Python not found: " + $Python)
    }

    Set-Location $ProjectRoot

    Add-Step "start" "ok" "Sanity audit started" $RunDir

    # Environment
    Run-Cmd "python_version" $Python @("--version") 60 | Out-Null
    Run-Cmd "pip_freeze" $Python @("-m", "pip", "freeze") 120 | Out-Null
    Run-Cmd "git_status" "git" @("status", "--short") 120 | Out-Null
    Run-Cmd "git_log_last_5" "git" @("log", "--oneline", "-5") 120 | Out-Null

    # File checks
    Check-Files
    Snapshot-Directory "data\audits" "audits_dir_snapshot"
    Snapshot-Directory "data\training\models" "training_models_snapshot"
    Snapshot-Directory "data\training\policy_backtests" "policy_backtests_snapshot"
    Check-StaticUI

    # Compile and tests
    Run-Cmd "compileall_app_scripts_tests" $Python @("-m", "compileall", "app", "scripts", "tests") 1800 -Critical | Out-Null
    Run-Cmd "unittest_discover" $Python @("-m", "unittest", "discover", "-s", "tests") 3600 -Critical | Out-Null

    # Backend
    Start-Backend-IfNeeded

    # Initial API snapshots
    Invoke-Api "api_healthz" ($ApiBase + "/api/healthz") | Out-Null
    $settings = Invoke-Api "api_settings_effective_initial" ($ApiBase + "/api/settings/effective")
    Analyze-EffectiveSettings $settings

    Invoke-Api "api_debug_storage_initial" ($ApiBase + "/api/debug/storage") | Out-Null
    Invoke-Api "api_debug_collection_initial" ($ApiBase + "/api/debug/collection") | Out-Null
    Invoke-Api "api_training_dataset_summary" ($ApiBase + "/api/debug/training-dataset") | Out-Null
    Invoke-Api "api_training_dataset_build_status" ($ApiBase + "/api/debug/training-dataset/build-status") | Out-Null
    Invoke-Api "api_pipeline_audit_recent_60m" ($ApiBase + "/api/pipeline/audit/recent?minutes=60") | Out-Null

    Run-NoOpPatchTest $settings

    # Storage scripts
    if (Test-Path (Join-Path $ProjectRoot "scripts\reconcile_storage.py")) {
        Run-Cmd "reconcile_storage_check" $Python @("scripts\reconcile_storage.py", "--check") 900 | Out-Null
    }

    if (Test-Path (Join-Path $ProjectRoot "scripts\audit_paper_trades.py")) {
        Run-Cmd "audit_paper_trades" $Python @("scripts\audit_paper_trades.py") 900 | Out-Null
    }

    # Dataset
    if (-not $SkipDatasetBuild) {
        if (Test-Path (Join-Path $ProjectRoot "scripts\build_training_dataset.py")) {
            Run-Cmd "build_training_dataset" $Python @("scripts\build_training_dataset.py") 1800 | Out-Null
            Invoke-Api "api_training_dataset_summary_after_build" ($ApiBase + "/api/debug/training-dataset") | Out-Null
        }
    } else {
        Add-Step "build_training_dataset" "skipped" "SkipDatasetBuild was set"
    }

    # Economic gate dry runs
    if (Test-Path (Join-Path $ProjectRoot "scripts\dry_run_economic_gate_recent.py")) {
        Run-Cmd "dry_run_economic_gate_30m_50" $Python @("scripts\dry_run_economic_gate_recent.py", "--minutes", "30", "--limit", "50") 1200 | Out-Null
        Run-Cmd "dry_run_economic_gate_240m_500" $Python @("scripts\dry_run_economic_gate_recent.py", "--minutes", "240", "--limit", "500") 1800 | Out-Null
        Run-Cmd "dry_run_economic_gate_downstream_rf001" $Python @("scripts\dry_run_economic_gate_recent.py", "--minutes", "30", "--limit", "50", "--rf-threshold", "0.01") 1200 | Out-Null
    } else {
        Add-Issue "ERROR" "economic_gate" "scripts/dry_run_economic_gate_recent.py is missing."
    }

    # RF replay sanity if available
    if (Test-Path (Join-Path $ProjectRoot "scripts\replay_rf_artifact_on_predictions.py")) {
        Run-Cmd "replay_rf_validation_5000" $Python @("scripts\replay_rf_artifact_on_predictions.py", "--split", "validation", "--limit", "5000") 1200 | Out-Null
        Run-Cmd "replay_rf_test_5000" $Python @("scripts\replay_rf_artifact_on_predictions.py", "--split", "test", "--limit", "5000") 1200 | Out-Null
    } else {
        Add-Issue "WARN" "rf" "scripts/replay_rf_artifact_on_predictions.py is missing. RF replay sanity skipped."
    }

    # Solana and optional Helius
    if (Test-Path (Join-Path $ProjectRoot "scripts\probe_solana_pool_activity.py")) {
        Run-Cmd "solana_probe_raw_doge_usdc" $Python @(
            "scripts\probe_solana_pool_activity.py",
            "--address", "6z7NWpKoKhaXR5emiNH6SDx4N1m63gRQPQ6ynR5EUoF2",
            "--limit", "25"
        ) 1800 | Out-Null

        if ($UseHelius) {
            if ([string]::IsNullOrWhiteSpace($env:HELIUS_API_KEY)) {
                Add-Issue "WARN" "helius" "UseHelius is set but HELIUS_API_KEY is not set."
            } else {
                Run-Cmd "solana_probe_helius_5" $Python @(
                    "scripts\probe_solana_pool_activity.py",
                    "--address", "6z7NWpKoKhaXR5emiNH6SDx4N1m63gRQPQ6ynR5EUoF2",
                    "--limit", "25",
                    "--validate-with-helius",
                    "--helius-validation-limit", "5"
                ) 1800 | Out-Null
            }
        } else {
            Add-Step "helius_probe" "skipped" "UseHelius not set. No Helius credits consumed."
        }
    } else {
        Add-Issue "WARN" "solana" "scripts/probe_solana_pool_activity.py is missing."
    }

    # TabICL smoke if available
    if ($SkipTabICLSmoke) {
        Add-Step "tabicl_smoke" "skipped" "SkipTabICLSmoke was set"
    } elseif ((Test-Path $TabPython) -and (Test-Path (Join-Path $ProjectRoot "scripts\evaluate_tabicl_v2.py"))) {
        Run-Cmd "tabicl_gpu_smoke_2000" $TabPython @(
            "scripts\evaluate_tabicl_v2.py",
            "--max-rows", "2000",
            "--max-features", "30",
            "--context-size", "512",
            "--max-train-context-rows", "512",
            "--batch-size", "128",
            "--context-strategy", "positive_enriched",
            "--output-suffix", "sanity_smoke"
        ) 2400 | Out-Null
    } else {
        Add-Step "tabicl_smoke" "skipped" ".venv-tabicl or evaluate_tabicl_v2.py not found"
    }

    # Main runtime monitoring
    Monitor-Runtime

    # Final snapshots
    Invoke-Api "api_debug_storage_final" ($ApiBase + "/api/debug/storage") | Out-Null
    Invoke-Api "api_debug_collection_final" ($ApiBase + "/api/debug/collection") | Out-Null
    Invoke-Api "api_settings_effective_final" ($ApiBase + "/api/settings/effective") | Out-Null
    Invoke-Api "api_pipeline_audit_recent_120m_final" ($ApiBase + "/api/pipeline/audit/recent?minutes=120") | Out-Null
    Snapshot-Directory "data\audits" "audits_dir_snapshot_final"

    $summaryJson = Build-FinalReport

    # Zip results
    try {
        if (Test-Path $ZipPath) {
            Remove-Item $ZipPath -Force
        }

        Compress-Archive -Path (Join-Path $RunDir "*") -DestinationPath $ZipPath -Force
        Add-Step "zip_results" "ok" ("Compressed results to " + $ZipPath) $ZipPath
    } catch {
        Add-Issue "WARN" "zip" ("Failed to create ZIP: " + $_.Exception.Message)
    }

    Save-Json ([ordered]@{
        run_dir = $RunDir
        zip_path = $ZipPath
        summary_json = $summaryJson
        issues = $Global:Issues
        steps = $Global:Steps
    }) (Join-Path $RunDir "final_paths.json") 50

    Write-Host ""
    Write-Host "DONE" -ForegroundColor Green
    Write-Host ("Results folder: " + $RunDir)
    Write-Host ("ZIP: " + $ZipPath)

} catch {
    Add-Issue "ERROR" "fatal" $_.Exception.Message

    Save-Json ([ordered]@{
        fatal_error = $_.Exception.Message
        steps = $Global:Steps
        issues = $Global:Issues
    }) (Join-Path $RunDir "fatal_error.json") 50

    Write-Host ("FATAL ERROR: " + $_.Exception.Message) -ForegroundColor Red
} finally {
    try {
        Build-FinalReport | Out-Null
    } catch {}

    if ($Global:ServerStartedByScript -and ($null -ne $Global:ServerProcess)) {
        try {
            Write-Host "Stopping backend started by sanity script..."
            Stop-Process -Id $Global:ServerProcess.Id -Force -ErrorAction SilentlyContinue
        } catch {}
    }

    try {
        Stop-Transcript | Out-Null
    } catch {}

    Write-Host ""
    Write-Host "Sanity run finished."
    Write-Host ("Results: " + $RunDir)
    Write-Host ("ZIP: " + $ZipPath)
}