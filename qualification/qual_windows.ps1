param(
    [ValidateSet("1.0", "0.5", "0.25")]
    [string]$Interval
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$intervalDirectories = [ordered]@{
    "1.0" = "1s"
    "0.5" = "0.5s"
    "0.25" = "0.25s"
}

function Invoke-BenchmarkCase {
    param(
        [string]$Mode,
        [string]$Number,
        [string]$OutputDirectory
    )

    Write-Host "Running pair $Number telemetry $($Mode.ToUpper())..."
    $arguments = @(
        "--ui", "none",
        "--tests", "llm",
        "--llm-models", "qwen3.5:4b-q4_K_M",
        "--max-prompt-tokens", "2048",
        "--warmup", "2",
        "--runs", "3"
    )
    if ($Mode -eq "on") {
        $arguments += "--memory-telemetry"
    }
    $arguments += @("--out", (Join-Path $OutputDirectory "$Mode-$Number.json"))

    & "$Root\run_bench.bat" @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Pair $Number telemetry $Mode failed with exit code $LASTEXITCODE."
    }
}

function Invoke-Interval {
    param([string]$Value)

    $env:LOCAL_AI_BENCH_MEMORY_INTERVAL_SEC = $Value
    $outputDirectory = Join-Path $Root (
        "results\qualification\rtx-5090-windows\" + $intervalDirectories[$Value]
    )
    New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null

    foreach ($pair in 1..20) {
        $number = $pair.ToString("00")
        $offFile = Join-Path $outputDirectory "off-$number.json"
        $onFile = Join-Path $outputDirectory "on-$number.json"
        $offExists = Test-Path -LiteralPath $offFile
        $onExists = Test-Path -LiteralPath $onFile

        if ($offExists -and $onExists) {
            Write-Host "Pair $number already complete - skipping."
            continue
        }
        if ($offExists -or $onExists) {
            throw "Partial pair $number exists in $outputDirectory. Remove or relocate it, then retry."
        }

        $order = if ($pair % 2 -eq 1) { @("off", "on") } else { @("on", "off") }
        Invoke-BenchmarkCase -Mode $order[0] -Number $number -OutputDirectory $outputDirectory
        Start-Sleep -Seconds 30
        Invoke-BenchmarkCase -Mode $order[1] -Number $number -OutputDirectory $outputDirectory
        if ($pair -lt 20) {
            Start-Sleep -Seconds 30
        }
    }
    Write-Host "Qualification interval $Value completed."
}

$selectedIntervals = if ($Interval) { @($Interval) } else { @($intervalDirectories.Keys) }
for ($index = 0; $index -lt $selectedIntervals.Count; $index++) {
    Invoke-Interval $selectedIntervals[$index]
    if ($index -lt $selectedIntervals.Count - 1) {
        Start-Sleep -Seconds 30
    }
}
Write-Host "Selected Windows qualification intervals completed."
