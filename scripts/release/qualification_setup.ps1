param(
    [Parameter(Mandatory = $true)][string]$Target,
    [Parameter(Mandatory = $true)][ValidateSet("llamacpp", "vllm")][string]$Engine,
    [Parameter(Mandatory = $true)][string]$EvidenceDir,
    [Parameter(Mandatory = $true)][string]$SetupPath
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $EvidenceDir | Out-Null
$log = Join-Path $EvidenceDir "setup.log"
$statusPath = Join-Path $EvidenceDir "setup-status.json"
$startedAt = [DateTime]::UtcNow.ToString("o")
[ordered]@{
    schema = "qualification-setup-v1"
    target = $Target
    status = "running"
    exit_code = $null
    started_at = $startedAt
    finished_at = $null
    log = "setup.log"
} | ConvertTo-Json | Set-Content -Encoding UTF8 -Path $statusPath
Add-Content -Path $log -Value "`n=== qualification setup attempt $startedAt ==="

& $SetupPath --qualification $Engine --qualification-target $Target 2>&1 |
    Tee-Object -FilePath $log -Append
$setupExit = $LASTEXITCODE
$status = if ($setupExit -eq 0) { "passed" } else { "failed" }
[ordered]@{
    schema = "qualification-setup-v1"
    target = $Target
    status = $status
    exit_code = $setupExit
    started_at = $startedAt
    finished_at = [DateTime]::UtcNow.ToString("o")
    log = "setup.log"
} | ConvertTo-Json | Set-Content -Encoding UTF8 -Path $statusPath

if ($setupExit -ne 0) {
    [Console]::Error.WriteLine("Qualification setup failed; evidence saved to $log")
}
exit $setupExit
