param([string]$Target = "")

$text = Get-Content -Raw (Join-Path $PSScriptRoot "qualification_targets.py")
$ids = @()
foreach ($match in [regex]::Matches($text, '"id": "([^"]+)"')) {
    $ids += $match.Groups[1].Value
}

if ($Target) {
    if ($ids -contains $Target) {
        exit 0
    }
    exit 1
}

foreach ($id in $ids) {
    Write-Output $id
}
