#!/usr/bin/env pwsh
# run_hot.ps1 — Windows equivalent of run_hot.sh
# Usage:
#   .\run_hot.ps1           # default port 8000
#   .\run_hot.ps1 9000      # custom port

param(
    [int]$Port = 8000
)

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path

# Kill any process already listening on $Port
$existing = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($existing) {
    $pids = $existing | Select-Object -ExpandProperty OwningProcess -Unique
    foreach ($p in $pids) {
        Write-Host "[hot] stopping existing process on :$Port -> PID $p"
        Stop-Process -Id $p -Force -ErrorAction SilentlyContinue
    }
    Start-Sleep -Milliseconds 300
}

Write-Host "[hot] starting dev server on :$Port"
Set-Location $Root
python "$Root\..\src\dev_server.py" $Port
