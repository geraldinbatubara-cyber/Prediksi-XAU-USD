#Requires -Version 5.1

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$configDirectory = Join-Path $env:LOCALAPPDATA "GoldPredictor"
$configPath = Join-Path $configDirectory "launcher.json"
$dashboardUrl = "https://klmshreuteappzrrjyp3kuh.streamlit.app/"

$mt5Candidates = @(
    (Join-Path $env:ProgramFiles "MetaTrader 5\terminal64.exe"),
    (Join-Path ${env:ProgramFiles(x86)} "MetaTrader 5\terminal64.exe")
) | Where-Object { $_ -and (Test-Path -LiteralPath $_) }
$defaultMt5Path = $mt5Candidates | Select-Object -First 1

$mt5Prompt = "Lokasi terminal64.exe MT5"
if ($defaultMt5Path) {
    $mt5Prompt += " [$defaultMt5Path]"
}
$mt5Path = Read-Host $mt5Prompt
if (-not $mt5Path) {
    $mt5Path = $defaultMt5Path
}
if (-not $mt5Path -or -not (Test-Path -LiteralPath $mt5Path)) {
    throw "terminal64.exe tidak ditemukan. Masukkan lokasi instalasi MT5 yang benar."
}

$supabaseUrl = (Read-Host "Supabase Project URL (https://PROJECT.supabase.co)").Trim().TrimEnd("/")
$supabaseUrl = $supabaseUrl -replace "/rest/v1$", ""
if ($supabaseUrl -notmatch "^https://.+\.supabase\.co$") {
    throw "Format Supabase Project URL tidak valid."
}

$secret = Read-Host "Supabase secret key (disimpan terenkripsi untuk user Windows ini)" -AsSecureString
$encryptedSecret = ConvertFrom-SecureString -SecureString $secret

New-Item -ItemType Directory -Path $configDirectory -Force | Out-Null
[ordered]@{
    project_root = $projectRoot
    mt5_path = $mt5Path
    dashboard_url = $dashboardUrl
    supabase_url = $supabaseUrl
    supabase_secret = $encryptedSecret
    symbol = "XAUUSD"
    interval_seconds = 60
} | ConvertTo-Json | Set-Content -LiteralPath $configPath -Encoding UTF8

function New-GoldPredictorShortcut {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$ScriptPath,
        [Parameter(Mandatory = $true)][string]$Description
    )

    $desktop = [Environment]::GetFolderPath("Desktop")
    $shortcutPath = Join-Path $desktop "$Name.lnk"
    $shell = New-Object -ComObject WScript.Shell
    $shortcut = $shell.CreateShortcut($shortcutPath)
    $shortcut.TargetPath = (Get-Command powershell.exe).Source
    $shortcut.Arguments = "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$ScriptPath`""
    $shortcut.WorkingDirectory = $projectRoot
    $shortcut.Description = $Description
    $shortcut.IconLocation = "$mt5Path,0"
    $shortcut.Save()
}

New-GoldPredictorShortcut `
    -Name "START Gold Predictor" `
    -ScriptPath (Join-Path $PSScriptRoot "start_gold_predictor.ps1") `
    -Description "Jalankan MT5, bridge Supabase, dan dashboard Gold Predictor"
New-GoldPredictorShortcut `
    -Name "STOP Gold Predictor" `
    -ScriptPath (Join-Path $PSScriptRoot "stop_gold_predictor.ps1") `
    -Description "Hentikan bridge Gold Predictor tanpa menutup MT5"

Write-Host "Setup selesai." -ForegroundColor Green
Write-Host "Shortcut START dan STOP Gold Predictor sudah dibuat di Desktop."
Write-Host "Secret hanya dapat didekripsi oleh user Windows ini pada komputer ini."
