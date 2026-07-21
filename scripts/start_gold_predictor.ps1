#Requires -Version 5.1

$ErrorActionPreference = "Stop"
$configDirectory = Join-Path $env:LOCALAPPDATA "GoldPredictor"
$configPath = Join-Path $configDirectory "launcher.json"
$pidPath = Join-Path $configDirectory "bridge.pid"
$logDirectory = Join-Path $configDirectory "logs"

function Show-LauncherMessage {
    param([string]$Message, [string]$Title = "Gold Predictor", [int]$Icon = 64)
    $shell = New-Object -ComObject WScript.Shell
    $null = $shell.Popup($Message, 8, $Title, $Icon)
}

try {
    if (-not (Test-Path -LiteralPath $configPath)) {
        throw "Konfigurasi belum tersedia. Jalankan scripts\setup_gold_predictor.ps1 satu kali."
    }
    $config = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
    $pythonPath = Join-Path $config.project_root ".venv\Scripts\python.exe"
    $bridgePath = Join-Path $config.project_root "scripts\mt5_data_bridge.py"
    if (-not (Test-Path -LiteralPath $pythonPath)) {
        throw "Python virtual environment tidak ditemukan: $pythonPath"
    }
    if (-not (Test-Path -LiteralPath $config.mt5_path)) {
        throw "MT5 tidak ditemukan: $($config.mt5_path)"
    }

    $mt5ProcessName = [IO.Path]::GetFileNameWithoutExtension($config.mt5_path)
    if (-not (Get-Process -Name $mt5ProcessName -ErrorAction SilentlyContinue)) {
        Start-Process -FilePath $config.mt5_path | Out-Null
        Start-Sleep -Seconds 8
    }

    $bridgeRunning = $false
    if (Test-Path -LiteralPath $pidPath) {
        $savedPid = Get-Content -LiteralPath $pidPath -ErrorAction SilentlyContinue
        $savedProcess = if ($savedPid) {
            Get-CimInstance Win32_Process -Filter "ProcessId = $savedPid" -ErrorAction SilentlyContinue
        }
        if ($savedProcess -and $savedProcess.CommandLine -like "*mt5_data_bridge.py*") {
            $bridgeRunning = $true
        } else {
            Remove-Item -LiteralPath $pidPath -Force -ErrorAction SilentlyContinue
        }
    }

    if (-not $bridgeRunning) {
        $secureSecret = ConvertTo-SecureString $config.supabase_secret
        $plainSecret = [Net.NetworkCredential]::new("", $secureSecret).Password
        $env:SUPABASE_URL = [string]$config.supabase_url
        $env:SUPABASE_SERVICE_ROLE_KEY = $plainSecret
        New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
        $timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
        $stdoutPath = Join-Path $logDirectory "bridge-$timestamp.log"
        $stderrPath = Join-Path $logDirectory "bridge-$timestamp-error.log"
        $arguments = @(
            "-u",
            "`"$bridgePath`"",
            "--symbol", [string]$config.symbol,
            "--interval", [string]$config.interval_seconds,
            "--publish-supabase"
        )
        $process = Start-Process `
            -FilePath $pythonPath `
            -ArgumentList $arguments `
            -WorkingDirectory $config.project_root `
            -WindowStyle Hidden `
            -RedirectStandardOutput $stdoutPath `
            -RedirectStandardError $stderrPath `
            -PassThru
        $process.Id | Set-Content -LiteralPath $pidPath -Encoding ASCII
        Remove-Variable plainSecret, secureSecret -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 5
        if ($process.HasExited) {
            $errorDetail = Get-Content -LiteralPath $stderrPath -Raw -ErrorAction SilentlyContinue
            throw "Bridge gagal berjalan. $errorDetail"
        }
    }

    Start-Process ([string]$config.dashboard_url) | Out-Null
    Show-LauncherMessage "MT5 dan bridge aktif. Dashboard dibuka di browser."
} catch {
    Show-LauncherMessage $_.Exception.Message "Gold Predictor gagal dimulai" 16
    exit 1
}
