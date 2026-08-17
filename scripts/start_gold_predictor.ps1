#Requires -Version 5.1

$ErrorActionPreference = "Stop"
$configDirectory = Join-Path $env:LOCALAPPDATA "GoldPredictor"
$configPath = Join-Path $configDirectory "launcher.json"
$pidPath = Join-Path $configDirectory "bridge.pid"
$notificationPidPath = Join-Path $configDirectory "notification.pid"
$logDirectory = Join-Path $configDirectory "logs"
$dashboardUrl = "https://goldpredictor.streamlit.app/"

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
    if ($config.dashboard_url -ne $dashboardUrl) {
        $config.dashboard_url = $dashboardUrl
        $config | ConvertTo-Json | Set-Content -LiteralPath $configPath -Encoding UTF8
    }
    $pythonPath = Join-Path $config.project_root ".venv\Scripts\python.exe"
    $bridgePath = Join-Path $config.project_root "scripts\mt5_data_bridge.py"
    $notificationPath = Join-Path $config.project_root "gold_forecast\email_notifications.py"
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

    $notificationConfigured = (
        $config.PSObject.Properties.Name -contains "email_app_password" -and
        $config.PSObject.Properties.Name -contains "email_sender" -and
        $config.PSObject.Properties.Name -contains "email_recipient"
    )
    $notificationRunning = $false
    if (Test-Path -LiteralPath $notificationPidPath) {
        $savedNotificationPid = Get-Content -LiteralPath $notificationPidPath -ErrorAction SilentlyContinue
        $savedNotificationProcess = if ($savedNotificationPid) {
            Get-CimInstance Win32_Process -Filter "ProcessId = $savedNotificationPid" -ErrorAction SilentlyContinue
        }
        if ($savedNotificationProcess -and $savedNotificationProcess.CommandLine -like "*email_notifications.py*") {
            $notificationRunning = $true
        } else {
            Remove-Item -LiteralPath $notificationPidPath -Force -ErrorAction SilentlyContinue
        }
    }

    if ($notificationConfigured -and -not $notificationRunning) {
        $secureSupabaseSecret = ConvertTo-SecureString $config.supabase_secret
        $plainSupabaseSecret = [Net.NetworkCredential]::new("", $secureSupabaseSecret).Password
        $secureEmailPassword = ConvertTo-SecureString $config.email_app_password
        $plainEmailPassword = [Net.NetworkCredential]::new("", $secureEmailPassword).Password
        $env:SUPABASE_URL = [string]$config.supabase_url
        $env:SUPABASE_SERVICE_ROLE_KEY = $plainSupabaseSecret
        $env:EMAIL_APP_PASSWORD = $plainEmailPassword
        $env:EMAIL_SENDER = [string]$config.email_sender
        $env:EMAIL_RECIPIENT = [string]$config.email_recipient
        $env:EMAIL_SMTP_HOST = [string]$config.email_smtp_host
        $env:EMAIL_SMTP_PORT = [string]$config.email_smtp_port
        $notificationInterval = if ($config.notification_interval_seconds) {
            [int]$config.notification_interval_seconds
        } else {
            30
        }
        $timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
        $notificationStdout = Join-Path $logDirectory "notification-$timestamp.log"
        $notificationStderr = Join-Path $logDirectory "notification-$timestamp-error.log"
        $notificationArguments = @(
            "-u",
            "`"$notificationPath`"",
            "--interval", [string]$notificationInterval
        )
        $notificationProcess = Start-Process `
            -FilePath $pythonPath `
            -ArgumentList $notificationArguments `
            -WorkingDirectory $config.project_root `
            -WindowStyle Hidden `
            -RedirectStandardOutput $notificationStdout `
            -RedirectStandardError $notificationStderr `
            -PassThru
        $notificationProcess.Id | Set-Content -LiteralPath $notificationPidPath -Encoding ASCII
        Remove-Item Env:EMAIL_APP_PASSWORD -ErrorAction SilentlyContinue
        Remove-Item Env:SUPABASE_SERVICE_ROLE_KEY -ErrorAction SilentlyContinue
        Remove-Variable plainSupabaseSecret, secureSupabaseSecret, plainEmailPassword, secureEmailPassword -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 2
        if ($notificationProcess.HasExited) {
            $errorDetail = Get-Content -LiteralPath $notificationStderr -Raw -ErrorAction SilentlyContinue
            throw "Dispatcher email gagal berjalan. $errorDetail"
        }
        $notificationRunning = $true
    }

    Start-Process $dashboardUrl | Out-Null
    $notificationStatus = if ($notificationRunning) {
        "Notifikasi email aktif."
    } elseif ($notificationConfigured) {
        "Notifikasi email belum aktif."
    } else {
        "Notifikasi email belum dikonfigurasi."
    }
    Show-LauncherMessage "MT5 dan bridge aktif. $notificationStatus Dashboard dibuka di browser."
} catch {
    Show-LauncherMessage $_.Exception.Message "Gold Predictor gagal dimulai" 16
    exit 1
}
