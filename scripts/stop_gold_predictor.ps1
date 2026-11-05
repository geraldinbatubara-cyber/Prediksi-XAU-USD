#Requires -Version 5.1

$ErrorActionPreference = "Stop"
$configDirectory = Join-Path $env:LOCALAPPDATA "GoldPredictor"
$pidPath = Join-Path $configDirectory "bridge.pid"
$paperWorkerPidPath = Join-Path $configDirectory "paper-worker.pid"
$notificationPidPath = Join-Path $configDirectory "notification.pid"
$shell = New-Object -ComObject WScript.Shell

try {
    $stopped = @()
    if (Test-Path -LiteralPath $pidPath) {
        $savedPid = Get-Content -LiteralPath $pidPath -ErrorAction Stop
        $processInfo = if ($savedPid) {
            Get-CimInstance Win32_Process -Filter "ProcessId = $savedPid" -ErrorAction SilentlyContinue
        }
        if ($processInfo -and $processInfo.CommandLine -like "*mt5_data_bridge.py*") {
            Stop-Process -Id $savedPid -ErrorAction Stop
            $stopped += "bridge MT5"
        }
        Remove-Item -LiteralPath $pidPath -Force -ErrorAction SilentlyContinue
    }

    if (Test-Path -LiteralPath $paperWorkerPidPath) {
        $savedWorkerPid = Get-Content -LiteralPath $paperWorkerPidPath -ErrorAction Stop
        $workerInfo = if ($savedWorkerPid) {
            Get-CimInstance Win32_Process -Filter "ProcessId = $savedWorkerPid" -ErrorAction SilentlyContinue
        }
        if ($workerInfo -and $workerInfo.CommandLine -like "*live_trading_job.py*") {
            Stop-Process -Id $savedWorkerPid -ErrorAction Stop
            $stopped += "worker Paper Trading"
        }
        Remove-Item -LiteralPath $paperWorkerPidPath -Force -ErrorAction SilentlyContinue
    }

    if (Test-Path -LiteralPath $notificationPidPath) {
        $savedNotificationPid = Get-Content -LiteralPath $notificationPidPath -ErrorAction Stop
        $notificationInfo = if ($savedNotificationPid) {
            Get-CimInstance Win32_Process -Filter "ProcessId = $savedNotificationPid" -ErrorAction SilentlyContinue
        }
        if ($notificationInfo -and $notificationInfo.CommandLine -like "*email_notifications.py*") {
            Stop-Process -Id $savedNotificationPid -ErrorAction Stop
            $stopped += "notifikasi email"
        }
        Remove-Item -LiteralPath $notificationPidPath -Force -ErrorAction SilentlyContinue
    }

    $message = if ($stopped.Count) {
        "Proses dihentikan: $($stopped -join ', '). MT5 tetap terbuka."
    } else {
        "Gold Predictor tidak sedang berjalan."
    }
    $null = $shell.Popup($message, 6, "Gold Predictor", 64)
} catch {
    $null = $shell.Popup($_.Exception.Message, 8, "Gold Predictor gagal dihentikan", 16)
    exit 1
}
