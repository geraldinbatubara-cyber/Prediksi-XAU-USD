#Requires -Version 5.1

$ErrorActionPreference = "Stop"
$configDirectory = Join-Path $env:LOCALAPPDATA "GoldPredictor"
$pidPath = Join-Path $configDirectory "bridge.pid"
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

    if (Test-Path -LiteralPath $notificationPidPath) {
        $savedNotificationPid = Get-Content -LiteralPath $notificationPidPath -ErrorAction Stop
        $notificationInfo = if ($savedNotificationPid) {
            Get-CimInstance Win32_Process -Filter "ProcessId = $savedNotificationPid" -ErrorAction SilentlyContinue
        }
        if ($notificationInfo -and $notificationInfo.CommandLine -like "*whatsapp_notifications.py*") {
            Stop-Process -Id $savedNotificationPid -ErrorAction Stop
            $stopped += "notifikasi WhatsApp"
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
