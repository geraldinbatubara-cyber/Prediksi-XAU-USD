#Requires -Version 5.1

$ErrorActionPreference = "Stop"
$configDirectory = Join-Path $env:LOCALAPPDATA "GoldPredictor"
$pidPath = Join-Path $configDirectory "bridge.pid"
$shell = New-Object -ComObject WScript.Shell

try {
    if (-not (Test-Path -LiteralPath $pidPath)) {
        $null = $shell.Popup("Bridge Gold Predictor tidak sedang berjalan.", 6, "Gold Predictor", 64)
        exit 0
    }
    $savedPid = Get-Content -LiteralPath $pidPath -ErrorAction Stop
    $processInfo = if ($savedPid) {
        Get-CimInstance Win32_Process -Filter "ProcessId = $savedPid" -ErrorAction SilentlyContinue
    }
    if ($processInfo -and $processInfo.CommandLine -like "*mt5_data_bridge.py*") {
        $process = Get-Process -Id $savedPid -ErrorAction Stop
        Stop-Process -Id $process.Id -ErrorAction Stop
        $process.WaitForExit(5000)
    }
    Remove-Item -LiteralPath $pidPath -Force -ErrorAction SilentlyContinue
    $null = $shell.Popup("Bridge dihentikan. MT5 tetap terbuka.", 6, "Gold Predictor", 64)
} catch {
    $null = $shell.Popup($_.Exception.Message, 8, "Gold Predictor gagal dihentikan", 16)
    exit 1
}
