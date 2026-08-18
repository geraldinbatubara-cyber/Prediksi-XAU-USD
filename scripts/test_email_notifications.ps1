#Requires -Version 5.1

$ErrorActionPreference = "Stop"
$configPath = Join-Path $env:LOCALAPPDATA "GoldPredictor\launcher.json"

if (-not (Test-Path -LiteralPath $configPath)) {
    throw "Konfigurasi Gold Predictor belum tersedia."
}

$config = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
$pythonPath = Join-Path $config.project_root ".venv\Scripts\python.exe"
$dispatcherPath = Join-Path $config.project_root "gold_forecast\email_notifications.py"

try {
    $secureSupabaseSecret = ConvertTo-SecureString $config.supabase_secret
    $secureEmailPassword = ConvertTo-SecureString $config.email_app_password
    $plainEmailPassword = [Net.NetworkCredential]::new("", $secureEmailPassword).Password -replace "\s", ""
    Write-Host "Preflight: sender=$($config.email_sender), App Password length=$($plainEmailPassword.Length)"
    if ($plainEmailPassword.Length -ne 16) {
        throw "App Password harus tepat 16 karakter setelah spasi dihapus."
    }
    $env:SUPABASE_URL = [string]$config.supabase_url
    $env:SUPABASE_SERVICE_ROLE_KEY = [Net.NetworkCredential]::new("", $secureSupabaseSecret).Password
    $env:EMAIL_APP_PASSWORD = $plainEmailPassword
    $env:EMAIL_SENDER = [string]$config.email_sender
    $env:EMAIL_RECIPIENT = [string]$config.email_recipient
    $env:EMAIL_SMTP_HOST = [string]$config.email_smtp_host
    $env:EMAIL_SMTP_PORT = [string]$config.email_smtp_port

    & $pythonPath $dispatcherPath --test-email
    if ($LASTEXITCODE -ne 0) {
        throw "Tes SMTP gagal dengan exit code $LASTEXITCODE."
    }
} finally {
    Remove-Item Env:SUPABASE_SERVICE_ROLE_KEY -ErrorAction SilentlyContinue
    Remove-Item Env:EMAIL_APP_PASSWORD -ErrorAction SilentlyContinue
    Remove-Variable secureSupabaseSecret, secureEmailPassword, plainEmailPassword -ErrorAction SilentlyContinue
}
