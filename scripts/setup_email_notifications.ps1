#Requires -Version 5.1

$ErrorActionPreference = "Stop"
$configDirectory = Join-Path $env:LOCALAPPDATA "GoldPredictor"
$configPath = Join-Path $configDirectory "launcher.json"

if (-not (Test-Path -LiteralPath $configPath)) {
    throw "Konfigurasi Gold Predictor belum tersedia. Jalankan setup_gold_predictor.ps1 terlebih dahulu."
}

$config = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
$sender = (Read-Host "Alamat Gmail pengirim").Trim()
$recipient = (Read-Host "Alamat email penerima [$sender]").Trim()
if (-not $recipient) {
    $recipient = $sender
}
if ($sender -notmatch "^[^@\s]+@[^@\s]+\.[^@\s]+$" -or $recipient -notmatch "^[^@\s]+@[^@\s]+\.[^@\s]+$") {
    throw "Alamat email pengirim atau penerima tidak valid."
}
$appPassword = Read-Host "Gmail App Password 16 karakter (disimpan terenkripsi)" -AsSecureString
$encryptedAppPassword = ConvertFrom-SecureString -SecureString $appPassword

$config | Add-Member -NotePropertyName email_sender -NotePropertyValue $sender -Force
$config | Add-Member -NotePropertyName email_recipient -NotePropertyValue $recipient -Force
$config | Add-Member -NotePropertyName email_app_password -NotePropertyValue $encryptedAppPassword -Force
$config | Add-Member -NotePropertyName email_smtp_host -NotePropertyValue "smtp.gmail.com" -Force
$config | Add-Member -NotePropertyName email_smtp_port -NotePropertyValue 587 -Force
$config | Add-Member -NotePropertyName notification_interval_seconds -NotePropertyValue 30 -Force
$whatsappProperties = @(
    "whatsapp_phone_number_id", "whatsapp_recipient", "whatsapp_template_name",
    "whatsapp_template_language", "whatsapp_graph_version", "whatsapp_access_token"
)
foreach ($property in $whatsappProperties) {
    $config.PSObject.Properties.Remove($property)
}
$config | ConvertTo-Json | Set-Content -LiteralPath $configPath -Encoding UTF8

Write-Host "Konfigurasi email tersimpan terenkripsi." -ForegroundColor Green
Write-Host "Restart STOP lalu START Gold Predictor untuk mengaktifkan notifikasi email."
