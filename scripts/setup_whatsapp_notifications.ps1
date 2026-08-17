#Requires -Version 5.1

$ErrorActionPreference = "Stop"
$configDirectory = Join-Path $env:LOCALAPPDATA "GoldPredictor"
$configPath = Join-Path $configDirectory "launcher.json"

if (-not (Test-Path -LiteralPath $configPath)) {
    throw "Konfigurasi Gold Predictor belum tersedia. Jalankan setup_gold_predictor.ps1 terlebih dahulu."
}

$config = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
$phoneNumberId = (Read-Host "WhatsApp Phone Number ID").Trim()
$recipient = (Read-Host "Nomor penerima dengan kode negara, tanpa tanda + (contoh 62812...)").Trim().Replace("+", "")
$templateName = (Read-Host "Nama template WhatsApp approved [kosong untuk pesan text saat pengujian]").Trim()
$templateLanguage = (Read-Host "Kode bahasa template [id]").Trim()
if (-not $templateLanguage) {
    $templateLanguage = "id"
}
$graphVersion = (Read-Host "Versi Meta Graph API [v23.0]").Trim()
if (-not $graphVersion) {
    $graphVersion = "v23.0"
}
$token = Read-Host "WhatsApp permanent access token (disimpan terenkripsi)" -AsSecureString
$encryptedToken = ConvertFrom-SecureString -SecureString $token

$config | Add-Member -NotePropertyName whatsapp_phone_number_id -NotePropertyValue $phoneNumberId -Force
$config | Add-Member -NotePropertyName whatsapp_recipient -NotePropertyValue $recipient -Force
$config | Add-Member -NotePropertyName whatsapp_template_name -NotePropertyValue $templateName -Force
$config | Add-Member -NotePropertyName whatsapp_template_language -NotePropertyValue $templateLanguage -Force
$config | Add-Member -NotePropertyName whatsapp_graph_version -NotePropertyValue $graphVersion -Force
$config | Add-Member -NotePropertyName whatsapp_access_token -NotePropertyValue $encryptedToken -Force
$config | Add-Member -NotePropertyName notification_interval_seconds -NotePropertyValue 30 -Force
$config | ConvertTo-Json | Set-Content -LiteralPath $configPath -Encoding UTF8

Write-Host "Konfigurasi WhatsApp tersimpan terenkripsi." -ForegroundColor Green
Write-Host "Jalankan whatsapp_notifications.sql di Supabase, lalu restart START Gold Predictor."
