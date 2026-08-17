# Notifikasi WhatsApp Paper Live Trading

Notifikasi dikirim untuk dua perubahan posisi yang tersimpan di Supabase:

- `ENTRY`: posisi baru berubah menjadi `OPEN`.
- `EXIT`: posisi berubah dari `OPEN` menjadi `CLOSED`, termasuk TP, CL/SL,
  break-even, dan ATR trailing.

Notifikasi lama tidak dibuat ulang. Kunci unik `strategy_id + position_id + event`
mencegah pesan ganda ketika Streamlit melakukan rerun.

## Persiapan Meta

Siapkan WhatsApp Cloud API pada akun Meta Business:

1. Phone Number ID.
2. Permanent access token.
3. Nomor penerima dalam format kode negara tanpa tanda `+`.
4. Template pesan approved dengan satu parameter body untuk notifikasi yang
   dimulai oleh aplikasi. Pesan text tanpa template hanya cocok untuk pengujian
   dalam conversation window WhatsApp.

Jangan menyimpan access token di GitHub, Streamlit Secrets, atau file `.env`.

## Aktivasi

1. Buka Supabase SQL Editor dan jalankan seluruh isi
   `supabase/whatsapp_notifications.sql`.
2. Dari PowerShell proyek, jalankan:

   ```powershell
   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\setup_whatsapp_notifications.ps1
   ```

3. Klik `STOP Gold Predictor`, lalu klik `START Gold Predictor`.
4. Popup START harus menampilkan `Notifikasi WhatsApp aktif`.

Log dispatcher tersimpan di:

```text
%LOCALAPPDATA%\GoldPredictor\logs\notification-*.log
```

Status antrean dapat diperiksa dari Supabase SQL Editor:

```sql
select
    notification_id,
    strategy_id,
    position_id,
    notification_type,
    status,
    attempt_count,
    last_error,
    created_at,
    sent_at
from public.paper_notification_outbox
order by notification_id desc
limit 50;
```

## Batas Operasional

Dispatcher tidak bergantung pada browser Streamlit setelah event tersedia di
Supabase. Namun, pada arsitektur lokal saat ini, MT5, laptop, dan START Gold
Predictor tetap harus aktif agar feed broker terus diperbarui. Notifikasi tidak
menggantikan pemeriksaan harga, spread, atau eksekusi broker.
