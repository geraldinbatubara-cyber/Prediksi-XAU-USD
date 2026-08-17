# Notifikasi Email Paper Live Trading

Notifikasi dikirim untuk dua perubahan posisi yang tersimpan di Supabase:

- `ENTRY`: posisi baru berubah menjadi `OPEN`.
- `EXIT`: posisi berubah dari `OPEN` menjadi `CLOSED`, termasuk TP, CL/SL,
  break-even, dan ATR trailing.

Notifikasi lama tidak dibuat ulang. Kunci unik `strategy_id + position_id + event`
mencegah pesan ganda ketika Streamlit melakukan rerun.

## Persiapan Gmail

Siapkan akun Gmail pengirim:

1. Aktifkan Verifikasi 2 Langkah pada akun Google.
2. Buat App Password khusus untuk Gold Predictor.
3. Tentukan alamat email penerima; alamat pengirim dan penerima boleh sama.

Jangan memakai password utama Gmail. App Password disimpan terenkripsi oleh
Windows dan tidak disimpan di repository.

## Aktivasi

1. Buka Supabase SQL Editor dan jalankan seluruh isi
   `supabase/whatsapp_notifications.sql` jika tabel antrean belum tersedia.
   Nama file SQL lama dipertahankan agar instalasi Supabase yang sudah berjalan
   tidak perlu diubah.
2. Dari PowerShell proyek, jalankan:

   ```powershell
   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\setup_email_notifications.ps1
   ```

3. Klik `STOP Gold Predictor`, lalu klik `START Gold Predictor`.
4. Popup START harus menampilkan `Notifikasi email aktif`.

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
