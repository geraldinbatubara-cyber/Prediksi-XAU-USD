# Prediksi XAU/USD

Dashboard estimasi harga emas untuk hari bursa berikutnya dan tujuh hari ke depan.

**Aplikasi live:** https://goldpredictor.streamlit.app/

## Menjalankan

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Data MVP memakai kontrak berjangka emas COMEX (`GC=F`) dalam USD per troy ounce.
Hasil model adalah estimasi statistik, bukan saran investasi.

## Data Broker Read-only

Halaman `Data Broker` disiapkan untuk mengaudit feed XAUUSD dari akun demo atau real MT5.
Integrasi ini hanya membaca bid, ask, spread, serta candle M1 dan tidak mengirim order.
Data broker tidak menggantikan `GC=F` sebelum hasil audit sumber, timestamp, spread,
dan contract size dinyatakan memadai. Bridge tetap read-only terhadap MT5 dan tidak
mempublikasikan nomor akun, nama pemilik, balance, equity, atau password broker.

Pada komputer Windows yang sudah menjalankan terminal MT5 dan login ke akun demo:

```powershell
pip install MetaTrader5
.venv\Scripts\python.exe scripts\mt5_data_bridge.py --symbol XAUUSD
```

Nama simbol harus mengikuti Market Watch broker dan dapat berbeda dari `XAUUSD`.
Gunakan `--once` untuk mengambil satu snapshot. Output disimpan di
`data/broker/xauusd_m1.csv` dan `data/broker/latest_quote.csv`; kedua file ini
diabaikan Git. Streamlit Cloud dapat membaca URL CSV read-only melalui secrets:

```toml
[broker_data]
bars_url = "https://storage.example/xauusd_m1.csv"
quote_url = "https://storage.example/latest_quote.csv"
```

### Feed MT5 ke Streamlit Cloud dengan Supabase

1. Buat proyek Supabase Free dan jalankan isi `supabase/broker_feed.sql` pada SQL Editor.
2. Pada terminal bridge, set secret hanya untuk sesi PowerShell saat ini:

```powershell
$env:SUPABASE_URL = "https://PROJECT.supabase.co"
$env:SUPABASE_SERVICE_ROLE_KEY = "SERVICE_ROLE_SECRET"
& ".\.venv\Scripts\python.exe" scripts\mt5_data_bridge.py --symbol XAUUSD --publish-supabase
```

3. Pada Streamlit Cloud Secrets, tambahkan key baca saja:

```toml
[supabase_broker]
url = "https://PROJECT.supabase.co"
publishable_key = "sb_publishable_..."
symbol = "XAUUSD"
```

Jangan menyimpan `SUPABASE_SERVICE_ROLE_KEY` di source code, `.streamlit/secrets.toml`,
Streamlit Cloud, atau GitHub. Dashboard hanya memerlukan publishable/anon key karena
RLS pada tabel broker memberikan izin `select` dan menolak penulisan publik.

### Launcher Windows Satu Klik

Setelah feed Supabase berhasil diuji, jalankan setup berikut satu kali:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\setup_gold_predictor.ps1
```

Setup meminta lokasi MT5, Project URL, dan secret key Supabase. Secret disimpan
terenkripsi dengan DPAPI dan hanya dapat dibuka oleh user Windows yang sama pada
komputer yang sama. Dua shortcut dibuat di Desktop:

- `START Gold Predictor` membuka MT5 bila perlu, menjalankan bridge secara
  tersembunyi, lalu membuka dashboard.
- `STOP Gold Predictor` menghentikan bridge tanpa menutup MT5.

Untuk proyek Supabase yang sudah ada, jalankan kembali isi
`supabase/broker_feed.sql` agar tabel status terminal tersedia. Panel kesiapan
akun real tetap bersifat read-only dan seluruh order dikonfirmasi manual di MT5.

## Metodologi

Model regresi ridge memakai harga terkini, lag harga, return, moving average, dan
volatilitas. Evaluasi memakai 20% data terbaru sebagai backtest berurutan.

Model 2 memakai gradient boosting dan menambahkan DXY, yield Treasury 10 tahun,
minyak, VIX, serta perak. Satu model direct dilatih untuk setiap horizon T+1 sampai
T+7. Dashboard membandingkan Model 1 dan Model 2 pada horizon T+1 yang setara.

## Simulasi Trading

Validasi terbaru membedakan tiga jenis pengujian. `Optimizer OOS` memakai OHLC
harian, sedangkan `Exact Broker-Aware OOS` mempertahankan sinyal dan aturan
harian lalu memakai candle M1 MT5 hanya untuk harga serta urutan eksekusi.

`Optimizer v1 Robustness Test` menguji Exact v1 pada sembilan kombinasi spread
aktual/1.5x/2x dan slippage adverse 2/4/6 points per sisi. Tab tersebut juga
menampilkan performa bulanan, bootstrap Monte Carlo 10.000 lintasan, serta audit
basis harga `GC=F` versus `XAUUSD MT5`. Ambang kandidat real-money ditetapkan
pada profit factor minimal 1.30, max drawdown maksimal 10%, dan pertumbuhan tetap
positif setelah stress biaya. Hasil saat ini belum memenuhi ambang tersebut,
sehingga v1 hanya digunakan untuk paper trading.

Tab `Simulasi` hanya memuat Optimizer v1, Optimizer v1 OOS, Optimizer v1 Exact
Broker-Aware OOS, dan Optimizer v1 Robustness Test. Tab `Live Trading` hanya
membuka posisi baru dengan Optimizer v1. Posisi v10 lama, bila masih ada, tetap
dipantau sampai TP/CL tetapi v10 tidak dapat membuka posisi baru. Ringkasan
eksperimen yang dikeluarkan dari deployment tersedia di
[`docs/archived_experiments.md`](docs/archived_experiments.md).

Asumsi dasar: equity awal USD 1.000, target tiap fase naik 20% dari start equity
fase tersebut, maksimal 8 BUY dan 10 SELL. Saat target fase tercapai, semua
posisi terbuka ditutup dan hasilnya dicatat sebagai fase selesai. Fase berikutnya
dimulai dari equity close-all fase sebelumnya. Proses berulang sampai batas data
uji 30 Juni 2026. Swap BUY dihitung USD 0.2 per hari per 0.01 lot, sehingga BUY
0.02 lot dikenakan USD 0.4 per hari. Swap SELL dianggap USD 0.0.

Karena data yang dipakai adalah OHLC harian `GC=F`, bukan data tick broker, jika
TP dan SL sama-sama tersentuh dalam candle harian yang sama maka simulasi memakai
asumsi konservatif: SL dianggap tersentuh lebih dulu. Dashboard juga menandai
kapan equity berada pada titik terendah dan tertinggi.

Bagian `Strategi Terbaik Optimizer` menguji strategi trend, breakout, dan
pullback berbasis indikator teknikal dari data OHLC historis. Periode uji
optimizer dikunci pada 1 Januari 2025 sampai 30 Juni 2026. Ranking strategi
memprioritaskan target equity USD 1.200 tercapai, tanggal target lebih cepat,
equity akhir lebih tinggi, dan drawdown lebih rendah.

Bagian `Strategi Terbaik v.2` mempertahankan acuan optimizer tersebut, lalu
mengeksplorasi variasi sinyal dengan lot dinamis 0.01 sampai 0.02. Lot 0.02
hanya dipakai saat confidence sinyal melewati cutoff kandidat; sinyal dengan
confidence lebih rendah tetap memakai 0.01 lot. Ranking tetap memakai periode
uji 1 Januari 2025 sampai 30 Juni 2026 dan target equity USD 1.200.

## Monitoring Estimasi

Tab `Monitoring Model 2` dan `Monitoring Model 1` membandingkan estimasi besok
yang disimpan pukul 23:59 WIT dengan harga `Open` candle intraday `GC=F`
pertama pada atau setelah pukul 08:00, 09:00, 10:00, 11:00, dan 12:00 WIT hari
berikutnya. Summary akurasi menampilkan jumlah perbandingan, MAE, MAPE, akurasi
arah, dan bias rata-rata per jam aktual. Data Model 2 disimpan di
`data/monitoring.csv`; data Model 1 disimpan di `data/monitoring_model1.csv`.
Keduanya diisi oleh GitHub Actions:

- `14:59 UTC` = `23:59 WIT` untuk menyimpan estimasi.
- `23:00 UTC` = `08:00 WIT` untuk mengisi aktual hari berikutnya.
- `00:00-03:00 UTC` = `09:00-12:00 WIT` untuk mengisi aktual lanjutan.
