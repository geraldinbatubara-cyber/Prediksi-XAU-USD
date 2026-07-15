# Prediksi XAU/USD

Dashboard estimasi harga emas untuk hari bursa berikutnya dan tujuh hari ke depan.

**Aplikasi live:** https://klmshreuteappzrrjyp3kuh.streamlit.app/

## Menjalankan

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Data MVP memakai kontrak berjangka emas COMEX (`GC=F`) dalam USD per troy ounce.
Hasil model adalah estimasi statistik, bukan saran investasi.

## Metodologi

Model regresi ridge memakai harga terkini, lag harga, return, moving average, dan
volatilitas. Evaluasi memakai 20% data terbaru sebagai backtest berurutan.

Model 2 memakai gradient boosting dan menambahkan DXY, yield Treasury 10 tahun,
minyak, VIX, serta perak. Satu model direct dilatih untuk setiap horizon T+1 sampai
T+7. Dashboard membandingkan Model 1 dan Model 2 pada horizon T+1 yang setara.

## Simulasi Trading

Tab `Simulasi` sekarang hanya menampilkan `Strategi Terbaik Optimizer` dan
`Strategi Terbaik v.2`. Simulasi Model 1 dan Model 2 skenario lama dihapus dari
tab ini karena hasilnya tidak memadai untuk pengembangan strategi.

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
