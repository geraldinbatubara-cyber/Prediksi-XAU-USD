# Optimizer v1 Signal Quality Lab - Adaptive Confirmation v2

Eksperimen ini mencari entry yang lebih berkualitas tanpa mengubah Optimizer v1
baseline, rule exit, TP/SL, lot, swap, target fase, atau ledger paper live.

## Tata kelola

- Development dan pemilihan kandidat: 1 Januari-31 Desember 2025.
- Validasi beku: 1 Januari-30 Juni 2026.
- Sinyal utama tetap berasal dari Optimizer v1 harian.
- Data M1 hanya memakai informasi yang sudah selesai sebelum waktu entry.
- Tiga kombinasi terbaik di development diuji sekali pada validation.
- Finalis menjalani stress 3 spread x 3 slippage dan Monte Carlo 10.000 jalur.

## Dimensi kualitas entry

- Conviction: expected change harus melampaui ambang v1 dengan margin tambahan.
- M15: harga, EMA cepat/lambat, dan momentum intraday harus searah sinyal.
- H1: harga dan EMA harus menunjukkan rezim tren yang searah.
- Momentum H1: perubahan enam jam terakhir harus mendukung arah entry.
- Stretch: entry ditolak jika harga terlalu jauh dari tren relatif terhadap ATR.
- Spread: entry ditolak jika spread melampaui P90 development 2025.
- Patient entry: kombinasi tertentu menunggu konfirmasi maksimal 12 jam.

## Adaptive Confirmation v2

Versi kedua mencari titik tengah dengan menerima entry ketika minimal dua dari
tiga bukti intraday mendukung sinyal harian: tren M15, tren H1, dan momentum H1.
Sebagian kandidat mewajibkan H1 sebagai jangkar, sedangkan waktu tunggu dibatasi
0, 6, atau 12 jam. Finalis dipilih pada development 2025 hanya jika retensi entry
65-90%, minimal tiga dari empat kuartal positif, growth positif, profit factor
minimal 1,25, dan drawdown maksimal 15%.

Validation Januari-Juni 2026 disebut secondary validation karena periodenya telah
diamati pada eksperimen sebelumnya. Bukti yang benar-benar baru tetap harus
berasal dari forward paper shadow setelah kandidat dibekukan.

Hasil Adaptive v2 menunjukkan trade-off yang tajam. AC-C 2of3 Wait-3h lolos
gerbang development dan menghasilkan 57 transaksi pada secondary validation,
tetapi hanya mencatat growth +2,386%, max drawdown 17,304%, profit factor 1,060,
dan probabilitas Monte Carlo rugi 44,43%. Target frekuensi tercapai dengan
mengorbankan kualitas terlalu besar, sehingga statusnya **BELUM LULUS**.

Kandidat tidak otomatis menggantikan baseline. Kandidat hanya dapat masuk paper
shadow terpisah setelah memenuhi seluruh kriteria kelulusan yang telah disepakati.
