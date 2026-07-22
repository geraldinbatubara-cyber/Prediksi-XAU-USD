# Optimizer v1 Signal Quality Lab

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

Kandidat tidak otomatis menggantikan baseline. Kandidat hanya dapat masuk paper
shadow terpisah setelah memenuhi seluruh kriteria kelulusan yang telah disepakati.
