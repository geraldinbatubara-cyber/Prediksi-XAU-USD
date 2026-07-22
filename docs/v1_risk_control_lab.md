# Optimizer v1 Risk-Control Lab

## Tujuan

Laboratorium ini menguji lapisan manajemen risiko tanpa mengubah sinyal inti
Optimizer v1. Baseline Live Trading dikunci sampai 31 Agustus 2026 dan tidak
ditulis oleh eksperimen ini.

## Desain Pengujian

- Development: 1 Januari-31 Desember 2025.
- Validation: 1 Januari-30 Juni 2026.
- Sinyal inti: Trend, MA 20/50, momentum 10 hari, threshold 0,15%, TP USD 25,
  SL USD 10, dan lot 0,01. Parameter ini merupakan pemenang train 2025 untuk
  pengujian Exact OOS, bukan perubahan pada parameter Live Baseline.
- Eksekusi: candle M1 XAUUSD MT5, bid/ask, spread historis, slippage, dan swap.
- Seleksi finalis hanya memakai development 2025.
- Tiga finalis dibekukan sebelum validation 2026H1.
- Finalis diuji pada sembilan kombinasi spread 1x/1,5x/2x dan slippage 2/4/6
  point per sisi, serta bootstrap Monte Carlo 10.000 lintasan.

## Kriteria Kelulusan

- Growth validation positif.
- Max drawdown maksimal 10%.
- Profit factor minimal 1,30.
- Probabilitas Monte Carlo berakhir di bawah modal awal maksimal 10%.
- Seluruh 9/9 stress scenario tetap profitable.
- Minimal 50 transaksi validation.

## Kandidat dan Hasil

Sebanyak 21 overlay satu faktor dan lima kombinasi diuji bersama baseline.
Tiga finalis berdasarkan development adalah RC-B Balanced, RC-D Profit
Protection, dan RC-C Break-Even.

RC-D Profit Protection menjadi kandidat terbaik relatif pada validation:

| Metrik | Baseline Exact v1 | RC-D |
| --- | ---: | ---: |
| Growth | +10,854% | +4,402% |
| Max drawdown | 15,598% | 9,635% |
| Profit factor | 1,221 | 1,157 |
| Transaksi | 73 | 49 |
| Probabilitas Monte Carlo rugi | 19,09% | 34,15% |

RC-D mempertahankan growth positif pada 9/9 stress scenario dan berhasil
menurunkan drawdown di bawah 10%. Namun kandidat gagal pada profit factor,
Monte Carlo, dan minimum transaksi. Status akhirnya adalah **BELUM LULUS**.

## Keputusan

Tidak ada Challenger yang dipromosikan. Optimizer v1 Baseline tetap menjadi
satu-satunya paper trading aktif sampai periode observasi selesai. Hasil ini
menunjukkan bahwa pengurangan risiko melalui filter dan profit protection dapat
menurunkan drawdown, tetapi juga memangkas terlalu banyak transaksi yang
menyumbang expectancy positif.
