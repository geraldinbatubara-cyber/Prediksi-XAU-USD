# Arsip Eksperimen Strategi

Dokumen ini menyimpan alasan eksperimen lama dikeluarkan dari deployment aktif.
Kode dan hasil historis tetap dapat ditelusuri melalui riwayat Git.

## Keputusan Aktif

- Baseline pengembangan: Optimizer v1.
- Simulasi aktif: v1, v1 OOS, v1 Exact Broker-Aware OOS, dan v1 Robustness Test.
- Paper live aktif: v1 saja.
- Posisi v10 lama tetap dipantau sampai TP/CL tanpa izin membuka posisi baru.

## Hasil yang Diarsipkan

| Eksperimen | Hasil utama | Keputusan |
| --- | ---: | --- |
| Optimizer v10 OOS harian | Growth +1,080.24% | Tidak dipakai karena hasil exact broker-aware berubah menjadi negatif. |
| Optimizer v10 Exact Broker-Aware OOS | Growth -24.64%; profit factor 0.851; max drawdown USD 713.43 | Diarsipkan. Tidak cukup tahan terhadap eksekusi M1, spread, dan slippage. |
| v1 Intraday Adaptation Broker-Aware OOS | Growth -4.82%; profit factor 0.943 | Diarsipkan. Strategi turunan intraday tidak mempertahankan keunggulan v1. |
| v10 Intraday Adaptation Broker-Aware OOS | Growth +5.18%; profit factor 1.036 | Diarsipkan. Positif tetapi lemah dan bukan pengujian exact terhadap rule harian asli. |
| Martingale dan eksperimen M1 lama | Lebih rendah dari baseline v1 | Diarsipkan agar aplikasi lebih ringan dan fokus pengembangan tetap jelas. |

## Catatan v1

Optimizer v1 Exact Broker-Aware OOS menghasilkan growth +10.85%, profit factor
1.221, max drawdown 15.60%, dan 73 transaksi. Robustness v1 tetap positif pada
9/9 skenario biaya, tetapi 0/9 memenuhi seluruh ambang kelayakan real-money.
Karena itu status v1 masih kandidat paper trading, bukan persetujuan trading riil.
