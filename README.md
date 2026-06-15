# Prediksi XAU/USD

Dashboard estimasi harga emas untuk hari bursa berikutnya dan tujuh hari ke depan.

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
