from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from gold_forecast.data import load_market_data
from gold_forecast.model import train_and_forecast
from gold_forecast.model_v2 import train_model_v2


st.set_page_config(page_title="Prediksi XAU/USD", page_icon=":material/monitoring:", layout="wide")
st.title("Prediksi Harga Emas")
st.caption("Estimasi hari bursa berikutnya dan tujuh hari ke depan")


@st.cache_data(ttl=3600)
def get_data() -> pd.DataFrame:
    return load_market_data()


with st.sidebar:
    st.header("Pengaturan")
    history_years = st.slider("Riwayat grafik (tahun)", 1, 10, 3)
    model_choice = st.radio(
        "Model prediksi",
        ["Model 2 - Lintas Pasar", "Model 1 - Harga Historis"],
    )
    st.info("Harga emas memakai COMEX `GC=F`, USD per troy ounce.")

try:
    market = get_data()
    model_1 = train_and_forecast(market["gold"])
    model_2 = train_model_v2(market)
except Exception as exc:
    st.error(f"Data belum dapat diproses: {exc}")
    st.stop()

result = model_2 if model_choice.startswith("Model 2") else model_1
gold = market["gold"]
latest = float(gold.iloc[-1])
previous = float(gold.iloc[-2])
tomorrow, day_seven = result.forecast.iloc[0], result.forecast.iloc[-1]

col1, col2, col3 = st.columns(3)
col1.metric("Harga terakhir", f"${latest:,.2f}", f"{latest - previous:+,.2f}")
col2.metric("Estimasi besok", f"${tomorrow['Estimasi']:,.2f}", f"{tomorrow['Estimasi'] - latest:+,.2f}")
col3.metric("Estimasi hari ke-7", f"${day_seven['Estimasi']:,.2f}", f"{day_seven['Estimasi'] - latest:+,.2f}")

cutoff = gold.index.max() - pd.DateOffset(years=history_years)
chart_prices = gold.loc[gold.index >= cutoff]
figure = go.Figure()
figure.add_trace(go.Scatter(x=chart_prices.index, y=chart_prices, name="Historis"))
figure.add_trace(go.Scatter(x=result.forecast.index, y=result.forecast["Batas atas"], mode="lines", line=dict(width=0), showlegend=False))
figure.add_trace(go.Scatter(x=result.forecast.index, y=result.forecast["Batas bawah"], mode="lines", fill="tonexty", fillcolor="rgba(245,158,11,.18)", line=dict(width=0), name="Interval 95%"))
figure.add_trace(go.Scatter(x=result.forecast.index, y=result.forecast["Estimasi"], name="Estimasi", line=dict(color="#f59e0b", width=3)))
figure.update_layout(title="Harga historis dan estimasi", yaxis_title="USD per troy ounce", hovermode="x unified", height=520)
st.plotly_chart(figure, use_container_width=True)

st.subheader("Estimasi 7 Hari Bursa")
st.dataframe(result.forecast.style.format("${:,.2f}"), use_container_width=True)

st.subheader("Perbandingan Model untuk Besok")
comparison = pd.DataFrame(
    {
        "Model 1": model_1.metrics,
        "Model 2": model_2.horizon_metrics.loc[1].to_dict(),
    }
).T
st.dataframe(
    comparison.style.format(
        {"MAE": "${:,.2f}", "RMSE": "${:,.2f}", "MAPE": "{:.2f}%", "Akurasi arah": "{:.1f}%"}
    ),
    use_container_width=True,
)
mae_improvement = (1 - comparison.loc["Model 2", "MAE"] / comparison.loc["Model 1", "MAE"]) * 100
if mae_improvement > 0:
    st.success(f"Pada backtest T+1, MAE Model 2 lebih rendah {mae_improvement:.1f}% dibanding Model 1.")
else:
    st.warning("Pada backtest terbaru, Model 2 belum mengalahkan MAE Model 1.")

st.subheader(f"Kualitas Backtest {model_choice.split(' - ')[0]}")
m1, m2, m3, m4 = st.columns(4)
m1.metric("MAE", f"${result.metrics['MAE']:,.2f}")
m2.metric("RMSE", f"${result.metrics['RMSE']:,.2f}")
m3.metric("MAPE", f"{result.metrics['MAPE']:.2f}%")
m4.metric("Akurasi arah", f"{result.metrics['Akurasi arah']:.1f}%")

if model_choice.startswith("Model 2"):
    with st.expander("Metrik Model 2 per horizon"):
        st.dataframe(
            model_2.horizon_metrics.style.format(
                {"MAE": "${:,.2f}", "RMSE": "${:,.2f}", "MAPE": "{:.2f}%", "Akurasi arah": "{:.1f}%"}
            ),
            use_container_width=True,
        )

with st.expander("Metodologi dan risiko"):
    st.markdown(
        """
        **Model 1** memakai regresi ridge dan hanya riwayat harga emas. **Model 2**
        memakai gradient boosting dengan DXY, yield Treasury 10 tahun, minyak, VIX,
        perak, momentum, dan volatilitas. Model 2 dilatih terpisah untuk setiap horizon
        agar prediksi T+7 tidak mengakumulasi prediksi hari sebelumnya.

        Backtest memakai 20% data terbaru secara berurutan dan tidak mengacak waktu.
        Interval berasal dari residual backtest, bukan jaminan cakupan 95%. Dashboard
        ini bukan saran investasi.
        """
    )
