from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from gold_forecast.data import load_gold_data
from gold_forecast.model import train_and_forecast


st.set_page_config(page_title="Prediksi XAU/USD", page_icon=":material/monitoring:", layout="wide")
st.title("Prediksi Harga Emas")
st.caption("Estimasi hari bursa berikutnya dan tujuh hari ke depan")


@st.cache_data(ttl=3600)
def get_data() -> pd.DataFrame:
    return load_gold_data()


with st.sidebar:
    st.header("Pengaturan")
    history_years = st.slider("Riwayat grafik (tahun)", 1, 10, 3)
    st.info("MVP memakai emas COMEX `GC=F`, USD per troy ounce.")

try:
    prices = get_data()
    result = train_and_forecast(prices["Close"])
except Exception as exc:
    st.error(f"Data belum dapat diproses: {exc}")
    st.stop()

latest = float(prices["Close"].iloc[-1])
previous = float(prices["Close"].iloc[-2])
tomorrow, day_seven = result.forecast.iloc[0], result.forecast.iloc[-1]

col1, col2, col3 = st.columns(3)
col1.metric("Harga terakhir", f"${latest:,.2f}", f"{latest - previous:+,.2f}")
col2.metric("Estimasi besok", f"${tomorrow['Estimasi']:,.2f}", f"{tomorrow['Estimasi'] - latest:+,.2f}")
col3.metric("Estimasi hari ke-7", f"${day_seven['Estimasi']:,.2f}", f"{day_seven['Estimasi'] - latest:+,.2f}")

cutoff = prices.index.max() - pd.DateOffset(years=history_years)
chart_prices = prices.loc[prices.index >= cutoff, "Close"]
figure = go.Figure()
figure.add_trace(go.Scatter(x=chart_prices.index, y=chart_prices, name="Historis"))
figure.add_trace(go.Scatter(x=result.forecast.index, y=result.forecast["Batas atas"], mode="lines", line=dict(width=0), showlegend=False))
figure.add_trace(go.Scatter(x=result.forecast.index, y=result.forecast["Batas bawah"], mode="lines", fill="tonexty", fillcolor="rgba(245,158,11,.18)", line=dict(width=0), name="Interval 95%"))
figure.add_trace(go.Scatter(x=result.forecast.index, y=result.forecast["Estimasi"], name="Estimasi", line=dict(color="#f59e0b", width=3)))
figure.update_layout(title="Harga historis dan estimasi", yaxis_title="USD per troy ounce", hovermode="x unified", height=520)
st.plotly_chart(figure, use_container_width=True)

st.subheader("Estimasi 7 Hari Bursa")
st.dataframe(result.forecast.style.format("${:,.2f}"), use_container_width=True)

st.subheader("Kualitas Backtest")
m1, m2, m3, m4 = st.columns(4)
m1.metric("MAE", f"${result.metrics['MAE']:,.2f}")
m2.metric("RMSE", f"${result.metrics['RMSE']:,.2f}")
m3.metric("MAPE", f"{result.metrics['MAPE']:.2f}%")
m4.metric("Akurasi arah", f"{result.metrics['Akurasi arah']:.1f}%")

with st.expander("Metodologi dan risiko"):
    st.write("Model memakai fitur harga historis dengan regresi ridge. Interval berasal dari residual backtest dan bukan jaminan cakupan 95%. Berita dan perubahan kondisi makro dapat membuat harga bergerak di luar interval. Dashboard ini bukan saran investasi.")
