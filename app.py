from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from gold_forecast.data import load_market_data
from gold_forecast.direction_model import train_direction_model
from gold_forecast.model import train_and_forecast
from gold_forecast.model_v2 import train_model_v2
from gold_forecast.signals import build_signal


st.set_page_config(page_title="Prediksi XAU/USD", page_icon=":material/monitoring:", layout="wide")
st.title("Prediksi Harga Emas")
st.caption("Estimasi hari bursa berikutnya dan tujuh hari ke depan")


@st.cache_data(ttl=3600)
def get_data() -> pd.DataFrame:
    return load_market_data()


@st.cache_data(ttl=3600)
def get_models(market_data: pd.DataFrame):
    return (
        train_and_forecast(market_data["gold"]),
        train_model_v2(market_data),
        train_direction_model(market_data),
    )


with st.sidebar:
    st.header("Pengaturan")
    history_years = st.slider("Riwayat grafik (tahun)", 1, 10, 3)
    model_choice = st.radio(
        "Model prediksi",
        ["Model 2 - Lintas Pasar", "Model 1 - Harga Historis"],
    )
    direction_threshold = st.select_slider(
        "Threshold sinyal arah",
        options=[0.50, 0.55, 0.60, 0.65, 0.70],
        value=0.65,
        format_func=lambda value: f"{value:.0%}",
    )
    st.info("Harga emas memakai COMEX `GC=F`, USD per troy ounce.")

try:
    market = get_data()
    model_1, model_2, direction_model = get_models(market)
except Exception as exc:
    st.error(f"Data belum dapat diproses: {exc}")
    st.stop()

result = model_2 if model_choice.startswith("Model 2") else model_1
gold = market["gold"]
latest = float(gold.iloc[-1])
previous = float(gold.iloc[-2])
tomorrow, day_seven = result.forecast.iloc[0], result.forecast.iloc[-1]
signal = build_signal(market, result.forecast)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Harga terakhir", f"${latest:,.2f}", f"{latest - previous:+,.2f}")
col2.metric("Estimasi besok", f"${tomorrow['Estimasi']:,.2f}", f"{tomorrow['Estimasi'] - latest:+,.2f}")
col3.metric("Estimasi hari ke-7", f"${day_seven['Estimasi']:,.2f}", f"{day_seven['Estimasi'] - latest:+,.2f}")
col4.metric("Sinyal", signal.label, f"Confidence {signal.confidence:.0f}%")

st.subheader("Ringkasan Sinyal Harian")
signal_col, driver_col = st.columns([1, 2])
with signal_col:
    if signal.label == "Bullish":
        st.success(f"**{signal.label}** dengan confidence **{signal.confidence:.0f}%**")
    elif signal.label == "Bearish":
        st.error(f"**{signal.label}** dengan confidence **{signal.confidence:.0f}%**")
    else:
        st.info(f"**{signal.label}** dengan confidence **{signal.confidence:.0f}%**")
    st.metric("Ekspektasi besok", f"{signal.expected_change_pct:+.2f}%", f"${signal.expected_change:+,.2f}")
    for item in signal.rationale:
        st.write(f"- {item}")

with driver_col:
    st.write("Faktor lintas pasar 5 hari terakhir")
    if signal.drivers.empty:
        st.caption("Faktor lintas pasar belum tersedia dari sumber data.")
    else:
        st.dataframe(
            signal.drivers.style.format({"Perubahan 5 hari": "{:+.2f}%"}),
            use_container_width=True,
            hide_index=True,
        )

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

st.subheader("Model Arah Berbasis Confidence")
direction_latest = direction_model.latest_probabilities.copy()
direction_latest["Sinyal aktif"] = direction_latest["Probabilitas naik"].apply(
    lambda probability: "Bullish"
    if probability >= direction_threshold * 100
    else ("Bearish" if probability <= (1 - direction_threshold) * 100 else "Netral")
)
selected_direction_metrics = direction_model.threshold_metrics.xs(
    direction_threshold, level="Threshold"
)
d1, d2, d3 = st.columns(3)
d1.metric(
    "Akurasi arah T+1",
    f"{selected_direction_metrics.loc[1, 'Akurasi actionable']:.1f}%",
    f"Coverage {selected_direction_metrics.loc[1, 'Coverage']:.1f}%",
)
d2.metric(
    "Akurasi arah T+7",
    f"{selected_direction_metrics.loc[7, 'Akurasi actionable']:.1f}%",
    f"Coverage {selected_direction_metrics.loc[7, 'Coverage']:.1f}%",
)
d3.metric(
    "Sinyal T+1 sekarang",
    direction_latest.loc[1, "Sinyal aktif"],
    f"Naik {direction_latest.loc[1, 'Probabilitas naik']:.1f}%",
)
st.caption(
    "Akurasi actionable hanya menghitung hari saat probabilitas melewati threshold. "
    "Coverage menunjukkan seberapa sering sinyal seperti itu muncul pada backtest."
)
st.dataframe(
    direction_latest.style.format(
        {"Probabilitas naik": "{:.1f}%", "Probabilitas turun": "{:.1f}%"}
    ),
    use_container_width=True,
)
with st.expander("Backtest model arah per threshold"):
    st.dataframe(
        direction_model.threshold_metrics.style.format(
            {
                "Akurasi semua hari": "{:.1f}%",
                "Akurasi actionable": "{:.1f}%",
                "Coverage": "{:.1f}%",
                "Jumlah sinyal": "{:.0f}",
            }
        ),
        use_container_width=True,
    )

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
        Model arah berbasis confidence tidak memaksa sinyal setiap hari; hari yang
        probabilitasnya rendah tetap dianggap netral. Interval berasal dari residual
        backtest, bukan jaminan cakupan 95%. Dashboard ini bukan saran investasi.
        """
    )
