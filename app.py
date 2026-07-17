from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:  # pragma: no cover - handled in deployed UI
    st_autorefresh = None

from gold_forecast.data import load_gold_data, load_market_data
from gold_forecast.direction_model import train_direction_model
from gold_forecast.intraday_audit import audit_intraday_data, load_intraday_csv
from gold_forecast import live_trading as live_trading_module
from gold_forecast.live_trading import (
    LIVE_INITIAL_EQUITY,
    LIVE_START_DATE,
    load_live_ledger,
    run_live_trading_update,
)
from gold_forecast.model import train_and_forecast
from gold_forecast.model_v2 import train_model_v2
from gold_forecast.monitoring import (
    ACTUAL_HOURS,
    DATA_PATH,
    MODEL_1_DATA_PATH,
    WIT,
    hour_suffix,
    load_monitoring,
    monitoring_summary,
)
from gold_forecast.signals import build_signal
from gold_forecast.strategy_optimizer import (
    OPTIMIZATION_END,
    OPTIMIZATION_START,
    _rsi,
    run_optimized_strategy,
    run_optimized_strategy_v2,
    run_optimized_strategy_v3,
    run_optimized_strategy_v4,
)


SIMULATION_CACHE_VERSION = "optimizer-multiphase-v4-dynamic-risk"

st.set_page_config(page_title="Prediksi XAU/USD", page_icon=":material/monitoring:", layout="wide")
st.title("Prediksi Harga Emas")
st.caption("Estimasi hari bursa berikutnya dan tujuh hari ke depan")


@st.cache_data(ttl=60)
def get_data() -> tuple[pd.DataFrame, pd.Timestamp]:
    return load_market_data(), pd.Timestamp.now(tz=WIT)


@st.cache_data(ttl=60)
def get_gold_ohlc() -> pd.DataFrame:
    return load_gold_data()


@st.cache_data(ttl=3600)
def get_models(market_data: pd.DataFrame):
    return (
        train_and_forecast(market_data["gold"]),
        train_model_v2(market_data),
        train_direction_model(market_data),
    )


@st.cache_data(ttl=3600)
def get_simulations(gold_ohlc: pd.DataFrame, simulation_version: str):
    optimized_result, optimization_leaderboard = run_optimized_strategy(gold_ohlc)
    optimized_v2_result, optimization_v2_leaderboard = run_optimized_strategy_v2(gold_ohlc)
    optimized_v3_result, optimization_v3_leaderboard = run_optimized_strategy_v3(gold_ohlc, optimization_leaderboard)
    optimized_v4_result, optimization_v4_leaderboard = run_optimized_strategy_v4(gold_ohlc, optimization_leaderboard)
    return (
        optimized_result,
        optimization_leaderboard,
        optimized_v2_result,
        optimization_v2_leaderboard,
        optimized_v3_result,
        optimization_v3_leaderboard,
        optimized_v4_result,
        optimization_v4_leaderboard,
    )


def render_dashboard(
    market: pd.DataFrame,
    data_fetched_at: pd.Timestamp,
    model_1,
    model_2,
    direction_model,
    model_choice: str,
    direction_threshold: float,
    history_years: int,
) -> None:
    result = model_2 if model_choice.startswith("Model 2") else model_1
    gold = market["gold"]
    latest = float(gold.iloc[-1])
    previous = float(gold.iloc[-2])
    tomorrow, day_seven = result.forecast.iloc[0], result.forecast.iloc[-1]
    signal = build_signal(market, result.forecast)
    market_last_date = pd.Timestamp(gold.index.max()).strftime("%d %b %Y")
    fetched_label = data_fetched_at.strftime("%d %b %Y %H:%M:%S WIT")

    st.info(f"Data pasar terakhir: **{market_last_date}** | Data diambil dashboard: **{fetched_label}**")

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
    figure.add_trace(
        go.Scatter(
            x=result.forecast.index,
            y=result.forecast["Batas atas"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
        )
    )
    figure.add_trace(
        go.Scatter(
            x=result.forecast.index,
            y=result.forecast["Batas bawah"],
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(245,158,11,.18)",
            line=dict(width=0),
            name="Interval 95%",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=result.forecast.index,
            y=result.forecast["Estimasi"],
            name="Estimasi",
            line=dict(color="#f59e0b", width=3),
        )
    )
    figure.update_layout(
        title="Harga historis dan estimasi",
        yaxis_title="USD per troy ounce",
        hovermode="x unified",
        height=520,
    )
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
    selected_direction_metrics = direction_model.threshold_metrics.xs(direction_threshold, level="Threshold")
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


def _render_simulation_result(title: str, result) -> None:
    st.subheader(title)
    summary = result.summary
    modal_awal = summary.get("Modal awal", 1000.0)
    balance_akhir = summary.get("Balance akhir", modal_awal)
    equity_akhir = summary.get("Equity akhir", balance_akhir)
    target_equity = summary.get("Target equity", 1200.0)
    target_tercapai = summary.get("Target tercapai", equity_akhir >= target_equity)
    total_swap = summary.get("Total swap", result.trades["Swap"].sum() if "Swap" in result.trades.columns else 0.0)

    equity_curve = result.equity_curve.copy()
    if not equity_curve.empty and "Equity" not in equity_curve.columns and "Balance" in equity_curve.columns:
        equity_curve["Equity"] = equity_curve["Balance"]
    if not equity_curve.empty and "Balance" not in equity_curve.columns and "Equity" in equity_curve.columns:
        equity_curve["Balance"] = equity_curve["Equity"]

    target_date = summary.get("Tanggal target")
    low_date = summary.get("Tanggal equity terendah")
    high_date = summary.get("Tanggal equity tertinggi")
    if not equity_curve.empty and "Equity" in equity_curve.columns:
        equity_series = pd.to_numeric(equity_curve["Equity"], errors="coerce")
        if pd.isna(low_date):
            low_date = equity_series.idxmin()
        if pd.isna(high_date):
            high_date = equity_series.idxmax()
    equity_terendah = summary.get("Equity terendah", equity_curve["Equity"].min() if "Equity" in equity_curve else equity_akhir)
    equity_tertinggi = summary.get("Equity tertinggi", equity_curve["Equity"].max() if "Equity" in equity_curve else equity_akhir)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Equity akhir", f"${equity_akhir:,.2f}", f"{equity_akhir - modal_awal:+,.2f}")
    c2.metric("Jumlah transaksi", f"{summary['Jumlah transaksi']:.0f}")
    c3.metric("Win rate", "-" if pd.isna(summary["Win rate"]) else f"{summary['Win rate']:.1f}%")
    c4.metric("Max drawdown", f"${summary['Max drawdown']:,.2f}")

    target_label = "-" if pd.isna(target_date) else pd.Timestamp(target_date).strftime("%d %b %Y")
    st.info(
        f"Target close-all: **${target_equity:,.2f}** | "
        f"Status: **{'Tercapai' if target_tercapai else 'Belum tercapai'}** | "
        f"Tanggal target: **{target_label}**"
    )

    detail = pd.DataFrame(
        [
            {"Metrik": "Total BUY", "Nilai": summary["Total BUY"]},
            {"Metrik": "Total SELL", "Nilai": summary["Total SELL"]},
            {"Metrik": "Modal awal", "Nilai": modal_awal},
            {"Metrik": "Balance akhir", "Nilai": balance_akhir},
            {
                "Metrik": "Equity terendah",
                "Nilai": f"${equity_terendah:,.2f} pada {pd.Timestamp(low_date).strftime('%d %b %Y')}"
                if not pd.isna(low_date)
                else "-",
            },
            {
                "Metrik": "Equity tertinggi",
                "Nilai": f"${equity_tertinggi:,.2f} pada {pd.Timestamp(high_date).strftime('%d %b %Y')}"
                if not pd.isna(high_date)
                else "-",
            },
            {"Metrik": "Total net P/L", "Nilai": summary["Total net P/L"]},
            {"Metrik": "Total swap", "Nilai": total_swap},
            {"Metrik": "Profit factor", "Nilai": summary.get("Profit factor", pd.NA)},
            {"Metrik": "Avg net P/L", "Nilai": summary.get("Avg net P/L", pd.NA)},
        ]
    )
    st.dataframe(detail, use_container_width=True, hide_index=True)

    if equity_curve.empty:
        st.info("Belum ada transaksi simulasi yang dapat dihitung.")
        return

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=equity_curve.index,
            y=equity_curve["Equity"],
            name="Equity",
            line=dict(width=3),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=equity_curve.index,
            y=equity_curve["Balance"],
            name="Balance cash",
            line=dict(width=1.5, dash="dot"),
        )
    )
    figure.add_hline(
        y=target_equity,
        line_dash="dash",
        line_color="#16a34a",
        annotation_text="Target equity $1,200",
    )
    if not pd.isna(low_date):
        figure.add_trace(
            go.Scatter(
                x=[low_date],
                y=[equity_terendah],
                mode="markers+text",
                name="Equity terendah",
                marker=dict(size=11, color="#dc2626"),
                text=["Low"],
                textposition="bottom center",
            )
        )
    if not pd.isna(high_date):
        figure.add_trace(
            go.Scatter(
                x=[high_date],
                y=[equity_tertinggi],
                mode="markers+text",
                name="Equity tertinggi",
                marker=dict(size=11, color="#16a34a"),
                text=["High"],
                textposition="top center",
            )
        )
    figure.update_layout(title=f"Equity Curve {title}", yaxis_title="USD", height=390)
    st.plotly_chart(figure, use_container_width=True)

    trades = result.trades.copy()
    numeric_columns = [
        "Lot",
        "Confidence",
        "Prediksi",
        "Expected change (%)",
        "Entry",
        "Exit",
        "TP (USD)",
        "SL (USD)",
        "Threshold entry (%)",
        "Gross P/L",
        "Swap",
        "Net P/L",
        "Balance",
    ]
    visible_columns = [
        column
        for column in [
            "Model",
            "Strategi",
            "Position ID",
            "Tanggal entry",
            "Tanggal tutup",
            "Arah",
            "Lot",
            "Confidence",
            "Prediksi",
            "Expected change (%)",
            "Entry",
            "Exit",
            "Alasan exit",
            "TP (USD)",
            "SL (USD)",
            "Threshold entry (%)",
            "Gross P/L",
            "Swap",
            "Net P/L",
            "Balance",
        ]
        if column in trades.columns
    ]
    st.dataframe(
        trades[visible_columns].style.format(
            {
                "Lot": "{:.2f}",
                "Confidence": "{:.1f}%",
                "Prediksi": "${:,.2f}",
                "Expected change (%)": "{:+.2f}%",
                "Entry": "${:,.2f}",
                "Exit": "${:,.2f}",
                "TP (USD)": "${:,.2f}",
                "SL (USD)": "${:,.2f}",
                "Threshold entry (%)": "{:.2f}%",
                "Gross P/L": "${:+,.2f}",
                "Swap": "${:+,.2f}",
                "Net P/L": "${:+,.2f}",
                "Balance": "${:,.2f}",
            },
            subset=[column for column in numeric_columns if column in trades.columns],
        ),
        use_container_width=True,
        hide_index=True,
    )


def _format_date(value) -> str:
    if pd.isna(value):
        return "-"
    try:
        return pd.Timestamp(value).strftime("%d %b %Y")
    except (TypeError, ValueError):
        return str(value)


def _build_phase_execution_summary(phases: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if phases.empty:
        return pd.DataFrame(rows)

    for _, phase in phases.iterrows():
        phase_id = phase.get("Fase")
        phase_trades = trades[trades["Fase"] == phase_id].copy() if not trades.empty and "Fase" in trades.columns else pd.DataFrame()

        if phase_trades.empty:
            total_tp = total_cl = total_swap = total_other = 0.0
            buy_count = sell_count = close_count = tp_count = cl_count = 0
        else:
            gross = pd.to_numeric(phase_trades.get("Gross P/L", 0.0), errors="coerce").fillna(0.0)
            swap = pd.to_numeric(phase_trades.get("Swap", 0.0), errors="coerce").fillna(0.0)
            reasons = phase_trades.get("Alasan exit", pd.Series("", index=phase_trades.index)).astype(str)

            tp_mask = reasons.eq("TP tersentuh")
            cl_mask = reasons.eq("SL tersentuh")
            total_tp = float(gross[tp_mask].sum())
            total_cl = float(gross[cl_mask].sum())
            total_other = float(gross[~tp_mask & ~cl_mask].sum())
            total_swap = float(swap.sum())
            directions = phase_trades.get("Arah", pd.Series("", index=phase_trades.index)).astype(str)
            buy_count = int(directions.eq("BUY").sum())
            sell_count = int(directions.eq("SELL").sum())
            close_count = int(len(phase_trades))
            tp_count = int(tp_mask.sum())
            cl_count = int(cl_mask.sum())

        rows.append(
            {
                "Fase": phase_id,
                "Start equity": phase.get("Start equity"),
                "Target equity": phase.get("Target equity"),
                "Equity close-all": phase.get("Equity close-all"),
                "Target tercapai": phase.get("Target tercapai"),
                "Tanggal target": phase.get("Tanggal target"),
                "Biaya SWAP": total_swap,
                "Nilai total TP": total_tp,
                "Nilai total CL/SL": total_cl,
                "Nilai close-all/lainnya": total_other,
                "Posisi BUY": buy_count,
                "Posisi SELL": sell_count,
                "Jumlah CLOSE": close_count,
                "Close TP": tp_count,
                "Close CL/SL": cl_count,
            }
        )
    return pd.DataFrame(rows)


def _render_strategy_explanation(title: str, summary: dict, phases: pd.DataFrame, trades: pd.DataFrame, leaderboard: pd.DataFrame) -> None:
    best = leaderboard.iloc[0].to_dict() if not leaderboard.empty else {}
    strategy_name = best.get("Strategi", title)
    threshold = best.get("Threshold entry (%)")
    tp_value = best.get("TP (USD)")
    sl_value = best.get("SL (USD)")
    fixed_lot = best.get("Lot", 0.01)
    risk_cap = best.get("Risk cap floating SL (%)")
    if pd.isna(fixed_lot):
        fixed_lot = 0.01
    lot_info = "lot dinamis 0.01-0.02 mengikuti confidence sinyal" if "v.2" in title else f"lot tetap {fixed_lot:.2f}"
    risk_info = ""
    if pd.notna(risk_cap):
        risk_info = (
            f" v4 tidak memakai batas 8 BUY/10 SELL; pembukaan posisi dibatasi oleh risk cap floating SL "
            f"**{risk_cap:.1f}%** dari equity fase."
        )

    strategy_parts = [
        f"Strategi terpilih: **{strategy_name}**.",
        f"Entry hanya dibuka saat sinyal melewati threshold **{threshold:.2f}%**." if pd.notna(threshold) else "",
        f"Setiap posisi memakai TP **USD {tp_value:,.2f}** dan CL/SL **USD {sl_value:,.2f}**." if pd.notna(tp_value) and pd.notna(sl_value) else "",
        f"Ukuran transaksi: **{lot_info}**.",
        risk_info,
    ]
    st.markdown("**Penjelasan Strategi dan Momen Kritis**")
    st.info(" ".join(part for part in strategy_parts if part))

    low_date = _format_date(summary.get("Tanggal equity terendah"))
    high_date = _format_date(summary.get("Tanggal equity tertinggi"))
    last_phase = phases.iloc[-1] if not phases.empty else pd.Series(dtype=object)
    target_reached = last_phase.get("Target tercapai", False)
    target_status = "tercapai" if pd.notna(target_reached) and bool(target_reached) else "belum tercapai"
    total_buy = int(summary.get("Total BUY", 0))
    total_sell = int(summary.get("Total SELL", 0))
    max_open = int(summary.get("Max open posisi", 0))
    total_swap = float(summary.get("Total swap", 0.0))

    critical_notes = [
        f"Equity terendah terjadi pada **{low_date}** di **USD {summary.get('Equity terendah', 0):,.2f}**.",
        f"Equity tertinggi terjadi pada **{high_date}** di **USD {summary.get('Equity tertinggi', 0):,.2f}**.",
        f"Fase terakhir **{target_status}**; tanggal target: **{_format_date(last_phase.get('Tanggal target'))}**.",
        f"Eksposur arah: **{total_buy} BUY** dan **{total_sell} SELL**; max open bersamaan **{max_open} posisi**; total swap simulasi **USD {total_swap:+,.2f}**.",
    ]
    st.warning(" ".join(critical_notes))

    phase_execution = _build_phase_execution_summary(phases, trades)
    if phase_execution.empty:
        st.info("Belum ada transaksi fase yang bisa diringkas.")
        return

    st.markdown("**Summary Eksekusi per Fase**")
    st.dataframe(
        phase_execution.style.format(
            {
                "Start equity": "${:,.2f}",
                "Target equity": "${:,.2f}",
                "Equity close-all": "${:,.2f}",
                "Biaya SWAP": "${:+,.2f}",
                "Nilai total TP": "${:+,.2f}",
                "Nilai total CL/SL": "${:+,.2f}",
                "Nilai close-all/lainnya": "${:+,.2f}",
                "Posisi BUY": "{:.0f}",
                "Posisi SELL": "{:.0f}",
                "Jumlah CLOSE": "{:.0f}",
                "Close TP": "{:.0f}",
                "Close CL/SL": "{:.0f}",
            },
            na_rep="-",
        ),
        use_container_width=True,
        hide_index=True,
    )


def _render_strategy_trade_chart(
    title: str,
    gold_ohlc: pd.DataFrame,
    trades: pd.DataFrame,
    leaderboard: pd.DataFrame,
) -> None:
    if gold_ohlc.empty or trades.empty or leaderboard.empty:
        return

    best = leaderboard.iloc[0]
    fast_window = int(best.get("Fast MA", 10))
    slow_window = int(best.get("Slow MA", 50))
    momentum_days = int(best.get("Momentum hari", 10))
    threshold = float(best.get("Threshold entry (%)", 0.15))

    chart_data = gold_ohlc.copy()
    chart_data = chart_data.loc[
        (chart_data.index >= OPTIMIZATION_START - pd.Timedelta(days=120))
        & (chart_data.index <= OPTIMIZATION_END)
    ].copy()
    if chart_data.empty:
        return

    close = chart_data["Close"].astype(float)
    chart_data["MA cepat"] = close.rolling(fast_window).mean()
    chart_data["MA lambat"] = close.rolling(slow_window).mean()
    chart_data["Momentum"] = close.pct_change(momentum_days) * 100

    trade_rows = trades.copy()
    trade_rows["Tanggal entry"] = pd.to_datetime(trade_rows["Tanggal entry"], errors="coerce")
    trade_rows["Tanggal tutup"] = pd.to_datetime(trade_rows["Tanggal tutup"], errors="coerce")
    trade_rows = trade_rows.dropna(subset=["Tanggal entry", "Tanggal tutup"])

    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.72, 0.28],
        subplot_titles=("Harga, MA, Entry dan Exit", f"Momentum {momentum_days} Hari"),
    )
    figure.add_trace(go.Scatter(x=chart_data.index, y=chart_data["Close"], name="Close", line=dict(width=2)), row=1, col=1)
    figure.add_trace(
        go.Scatter(x=chart_data.index, y=chart_data["MA cepat"], name=f"MA cepat {fast_window}", line=dict(width=1.8)),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(x=chart_data.index, y=chart_data["MA lambat"], name=f"MA lambat {slow_window}", line=dict(width=1.8)),
        row=1,
        col=1,
    )

    buy_entries = trade_rows[trade_rows["Arah"].eq("BUY")]
    sell_entries = trade_rows[trade_rows["Arah"].eq("SELL")]
    if not buy_entries.empty:
        figure.add_trace(
            go.Scatter(
                x=buy_entries["Tanggal entry"],
                y=buy_entries["Entry"],
                mode="markers",
                name="Entry BUY",
                marker=dict(symbol="triangle-up", size=10, color="#22c55e", line=dict(width=1, color="#052e16")),
                customdata=buy_entries[["Position ID", "Fase", "Expected change (%)"]].to_numpy(),
                hovertemplate="Entry BUY<br>Tanggal=%{x|%d %b %Y}<br>Harga=%{y:,.2f}<br>ID=%{customdata[0]}<br>Fase=%{customdata[1]}<br>Expected=%{customdata[2]:+.2f}%<extra></extra>",
            ),
            row=1,
            col=1,
        )
    if not sell_entries.empty:
        figure.add_trace(
            go.Scatter(
                x=sell_entries["Tanggal entry"],
                y=sell_entries["Entry"],
                mode="markers",
                name="Entry SELL",
                marker=dict(symbol="triangle-down", size=10, color="#ef4444", line=dict(width=1, color="#450a0a")),
                customdata=sell_entries[["Position ID", "Fase", "Expected change (%)"]].to_numpy(),
                hovertemplate="Entry SELL<br>Tanggal=%{x|%d %b %Y}<br>Harga=%{y:,.2f}<br>ID=%{customdata[0]}<br>Fase=%{customdata[1]}<br>Expected=%{customdata[2]:+.2f}%<extra></extra>",
            ),
            row=1,
            col=1,
        )

    exit_styles = {
        "TP tersentuh": ("Exit TP", "#f59e0b", "circle"),
        "SL tersentuh": ("Exit CL/SL", "#dc2626", "x"),
        "Target equity tercapai": ("Close-all target", "#2563eb", "diamond"),
        "Akhir periode data": ("Exit akhir data", "#64748b", "square"),
    }
    for reason, (label, color, symbol) in exit_styles.items():
        exits = trade_rows[trade_rows["Alasan exit"].eq(reason)]
        if exits.empty:
            continue
        figure.add_trace(
            go.Scatter(
                x=exits["Tanggal tutup"],
                y=exits["Exit"],
                mode="markers",
                name=label,
                marker=dict(symbol=symbol, size=9, color=color),
                customdata=exits[["Position ID", "Arah", "Net P/L"]].to_numpy(),
                hovertemplate=f"{label}<br>Tanggal=%{{x|%d %b %Y}}<br>Harga=%{{y:,.2f}}<br>ID=%{{customdata[0]}}<br>Arah=%{{customdata[1]}}<br>Net P/L=%{{customdata[2]:+,.2f}}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    figure.add_trace(
        go.Scatter(x=chart_data.index, y=chart_data["Momentum"], name=f"Momentum {momentum_days}", line=dict(width=2, color="#a855f7")),
        row=2,
        col=1,
    )
    figure.add_hline(y=threshold, line_dash="dash", line_color="#22c55e", annotation_text=f"+{threshold:.2f}%", row=2, col=1)
    figure.add_hline(y=-threshold, line_dash="dash", line_color="#ef4444", annotation_text=f"-{threshold:.2f}%", row=2, col=1)
    figure.add_hline(y=0, line_dash="dot", line_color="#94a3b8", row=2, col=1)
    figure.update_layout(
        title=f"Visual Strategi {title}: MA {fast_window}/{slow_window}, Momentum {momentum_days}",
        height=720,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    figure.update_yaxes(title_text="Harga XAU/USD", row=1, col=1)
    figure.update_yaxes(title_text="Momentum (%)", row=2, col=1)
    figure.update_xaxes(range=[OPTIMIZATION_START, OPTIMIZATION_END], row=2, col=1)

    st.markdown("**Visual Strategi: Harga, MA, Momentum, Entry dan Exit**")
    st.caption(
        f"Timeframe simulasi memakai candle harian. BUY menunggu Close > MA {fast_window}, MA {fast_window} > MA {slow_window}, "
        f"dan momentum > +{threshold:.2f}%. SELL menunggu kondisi kebalikannya dengan momentum < -{threshold:.2f}%."
    )
    st.plotly_chart(figure, use_container_width=True)


def _render_multiphase_result(title: str, result, leaderboard: pd.DataFrame, gold_ohlc: pd.DataFrame | None = None) -> None:
    st.subheader(title)
    summary = result.summary
    if not hasattr(result, "phases"):
        target_equity = summary.get("Target equity", 1200.0)
        modal_awal = summary.get("Modal awal", 1000.0)
        phases = pd.DataFrame(
            [
                {
                    "Fase": 1,
                    "Start equity": modal_awal,
                    "Target equity": target_equity,
                    "Equity close-all": summary.get("Equity akhir", modal_awal),
                    "Target tercapai": summary.get("Target tercapai", False),
                    "Tanggal target": summary.get("Tanggal target"),
                    "Equity terendah": summary.get("Equity terendah", modal_awal),
                    "Tanggal equity terendah": summary.get("Tanggal equity terendah"),
                    "Equity tertinggi": summary.get("Equity tertinggi", modal_awal),
                    "Tanggal equity tertinggi": summary.get("Tanggal equity tertinggi"),
                    "Total net P/L": summary.get("Total net P/L", 0.0),
                    "Total swap": summary.get("Total swap", 0.0),
                    "Jumlah transaksi": summary.get("Jumlah transaksi", 0.0),
                    "Total BUY": summary.get("Total BUY", 0.0),
                    "Total SELL": summary.get("Total SELL", 0.0),
                    "Win rate": summary.get("Win rate", pd.NA),
                    "Max drawdown": summary.get("Max drawdown", 0.0),
                    "Profit factor": summary.get("Profit factor", pd.NA),
                    "Status": "Selesai" if summary.get("Target tercapai", False) else "Cache lama / belum multi-fase",
                }
            ]
        )
        summary = {
            **summary,
            "Fase selesai": 1.0 if summary.get("Target tercapai", False) else 0.0,
            "Fase total": 1.0,
            "Growth total": (summary.get("Equity akhir", modal_awal) / modal_awal - 1) * 100 if modal_awal else 0.0,
        }
    else:
        phases = result.phases.copy()
    trades = result.trades.copy()
    equity_curve = result.equity_curve.copy()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Equity akhir", f"${summary['Equity akhir']:,.2f}", f"{summary['Growth total']:+.1f}%")
    c2.metric("Fase selesai", f"{summary['Fase selesai']:.0f}", f"Total fase {summary['Fase total']:.0f}")
    c3.metric("Jumlah transaksi", f"{summary['Jumlah transaksi']:.0f}")
    c4.metric("Max drawdown", f"${summary['Max drawdown']:,.2f}")

    st.info(
        "Target tiap fase: **+20% dari start equity fase tersebut**. "
        "Saat target tercapai, semua posisi ditutup dan fase berikutnya dimulai dari equity close-all."
    )

    st.markdown("**Ringkasan Fase**")
    if phases.empty:
        st.warning("Belum ada fase simulasi yang dapat dihitung.")
        return

    st.dataframe(
        phases.style.format(
            {
                "Start equity": "${:,.2f}",
                "Target equity": "${:,.2f}",
                "Equity close-all": "${:,.2f}",
                "Equity terendah": "${:,.2f}",
                "Equity tertinggi": "${:,.2f}",
                "Total net P/L": "${:+,.2f}",
                "Total swap": "${:+,.2f}",
                "Jumlah transaksi": "{:.0f}",
                "Total BUY": "{:.0f}",
                "Total SELL": "{:.0f}",
                "Max open posisi": "{:.0f}",
                "Win rate": "{:.1f}%",
                "Max drawdown": "${:,.2f}",
                "Profit factor": "{:.2f}",
            },
            na_rep="-",
        ),
        use_container_width=True,
        hide_index=True,
    )

    if gold_ohlc is not None:
        _render_strategy_trade_chart(title, gold_ohlc, trades, leaderboard)

    if not equity_curve.empty:
        figure = go.Figure()
        figure.add_trace(
            go.Scatter(
                x=equity_curve.index,
                y=equity_curve["Equity"],
                name="Equity",
                line=dict(width=3),
            )
        )
        figure.add_trace(
            go.Scatter(
                x=equity_curve.index,
                y=equity_curve["Balance"],
                name="Balance cash",
                line=dict(width=1.5, dash="dot"),
            )
        )
        completed_phases = phases[phases["Target tercapai"]]
        if not completed_phases.empty:
            figure.add_trace(
                go.Scatter(
                    x=completed_phases["Tanggal target"],
                    y=completed_phases["Equity close-all"],
                    mode="markers+text",
                    name="Close-all fase",
                    marker=dict(size=10, color="#16a34a"),
                    text=[f"Fase {int(value)}" for value in completed_phases["Fase"]],
                    textposition="top center",
                )
            )
        figure.update_layout(title=f"Equity Curve Multi-Fase {title}", yaxis_title="USD", height=420)
        st.plotly_chart(figure, use_container_width=True)
        _render_strategy_explanation(title, summary, phases, trades, leaderboard)

    if not trades.empty:
        st.markdown("**Detail Transaksi**")
        numeric_columns = [
            "Lot",
            "Confidence",
            "Prediksi",
            "Expected change (%)",
            "Entry",
            "Exit",
            "TP (USD)",
            "SL (USD)",
            "Threshold entry (%)",
            "Gross P/L",
            "Swap",
            "Net P/L",
            "Balance",
        ]
        visible_columns = [
            column
            for column in [
                "Fase",
                "Model",
                "Strategi",
                "Position ID",
                "Tanggal entry",
                "Tanggal tutup",
                "Arah",
                "Lot",
                "Confidence",
                "Prediksi",
                "Expected change (%)",
                "Entry",
                "Exit",
                "Alasan exit",
                "TP (USD)",
                "SL (USD)",
                "Threshold entry (%)",
                "Gross P/L",
                "Swap",
                "Net P/L",
                "Balance",
            ]
            if column in trades.columns
        ]
        st.dataframe(
            trades[visible_columns].style.format(
                {
                    "Lot": "{:.2f}",
                    "Confidence": "{:.1f}%",
                    "Prediksi": "${:,.2f}",
                    "Expected change (%)": "{:+.2f}%",
                    "Entry": "${:,.2f}",
                    "Exit": "${:,.2f}",
                    "TP (USD)": "${:,.2f}",
                    "SL (USD)": "${:,.2f}",
                    "Threshold entry (%)": "{:.2f}%",
                    "Gross P/L": "${:+,.2f}",
                    "Swap": "${:+,.2f}",
                    "Net P/L": "${:+,.2f}",
                    "Balance": "${:,.2f}",
                },
                subset=[column for column in numeric_columns if column in trades.columns],
            ),
            use_container_width=True,
            hide_index=True,
        )

    with st.expander(f"Leaderboard kandidat {title}"):
        if leaderboard.empty:
            st.info("Leaderboard belum tersedia.")
        else:
            st.dataframe(
                leaderboard.head(25).style.format(
                    {
                        "Threshold entry (%)": "{:.2f}%",
                        "Confidence cutoff": "{:.0%}",
                        "Risk cap floating SL (%)": "{:.1f}%",
                        "TP (USD)": "${:,.2f}",
                        "SL (USD)": "${:,.2f}",
                        "Lot": "{:.2f}",
                        "Lot minimum": "{:.2f}",
                        "Lot maksimum": "{:.2f}",
                        "Lot rata-rata sinyal": "{:.3f}",
                        "Confidence rata-rata": "{:.1f}%",
                        "Fase selesai": "{:.0f}",
                        "Fase total": "{:.0f}",
                        "Equity akhir": "${:,.2f}",
                        "Growth total": "{:+.1f}%",
                        "Equity terendah": "${:,.2f}",
                        "Equity tertinggi": "${:,.2f}",
                        "Max drawdown": "${:,.2f}",
                        "Jumlah transaksi": "{:.0f}",
                        "Total BUY": "{:.0f}",
                        "Total SELL": "{:.0f}",
                        "Max open posisi": "{:.0f}",
                        "Total swap": "${:+,.2f}",
                        "Win rate": "{:.1f}%",
                        "Profit factor": "{:.2f}",
                        "Avg net P/L": "${:+,.2f}",
                    },
                    na_rep="-",
                ),
                use_container_width=True,
                hide_index=True,
            )


def render_simulation(
    optimized_result,
    optimization_leaderboard: pd.DataFrame,
    optimized_v2_result,
    optimization_v2_leaderboard: pd.DataFrame,
    optimized_v3_result,
    optimization_v3_leaderboard: pd.DataFrame,
    optimized_v4_result,
    optimization_v4_leaderboard: pd.DataFrame,
    gold_ohlc: pd.DataFrame,
) -> None:
    st.subheader("Simulasi Trading XAU/USD Multi-Fase")
    st.caption(
        "Simulasi hanya menampilkan Strategi Terbaik Optimizer dan Strategi Terbaik v.2. "
        "Model 1 dan Model 2 lama dihapus dari tab ini karena performanya tidak memadai."
    )
    st.warning(
        "Asumsi: equity awal USD 1.000, target tiap fase +20%, maksimal 8 BUY dan 10 SELL. "
        "Swap BUY USD 0.2 per hari per 0.01 lot; SELL dianggap USD 0.0. "
        "Data memakai OHLC harian GC=F, sehingga jika TP dan SL tersentuh dalam candle yang sama, SL dianggap lebih dulu."
    )

    optimizer_tab, optimizer_v2_tab, optimizer_v3_tab, optimizer_v4_tab = st.tabs(
        ["Strategi Terbaik Optimizer", "Strategi Terbaik v.2", "Strategi Optimizer v3", "Strategi Optimizer v4"]
    )
    with optimizer_tab:
        _render_multiphase_result("Strategi Terbaik Optimizer", optimized_result, optimization_leaderboard, gold_ohlc)
    with optimizer_v2_tab:
        _render_multiphase_result("Strategi Terbaik v.2", optimized_v2_result, optimization_v2_leaderboard, gold_ohlc)
    with optimizer_v3_tab:
        st.info(
            "v3 memakai parameter Strategi Terbaik Optimizer, lalu backtest ulang dengan rule Live Trading: "
            "anti-duplikat posisi aktif, re-entry setelah CL dengan buffer USD 3, dan guard candle entry."
        )
        _render_multiphase_result("Strategi Optimizer v3", optimized_v3_result, optimization_v3_leaderboard, gold_ohlc)
    with optimizer_v4_tab:
        st.info(
            "v4 memakai parameter Strategi Terbaik Optimizer, tetapi batas posisi BUY/SELL 8/10 diganti "
            "menjadi risk cap dinamis. Posisi boleh lebih banyak selama total potensi SL posisi terbuka "
            "masih berada di bawah batas risiko terhadap equity fase."
        )
        _render_multiphase_result("Strategi Optimizer v4", optimized_v4_result, optimization_v4_leaderboard, gold_ohlc)


def _render_signal_checklist(title: str, checklist: list[dict[str, object]], ready_status: str) -> None:
    total = len(checklist)
    passed = sum(bool(item.get("Lolos")) for item in checklist)
    if passed == total and total > 0:
        st.success(f"**{title}: Siap** ({passed}/{total} syarat)")
    else:
        st.warning(f"**{title}: Menunggu** ({passed}/{total} syarat)")

    rows = []
    for item in checklist:
        rows.append(
            {
                "Status": "LOLOS" if item.get("Lolos") else "BELUM",
                "Syarat": item.get("Syarat", "-"),
                "Nilai saat ini": item.get("Nilai saat ini", "-"),
            }
        )
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption(ready_status)


def _optimizer_best_params(leaderboard: pd.DataFrame) -> dict[str, object]:
    if leaderboard.empty:
        return {
            "Mode": "Trend",
            "Fast MA": 20,
            "Slow MA": 50,
            "Momentum hari": 10,
            "Threshold entry (%)": 0.15,
            "TP (USD)": 25.0,
            "SL (USD)": 18.0,
            "Strategi": "Fallback Optimizer",
            "Lot": 0.01,
        }
    best = leaderboard.iloc[0].to_dict()
    return {
        "Mode": best.get("Mode", "Trend"),
        "Fast MA": int(best.get("Fast MA", 20)),
        "Slow MA": int(best.get("Slow MA", 50)),
        "Momentum hari": int(best.get("Momentum hari", 10)),
        "Threshold entry (%)": float(best.get("Threshold entry (%)", 0.15)),
        "TP (USD)": float(best.get("TP (USD)", 25.0)),
        "SL (USD)": float(best.get("SL (USD)", 18.0)),
        "Strategi": best.get("Strategi", "Strategi Terbaik Optimizer"),
        "Lot": float(best.get("Lot", 0.01)),
    }


def _entry_checklist_rows(gold_ohlc: pd.DataFrame, params: dict[str, object], limit: int | None) -> pd.DataFrame:
    if gold_ohlc.empty:
        return pd.DataFrame()

    mode = str(params["Mode"])
    fast_window = int(params["Fast MA"])
    slow_window = int(params["Slow MA"])
    momentum_days = int(params["Momentum hari"])
    threshold = float(params["Threshold entry (%)"])

    frame = gold_ohlc.loc[gold_ohlc.index >= OPTIMIZATION_START].copy()
    close = frame["Close"].astype(float)
    high = frame["High"].astype(float)
    low = frame["Low"].astype(float)
    fast_ma = close.rolling(fast_window).mean()
    slow_ma = close.rolling(slow_window).mean()
    momentum = close.pct_change(momentum_days) * 100
    rsi = _rsi(close)
    previous_high = high.rolling(slow_window).max().shift(1)
    previous_low = low.rolling(slow_window).min().shift(1)

    rows: list[dict[str, object]] = []
    for current_date in frame.index:
        values = {
            "Close": close.loc[current_date],
            "MA cepat": fast_ma.loc[current_date],
            "MA lambat": slow_ma.loc[current_date],
            "Momentum": momentum.loc[current_date],
            "RSI": rsi.loc[current_date],
            "High acuan": previous_high.loc[current_date],
            "Low acuan": previous_low.loc[current_date],
        }

        if mode == "Trend":
            buy_checks = [
                ("Close > MA cepat", values["Close"], ">", values["MA cepat"], values["Close"] > values["MA cepat"]),
                ("MA cepat > MA lambat", values["MA cepat"], ">", values["MA lambat"], values["MA cepat"] > values["MA lambat"]),
                (f"Momentum {momentum_days} hari > +{threshold:.2f}%", values["Momentum"], ">", threshold, values["Momentum"] > threshold),
            ]
            sell_checks = [
                ("Close < MA cepat", values["Close"], "<", values["MA cepat"], values["Close"] < values["MA cepat"]),
                ("MA cepat < MA lambat", values["MA cepat"], "<", values["MA lambat"], values["MA cepat"] < values["MA lambat"]),
                (f"Momentum {momentum_days} hari < -{threshold:.2f}%", values["Momentum"], "<", -threshold, values["Momentum"] < -threshold),
            ]
        elif mode == "Breakout":
            buy_checks = [
                (f"Close > high {slow_window} hari sebelumnya", values["Close"], ">", values["High acuan"], values["Close"] > values["High acuan"]),
                (f"Momentum {momentum_days} hari > 0", values["Momentum"], ">", 0.0, values["Momentum"] > 0),
            ]
            sell_checks = [
                (f"Close < low {slow_window} hari sebelumnya", values["Close"], "<", values["Low acuan"], values["Close"] < values["Low acuan"]),
                (f"Momentum {momentum_days} hari < 0", values["Momentum"], "<", 0.0, values["Momentum"] < 0),
            ]
        elif mode == "Pullback":
            buy_checks = [
                ("Close > MA lambat", values["Close"], ">", values["MA lambat"], values["Close"] > values["MA lambat"]),
                ("RSI < 42", values["RSI"], "<", 42.0, values["RSI"] < 42),
                (f"Momentum {momentum_days} hari > -{threshold:.2f}%", values["Momentum"], ">", -threshold, values["Momentum"] > -threshold),
            ]
            sell_checks = [
                ("Close < MA lambat", values["Close"], "<", values["MA lambat"], values["Close"] < values["MA lambat"]),
                ("RSI > 58", values["RSI"], ">", 58.0, values["RSI"] > 58),
                (f"Momentum {momentum_days} hari < +{threshold:.2f}%", values["Momentum"], "<", threshold, values["Momentum"] < threshold),
            ]
        else:
            buy_checks = [("Mode strategi valid", pd.NA, "=", mode, False)]
            sell_checks = [("Mode strategi valid", pd.NA, "=", mode, False)]

        for direction, checks in [("BUY", buy_checks), ("SELL", sell_checks)]:
            passed = sum(bool(item[4]) for item in checks)
            total = len(checks)
            checklist = " | ".join(
                f"{item[0]} ({'LOLOS' if item[4] else 'BELUM'})" for item in checks
            )
            def _format_signal_value(value) -> str:
                numeric_value = pd.to_numeric(value, errors="coerce")
                if pd.isna(numeric_value):
                    return "-"
                return f"{numeric_value:,.2f}"

            row = {
                "Tanggal": current_date,
                "Strategi": "ENTRY",
                "Mode": mode,
                "Arah": direction,
                "Checklist sinyal harian": checklist,
                "Status": "LOLOS" if passed == total and total > 0 else "BELUM",
                "Keterangan": f"{passed}/{total} syarat lolos",
                "Close": values["Close"],
                "MA cepat": values["MA cepat"],
                "MA lambat": values["MA lambat"],
                "Momentum (%)": values["Momentum"],
                "RSI": values["RSI"],
            }
            for signal_number, item in enumerate(checks, start=1):
                row[f"Sinyal {signal_number}"] = item[0]
                row[f"Status Sinyal {signal_number}"] = "LOLOS" if item[4] else "BELUM"
                row[f"Nilai Sinyal {signal_number}"] = (
                    f"{_format_signal_value(item[1])} {item[2]} {_format_signal_value(item[3])}"
                )
            rows.append(row)

    result = pd.DataFrame(rows)
    result = result.sort_values("Tanggal", ascending=False)
    if limit is not None:
        result = result.head(limit)
    return result


def _exit_checklist_rows(gold_ohlc: pd.DataFrame, params: dict[str, object]) -> pd.DataFrame:
    ledger = load_live_ledger()
    if ledger.empty:
        return pd.DataFrame()

    latest_price = float(gold_ohlc.iloc[-1]["Close"]) if not gold_ohlc.empty else pd.NA
    latest_high = float(gold_ohlc.iloc[-1]["High"]) if not gold_ohlc.empty else pd.NA
    latest_low = float(gold_ohlc.iloc[-1]["Low"]) if not gold_ohlc.empty else pd.NA
    latest_date = gold_ohlc.index[-1] if not gold_ohlc.empty else pd.NaT
    rows: list[dict[str, object]] = []

    for _, position in ledger.iterrows():
        entry_price = pd.to_numeric(position.get("entry_price"), errors="coerce")
        lot = pd.to_numeric(position.get("lot"), errors="coerce")
        tp_usd = pd.to_numeric(position.get("tp_usd", params["TP (USD)"]), errors="coerce")
        cl_usd = pd.to_numeric(position.get("cl_usd", params["SL (USD)"]), errors="coerce")
        direction = str(position.get("arah", "-"))
        status = str(position.get("status", "-"))
        if pd.isna(entry_price) or pd.isna(lot) or lot <= 0:
            continue

        units = lot * 100
        tp_points = tp_usd / units
        cl_points = cl_usd / units
        if direction == "BUY":
            tp_price = entry_price + tp_points
            cl_price = entry_price - cl_points
            tp_hit = status == "CLOSED" and str(position.get("exit_reason", "")) == "TP tersentuh"
            cl_hit = status == "CLOSED" and str(position.get("exit_reason", "")) == "CL tersentuh"
            if status == "OPEN":
                tp_hit = pd.notna(latest_high) and latest_high >= tp_price
                cl_hit = pd.notna(latest_low) and latest_low <= cl_price
            tp_check = f"BUY TP: High/latest >= {tp_price:,.2f}"
            cl_check = f"BUY CL: Low/latest <= {cl_price:,.2f}"
        else:
            tp_price = entry_price - tp_points
            cl_price = entry_price + cl_points
            tp_hit = status == "CLOSED" and str(position.get("exit_reason", "")) == "TP tersentuh"
            cl_hit = status == "CLOSED" and str(position.get("exit_reason", "")) == "CL tersentuh"
            if status == "OPEN":
                tp_hit = pd.notna(latest_low) and latest_low <= tp_price
                cl_hit = pd.notna(latest_high) and latest_high >= cl_price
            tp_check = f"SELL TP: Low/latest <= {tp_price:,.2f}"
            cl_check = f"SELL CL: High/latest >= {cl_price:,.2f}"

        for exit_type, checklist, hit, target_price in [
            ("TP", tp_check, tp_hit, tp_price),
            ("CL/SL", cl_check, cl_hit, cl_price),
        ]:
            rows.append(
                {
                    "Position ID": position.get("position_id"),
                    "Strategi": "EXIT",
                    "Arah": direction,
                    "Status posisi": status,
                    "Tanggal entry": position.get("entry_time_wit"),
                    "Checklist exit": checklist,
                    "Status": "LOLOS" if hit else "BELUM",
                    "Keterangan": "Sudah memenuhi rule exit" if hit else "Belum memenuhi rule exit",
                    "Entry": entry_price,
                    "Target harga": target_price,
                    "Harga terbaru": latest_price,
                    "Tanggal data terbaru": latest_date,
                    "Tipe exit": exit_type,
                }
            )

    return pd.DataFrame(rows)


def _optimizer_logic_summary_rows(
    gold_ohlc: pd.DataFrame,
    params: dict[str, object],
    entry_rows: pd.DataFrame,
    exit_rows: pd.DataFrame,
) -> pd.DataFrame:
    latest_date = gold_ohlc.index.max() if not gold_ohlc.empty else pd.NaT
    latest_close = float(gold_ohlc.iloc[-1]["Close"]) if not gold_ohlc.empty else pd.NA
    latest_entry = pd.DataFrame()
    if not entry_rows.empty:
        latest_entry = entry_rows[entry_rows["Tanggal"].eq(entry_rows["Tanggal"].max())].copy()

    buy_entry = latest_entry[latest_entry["Arah"].eq("BUY")] if not latest_entry.empty else pd.DataFrame()
    sell_entry = latest_entry[latest_entry["Arah"].eq("SELL")] if not latest_entry.empty else pd.DataFrame()
    buy_ready = not buy_entry.empty and bool(buy_entry.iloc[0]["Status"] == "LOLOS")
    sell_ready = not sell_entry.empty and bool(sell_entry.iloc[0]["Status"] == "LOLOS")
    if buy_ready:
        entry_status = "BUY"
        entry_note = str(buy_entry.iloc[0]["Keterangan"])
    elif sell_ready:
        entry_status = "SELL"
        entry_note = str(sell_entry.iloc[0]["Keterangan"])
    else:
        entry_status = "NETRAL / BELUM"
        entry_note = "Belum ada arah BUY/SELL yang seluruh sinyalnya lolos pada data terbaru."

    if exit_rows.empty:
        exit_status = "BELUM ADA POSISI"
        exit_note = "Belum ada ledger Live Trading untuk dievaluasi."
    else:
        exit_ready = exit_rows[exit_rows["Status"].eq("LOLOS")]
        if exit_ready.empty:
            exit_status = "BELUM EXIT"
            exit_note = "Belum ada TP atau CL/SL yang memenuhi rule exit."
        else:
            exit_directions = sorted(exit_ready["Arah"].dropna().astype(str).unique())
            exit_status = " / ".join(f"EXIT {direction}" for direction in exit_directions)
            exit_note = f"{len(exit_ready)} checklist exit sudah LOLOS."

    return pd.DataFrame(
        [
            {
                "Elemen Strategi": "1. Data yang Dipakai",
                "Keterangan": (
                    "Data utama memakai OHLC harian GC=F. Evaluasi terbaru: "
                    f"{_format_date(latest_date)} dengan close "
                    f"{'-' if pd.isna(latest_close) else f'USD {latest_close:,.2f}'}."
                ),
                "Status BUY / SELL": "NETRAL / INFORMASI",
            },
            {
                "Elemen Strategi": "2. Mode Strategi",
                "Keterangan": (
                    f"Mode aktif Optimizer adalah {params['Mode']}. Mode ini menentukan apakah rule yang dipakai "
                    "berbasis tren, breakout, atau pullback."
                ),
                "Status BUY / SELL": "BUY / SELL",
            },
            {
                "Elemen Strategi": "3. Parameter Strategi",
                "Keterangan": (
                    f"MA {params['Fast MA']}/{params['Slow MA']}, momentum {params['Momentum hari']} hari, "
                    f"threshold {params['Threshold entry (%)']:.2f}%, TP USD {params['TP (USD)']:,.2f}, "
                    f"CL/SL USD {params['SL (USD)']:,.2f}, lot {params['Lot']:.2f}."
                ),
                "Status BUY / SELL": "NETRAL / PARAMETER",
            },
            {
                "Elemen Strategi": "4. Sinyal Entry",
                "Keterangan": f"Entry membaca seluruh checklist BUY dan SELL pada candle harian terbaru. {entry_note}",
                "Status BUY / SELL": entry_status,
            },
            {
                "Elemen Strategi": "5. Sinyal Exit",
                "Keterangan": f"Exit membaca TP dan CL/SL dari posisi Live Trading yang tercatat. {exit_note}",
                "Status BUY / SELL": exit_status,
            },
            {
                "Elemen Strategi": "6. Multi-Fase Equity",
                "Keterangan": (
                    "Simulasi Optimizer mengejar target +20% per fase. Saat target fase tercapai, semua posisi "
                    "ditutup dan fase berikutnya dimulai dari equity baru."
                ),
                "Status BUY / SELL": "MANAJEMEN MODAL",
            },
            {
                "Elemen Strategi": "7. Penilaian Strategi",
                "Keterangan": (
                    "Strategi terbaik dipilih dari kombinasi fase selesai, equity akhir, max drawdown, "
                    "profit factor, dan jumlah transaksi."
                ),
                "Status BUY / SELL": "EVALUASI PERFORMA",
            },
            {
                "Elemen Strategi": "8. Batas Posisi",
                "Keterangan": (
                    "Live Trading membatasi maksimal 8 BUY dan 10 SELL, serta tidak mencatat ulang kombinasi "
                    "tanggal sinyal dan arah yang sama."
                ),
                "Status BUY / SELL": "KONTROL BUY / SELL",
            },
        ]
    )


def render_optimizer_signals(gold_ohlc: pd.DataFrame, optimization_leaderboard: pd.DataFrame) -> None:
    params = _optimizer_best_params(optimization_leaderboard)
    st.subheader("Sinyal Optimizer")
    st.caption(
        "Tab ini memisahkan checklist strategi ENTRY dan EXIT agar proses Strategi Terbaik Optimizer "
        "lebih mudah diawasi dan dipelajari."
    )
    st.info(
        f"Strategi aktif: **{params['Strategi']}** | Mode **{params['Mode']}** | "
        f"MA {params['Fast MA']}/{params['Slow MA']} | Momentum {params['Momentum hari']} hari | "
        f"Threshold {params['Threshold entry (%)']:.2f}% | TP USD {params['TP (USD)']:,.2f} | CL/SL USD {params['SL (USD)']:,.2f}"
    )

    summary_tab, evaluation_entry_tab, exit_tab = st.tabs(
        ["Ringkasan Logika Optimizer", "Evaluasi Strategi Entry", "Evaluasi Strategi Exit"]
    )
    with summary_tab:
        entry_rows = _entry_checklist_rows(gold_ohlc, params, None)
        exit_rows = _exit_checklist_rows(gold_ohlc, params)
        summary_rows = _optimizer_logic_summary_rows(gold_ohlc, params, entry_rows, exit_rows)
        st.dataframe(summary_rows, use_container_width=True, hide_index=True)
        st.caption(
            "Kolom status menunjukkan apakah elemen tersebut sedang mengarah ke BUY, SELL, EXIT, "
            "atau hanya menjadi informasi/manajemen risiko."
        )

    with evaluation_entry_tab:
        limit_choice = st.selectbox(
            "Jumlah evaluasi ENTRY yang ditampilkan",
            ["120 baris terbaru", "240 baris terbaru", "Semua"],
            index=0,
        )
        limit = None if limit_choice == "Semua" else int(limit_choice.split()[0])
        entry_rows = _entry_checklist_rows(gold_ohlc, params, limit)
        if entry_rows.empty:
            st.warning("Belum ada data ENTRY yang dapat dievaluasi.")
        else:
            completed = entry_rows[entry_rows["Status"].eq("LOLOS")]
            c1, c2, c3 = st.columns(3)
            c1.metric("Evaluasi ENTRY tampil", f"{len(entry_rows):,.0f}")
            c2.metric("Sinyal LOLOS", f"{len(completed):,.0f}")
            c3.metric("Sinyal BELUM", f"{len(entry_rows) - len(completed):,.0f}")
            st.dataframe(
                entry_rows.style.format(
                    {
                        "Close": "${:,.2f}",
                        "MA cepat": "${:,.2f}",
                        "MA lambat": "${:,.2f}",
                        "Momentum (%)": "{:+.2f}%",
                        "RSI": "{:.1f}",
                    },
                    na_rep="-",
                ),
                use_container_width=True,
                hide_index=True,
            )

    with exit_tab:
        exit_rows = _exit_checklist_rows(gold_ohlc, params)
        if exit_rows.empty:
            st.info("Belum ada ledger Live Trading untuk dievaluasi pada strategi EXIT.")
        else:
            completed = exit_rows[exit_rows["Status"].eq("LOLOS")]
            c1, c2, c3 = st.columns(3)
            c1.metric("Checklist EXIT", f"{len(exit_rows):,.0f}")
            c2.metric("EXIT LOLOS", f"{len(completed):,.0f}")
            c3.metric("EXIT BELUM", f"{len(exit_rows) - len(completed):,.0f}")
            st.dataframe(
                exit_rows.sort_values(["Status posisi", "Position ID", "Tipe exit"]).style.format(
                    {
                        "Entry": "${:,.2f}",
                        "Target harga": "${:,.2f}",
                        "Harga terbaru": "${:,.2f}",
                    },
                    na_rep="-",
                ),
                use_container_width=True,
                hide_index=True,
            )


def _render_manual_exit_comparison(live: dict[str, object]) -> None:
    summary = live["summary"]
    ledger = live["ledger"].copy()
    manual_loader = getattr(live_trading_module, "load_manual_exit_ledger", None)
    manual_recorder = getattr(live_trading_module, "record_manual_exit", None)
    manual_exits = manual_loader() if manual_loader is not None else pd.DataFrame()
    latest_price = summary["Latest price"]

    st.markdown("**Perbandingan Optimizer vs Intervensi Manual**")
    st.caption(
        "Tombol Exit Manual hanya mencatat keputusan manusia pada harga data terbaru. "
        "Posisi Optimizer tidak ditutup oleh tombol ini dan tetap berjalan sampai TP/CL algoritma tersentuh."
    )

    trade_rows = ledger[
        ledger["status"].isin(["OPEN", "CLOSED"])
        & pd.to_numeric(ledger["entry_price"], errors="coerce").notna()
    ].copy()
    if trade_rows.empty:
        st.info("Belum ada posisi Optimizer yang bisa dibandingkan dengan keputusan manual.")
        return

    manual_by_position = {}
    if not manual_exits.empty:
        manual_sorted = manual_exits.copy()
        manual_sorted["position_id_numeric"] = pd.to_numeric(manual_sorted["position_id"], errors="coerce")
        for _, manual_row in manual_sorted.dropna(subset=["position_id_numeric"]).iterrows():
            manual_by_position[int(manual_row["position_id_numeric"])] = manual_row

    rows = []
    optimizer_current_total = 0.0
    optimizer_closed_total = 0.0
    human_total = 0.0
    paired_optimizer_total = 0.0
    for _, row in trade_rows.iterrows():
        position_id = int(pd.to_numeric(row["position_id"], errors="coerce"))
        direction = str(row["arah"])
        lot = float(pd.to_numeric(row["lot"], errors="coerce") or 0.0)
        entry_price = float(pd.to_numeric(row["entry_price"], errors="coerce") or 0.0)
        swap = float(pd.to_numeric(row.get("swap", 0.0), errors="coerce") or 0.0)
        if row["status"] == "CLOSED":
            optimizer_net = float(pd.to_numeric(row.get("net_pl", 0.0), errors="coerce") or 0.0)
            optimizer_exit_price = float(pd.to_numeric(row.get("exit_price", 0.0), errors="coerce") or 0.0)
            optimizer_net_status = "Closed"
            optimizer_closed_total += optimizer_net
        elif pd.isna(latest_price):
            optimizer_net = 0.0
            optimizer_exit_price = pd.NA
            optimizer_net_status = "Floating - harga belum tersedia"
        else:
            if direction == "BUY":
                floating = (float(latest_price) - entry_price) * lot * 100
            else:
                floating = (entry_price - float(latest_price)) * lot * 100
            optimizer_net = floating + swap
            optimizer_exit_price = pd.NA
            optimizer_net_status = "Floating"
        optimizer_current_total += optimizer_net

        manual_row = manual_by_position.get(position_id)
        if manual_row is None:
            manual_exit_price = pd.NA
            manual_net = pd.NA
            manual_label = "Belum exit manual"
        else:
            manual_exit_price = float(pd.to_numeric(manual_row.get("manual_exit_price", 0.0), errors="coerce") or 0.0)
            manual_net = float(pd.to_numeric(manual_row.get("manual_net_pl", 0.0), errors="coerce") or 0.0)
            manual_label = f"{manual_row.get('manual_result_label', 'Manual')} ({manual_net:+,.2f})"
            human_total += manual_net
            paired_optimizer_total += optimizer_net

        rows.append(
            {
                "Position ID": position_id,
                "Posisi Entry Optimizer (Buy/Sell)": direction,
                "Harga Entry": entry_price,
                "Nilai TP/CL Optimizer": f"TP ${float(row['tp_usd']):,.2f} | CL ${float(row['cl_usd']):,.2f}",
                "Status Optimizer": row["status"],
                "Exit Price Optimizer": optimizer_exit_price,
                "Net Profit Optimizer": optimizer_net,
                "Status Net Optimizer": optimizer_net_status,
                "Exit Price Manual": manual_exit_price,
                "TP/CL Manusia": manual_label,
                "Net Profit Manusia": manual_net,
            }
        )

    paired_delta = human_total - paired_optimizer_total
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Net Optimizer saat ini", f"${optimizer_current_total:+,.2f}")
    m2.metric("Net Optimizer closed", f"${optimizer_closed_total:+,.2f}")
    m3.metric("Net Manusia tercatat", f"${human_total:+,.2f}")
    m4.metric("Selisih posisi yang sama", f"${paired_delta:+,.2f}")

    comparison_frame = pd.DataFrame(rows)
    money_columns = [
        "Harga Entry",
        "Exit Price Optimizer",
        "Net Profit Optimizer",
        "Exit Price Manual",
        "Net Profit Manusia",
    ]
    st.dataframe(
        comparison_frame.style.format(
            {column: "${:,.2f}" for column in money_columns if column in comparison_frame.columns},
            subset=[column for column in money_columns if column in comparison_frame.columns],
            na_rep="-",
        ),
        use_container_width=True,
        hide_index=True,
    )

    open_action_rows = trade_rows[trade_rows["status"].eq("OPEN")].copy()
    open_action_rows = open_action_rows[
        ~pd.to_numeric(open_action_rows["position_id"], errors="coerce").isin(list(manual_by_position.keys()))
    ]
    if open_action_rows.empty:
        st.info("Tidak ada posisi terbuka yang menunggu exit manual baru.")
        return
    if pd.isna(latest_price):
        st.warning("Harga terbaru belum tersedia, tombol exit manual dinonaktifkan.")
        return
    if manual_recorder is None:
        st.warning("Fungsi exit manual belum siap pada runtime ini. Refresh/redeploy aplikasi, lalu coba lagi.")
        return

    st.markdown("**Aksi Exit Manual**")
    for _, row in open_action_rows.iterrows():
        position_id = int(pd.to_numeric(row["position_id"], errors="coerce"))
        direction = str(row["arah"])
        entry_price = float(pd.to_numeric(row["entry_price"], errors="coerce") or 0.0)
        if direction == "BUY":
            floating = (float(latest_price) - entry_price) * float(row["lot"]) * 100
        else:
            floating = (entry_price - float(latest_price)) * float(row["lot"]) * 100
        swap = float(pd.to_numeric(row.get("swap", 0.0), errors="coerce") or 0.0)
        net_now = floating + swap
        action_cols = st.columns([2, 2, 2, 2])
        action_cols[0].write(f"#{position_id} | {direction}")
        action_cols[1].write(f"Entry ${entry_price:,.2f}")
        action_cols[2].write(f"Manual net sekarang ${net_now:+,.2f}")
        if action_cols[3].button(f"Exit Manual #{position_id}", key=f"manual_exit_{position_id}"):
            _, message, success = manual_recorder(position_id, float(latest_price))
            if success:
                st.success(message)
            else:
                st.warning(message)
            st.rerun()


def render_live_trading(gold_ohlc: pd.DataFrame, optimization_leaderboard: pd.DataFrame) -> None:
    live = run_live_trading_update(gold_ohlc, optimization_leaderboard)
    summary = live["summary"]
    params = live["params"]
    signal = live["signal"]
    waiting_state = live["waiting_state"]
    trigger_state = live["trigger_state"]
    signals = live["signals"].copy()
    open_positions = live["open_positions"].copy()
    closed_positions = live["closed_positions"].copy()

    st.subheader("Live Trading - Strategi Terbaik Optimizer")
    st.caption(
        "Mode ini adalah paper-trading live berbasis data dashboard. Posisi dibuka mengikuti pola "
        "Strategi Terbaik Optimizer, bukan order broker sungguhan."
    )
    st.warning(
        "Asumsi live: equity awal USD 1.000, lot tetap 0.01, swap BUY USD 0.02 per 0.01 lot per hari, "
        "swap SELL USD 0. Jam trading Senin-Jumat 07:00 WIT sampai 06:00 WIT hari berikutnya; "
        f"Sabtu dan Minggu tidak membuka posisi baru. Ledger live dimulai **{LIVE_START_DATE.strftime('%d %b %Y')}**."
    )
    st.info(
        "Live Trading sekarang memakai **Strategi Terbaik Optimizer secara penuh**. "
        "Entry mengikuti sinyal Optimizer, boleh menambah posisi dari sinyal hari berbeda sampai batas "
        "maksimal 8 BUY dan 10 SELL. Re-entry guard USD 3 tidak dipakai untuk eksekusi live."
    )

    status_text = "Aktif" if summary["Can trade"] else "Tidak membuka posisi baru"
    st.info(
        f"Waktu dashboard: **{summary['Now WIT'].strftime('%d %b %Y %H:%M:%S WIT')}** | "
        f"Status sesi: **{status_text}** | {summary['Session note']}"
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Equity live", f"${summary['Equity']:,.2f}", f"{summary['Equity'] - LIVE_INITIAL_EQUITY:+,.2f}")
    c2.metric("Balance live", f"${summary['Balance']:,.2f}")
    c3.metric("Floating P/L", f"${summary['Floating P/L']:+,.2f}")
    c4.metric("Open posisi", f"{summary['Open BUY'] + summary['Open SELL']}", f"BUY {summary['Open BUY']} | SELL {summary['Open SELL']}")

    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Harga data terbaru", "-" if pd.isna(summary["Latest price"]) else f"${summary['Latest price']:,.2f}")
    p2.metric("Tanggal data", "-" if pd.isna(summary["Latest data date"]) else pd.Timestamp(summary["Latest data date"]).strftime("%d %b %Y"))
    p3.metric("Closed net P/L", f"${summary['Closed net P/L']:+,.2f}")
    p4.metric("Swap posisi terbuka", f"${summary['Open swap']:+,.2f}")

    strategy_frame = pd.DataFrame(
        [
            {"Parameter": "Strategi", "Nilai": params["Strategi"]},
            {"Parameter": "Mode", "Nilai": params["Mode"]},
            {"Parameter": "MA cepat/lambat", "Nilai": f"{params['Fast MA']} / {params['Slow MA']}"},
            {"Parameter": "Momentum hari", "Nilai": params["Momentum hari"]},
            {"Parameter": "Threshold entry", "Nilai": f"{params['Threshold entry (%)']:.2f}%"},
            {"Parameter": "TP per posisi", "Nilai": f"USD {params['TP (USD)']:,.2f}"},
            {"Parameter": "CL/SL per posisi", "Nilai": f"USD {params['SL (USD)']:,.2f}"},
            {"Parameter": "Lot live", "Nilai": "0.01 tetap"},
        ]
    )
    st.markdown("**Pola Strategi yang Dipakai**")
    st.dataframe(strategy_frame, use_container_width=True, hide_index=True)

    st.markdown("**Sinyal Strategi Saat Ini**")
    if signal is None:
        st.info(
            "Belum ada sinyal optimizer yang memenuhi threshold dari data terbaru. "
            f"Status sekarang: **{waiting_state['Status sinyal']}**."
        )
    else:
        signal_direction = signal["arah"]
        signal_message = (
            f"Sinyal terakhir **{signal_direction}** pada **{pd.Timestamp(signal['signal_date']).strftime('%d %b %Y')}** "
            f"dengan prediksi **${signal['prediction']:,.2f}**, harga referensi **${signal['reference_price']:,.2f}**, "
            f"expected change **{signal['expected_change_pct']:+.2f}%**."
        )
        if signal_direction == "BUY":
            st.success(signal_message)
        elif signal_direction == "SELL":
            st.error(signal_message)
        else:
            st.info(signal_message)

    st.markdown("**Status Trigger Optimizer**")
    trigger_note = trigger_state["Catatan"]
    trigger_status = trigger_state["Status trigger"]
    if str(trigger_status).startswith("Siap buka"):
        st.success(f"**{trigger_status}** | {trigger_note}")
    elif trigger_status in {"Sinyal sudah dieksekusi/dicatat", "Slot BUY penuh", "Slot SELL penuh"}:
        st.warning(f"**{trigger_status}** | {trigger_note}")
    elif trigger_status == "Menunggu threshold arah":
        st.info(f"**{trigger_status}** | {trigger_note}")
    else:
        st.info(f"**{trigger_status}** | {trigger_note}")

    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Arah sinyal Optimizer", trigger_state["Arah sinyal"])
    t2.metric(
        "Expected change",
        "-" if pd.isna(trigger_state["Expected change (%)"]) else f"{trigger_state['Expected change (%)']:+.2f}%",
        f"Threshold {trigger_state['Threshold entry (%)']:.2f}%",
    )
    t3.metric(
        "Harga referensi",
        "-" if pd.isna(trigger_state["Harga referensi"]) else f"${trigger_state['Harga referensi']:,.2f}",
    )
    t4.metric(
        "Prediksi Optimizer",
        "-" if pd.isna(trigger_state["Prediksi"]) else f"${trigger_state['Prediksi']:,.2f}",
    )

    slot_frame = pd.DataFrame(
        [
            {"Parameter": "Tanggal sinyal", "Nilai": "-" if pd.isna(trigger_state["Tanggal sinyal"]) else pd.Timestamp(trigger_state["Tanggal sinyal"]).strftime("%d %b %Y")},
            {"Parameter": "Posisi BUY terbuka", "Nilai": trigger_state["Posisi BUY terbuka"]},
            {"Parameter": "Sisa slot BUY", "Nilai": trigger_state["Sisa slot BUY"]},
            {"Parameter": "Posisi SELL terbuka", "Nilai": trigger_state["Posisi SELL terbuka"]},
            {"Parameter": "Sisa slot SELL", "Nilai": trigger_state["Sisa slot SELL"]},
            {"Parameter": "Sinyal tanggal/arah sudah dicatat", "Nilai": "Ya" if trigger_state["Sudah dieksekusi"] else "Belum"},
        ]
    )
    st.dataframe(slot_frame, use_container_width=True, hide_index=True)

    trigger_checklist = pd.DataFrame(trigger_state["Checklist"])
    if not trigger_checklist.empty:
        st.dataframe(trigger_checklist, use_container_width=True, hide_index=True)
    st.caption(
        "Catatan pembelajaran: Strategi Terbaik Optimizer memakai sinyal harian. "
        "Posisi baru dibuka hanya ketika arah sinyal melewati threshold, sesi trading aktif, slot posisi tersedia, "
        "dan kombinasi tanggal sinyal + arah belum pernah dicatat di ledger."
    )

    st.markdown("**Sinyal yang Sedang Ditunggu**")
    st.write(waiting_state["Yang ditunggu"])
    w1, w2, w3, w4 = st.columns(4)
    w1.metric(
        "Momentum sekarang",
        "-" if pd.isna(waiting_state["Momentum saat ini"]) else f"{waiting_state['Momentum saat ini']:+.2f}%",
        f"Threshold {waiting_state['Threshold']:.2f}%",
    )
    w2.metric("MA cepat", "-" if pd.isna(waiting_state["MA cepat"]) else f"${waiting_state['MA cepat']:,.2f}")
    w3.metric("MA lambat", "-" if pd.isna(waiting_state["MA lambat"]) else f"${waiting_state['MA lambat']:,.2f}")
    w4.metric("RSI", "-" if pd.isna(waiting_state["RSI"]) else f"{waiting_state['RSI']:.1f}")

    buy_col, sell_col = st.columns(2)
    with buy_col:
        _render_signal_checklist(
            "Kondisi BUY",
            waiting_state["Checklist BUY"],
            "BUY hanya dibuka jika seluruh syarat BUY berstatus LOLOS.",
        )
    with sell_col:
        _render_signal_checklist(
            "Kondisi SELL",
            waiting_state["Checklist SELL"],
            "SELL hanya dibuka jika seluruh syarat SELL berstatus LOLOS.",
        )

    if waiting_state["Status sinyal"] == "Kondisi BUY siap":
        st.success(f"**Interpretasi:** {waiting_state['Interpretasi']}")
    elif waiting_state["Status sinyal"] == "Kondisi SELL siap":
        st.error(f"**Interpretasi:** {waiting_state['Interpretasi']}")
    else:
        st.info(f"**Kondisi Netral / Tunggu:** {waiting_state['Interpretasi']}")

    format_columns = {
        "lot": "{:.2f}",
        "prediction": "${:,.2f}",
        "reference_price": "${:,.2f}",
        "expected_change_pct": "{:+.2f}%",
        "threshold_entry_pct": "{:.2f}%",
        "tp_usd": "${:,.2f}",
        "cl_usd": "${:,.2f}",
        "entry_price": "${:,.2f}",
        "exit_price": "${:,.2f}",
        "gross_pl": "${:+,.2f}",
        "swap": "${:+,.2f}",
        "net_pl": "${:+,.2f}",
    }

    st.markdown("**Sinyal dan Status Posisi**")
    signal_columns = [
        "position_id",
        "signal_date",
        "detected_at_wit",
        "status",
        "arah",
        "lot",
        "prediction",
        "reference_price",
        "expected_change_pct",
        "entry_time_wit",
        "entry_price",
        "catatan",
    ]
    if signals.empty:
        st.info("Belum ada sinyal live yang tercatat sejak ledger dimulai.")
    else:
        signal_display = signals[[column for column in signal_columns if column in signals.columns]].tail(50)
        st.dataframe(
            signal_display.style.format(
                format_columns,
                subset=[column for column in format_columns if column in signal_display.columns],
                na_rep="-",
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("**Posisi Terbuka**")
    if open_positions.empty:
        st.info("Tidak ada posisi terbuka saat ini.")
    else:
        open_display = open_positions.copy()
        open_display["floating_pl"] = open_display.apply(
            lambda row: (
                (summary["Latest price"] - float(row["entry_price"])) * float(row["lot"]) * 100
                if row["arah"] == "BUY"
                else (float(row["entry_price"]) - summary["Latest price"]) * float(row["lot"]) * 100
            ),
            axis=1,
        )
        open_columns = [
            "position_id",
            "status",
            "arah",
            "lot",
            "entry_time_wit",
            "entry_price",
            "tp_usd",
            "cl_usd",
            "swap",
            "floating_pl",
            "catatan",
        ]
        open_table = open_display[[column for column in open_columns if column in open_display.columns]]
        st.dataframe(
            open_table.style.format(
                {**format_columns, "floating_pl": "${:+,.2f}"},
                subset=[column for column in [*format_columns, "floating_pl"] if column in open_table.columns],
                na_rep="-",
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("**Riwayat Posisi Close**")
    if closed_positions.empty:
        st.info("Belum ada posisi yang close.")
    else:
        closed_columns = [
            "position_id",
            "signal_date",
            "arah",
            "lot",
            "entry_time_wit",
            "entry_price",
            "exit_time_wit",
            "exit_price",
            "exit_reason",
            "gross_pl",
            "swap",
            "net_pl",
        ]
        closed_table = closed_positions[[column for column in closed_columns if column in closed_positions.columns]].tail(100)
        st.dataframe(
            closed_table.style.format(
                format_columns,
                subset=[column for column in format_columns if column in closed_table.columns],
                na_rep="-",
            ),
            use_container_width=True,
            hide_index=True,
        )

    _render_manual_exit_comparison(live)


def render_monitoring(title: str, data_path) -> None:
    st.subheader(f"{title}: Estimasi vs Aktual Intraday")
    st.caption(
        "Estimasi disimpan otomatis pada 23:59 WIT. Aktual diisi dari candle intraday "
        "GC=F pertama pada/setelah 08:00, 09:00, 10:00, 11:00, dan 12:00 WIT hari berikutnya."
    )

    frame = load_monitoring(data_path)
    if frame.empty:
        st.info("Belum ada data monitoring. Baris pertama akan dibuat oleh job 23:59 WIT.")
        return

    timestamp_columns = ["forecast_timestamp_wit"] + [f"actual_timestamp_{hour_suffix(hour)}" for hour in ACTUAL_HOURS]
    timestamp_candidates = pd.concat(
        [pd.to_datetime(frame[column], errors="coerce") for column in timestamp_columns if column in frame.columns]
    ).dropna()
    if timestamp_candidates.empty:
        st.warning("Data monitoring sudah tersedia, tetapi belum ada timestamp update yang valid.")
    else:
        last_monitoring_update = timestamp_candidates.max()
        st.info(f"Data monitoring terakhir diperbarui: **{last_monitoring_update.strftime('%d %b %Y %H:%M:%S WIT')}**")

    summary = monitoring_summary(frame)
    completed_summary = summary[pd.to_numeric(summary["Jumlah selesai"], errors="coerce") > 0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Total perbandingan selesai", f"{int(completed_summary['Jumlah selesai'].sum())}" if not completed_summary.empty else "0")
    if completed_summary.empty:
        c2.metric("Jam MAE terbaik", "-")
        c3.metric("Jam arah terbaik", "-")
    else:
        best_mae = completed_summary.loc[pd.to_numeric(completed_summary["MAE"], errors="coerce").idxmin()]
        best_direction = completed_summary.loc[pd.to_numeric(completed_summary["Akurasi arah"], errors="coerce").idxmax()]
        c2.metric("Jam MAE terbaik", f"{best_mae['Jam WIT']} - ${best_mae['MAE']:,.2f}")
        c3.metric("Jam arah terbaik", f"{best_direction['Jam WIT']} - {best_direction['Akurasi arah']:.1f}%")

    st.markdown("**Summary Akurasi Per Jam Aktual**")
    st.dataframe(
        summary.style.format(
            {
                "MAE": "${:,.2f}",
                "MAPE": "{:.2f}%",
                "Akurasi arah": "{:.1f}%",
                "Bias rata-rata": "${:+,.2f}",
            },
            na_rep="-",
        ),
        use_container_width=True,
        hide_index=True,
    )

    display = frame.copy()
    numeric_columns = [
        "estimate_tomorrow",
        "reference_price",
        "estimate_lower",
        "estimate_upper",
        "confidence",
    ]
    for hour in ACTUAL_HOURS:
        suffix = hour_suffix(hour)
        numeric_columns.extend([f"actual_open_{suffix}", f"delta_{suffix}", f"delta_pct_{suffix}"])
    for column in numeric_columns:
        display[column] = pd.to_numeric(display[column], errors="coerce")

    actual_column_order = [
        field
        for hour in ACTUAL_HOURS
        for suffix in (hour_suffix(hour),)
        for field in (
            f"actual_timestamp_{suffix}",
            f"actual_open_{suffix}",
            f"delta_{suffix}",
            f"delta_pct_{suffix}",
            f"actual_direction_{suffix}",
            f"direction_correct_{suffix}",
        )
    ]
    column_order = [
        "forecast_date_wit",
        "target_date_wit",
        "forecast_timestamp_wit",
        "reference_price",
        "estimate_tomorrow",
        "estimated_direction",
        "signal",
        "confidence",
        "status",
        "notes",
    ] + actual_column_order
    format_columns = {
        "estimate_tomorrow": "${:,.2f}",
        "reference_price": "${:,.2f}",
        "confidence": "{:.0f}%",
    }
    for hour in ACTUAL_HOURS:
        suffix = hour_suffix(hour)
        format_columns[f"actual_open_{suffix}"] = "${:,.2f}"
        format_columns[f"delta_{suffix}"] = "${:+,.2f}"
        format_columns[f"delta_pct_{suffix}"] = "{:+.2f}%"
    st.dataframe(
        display[column_order].style.format(
            format_columns,
            na_rep="-",
        ),
        use_container_width=True,
        hide_index=True,
    )


def render_intraday_audit(gold_ohlc: pd.DataFrame) -> None:
    st.subheader("Audit Data Intraday Pihak Ketiga")
    st.warning(
        "Dataset 1-minute XAUUSD ini dicatat sebagai **data pihak ketiga**. "
        "Hasil audit di tab ini dipakai untuk validasi kualitas data, belum untuk mengganti model utama atau Live Trading."
    )

    source_mode = st.radio("Sumber CSV", ["Upload CSV", "URL CSV"], horizontal=True)
    raw_source = None
    if source_mode == "Upload CSV":
        uploaded_file = st.file_uploader("Upload file CSV intraday", type=["csv"])
        if uploaded_file is not None:
            raw_source = uploaded_file.getvalue()
    else:
        csv_url = st.text_input(
            "URL CSV mentah",
            placeholder="Contoh: https://raw.githubusercontent.com/.../XAUUSD_1m.csv",
        )
        if csv_url:
            raw_source = csv_url.strip()

    st.caption(
        "Kolom yang didukung: `time/open/high/low/close/volume`. Alias umum seperti "
        "`datetime`, `timestamp`, `o/h/l/c`, dan `vol` akan dinormalisasi otomatis."
    )

    if raw_source is None:
        st.info("Masukkan file atau URL CSV untuk mulai audit.")
        return

    try:
        intraday = load_intraday_csv(raw_source)
        audit = audit_intraday_data(intraday)
    except Exception as exc:
        st.error(f"CSV intraday belum bisa diaudit: {exc}")
        return

    metrics = audit.metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jumlah baris", f"{metrics.get('Jumlah baris', 0):,}")
    c2.metric("Hari agregasi", f"{metrics.get('Jumlah hari agregasi', 0):,}")
    c3.metric("Gap > 1 menit", f"{metrics.get('Gap > 1 menit', 0):,}")
    c4.metric("Duplikat timestamp", f"{metrics.get('Duplikat timestamp', 0):,}")

    period_start = metrics.get("Periode awal UTC")
    period_end = metrics.get("Periode akhir UTC")
    if pd.notna(period_start) and pd.notna(period_end):
        st.info(
            "Periode data intraday: "
            f"**{pd.Timestamp(period_start).strftime('%d %b %Y %H:%M UTC')}** sampai "
            f"**{pd.Timestamp(period_end).strftime('%d %b %Y %H:%M UTC')}**. "
            "Untuk Live Trading WIT, timestamp ini harus dikonversi dan dicek lagi."
        )

    st.markdown("**Checklist Kualitas Data**")
    st.dataframe(audit.issues, use_container_width=True, hide_index=True)

    if not audit.daily_ohlc.empty:
        st.markdown("**Agregasi Harian dari Data 1 Menit**")
        daily_preview = audit.daily_ohlc.tail(20).copy()
        st.dataframe(
            daily_preview.style.format(
                {
                    "open": "${:,.2f}",
                    "high": "${:,.2f}",
                    "low": "${:,.2f}",
                    "close": "${:,.2f}",
                    "volume": "{:,.0f}",
                }
            ),
            use_container_width=True,
        )

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=audit.daily_ohlc.index, y=audit.daily_ohlc["close"], name="Close intraday -> daily"))
        fig.update_layout(
            title="Close Harian Hasil Agregasi Data 1 Menit",
            yaxis_title="XAUUSD",
            height=420,
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

        daily_compare = audit.daily_ohlc[["close"]].copy()
        daily_compare.index = pd.to_datetime(daily_compare.index).tz_convert(None).normalize()
        main_daily = gold_ohlc[["Close"]].copy()
        main_daily.index = pd.to_datetime(main_daily.index).normalize()
        comparison = daily_compare.join(main_daily, how="inner").rename(columns={"close": "Close intraday", "Close": "Close data utama"})
        if comparison.empty:
            st.warning("Belum ada overlap tanggal antara data intraday dan data utama dashboard.")
        else:
            comparison["Selisih"] = comparison["Close intraday"] - comparison["Close data utama"]
            comparison["Selisih %"] = comparison["Selisih"] / comparison["Close data utama"] * 100
            direction_match = (
                comparison["Close intraday"].pct_change().gt(0)
                == comparison["Close data utama"].pct_change().gt(0)
            )
            direction_accuracy = float(direction_match.iloc[1:].mean() * 100) if len(direction_match) > 1 else pd.NA
            m1, m2, m3 = st.columns(3)
            m1.metric("Hari overlap", f"{len(comparison):,}")
            m2.metric("MAE close vs data utama", f"${comparison['Selisih'].abs().mean():,.2f}")
            m3.metric("Kecocokan arah harian", "-" if pd.isna(direction_accuracy) else f"{direction_accuracy:.1f}%")

            st.markdown("**Perbandingan dengan Data Utama Dashboard**")
            st.dataframe(
                comparison.tail(30).style.format(
                    {
                        "Close intraday": "${:,.2f}",
                        "Close data utama": "${:,.2f}",
                        "Selisih": "${:+,.2f}",
                        "Selisih %": "{:+.2f}%",
                    }
                ),
                use_container_width=True,
            )


with st.sidebar:
    st.header("Pengaturan")
    auto_refresh = st.checkbox("Auto refresh dashboard", value=True)
    refresh_interval_seconds = st.selectbox(
        "Interval auto refresh",
        options=[60, 300, 900],
        index=0,
        format_func=lambda value: f"{value // 60} menit" if value >= 60 else f"{value} detik",
        disabled=not auto_refresh,
    )
    if auto_refresh:
        if st_autorefresh is None:
            st.warning("Auto refresh belum aktif karena dependency belum tersedia.")
        else:
            refresh_count = st_autorefresh(
                interval=refresh_interval_seconds * 1000,
                key="dashboard_auto_refresh",
            )
            st.caption(
                f"Auto refresh aktif tiap {refresh_interval_seconds // 60} menit. "
                f"Refresh ke-{refresh_count} | {pd.Timestamp.now(tz=WIT).strftime('%H:%M:%S WIT')}"
            )
    if st.button("Refresh data sekarang", use_container_width=True):
        get_data.clear()
        get_gold_ohlc.clear()
        get_models.clear()
        get_simulations.clear()
        st.rerun()
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
    market, data_fetched_at = get_data()
    gold_ohlc = get_gold_ohlc()
    model_1, model_2, direction_model = get_models(market)
    (
        optimized_result,
        optimization_leaderboard,
        optimized_v2_result,
        optimization_v2_leaderboard,
        optimized_v3_result,
        optimization_v3_leaderboard,
        optimized_v4_result,
        optimization_v4_leaderboard,
    ) = get_simulations(
        gold_ohlc,
        SIMULATION_CACHE_VERSION,
    )
except Exception as exc:
    st.error(f"Data belum dapat diproses: {exc}")
    st.stop()

dashboard_tab, simulation_tab, live_trading_tab, optimizer_signals_tab, monitoring_model_2_tab, monitoring_model_1_tab, intraday_audit_tab = st.tabs(
    ["Dashboard", "Simulasi", "Live Trading", "Sinyal Optimizer", "Monitoring Model 2", "Monitoring Model 1", "Audit Intraday"]
)
with dashboard_tab:
    render_dashboard(
        market,
        data_fetched_at,
        model_1,
        model_2,
        direction_model,
        model_choice,
        direction_threshold,
        history_years,
    )

with monitoring_model_2_tab:
    render_monitoring("Monitoring Model 2", DATA_PATH)

with monitoring_model_1_tab:
    render_monitoring("Monitoring Model 1", MODEL_1_DATA_PATH)

with simulation_tab:
    render_simulation(
        optimized_result,
        optimization_leaderboard,
        optimized_v2_result,
        optimization_v2_leaderboard,
        optimized_v3_result,
        optimization_v3_leaderboard,
        optimized_v4_result,
        optimization_v4_leaderboard,
        gold_ohlc,
    )

with live_trading_tab:
    render_live_trading(gold_ohlc, optimization_leaderboard)

with optimizer_signals_tab:
    render_optimizer_signals(gold_ohlc, optimization_leaderboard)

with intraday_audit_tab:
    render_intraday_audit(gold_ohlc)
