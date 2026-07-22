from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:  # pragma: no cover - handled in deployed UI
    st_autorefresh = None

from gold_forecast.broker_data import (
    apply_broker_clock_offset,
    BROKER_BARS_PATH,
    BROKER_QUOTE_PATH,
    audit_broker_feed,
    load_broker_bars,
    load_broker_quote,
)
from gold_forecast.data import load_gold_data, load_market_data
from gold_forecast.dashboard_snapshot import (
    DASHBOARD_SNAPSHOT_VERSION,
    load_dashboard_snapshot,
)
from gold_forecast.intraday_audit import audit_intraday_data, load_intraday_csv
from gold_forecast import live_trading as live_trading_module
from gold_forecast.live_trading import (
    LIVE_INITIAL_EQUITY,
    LIVE_START_DATE,
    LIVE_MANUAL_EXIT_PATH,
    LIVE_MANUAL_EXIT_V10_PATH,
    LIVE_TRADING_PATH,
    LIVE_TRADING_V10_PATH,
    LIVE_V10_START_DATE,
    load_live_ledger,
    run_live_trading_update,
)
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
from gold_forecast import strategy_optimizer as strategy_optimizer_module
from gold_forecast.supabase_broker import load_supabase_broker_feed
try:
    from gold_forecast.supabase_broker import load_supabase_terminal_status
except ImportError:  # pragma: no cover - compatibility during Streamlit rolling deploys
    def load_supabase_terminal_status(
        base_url: str,
        read_key: str,
        symbol: str = "XAUUSD",
    ) -> dict[str, object]:
        return {}


OPTIMIZATION_END = strategy_optimizer_module.OPTIMIZATION_END
OPTIMIZATION_START = strategy_optimizer_module.OPTIMIZATION_START
_rsi = strategy_optimizer_module._rsi


SIMULATION_CACHE_VERSION = "optimizer-v1-v10-2025q1-2026q2"
PRECOMPUTED_SIMULATION_PATH = Path("data/precomputed/simulations.pkl")
M1_BACKTEST_VERSION = "v1-full-history-v10-oos-2026-07-21"
M1_BACKTEST_PATH = Path("data/precomputed/m1_backtests.pkl")
MARTINGALE_BACKTEST_VERSION = "martingale-v1-daily-2025q1-2026q2"
MARTINGALE_BACKTEST_PATH = Path("data/precomputed/martingale_v1.pkl")
MARTINGALE_V2_VERSION = "martingale-v2-adaptive-train2025-oos2026h1"
MARTINGALE_V2_PATH = Path("data/precomputed/martingale_v2.pkl")

st.set_page_config(page_title="Prediksi XAU/USD", page_icon=":material/monitoring:", layout="wide")
st.title("Prediksi Harga Emas")
st.caption("Estimasi hari bursa berikutnya dan tujuh hari ke depan")


@st.cache_data(ttl=300)
def get_data() -> tuple[pd.DataFrame, pd.Timestamp]:
    return load_market_data(), pd.Timestamp.now(tz=WIT)


@st.cache_data(ttl=300)
def get_gold_ohlc() -> pd.DataFrame:
    return load_gold_data()


@st.cache_resource(ttl=300)
def get_dashboard_snapshot(snapshot_version: str):
    return load_dashboard_snapshot()


def load_precomputed_simulations(simulation_version: str):
    if not PRECOMPUTED_SIMULATION_PATH.exists():
        return None
    try:
        with PRECOMPUTED_SIMULATION_PATH.open("rb") as file:
            saved = pickle.load(file)
        if saved.get("version") == simulation_version:
            return saved["payload"]
    except Exception:
        PRECOMPUTED_SIMULATION_PATH.unlink(missing_ok=True)
    return None


@st.cache_resource
def load_precomputed_m1_backtests(backtest_version: str):
    if not M1_BACKTEST_PATH.exists():
        return None
    try:
        with M1_BACKTEST_PATH.open("rb") as file:
            saved = pickle.load(file)
        if saved.get("version") == backtest_version:
            return saved["payload"]
    except Exception:
        return None
    return None


@st.cache_resource
def load_precomputed_martingale(backtest_version: str):
    if not MARTINGALE_BACKTEST_PATH.exists():
        return None
    try:
        with MARTINGALE_BACKTEST_PATH.open("rb") as file:
            saved = pickle.load(file)
        if saved.get("version") == backtest_version:
            return saved["payload"]
    except Exception:
        return None
    return None


@st.cache_resource
def load_precomputed_martingale_v2(backtest_version: str):
    if not MARTINGALE_V2_PATH.exists():
        return None
    try:
        with MARTINGALE_V2_PATH.open("rb") as file:
            saved = pickle.load(file)
        if saved.get("version") == backtest_version:
            return saved["payload"]
    except Exception:
        return None
    return None


@st.cache_data(ttl=3600)
def get_base_optimizer(gold_ohlc: pd.DataFrame):
    return strategy_optimizer_module.run_optimized_strategy(gold_ohlc)


@st.cache_data(ttl=3600)
def get_simulations(simulation_version: str):
    cached = load_precomputed_simulations(simulation_version)
    if cached is not None:
        return cached

    gold_ohlc = get_gold_ohlc()
    optimized_result, optimization_leaderboard = get_base_optimizer(gold_ohlc)
    optimized_v10_runner = getattr(
        strategy_optimizer_module,
        "run_optimized_strategy_v10",
        strategy_optimizer_module.run_optimized_strategy,
    )
    optimized_v10_result, optimization_v10_leaderboard = optimized_v10_runner(gold_ohlc)
    payload = (
        optimized_result,
        optimization_leaderboard,
        optimized_v10_result,
        optimization_v10_leaderboard,
    )
    try:
        PRECOMPUTED_SIMULATION_PATH.parent.mkdir(parents=True, exist_ok=True)
        with PRECOMPUTED_SIMULATION_PATH.open("wb") as file:
            pickle.dump({"version": simulation_version, "payload": payload}, file)
    except Exception:
        pass
    return payload


def get_v10_leaderboard_for_live(simulation_version: str) -> pd.DataFrame:
    snapshot = get_dashboard_snapshot(DASHBOARD_SNAPSHOT_VERSION)
    if snapshot is not None:
        leaderboard = snapshot.get("v10_leaderboard")
        if isinstance(leaderboard, pd.DataFrame) and not leaderboard.empty:
            return leaderboard
    cached = load_precomputed_simulations(simulation_version)
    if cached is None or len(cached) < 4:
        return pd.DataFrame()
    leaderboard = cached[3]
    if isinstance(leaderboard, pd.DataFrame):
        return leaderboard
    return pd.DataFrame()


def get_v1_leaderboard_for_live(simulation_version: str) -> pd.DataFrame:
    cached = load_precomputed_simulations(simulation_version)
    if cached is None or len(cached) < 2:
        return pd.DataFrame()
    leaderboard = cached[1]
    if isinstance(leaderboard, pd.DataFrame):
        return leaderboard
    return pd.DataFrame()


def _optimizer_v10_latest_prediction(gold_ohlc: pd.DataFrame, leaderboard: pd.DataFrame) -> dict[str, object]:
    params = _optimizer_best_params(leaderboard)
    if gold_ohlc.empty:
        return {"params": params, "direction": "WAIT", "note": "Data OHLC belum tersedia."}

    frame = gold_ohlc.copy()
    latest_date = pd.Timestamp(frame.index.max())
    latest = frame.loc[latest_date]
    latest_close = float(latest["Close"])
    threshold = float(params["Threshold entry (%)"])
    lot = float(params.get("Lot", 0.01) or 0.01)
    units = max(lot * 100, 1)

    signals = strategy_optimizer_module._indicator_predictions(
        frame,
        str(params["Mode"]),
        int(params["Fast MA"]),
        int(params["Slow MA"]),
        int(params["Momentum hari"]),
        threshold,
        test_start=latest_date,
        test_end=latest_date,
    )
    if signals.empty:
        prediction = latest_close
        expected_change_pct = 0.0
        direction = "WAIT"
        note = "Candle terbaru belum memenuhi threshold entry Optimizer v10."
    else:
        prediction = float(signals.iloc[-1])
        expected_change_pct = (prediction / latest_close - 1) * 100
        if expected_change_pct >= threshold:
            direction = "BUY"
            note = "Sinyal BUY v10 lolos pada candle harian terbaru."
        elif expected_change_pct <= -threshold:
            direction = "SELL"
            note = "Sinyal SELL v10 lolos pada candle harian terbaru."
        else:
            direction = "WAIT"
            note = "Expected change belum melewati threshold Optimizer v10."

    tp_points = float(params["TP (USD)"]) / units
    sl_points = float(params["SL (USD)"]) / units
    if direction == "BUY":
        target_price = latest_close + tp_points
        risk_price = latest_close - sl_points
    elif direction == "SELL":
        target_price = latest_close - tp_points
        risk_price = latest_close + sl_points
    else:
        target_price = pd.NA
        risk_price = pd.NA

    return {
        "params": params,
        "latest_date": latest_date,
        "latest_close": latest_close,
        "prediction": prediction,
        "expected_change_pct": expected_change_pct,
        "direction": direction,
        "target_price": target_price,
        "risk_price": risk_price,
        "note": note,
    }


def render_optimizer_v10_dashboard(
    market: pd.DataFrame,
    data_fetched_at: pd.Timestamp,
    gold_ohlc: pd.DataFrame,
    optimization_v10_leaderboard: pd.DataFrame,
    history_years: int,
) -> None:
    gold = market["gold"]
    latest_market = float(gold.iloc[-1])
    previous_market = float(gold.iloc[-2])
    market_last_date = pd.Timestamp(gold.index.max()).strftime("%d %b %Y")
    fetched_label = data_fetched_at.strftime("%d %b %Y %H:%M:%S WIT")
    prediction = _optimizer_v10_latest_prediction(gold_ohlc, optimization_v10_leaderboard)
    params = prediction["params"]

    st.info(f"Data pasar terakhir: **{market_last_date}** | Data diambil dashboard: **{fetched_label}**")
    if optimization_v10_leaderboard.empty:
        st.warning(
            "Parameter precomputed Optimizer v10 belum tersedia. Dashboard memakai parameter fallback dan tidak "
            "menghitung ulang optimizer berat saat halaman dibuka."
        )

    signal_direction = str(prediction["direction"])
    signal_label = signal_direction if signal_direction in {"BUY", "SELL"} else "WAIT"
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Harga terakhir", f"${latest_market:,.2f}", f"{latest_market - previous_market:+,.2f}")
    c2.metric("Sinyal Optimizer v10", signal_label, str(prediction["note"]))
    c3.metric(
        "Prediksi strategi",
        f"${float(prediction['prediction']):,.2f}",
        f"{float(prediction['expected_change_pct']):+.2f}%",
    )
    target_value = prediction["target_price"]
    c4.metric("Target TP v10", "-" if pd.isna(target_value) else f"${float(target_value):,.2f}")

    st.subheader("Prediksi Menurut Optimizer v10")
    if signal_direction == "BUY":
        st.success("Optimizer v10 membaca peluang **BUY** pada candle harian terbaru.")
    elif signal_direction == "SELL":
        st.error("Optimizer v10 membaca peluang **SELL** pada candle harian terbaru.")
    else:
        st.info("Optimizer v10 belum memberi sinyal entry. Posisi yang disarankan: **WAIT / TUNGGU**.")

    detail = pd.DataFrame(
        [
            {"Parameter": "Tanggal candle", "Nilai": _format_date(prediction.get("latest_date")), "Keterangan": "Candle harian terbaru yang dievaluasi."},
            {"Parameter": "Strategi", "Nilai": params["Strategi"], "Keterangan": "Kandidat terbaik dari Optimizer v10 atau fallback jika belum ada precomputed."},
            {"Parameter": "Mode", "Nilai": params["Mode"], "Keterangan": "Jenis rule sinyal: Trend, Breakout, atau Pullback."},
            {"Parameter": "MA cepat/lambat", "Nilai": f"{params['Fast MA']} / {params['Slow MA']}", "Keterangan": "Moving average harian yang dipakai strategi."},
            {"Parameter": "Momentum hari", "Nilai": params["Momentum hari"], "Keterangan": "Jendela momentum harian."},
            {"Parameter": "Threshold entry", "Nilai": f"{params['Threshold entry (%)']:.2f}%", "Keterangan": "Minimal expected change agar entry valid."},
            {"Parameter": "TP / CL", "Nilai": f"USD {params['TP (USD)']:,.2f} / USD {params['SL (USD)']:,.2f}", "Keterangan": "Target profit dan batas rugi per posisi dari strategi."},
            {"Parameter": "Lot backtest", "Nilai": f"{float(params.get('Lot', 0.01)):,.2f}", "Keterangan": "Lot dari kandidat v10. Paper live tetap memakai rule live yang sudah ditetapkan."},
            {"Parameter": "Harga CL risiko", "Nilai": "-" if pd.isna(prediction["risk_price"]) else f"${float(prediction['risk_price']):,.2f}", "Keterangan": "Harga risiko jika sinyal BUY/SELL aktif."},
        ]
    )
    st.dataframe(detail, use_container_width=True, hide_index=True)

    st.caption(
        "Catatan: Model 3 adalah prediksi berbasis strategi Optimizer v10. Ia menilai apakah candle harian terbaru "
        "cukup kuat untuk BUY/SELL/WAIT, bukan model probabilistik harga seperti Model 1 dan Model 2."
    )


def render_dashboard(
    market: pd.DataFrame,
    data_fetched_at: pd.Timestamp,
    model_1,
    model_2,
    direction_model,
    direction_threshold: float,
    gold_ohlc: pd.DataFrame,
    optimization_v10_leaderboard: pd.DataFrame,
    snapshot_generated_at: str,
) -> None:
    gold = market["gold"]
    latest = float(gold.iloc[-1])
    previous = float(gold.iloc[-2])
    market_last_date = pd.Timestamp(gold.index.max()).strftime("%d %b %Y")
    fetched_label = data_fetched_at.strftime("%d %b %Y %H:%M:%S WIT")
    v10_prediction = _optimizer_v10_latest_prediction(gold_ohlc, optimization_v10_leaderboard)
    v10_params = v10_prediction["params"]

    generated_at = pd.Timestamp(snapshot_generated_at)
    if generated_at.tzinfo is None:
        generated_at = generated_at.tz_localize("UTC")
    snapshot_label = generated_at.tz_convert(WIT).strftime("%d %b %Y %H:%M:%S WIT")
    st.info(
        f"Data pasar terakhir: **{market_last_date}** | Data diambil dashboard: **{fetched_label}** | "
        f"Snapshot model dibuat: **{snapshot_label}**"
    )

    m1_tomorrow, m1_day_seven = model_1.forecast.iloc[0], model_1.forecast.iloc[-1]
    m2_tomorrow, m2_day_seven = model_2.forecast.iloc[0], model_2.forecast.iloc[-1]
    signal_1 = build_signal(market, model_1.forecast)
    signal_2 = build_signal(market, model_2.forecast)

    st.subheader("Perbandingan Prediksi Antar Model")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Model 1 - Harga Historis**")
        st.metric("Estimasi besok", f"${m1_tomorrow['Estimasi']:,.2f}", f"{m1_tomorrow['Estimasi'] - latest:+,.2f}")
        st.metric("Estimasi hari ke-7", f"${m1_day_seven['Estimasi']:,.2f}", f"{m1_day_seven['Estimasi'] - latest:+,.2f}")
        st.metric("Sinyal", signal_1.label, f"Confidence {signal_1.confidence:.0f}%")
        st.caption("Regresi ridge berbasis riwayat harga emas.")
    with col2:
        st.markdown("**Model 2 - Lintas Pasar**")
        st.metric("Estimasi besok", f"${m2_tomorrow['Estimasi']:,.2f}", f"{m2_tomorrow['Estimasi'] - latest:+,.2f}")
        st.metric("Estimasi hari ke-7", f"${m2_day_seven['Estimasi']:,.2f}", f"{m2_day_seven['Estimasi'] - latest:+,.2f}")
        st.metric("Sinyal", signal_2.label, f"Confidence {signal_2.confidence:.0f}%")
        st.caption("Gradient boosting memakai emas dan faktor lintas pasar.")
    with col3:
        signal_direction = str(v10_prediction["direction"])
        signal_label = signal_direction if signal_direction in {"BUY", "SELL"} else "WAIT"
        target_value = v10_prediction["target_price"]
        st.markdown("**Model 3 - Optimizer v10**")
        st.metric(
            "Prediksi strategi",
            f"${float(v10_prediction['prediction']):,.2f}",
            f"{float(v10_prediction['expected_change_pct']):+.2f}%",
        )
        st.metric("Target TP", "-" if pd.isna(target_value) else f"${float(target_value):,.2f}")
        st.metric("Sinyal", signal_label, str(v10_prediction["note"]))
        st.caption("Prediksi berbasis rule Optimizer v10, bukan forecast probabilistik harga.")

    if optimization_v10_leaderboard.empty:
        st.warning(
            "Parameter precomputed Optimizer v10 belum tersedia. Model 3 memakai parameter fallback dan tidak "
            "menghitung ulang optimizer berat saat Dashboard dibuka."
        )

    st.subheader("Ringkasan Angka Pembanding")
    comparison = pd.DataFrame(
        [
            {
                "Model": "Model 1 - Harga Historis",
                "Output utama": f"${m1_tomorrow['Estimasi']:,.2f}",
                "Arah / Sinyal": signal_1.label,
                "Perubahan vs terakhir": m1_tomorrow["Estimasi"] - latest,
                "Confidence / Expected": f"{signal_1.confidence:.0f}%",
                "MAE T+1": model_1.metrics.get("MAE", pd.NA),
                "Akurasi arah T+1": model_1.metrics.get("Akurasi arah", pd.NA),
            },
            {
                "Model": "Model 2 - Lintas Pasar",
                "Output utama": f"${m2_tomorrow['Estimasi']:,.2f}",
                "Arah / Sinyal": signal_2.label,
                "Perubahan vs terakhir": m2_tomorrow["Estimasi"] - latest,
                "Confidence / Expected": f"{signal_2.confidence:.0f}%",
                "MAE T+1": model_2.horizon_metrics.loc[1, "MAE"],
                "Akurasi arah T+1": model_2.horizon_metrics.loc[1, "Akurasi arah"],
            },
            {
                "Model": "Model 3 - Optimizer v10",
                "Output utama": f"${float(v10_prediction['prediction']):,.2f}",
                "Arah / Sinyal": signal_label,
                "Perubahan vs terakhir": float(v10_prediction["prediction"]) - latest,
                "Confidence / Expected": f"{float(v10_prediction['expected_change_pct']):+.2f}%",
                "MAE T+1": pd.NA,
                "Akurasi arah T+1": pd.NA,
            },
        ]
    )
    st.dataframe(
        comparison.style.format(
            {
                "Perubahan vs terakhir": "${:+,.2f}",
                "MAE T+1": "${:,.2f}",
                "Akurasi arah T+1": "{:.1f}%",
            },
            na_rep="-",
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Estimasi 7 Hari Bursa Model 1 dan Model 2")
    forecast_comparison = pd.DataFrame(
        {
            "Model 1 - Estimasi": model_1.forecast["Estimasi"],
            "Model 2 - Estimasi": model_2.forecast["Estimasi"],
        }
    )
    st.dataframe(
        forecast_comparison.style.format("${:,.2f}"),
        use_container_width=True,
    )
    mae_improvement = (1 - model_2.horizon_metrics.loc[1, "MAE"] / model_1.metrics["MAE"]) * 100
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

    with st.expander("Parameter Optimizer v10"):
        v10_detail = pd.DataFrame(
            [
                {"Parameter": "Strategi", "Nilai": v10_params["Strategi"]},
                {"Parameter": "Mode", "Nilai": v10_params["Mode"]},
                {"Parameter": "MA cepat/lambat", "Nilai": f"{v10_params['Fast MA']} / {v10_params['Slow MA']}"},
                {"Parameter": "Momentum hari", "Nilai": v10_params["Momentum hari"]},
                {"Parameter": "Threshold entry", "Nilai": f"{v10_params['Threshold entry (%)']:.2f}%"},
                {"Parameter": "TP / CL", "Nilai": f"USD {v10_params['TP (USD)']:,.2f} / USD {v10_params['SL (USD)']:,.2f}"},
                {"Parameter": "Lot backtest", "Nilai": f"{float(v10_params.get('Lot', 0.01)):,.2f}"},
                {
                    "Parameter": "Harga CL risiko",
                    "Nilai": "-" if pd.isna(v10_prediction["risk_price"]) else f"${float(v10_prediction['risk_price']):,.2f}",
                },
            ]
        )
        st.dataframe(v10_detail, use_container_width=True, hide_index=True)

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
            probabilitasnya rendah tetap dianggap netral. **Model 3** adalah pembacaan
            sinyal strategi Optimizer v10 pada candle harian terbaru, bukan model
            probabilistik harga. Dashboard ini bukan saran investasi.
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
        "Floating profit close (USD)",
        "Profit protection aktif (USD)",
        "Profit protection floor (USD)",
        "Profit protection trail (USD)",
        "Peak floating profit (USD)",
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
            "Floating profit close (USD)",
            "Profit protection aktif (USD)",
            "Profit protection floor (USD)",
            "Profit protection trail (USD)",
            "Peak floating profit (USD)",
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
                "Floating profit close (USD)": "${:,.2f}",
                "Profit protection aktif (USD)": "${:,.2f}",
                "Profit protection floor (USD)": "${:,.2f}",
                "Profit protection trail (USD)": "${:,.2f}",
                "Peak floating profit (USD)": "${:,.2f}",
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
            total_profit = total_loss = total_net = 0.0
            buy_count = sell_count = close_count = tp_count = cl_count = 0
        else:
            gross = pd.to_numeric(phase_trades.get("Gross P/L", 0.0), errors="coerce").fillna(0.0)
            net = pd.to_numeric(phase_trades.get("Net P/L", 0.0), errors="coerce").fillna(0.0)
            swap = pd.to_numeric(phase_trades.get("Swap", 0.0), errors="coerce").fillna(0.0)
            reasons = phase_trades.get("Alasan exit", pd.Series("", index=phase_trades.index)).astype(str)

            tp_mask = reasons.eq("TP tersentuh")
            cl_mask = reasons.eq("SL tersentuh")
            total_tp = float(gross[tp_mask].sum())
            total_cl = float(gross[cl_mask].sum())
            total_other = float(gross[~tp_mask & ~cl_mask].sum())
            total_swap = float(swap.sum())
            total_profit = float(net[net > 0].sum())
            total_loss = float(net[net < 0].sum())
            total_net = float(net.sum())
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
                "Total Profit": total_profit,
                "Total Loss": total_loss,
                "Total Net P/L": total_net,
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
                "Total Profit": "${:+,.2f}",
                "Total Loss": "${:+,.2f}",
                "Total Net P/L": "${:+,.2f}",
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

    trade_rows = trades.copy()
    trade_rows["Tanggal entry"] = pd.to_datetime(trade_rows["Tanggal entry"], errors="coerce")
    trade_rows["Tanggal tutup"] = pd.to_datetime(trade_rows["Tanggal tutup"], errors="coerce")
    trade_rows = trade_rows.dropna(subset=["Tanggal entry", "Tanggal tutup"])
    if trade_rows.empty:
        return

    chart_start = trade_rows["Tanggal entry"].min() - pd.Timedelta(days=max(slow_window, momentum_days) + 30)
    chart_end = trade_rows["Tanggal tutup"].max()
    chart_data = gold_ohlc.copy()
    chart_data = chart_data.loc[(chart_data.index >= chart_start) & (chart_data.index <= chart_end)].copy()
    if chart_data.empty:
        return

    close = chart_data["Close"].astype(float)
    chart_data["MA cepat"] = close.rolling(fast_window).mean()
    chart_data["MA lambat"] = close.rolling(slow_window).mean()
    chart_data["Momentum"] = close.pct_change(momentum_days) * 100

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
    figure.update_xaxes(range=[chart_start, chart_end], row=2, col=1)

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

    if "Intraday" in str(summary.get("Timeframe", "")) or "M1 execution" in str(summary.get("Timeframe", "")):
        st.info(
            "Simulasi intraday berjalan kontinu selama periode out-of-sample. Equity USD 1.200 hanya menjadi referensi; "
            "tidak ada close-all multi-fase pada eksperimen ini."
        )
    else:
        st.info(
            "Target tiap fase: **+20% dari start equity fase tersebut**. "
            "Saat target tercapai, semua posisi ditutup dan fase berikutnya dimulai dari equity close-all."
        )
    if summary.get("Periode uji"):
        st.warning(
            f"Periode uji khusus: **{summary['Periode uji']}** | "
            f"Sumber parameter: **{summary.get('Sumber parameter', '-')}**"
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
                "Total Profit": "${:+,.2f}",
                "Total Loss": "${:+,.2f}",
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
            "Floating profit close (USD)",
            "Profit protection aktif (USD)",
            "Profit protection floor (USD)",
            "Profit protection trail (USD)",
            "Peak floating profit (USD)",
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
                "Floating profit close (USD)",
                "Profit protection aktif (USD)",
                "Profit protection floor (USD)",
                "Profit protection trail (USD)",
                "Peak floating profit (USD)",
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
                    "Floating profit close (USD)": "${:,.2f}",
                    "Profit protection aktif (USD)": "${:,.2f}",
                    "Profit protection floor (USD)": "${:,.2f}",
                    "Profit protection trail (USD)": "${:,.2f}",
                    "Peak floating profit (USD)": "${:,.2f}",
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
                        "Target fase (%)": "{:.0f}%",
                        "Confidence cutoff": "{:.0%}",
                        "Risk cap floating SL (%)": "{:.1f}%",
                        "TP (USD)": "${:,.2f}",
                        "SL (USD)": "${:,.2f}",
                        "Floating profit close (USD)": "${:,.2f}",
                        "Profit protection aktif (USD)": "${:,.2f}",
                        "Profit protection floor (USD)": "${:,.2f}",
                        "Profit protection trail (USD)": "${:,.2f}",
                        "Close-all target equity": "{}",
                        "Protection preset": "{}",
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


def _render_martingale_tab(payload) -> None:
    st.subheader("Martingale v1")
    if payload is None:
        st.warning("Hasil Martingale v1 belum tersedia pada artifact precomputed.")
        return

    result, leaderboard = payload
    summary = result.summary
    selected = leaderboard.iloc[0] if not leaderboard.empty else pd.Series(dtype=object)
    risk_pass = bool(selected.get("Lolos batas risiko", False))
    st.warning(
        "Eksperimen memakai sinyal harian Optimizer v1 dua arah, equity awal USD 10.000, lot awal 0.10, "
        "lot berikutnya berlipat dua, leverage 1:100, dan stop-out margin level 50%. Jika level averaging "
        "dan BEP tersentuh pada candle yang sama, jalur adverse-first dipakai: risiko diperiksa sebelum close BEP."
    )
    if risk_pass:
        st.success("Kandidat terpilih lolos batas internal: tanpa stop-out dan max drawdown tidak lebih dari 20%.")
    else:
        st.error(
            "Tidak ada kandidat yang lolos batas risiko internal. Tab tetap menampilkan kombinasi dengan skor relatif "
            "terbaik sebagai hasil eksperimen, bukan rekomendasi trading."
        )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Equity akhir", f"${summary['Equity akhir']:,.2f}", f"{summary['Growth total']:+.2f}%")
    c2.metric("Max drawdown", f"${summary['Max drawdown']:,.2f}")
    c3.metric("Jumlah basket", f"{summary['Jumlah basket']:.0f}")
    c4.metric("Biaya swap BUY", f"${abs(summary['Total swap']):,.2f}")
    c5.metric("Margin minimum", f"{summary['Margin level minimum (%)']:.1f}%")

    parameter_rows = [
        {"Parameter": "Sumber sinyal", "Nilai": summary["Sumber sinyal"]},
        {"Parameter": "Periode uji", "Nilai": summary["Periode uji"]},
        {"Parameter": "Lot berjenjang", "Nilai": "0.10, 0.20, 0.40, dan seterusnya"},
        {"Parameter": "Max posisi per basket", "Nilai": int(summary["Max posisi per basket"])},
        {"Parameter": "Lot posisi maksimum", "Nilai": f"{summary['Lot posisi maksimum']:.2f}"},
        {"Parameter": "Hard basket loss", "Nilai": f"{summary['Hard basket loss (%)']:.1f}% equity awal basket"},
        {"Parameter": "Leverage / stop-out", "Nilai": f"1:{summary['Leverage']:.0f} / {summary['Stop-out margin level (%)']:.0f}%"},
        {"Parameter": "Close normal", "Nilai": "Close seluruh basket saat harga kembali ke entry posisi awal"},
        {"Parameter": "Asumsi candle", "Nilai": summary["Asumsi intrabar"]},
    ]
    st.dataframe(pd.DataFrame(parameter_rows), use_container_width=True, hide_index=True)

    detail = pd.DataFrame(
        [
            {"Metrik": "Basket close di BEP", "Nilai": summary["Basket BEP"]},
            {"Metrik": "Basket hard loss", "Nilai": summary["Basket hard loss"]},
            {"Metrik": "Basket margin stop-out", "Nilai": summary["Stop-out basket"]},
            {"Metrik": "Basket ditutup akhir data", "Nilai": summary["Basket akhir data"]},
            {"Metrik": "Total posisi BUY", "Nilai": summary["Total BUY"]},
            {"Metrik": "Total posisi SELL", "Nilai": summary["Total SELL"]},
            {"Metrik": "Sinyal BUY v1 tersedia", "Nilai": summary["Sinyal BUY v1"]},
            {"Metrik": "Sinyal SELL v1 tersedia", "Nilai": summary["Sinyal SELL v1"]},
            {"Metrik": "Max posisi bersamaan", "Nilai": summary["Max open posisi"]},
            {"Metrik": "Total lot maksimum basket", "Nilai": summary["Total lot maksimum"]},
            {"Metrik": "Used margin maksimum", "Nilai": summary["Used margin maksimum"]},
            {"Metrik": "Penambahan ditolak karena margin", "Nilai": summary["Penambahan ditolak margin"]},
            {"Metrik": "Entry awal ditolak karena margin", "Nilai": summary["Entry awal ditolak margin"]},
            {"Metrik": "Total net P/L", "Nilai": summary["Total net P/L"]},
            {"Metrik": "Profit factor", "Nilai": summary["Profit factor"]},
        ]
    )
    st.markdown("**Summary Risiko dan Transaksi**")
    st.dataframe(detail, use_container_width=True, hide_index=True)
    if summary["Sinyal SELL v1"] > 0 and summary["Total SELL"] == 0:
        st.info(
            "Mesin mendeteksi sinyal SELL v1, tetapi tidak ada SELL yang dapat dieksekusi. "
            "Ketika regime SELL pertama muncul, equity yang tersisa sudah tidak cukup untuk margin posisi awal 0.10 lot."
        )

    equity_curve = result.equity_curve.copy()
    if not equity_curve.empty:
        figure = go.Figure()
        figure.add_trace(go.Scatter(x=equity_curve.index, y=equity_curve["Equity"], name="Equity", line=dict(width=3)))
        figure.add_trace(
            go.Scatter(
                x=equity_curve.index,
                y=equity_curve["Balance"],
                name="Balance",
                line=dict(width=1.5, dash="dot"),
            )
        )
        figure.add_hline(y=summary["Modal awal"], line_dash="dash", line_color="#f59e0b", annotation_text="Equity awal")
        figure.update_layout(title="Equity Curve Martingale v1", yaxis_title="USD", height=430)
        st.plotly_chart(figure, use_container_width=True)

    basket_summary = summary.get("Basket summary", pd.DataFrame())
    st.markdown("**Summary Setiap Basket**")
    if isinstance(basket_summary, pd.DataFrame) and not basket_summary.empty:
        st.dataframe(
            basket_summary.style.format(
                {
                    "Anchor": "${:,.2f}",
                    "Total lot": "{:.2f}",
                    "Lot maksimum": "{:.2f}",
                    "Gross P/L": "${:+,.2f}",
                    "Swap": "${:+,.2f}",
                    "Net P/L": "${:+,.2f}",
                    "Balance akhir": "${:,.2f}",
                },
                na_rep="-",
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Tidak ada basket yang selesai selama periode pengujian.")

    with st.expander("Perbandingan kandidat batas risiko"):
        st.dataframe(
            leaderboard.style.format(
                {
                    "Lot maksimum": "{:.2f}",
                    "Hard basket loss (%)": "{:.1f}%",
                    "Equity akhir": "${:,.2f}",
                    "Growth total": "{:+.2f}%",
                    "Max drawdown": "${:,.2f}",
                    "Max drawdown (%)": "{:.2f}%",
                    "Margin minimum (%)": "{:.1f}%",
                    "Total swap": "${:+,.2f}",
                    "Risk-adjusted score": "{:.3f}",
                },
                na_rep="-",
            ),
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("Detail seluruh posisi Martingale"):
        trades = result.trades.copy()
        if trades.empty:
            st.info("Tidak ada posisi yang ditutup.")
        else:
            st.dataframe(
                trades.style.format(
                    {
                        "Lot": "{:.2f}",
                        "Entry": "${:,.2f}",
                        "Exit": "${:,.2f}",
                        "Gross P/L": "${:+,.2f}",
                        "Swap": "${:+,.2f}",
                        "Net P/L": "${:+,.2f}",
                        "Balance": "${:,.2f}",
                    },
                    na_rep="-",
                ),
                use_container_width=True,
                hide_index=True,
            )


def _render_martingale_v2_tab(payload) -> None:
    st.subheader("Martingale v2")
    if payload is None:
        st.warning("Hasil Martingale v2 belum tersedia pada artifact precomputed.")
        return

    full_result, leaderboard, train_result, oos_result = payload
    summary = full_result.summary
    st.info(
        "v2 menutup kelemahan v1 dengan fresh-signal entry, filter volatilitas, ATR spacing, lot bertahap yang dibatasi, "
        "weighted basket target, hard risk 1%, minimum margin entry, time-stop, dan regime-flip exit. Parameter dipilih "
        "hanya pada 2025 lalu dibekukan untuk pengujian Januari-Juni 2026."
    )
    if summary["Status kelayakan"].startswith("LAYAK"):
        st.success("Hasil OOS memenuhi kriteria kandidat paper test.")
    elif summary["OOS growth (%)"] > 0 and summary["OOS jumlah basket"] < 3:
        st.warning(
            "OOS menghasilkan profit, tetapi belum dinyatakan layak karena hanya memiliki "
            f"{summary['OOS jumlah basket']:.0f} basket. Sampel minimum internal adalah 3 basket."
        )
    else:
        st.warning(
            f"OOS belum lolos: growth {summary['OOS growth (%)']:+.2f}% pada "
            f"{summary['OOS jumlah basket']:.0f} basket. v2 lebih terkendali daripada v1, tetapi belum layak untuk paper live."
        )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Equity akhir full", f"${summary['Equity akhir']:,.2f}", f"{summary['Growth total']:+.2f}%")
    c2.metric("Equity akhir OOS", f"${summary['OOS equity akhir']:,.2f}", f"{summary['OOS growth (%)']:+.2f}%")
    c3.metric("Max drawdown full", f"${summary['Max drawdown']:,.2f}")
    c4.metric("Jumlah basket full", f"{summary['Jumlah basket']:.0f}")
    c5.metric("Biaya swap full", f"${abs(summary['Total swap']):,.2f}")

    selected_parameters = pd.DataFrame(
        [
            {"Parameter": "Arah terpilih", "Nilai": summary["Arah diizinkan"]},
            {"Parameter": "Fresh signal", "Nilai": "Entry hanya saat sinyal v1 baru/berubah arah"},
            {"Parameter": "Batas ATR/Close", "Nilai": f"Maksimum {summary['Maks ATR/Close (%)']:.2f}%"},
            {"Parameter": "Lot bertahap", "Nilai": f"0.10 x {summary['Lot multiplier']:.2f} per level"},
            {"Parameter": "Maks posisi / lot posisi", "Nilai": f"{summary['Max posisi per basket']:.0f} / {summary['Lot posisi maksimum']:.2f}"},
            {"Parameter": "Jarak averaging", "Nilai": f"{summary['Jarak entry (ATR)']:.2f} ATR per level"},
            {"Parameter": "Target basket", "Nilai": f"USD {summary['Target basket (USD)']:,.2f} dari weighted entry"},
            {"Parameter": "Hard basket loss", "Nilai": f"{summary['Hard basket loss (%)']:.2f}% equity awal basket"},
            {"Parameter": "Minimum margin entry", "Nilai": f"{summary['Minimum margin entry (%)']:.0f}%"},
            {"Parameter": "Leverage / stop-out", "Nilai": f"1:{summary['Leverage']:.0f} / {summary['Stop-out margin level (%)']:.0f}%"},
        ]
    )
    st.markdown("**Parameter Terpilih dari Train 2025**")
    st.dataframe(selected_parameters, use_container_width=True, hide_index=True)

    validation = pd.DataFrame(
        [
            {
                "Segmen": "Train",
                "Periode": summary["Periode train"],
                "Equity awal": train_result.summary["Modal awal"],
                "Equity akhir": train_result.summary["Equity akhir"],
                "Growth (%)": train_result.summary["Growth total"],
                "Max drawdown": train_result.summary["Max drawdown"],
                "Basket": train_result.summary["Jumlah basket"],
                "Basket target": train_result.summary["Basket target"],
                "Basket hard loss": train_result.summary["Basket hard loss"],
                "Swap": train_result.summary["Total swap"],
            },
            {
                "Segmen": "Out-of-sample",
                "Periode": summary["Periode OOS"],
                "Equity awal": oos_result.summary["Modal awal"],
                "Equity akhir": oos_result.summary["Equity akhir"],
                "Growth (%)": oos_result.summary["Growth total"],
                "Max drawdown": oos_result.summary["Max drawdown"],
                "Basket": oos_result.summary["Jumlah basket"],
                "Basket target": oos_result.summary["Basket target"],
                "Basket hard loss": oos_result.summary["Basket hard loss"],
                "Swap": oos_result.summary["Total swap"],
            },
            {
                "Segmen": "Full replay",
                "Periode": summary["Periode uji"],
                "Equity awal": summary["Modal awal"],
                "Equity akhir": summary["Equity akhir"],
                "Growth (%)": summary["Growth total"],
                "Max drawdown": summary["Max drawdown"],
                "Basket": summary["Jumlah basket"],
                "Basket target": summary["Basket target"],
                "Basket hard loss": summary["Basket hard loss"],
                "Swap": summary["Total swap"],
            },
        ]
    )
    st.markdown("**Perbandingan Train, OOS, dan Full Replay**")
    st.dataframe(
        validation.style.format(
            {
                "Equity awal": "${:,.2f}",
                "Equity akhir": "${:,.2f}",
                "Growth (%)": "{:+.2f}%",
                "Max drawdown": "${:,.2f}",
                "Basket": "{:.0f}",
                "Basket target": "{:.0f}",
                "Basket hard loss": "{:.0f}",
                "Swap": "${:+,.2f}",
            },
            na_rep="-",
        ),
        use_container_width=True,
        hide_index=True,
    )

    equity_curve = full_result.equity_curve.copy()
    if not equity_curve.empty:
        figure = go.Figure()
        figure.add_trace(go.Scatter(x=equity_curve.index, y=equity_curve["Equity"], name="Equity", line=dict(width=3)))
        figure.add_trace(
            go.Scatter(x=equity_curve.index, y=equity_curve["Balance"], name="Balance", line=dict(width=1.5, dash="dot"))
        )
        figure.add_vline(x=pd.Timestamp("2026-01-01"), line_dash="dash", line_color="#f59e0b")
        figure.add_hline(y=summary["Modal awal"], line_dash="dot", annotation_text="Equity awal")
        figure.update_layout(title="Equity Curve Martingale v2 - Full Replay", yaxis_title="USD", height=430)
        st.plotly_chart(figure, use_container_width=True)

    baskets = summary.get("Basket summary", pd.DataFrame())
    st.markdown("**Summary Setiap Basket Full Replay**")
    if isinstance(baskets, pd.DataFrame) and not baskets.empty:
        st.dataframe(
            baskets.style.format(
                {
                    "Anchor": "${:,.2f}",
                    "ATR entry": "${:,.2f}",
                    "Total lot": "{:.2f}",
                    "Lot maksimum": "{:.2f}",
                    "Gross P/L": "${:+,.2f}",
                    "Swap": "${:+,.2f}",
                    "Net P/L": "${:+,.2f}",
                    "Balance akhir": "${:,.2f}",
                },
                na_rep="-",
            ),
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("Seleksi filter hanya pada train 2025"):
        st.dataframe(
            leaderboard.style.format(
                {
                    "Maks ATR/Close (%)": lambda value: "Tanpa batas" if pd.isna(value) or value == float("inf") else f"{value:.2f}%",
                    "Lot multiplier": "{:.2f}",
                    "Lot maksimum": "{:.2f}",
                    "Jarak entry (ATR)": "{:.2f}",
                    "Hard basket loss (%)": "{:.2f}%",
                    "Target basket (USD)": "${:,.2f}",
                    "Train equity akhir": "${:,.2f}",
                    "Train growth (%)": "{:+.2f}%",
                    "Train max drawdown": "${:,.2f}",
                    "Train max drawdown (%)": "{:.2f}%",
                    "Train total swap": "${:+,.2f}",
                    "Train risk-adjusted score": "{:.3f}",
                },
                na_rep="-",
            ),
            use_container_width=True,
            hide_index=True,
        )


def render_simulation(
    optimized_result,
    optimization_leaderboard: pd.DataFrame,
    optimized_v10_result,
    optimization_v10_leaderboard: pd.DataFrame,
    gold_ohlc: pd.DataFrame,
) -> None:
    st.subheader("Simulasi Trading XAU/USD Multi-Fase")
    st.caption(
        "Simulasi menampilkan strategi harian serta eksperimen fixed-parameter pada candle M1."
    )
    st.warning(
        "Asumsi strategi utama: equity awal USD 1.000, target tiap fase +20%, maksimal 8 BUY dan 10 SELL. "
        "Simulasi memakai dataset 1 Jan 2025-30 Jun 2026. Data sebelum 1 Jan 2025 tidak dipakai agar proses tetap ringan. "
        "Swap BUY USD 0.2 per hari per 0.01 lot; SELL dianggap USD 0.0. "
        "Martingale v1 memakai equity awal USD 10.000 dan aturan margin tersendiri. "
        "Dua tab strategi harian memakai OHLC harian GC=F; tab M1 memiliki penjelasan dataset tersendiri. "
        "Jika TP dan SL tersentuh dalam candle yang sama, SL dianggap lebih dulu."
    )

    optimizer_tab, optimizer_v10_tab, optimizer_v1_m1_tab, optimizer_v10_m1_tab, martingale_tab, martingale_v2_tab = st.tabs(
        [
            "Strategi Terbaik Optimizer",
            "Strategi Optimizer v10",
            "v1 Intraday M1",
            "v10 Intraday M1",
            "Martingale v1",
            "Martingale v2",
        ]
    )
    with optimizer_tab:
        _render_multiphase_result("Strategi Terbaik Optimizer", optimized_result, optimization_leaderboard, gold_ohlc)
    with optimizer_v10_tab:
        _render_multiphase_result("Strategi Optimizer v10", optimized_v10_result, optimization_v10_leaderboard, gold_ohlc)
    m1_payload = load_precomputed_m1_backtests(M1_BACKTEST_VERSION)
    with optimizer_v1_m1_tab:
        _render_m1_backtest_tab(m1_payload, payload_offset=0, title="Optimizer v1 Intraday M1")
    with optimizer_v10_m1_tab:
        _render_m1_backtest_tab(m1_payload, payload_offset=2, title="Optimizer v10 Intraday M1")
    with martingale_tab:
        _render_martingale_tab(load_precomputed_martingale(MARTINGALE_BACKTEST_VERSION))
    with martingale_v2_tab:
        _render_martingale_v2_tab(load_precomputed_martingale_v2(MARTINGALE_V2_VERSION))


def _render_m1_backtest_tab(m1_payload, *, payload_offset: int, title: str) -> None:
    st.subheader(title)
    if m1_payload is None:
        st.warning("Hasil M1 precomputed belum tersedia. Jalankan scripts/build_m1_backtests.py dari laptop yang terhubung ke MT5.")
        return

    result = m1_payload[payload_offset]
    leaderboard = m1_payload[payload_offset + 1]
    summary = result.summary
    coverage = bool(summary.get("Cakupan lengkap", False))
    monthly = summary.get("Pertumbuhan bulanan")
    has_monthly = isinstance(monthly, pd.DataFrame) and not monthly.empty
    if has_monthly:
        st.warning(
            f"Extended backtest memakai **{summary.get('Jumlah candle', 0):,.0f} candle M1 MT5** pada "
            f"**{summary.get('Periode uji', '-')}**. Parameter dibekukan dari pemenang train April-Mei 2026 dan tidak "
            "dioptimasi ulang pada histori penuh. "
            + ("Seluruh 18 bulan tersedia." if coverage else "Ada bulan yang belum tersedia lengkap.")
        )
    else:
        st.warning(
            f"Parameter dipilih hanya dari train **{summary.get('Periode train', '-')}**, lalu diuji tanpa optimasi ulang pada "
            f"out-of-sample **{summary.get('Periode test', '-')}** ({summary.get('Jumlah candle', 0):,.0f} candle M1). "
            + ("Cakupan data lengkap." if coverage else "Cakupan belum lengkap karena histori terminal terbatas.")
        )
    status = str(summary.get("Status kelayakan", "BELUM DINILAI"))
    if status.startswith("LAYAK"):
        st.success(f"Status hasil out-of-sample: **{status}**. Ini kandidat untuk paper test, belum persetujuan real-money.")
    elif status.startswith("EXTENDED"):
        st.info(f"Status validasi: **{status}**. Bulan ditandai menurut backward validation, calibration, dan out-of-sample.")
    else:
        st.error(f"Status hasil out-of-sample: **{status}**.")
    model_detail = (
        "v10 memakai regime daily, konfirmasi tren H1, breakout M1, ATR trailing, dan lot confidence 0.01-0.02."
        if "v10" in title.lower()
        else "v1 memakai regime daily, alignment EMA dan momentum M1, lot tetap 0.01, serta maksimal satu posisi."
    )
    st.info(
        f"{model_detail} TP/SL memakai ATR intraday. Spread historis MT5 dan slippage dua point per sisi sudah "
        "dimasukkan; swap BUY dihitung per hari menginap dan SELL nol. Latency serta kualitas fill riil belum dimodelkan. "
        "Summary memakai seluruh transaksi; tabel detail dibatasi pada 1.000 transaksi terakhir agar dashboard tetap ringan."
    )
    if has_monthly:
        st.markdown("**Pertumbuhan dan Kinerja per Bulan**")
        st.caption(
            "Growth bulanan memakai perubahan equity month-end dan membawa posisi/balance secara kontinu. "
            "Net closed P/L hanya mencakup posisi yang ditutup pada bulan tersebut."
        )
        monthly_display = monthly.copy()
        monthly_display["Bulan"] = pd.to_datetime(monthly_display["Bulan"]).dt.strftime("%b %Y")
        st.dataframe(
            monthly_display.style.format(
                {
                    "Equity awal": "${:,.2f}", "Equity akhir": "${:,.2f}",
                    "Growth bulanan (%)": "{:+.2f}%", "Growth kumulatif (%)": "{:+.2f}%",
                    "Total Profit": "${:+,.2f}", "Total Loss": "${:+,.2f}",
                    "Net closed P/L": "${:+,.2f}", "Swap": "${:+,.2f}",
                    "Transaksi": "{:.0f}", "BUY": "{:.0f}", "SELL": "{:.0f}",
                    "Win rate (%)": "{:.1f}%", "Profit factor": "{:.2f}",
                    "Max drawdown": "${:,.2f}", "Jumlah candle": "{:,.0f}",
                    "Celah intraday >5 menit": "{:,.0f}",
                },
                na_rep="-",
            ),
            use_container_width=True,
            hide_index=True,
        )
        chart_monthly = monthly.copy()
        chart_monthly["Bulan"] = pd.to_datetime(chart_monthly["Bulan"])
        figure = go.Figure()
        figure.add_trace(
            go.Bar(
                x=chart_monthly["Bulan"], y=chart_monthly["Growth bulanan (%)"],
                name="Growth bulanan", marker_color=["#16a34a" if value >= 0 else "#dc2626" for value in chart_monthly["Growth bulanan (%)"]],
            )
        )
        figure.add_trace(
            go.Scatter(
                x=chart_monthly["Bulan"], y=chart_monthly["Growth kumulatif (%)"],
                name="Growth kumulatif", mode="lines+markers", yaxis="y2", line=dict(width=3, color="#2563eb"),
            )
        )
        figure.update_layout(
            title="Growth Bulanan dan Kumulatif v1 Intraday M1",
            yaxis=dict(title="Growth bulanan (%)"),
            yaxis2=dict(title="Growth kumulatif (%)", overlaying="y", side="right"),
            height=430,
            hovermode="x unified",
        )
        st.plotly_chart(figure, use_container_width=True)
    _render_multiphase_result(title, result, leaderboard)


def _render_v10_walk_forward_result(result, leaderboard: pd.DataFrame) -> None:
    st.subheader("Strategi Optimizer v10 - Walk-Forward Test")
    st.caption(
        "Dataset walk-forward memakai 1 Jan 2023 sampai 30 Jun 2026. Setiap fold mengoptimasi parameter v10 "
        "hanya sampai akhir periode train, lalu parameter terbaik diuji pada kuartal berikutnya tanpa optimasi ulang."
    )
    st.warning(
        "Interpretasi: jika train growth tinggi tetapi test growth negatif, fold tersebut diberi flag overfitting. "
        "Strategi yang sehat seharusnya tetap masuk akal di beberapa fold out-of-sample, bukan hanya menang di periode optimasi."
    )

    if leaderboard.empty:
        st.info("Belum ada hasil walk-forward. Bangun ulang simulasi setelah data cache market tersedia.")
        return

    expected_folds = 10
    actual_folds = len(leaderboard)
    if actual_folds < expected_folds:
        st.warning(
            f"Hasil walk-forward baru berisi {actual_folds}/{expected_folds} fold. "
            "Ini biasanya berarti data OHLC historis belum lengkap dari 1 Jan 2023 atau hasil simulasi masih perlu dibangun ulang."
        )

    summary = result.summary
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Fold profitable", f"{summary.get('Fold profitable', 0):.0f}/{expected_folds}")
    c2.metric("Rata-rata test growth", f"{summary.get('Growth total', 0):+.1f}%")
    c3.metric("Worst fold growth", f"{summary.get('Worst fold growth (%)', 0):+.1f}%")
    c4.metric("Fold overfitting", f"{summary.get('Fold overfitting', 0):.0f}")

    st.markdown("**Ringkasan Walk-Forward per Fold**")
    display = leaderboard.copy()
    date_columns = ["Train mulai", "Train akhir", "Test mulai", "Test akhir"]
    for column in date_columns:
        if column in display.columns:
            display[column] = pd.to_datetime(display[column], errors="coerce").dt.strftime("%d %b %Y")
    format_columns = {
        "Train equity akhir": "${:,.2f}",
        "Train growth (%)": "{:+.1f}%",
        "Test equity akhir": "${:,.2f}",
        "Test growth (%)": "{:+.1f}%",
        "Test max drawdown": "${:,.2f}",
        "Test jumlah transaksi": "{:.0f}",
        "Test total BUY": "{:.0f}",
        "Test total SELL": "{:.0f}",
        "Test win rate": "{:.1f}%",
        "Test profit factor": "{:.2f}",
        "Test avg net P/L": "${:+,.2f}",
        "Threshold entry (%)": "{:.2f}%",
        "TP (USD)": "${:,.2f}",
        "SL (USD)": "${:,.2f}",
        "Lot": "{:.2f}",
        "Risk cap floating SL (%)": "{:.1f}%",
        "Profit protection aktif (USD)": "${:,.2f}",
        "Profit protection floor (USD)": "${:,.2f}",
        "Profit protection trail (USD)": "${:,.2f}",
    }
    st.dataframe(
        display.style.format(
            {column: formatter for column, formatter in format_columns.items() if column in display.columns},
            na_rep="-",
        ),
        use_container_width=True,
        hide_index=True,
    )

    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=leaderboard["Fold"],
            y=pd.to_numeric(leaderboard["Train growth (%)"], errors="coerce"),
            name="Train growth",
            marker_color="#64748b",
        )
    )
    figure.add_trace(
        go.Bar(
            x=leaderboard["Fold"],
            y=pd.to_numeric(leaderboard["Test growth (%)"], errors="coerce"),
            name="Test growth",
            marker_color="#f59e0b",
        )
    )
    figure.add_hline(y=0, line_dash="dot", line_color="#94a3b8")
    figure.update_layout(
        title="Train vs Test Growth per Fold",
        yaxis_title="Growth (%)",
        barmode="group",
        height=420,
    )
    st.plotly_chart(figure, use_container_width=True)

    overfit_rows = leaderboard[leaderboard["Overfitting flag"].eq("YA")]
    if overfit_rows.empty:
        st.success("Tidak ada fold dengan flag overfitting sederhana: train tinggi dan test negatif.")
    else:
        st.error(
            f"Ada {len(overfit_rows)} fold dengan indikasi overfitting. "
            "Perhatikan parameter pada fold tersebut sebelum memakai v10 untuk real-money."
        )


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


def _render_manual_exit_comparison(
    live: dict[str, object],
    live_path=LIVE_TRADING_PATH,
    manual_path=LIVE_MANUAL_EXIT_PATH,
    key_prefix: str = "v1",
) -> None:
    summary = live["summary"]
    ledger = live["ledger"].copy()
    manual_loader = getattr(live_trading_module, "load_manual_exit_ledger", None)
    manual_recorder = getattr(live_trading_module, "record_manual_exit", None)
    manual_exits = manual_loader(manual_path) if manual_loader is not None else pd.DataFrame()
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
        if action_cols[3].button(f"Exit Manual #{position_id}", key=f"{key_prefix}_manual_exit_{position_id}"):
            _, message, success = manual_recorder(
                position_id,
                float(latest_price),
                live_path=live_path,
                manual_path=manual_path,
            )
            if success:
                st.success(message)
            else:
                st.warning(message)
            st.rerun()


def _format_live_param_value(value, value_type: str = "plain") -> str:
    if pd.isna(value):
        return "-"
    if value_type == "money":
        return f"USD {float(value):,.2f}"
    if value_type == "percent":
        return f"{float(value):,.2f}%"
    if value_type == "bool":
        return "Ya" if bool(value) else "Tidak"
    if value_type == "number":
        return f"{float(value):,.2f}".rstrip("0").rstrip(".")
    return str(value)


def _build_live_strategy_frame(params: dict[str, object], leaderboard: pd.DataFrame, title: str) -> pd.DataFrame:
    rows = [
        {"Parameter": "Strategi", "Nilai": params["Strategi"], "Keterangan": "Nama kandidat strategi yang dipakai."},
        {"Parameter": "Mode", "Nilai": params["Mode"], "Keterangan": "Jenis sinyal utama: Trend, Breakout, atau Pullback."},
        {
            "Parameter": "MA cepat/lambat",
            "Nilai": f"{params['Fast MA']} / {params['Slow MA']}",
            "Keterangan": "Moving average harian yang menjadi filter arah.",
        },
        {"Parameter": "Momentum hari", "Nilai": params["Momentum hari"], "Keterangan": "Jendela momentum harian."},
        {
            "Parameter": "Threshold entry",
            "Nilai": f"{params['Threshold entry (%)']:.2f}%",
            "Keterangan": "Minimal expected change agar sinyal bisa dianggap valid.",
        },
        {"Parameter": "TP per posisi", "Nilai": f"USD {params['TP (USD)']:,.2f}", "Keterangan": "Target profit algoritma."},
        {"Parameter": "CL/SL per posisi", "Nilai": f"USD {params['SL (USD)']:,.2f}", "Keterangan": "Batas rugi algoritma."},
        {"Parameter": "Lot live", "Nilai": "0.01 tetap", "Keterangan": "Override live sesuai rule paper trading saat ini."},
        {"Parameter": "Max BUY live", "Nilai": params.get("Max BUY", "-"), "Keterangan": "Batas posisi BUY yang dipakai engine live."},
        {"Parameter": "Max SELL live", "Nilai": params.get("Max SELL", "-"), "Keterangan": "Batas posisi SELL yang dipakai engine live."},
    ]

    if leaderboard.empty:
        return pd.DataFrame(rows)

    best = leaderboard.iloc[0].to_dict()
    extra_specs = [
        ("Eksplorasi", "plain", "Sumber/preset hasil pencarian optimizer."),
        ("Lot", "number", "Lot yang dipakai saat backtest v10; live tetap mengikuti override 0.01."),
        ("Max BUY", "number", "Batas posisi BUY pada kandidat backtest."),
        ("Max SELL", "number", "Batas posisi SELL pada kandidat backtest."),
        ("Risk cap floating SL (%)", "percent", "Batas risiko floating sebelum strategi menahan penambahan risiko."),
        ("Target fase (%)", "percent", "Target growth per fase pada simulasi optimizer."),
        ("Profit protection aktif (USD)", "money", "Floating profit minimum sebelum proteksi profit aktif."),
        ("Profit protection floor (USD)", "money", "Profit minimum yang ingin diamankan setelah proteksi aktif."),
        ("Profit protection trail (USD)", "money", "Jarak trailing dari peak floating profit."),
        ("Close-all target equity", "bool", "Apakah simulasi menutup semua posisi saat target equity fase tercapai."),
        ("Max open posisi", "number", "Jumlah posisi terbuka maksimum yang terjadi pada backtest."),
        ("Profit factor", "number", "Rasio total profit terhadap total loss pada backtest."),
        ("Max drawdown", "money", "Drawdown maksimum pada backtest kandidat."),
        ("Equity akhir", "money", "Equity akhir backtest kandidat."),
    ]
    for column, value_type, note in extra_specs:
        if column in best:
            rows.append(
                {
                    "Parameter": f"{column} ({title})",
                    "Nilai": _format_live_param_value(best[column], value_type),
                    "Keterangan": note,
                }
            )
    return pd.DataFrame(rows)


def render_live_trading(
    gold_ohlc: pd.DataFrame,
    optimization_leaderboard: pd.DataFrame,
    title: str = "Optimizer v1",
    start_date: pd.Timestamp = LIVE_START_DATE,
    live_path=LIVE_TRADING_PATH,
    manual_path=LIVE_MANUAL_EXIT_PATH,
    key_prefix: str = "v1",
    strategy_note: str | None = None,
    broker_quote: pd.Series | None = None,
) -> None:
    if optimization_leaderboard.empty:
        st.warning(
            f"Parameter {title} belum tersedia. Bangun ulang simulasi precomputed lebih dulu agar Live Trading "
            "bisa memakai parameter strategi yang benar."
        )
        return

    live = run_live_trading_update(
        gold_ohlc,
        optimization_leaderboard,
        path=live_path,
        start_date=start_date,
        broker_quote=broker_quote,
    )
    summary = live["summary"]
    params = live["params"]
    signal = live["signal"]
    waiting_state = live["waiting_state"]
    trigger_state = live["trigger_state"]
    signals = live["signals"].copy()
    open_positions = live["open_positions"].copy()
    closed_positions = live["closed_positions"].copy()

    st.subheader(f"Live Trading - {title}")
    st.caption(
        "Mode ini adalah paper-trading live berbasis data dashboard. Posisi dibuka mengikuti pola "
        f"{title}, bukan order broker sungguhan."
    )
    st.warning(
        "Asumsi live: equity awal USD 1.000, lot tetap 0.01, swap BUY USD 0.02 per 0.01 lot per hari, "
        "swap SELL USD 0. Jam trading Senin-Jumat 07:00 WIT sampai 06:00 WIT hari berikutnya; "
        f"Sabtu dan Minggu tidak membuka posisi baru. Ledger live dimulai **{pd.Timestamp(start_date).strftime('%d %b %Y')}**."
    )
    if strategy_note is None:
        strategy_note = (
            "Entry mengikuti sinyal Optimizer, boleh menambah posisi dari sinyal hari berbeda sampai batas "
            "maksimal 8 BUY dan 10 SELL. Re-entry guard USD 3 tidak dipakai untuk eksekusi live."
        )
    st.info(f"Live Trading memakai **{title} secara penuh**.")

    status_text = "Aktif" if summary["Can trade"] else "Tidak membuka posisi baru"
    st.info(
        f"Waktu dashboard: **{summary['Now WIT'].strftime('%d %b %Y %H:%M:%S WIT')}** | "
        f"Status sesi: **{status_text}** | {summary['Session note']}"
    )
    if pd.notna(summary.get("Latest bid")) and pd.notna(summary.get("Latest ask")):
        quote_status = "AKTIF" if summary.get("Broker quote fresh") else "STALE"
        st.info(
            f"Feed eksekusi: **{summary.get('Price source', 'MT5 broker')}** | Status: **{quote_status}** | "
            f"Bid **${float(summary['Latest bid']):,.2f}** | Ask **${float(summary['Latest ask']):,.2f}** | "
            f"Usia **{float(summary.get('Broker quote age minutes', 0.0)):.1f} menit**"
        )
    else:
        st.warning("Feed bid/ask broker belum tersedia; paper trading masih memakai fallback GC=F harian.")

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

    with st.expander("Dokumentasi Strategi Live Trading"):
        st.write(strategy_note)
        strategy_frame = _build_live_strategy_frame(params, optimization_leaderboard, title)
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
            {"Parameter": "Batas BUY strategi", "Nilai": trigger_state["Max BUY"]},
            {"Parameter": "Posisi BUY terbuka", "Nilai": trigger_state["Posisi BUY terbuka"]},
            {"Parameter": "Sisa slot BUY", "Nilai": trigger_state["Sisa slot BUY"]},
            {"Parameter": "Batas SELL strategi", "Nilai": trigger_state["Max SELL"]},
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
            "Kondisi Market BUY",
            waiting_state["Checklist BUY"],
            "Ini adalah syarat market BUY. Eksekusi akhir tetap dicek lagi pada Status Trigger Optimizer.",
        )
    with sell_col:
        _render_signal_checklist(
            "Kondisi Market SELL",
            waiting_state["Checklist SELL"],
            "Ini adalah syarat market SELL. Eksekusi akhir tetap dicek lagi pada Status Trigger Optimizer.",
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

    _render_manual_exit_comparison(live, live_path=live_path, manual_path=manual_path, key_prefix=key_prefix)


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


def _broker_feed_sources():
    try:
        config = st.secrets.get("broker_data", {})
    except Exception:
        config = {}
    bars_source = config.get("bars_url") if config else None
    quote_source = config.get("quote_url") if config else None
    return bars_source or BROKER_BARS_PATH, quote_source or BROKER_QUOTE_PATH


def _supabase_broker_config() -> tuple[str, str, str]:
    try:
        config = st.secrets.get("supabase_broker", {})
    except Exception:
        config = {}
    if not config:
        return "", "", "XAUUSD"
    read_key = config.get("publishable_key") or config.get("anon_key") or ""
    return str(config.get("url", "")).strip(), str(read_key).strip(), str(config.get("symbol", "XAUUSD")).strip()


@st.cache_data(ttl=60, show_spinner=False)
def get_supabase_broker_feed(base_url: str, read_key: str, symbol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    return load_supabase_broker_feed(base_url, read_key, symbol=symbol)


@st.cache_data(ttl=60, show_spinner=False)
def get_supabase_terminal_status(base_url: str, read_key: str, symbol: str) -> dict[str, object]:
    return load_supabase_terminal_status(base_url, read_key, symbol=symbol)


def _latest_broker_quote() -> tuple[pd.Series | None, str | None]:
    supabase_url, supabase_read_key, supabase_symbol = _supabase_broker_config()
    if supabase_url and supabase_read_key:
        try:
            _, quotes = get_supabase_broker_feed(supabase_url, supabase_read_key, supabase_symbol)
            return (quotes.iloc[-1] if not quotes.empty else None), None
        except Exception as exc:
            fallback = load_broker_quote(BROKER_QUOTE_PATH)
            return (fallback.iloc[-1] if not fallback.empty else None), str(exc)
    quotes = load_broker_quote(BROKER_QUOTE_PATH)
    return (quotes.iloc[-1] if not quotes.empty else None), None


def _render_real_trading_readiness(terminal_status: dict[str, object], audit: dict[str, object]) -> None:
    st.markdown("**Kesiapan Trading MT5**")
    if not terminal_status:
        st.info(
            "Status terminal belum tersedia. Jalankan pembaruan `supabase/broker_feed.sql`, lalu mulai ulang bridge."
        )
        return

    updated_at = pd.to_datetime(terminal_status.get("updated_at"), errors="coerce", utc=True)
    status_age = (
        (pd.Timestamp.now(tz="UTC") - updated_at).total_seconds() / 60
        if pd.notna(updated_at)
        else float("nan")
    )
    status_fresh = pd.notna(status_age) and -1 <= status_age <= 5
    account_mode = str(terminal_status.get("account_mode", "UNKNOWN")).upper()
    terminal_connected = bool(terminal_status.get("terminal_connected", False))
    account_trade_allowed = bool(terminal_status.get("account_trade_allowed", False))
    symbol_trade_mode = str(terminal_status.get("symbol_trade_mode", "UNKNOWN")).upper()
    manual_only = bool(terminal_status.get("manual_execution_only", True))
    feed_ready = bool(audit.get("connected")) and not bool(audit.get("stale"))
    operational_ready = all(
        [status_fresh, terminal_connected, account_trade_allowed, symbol_trade_mode == "FULL", manual_only, feed_ready]
    )

    if account_mode == "REAL" and operational_ready:
        readiness = "SIAP MANUAL"
        st.error(
            "Akun REAL terdeteksi. Gold Predictor hanya memberi data dan sinyal; seluruh order tetap dikonfirmasi "
            "manual di MT5."
        )
    elif account_mode == "DEMO" and operational_ready:
        readiness = "LATIHAN SIAP"
        st.info("Akun DEMO aktif. Gunakan tahap ini untuk memvalidasi feed, sinyal, dan jurnal sebelum uang riil.")
    else:
        readiness = "BELUM SIAP"
        st.warning("Satu atau lebih pemeriksaan terminal belum lolos. Jangan membuka posisi baru.")

    broker_label = str(terminal_status.get("broker_company") or terminal_status.get("broker_server") or "-")
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Mode akun", account_mode)
    r2.metric("Broker", broker_label)
    r3.metric("Leverage", f"1:{int(terminal_status.get('leverage') or 0)}")
    r4.metric("Status", readiness)

    terminal_api_allowed = bool(terminal_status.get("terminal_trade_allowed", False))
    checks = pd.DataFrame(
        [
            {
                "Pemeriksaan": "Status terminal terbaru",
                "Nilai": "-" if pd.isna(status_age) else f"{status_age:.1f} menit",
                "Status": "Lolos" if status_fresh else "Belum",
            },
            {
                "Pemeriksaan": "Terminal MT5 terhubung",
                "Nilai": "Terhubung" if terminal_connected else "Terputus",
                "Status": "Lolos" if terminal_connected else "Belum",
            },
            {
                "Pemeriksaan": "Akun mengizinkan trading manual",
                "Nilai": "Diizinkan" if account_trade_allowed else "Diblokir",
                "Status": "Lolos" if account_trade_allowed else "Belum",
            },
            {
                "Pemeriksaan": "Mode trading simbol",
                "Nilai": symbol_trade_mode,
                "Status": "Lolos" if symbol_trade_mode == "FULL" else "Belum",
            },
            {
                "Pemeriksaan": "Eksekusi Gold Predictor",
                "Nilai": "Manual/read-only" if manual_only else "Otomatis",
                "Status": "Lolos" if manual_only else "Diblokir",
            },
            {
                "Pemeriksaan": "Izin API trading terminal",
                "Nilai": "Aktif" if terminal_api_allowed else "Nonaktif",
                "Status": "Perhatian" if terminal_api_allowed else "Aman",
            },
        ]
    )
    st.dataframe(checks, use_container_width=True, hide_index=True)

    specifications = pd.DataFrame(
        [
            {"Parameter": "Server broker", "Nilai": terminal_status.get("broker_server", "-")},
            {"Parameter": "Contract size", "Nilai": terminal_status.get("contract_size", "-")},
            {"Parameter": "Lot minimum", "Nilai": terminal_status.get("volume_min", "-")},
            {"Parameter": "Lot maksimum", "Nilai": terminal_status.get("volume_max", "-")},
            {"Parameter": "Langkah lot", "Nilai": terminal_status.get("volume_step", "-")},
            {"Parameter": "Stops level (points)", "Nilai": terminal_status.get("stops_level_points", "-")},
            {"Parameter": "Spread (points)", "Nilai": terminal_status.get("spread_points", "-")},
            {"Parameter": "Swap BUY", "Nilai": terminal_status.get("swap_long", "-")},
            {"Parameter": "Swap SELL", "Nilai": terminal_status.get("swap_short", "-")},
            {"Parameter": "Mata uang akun", "Nilai": terminal_status.get("currency", "-")},
        ]
    )
    with st.expander("Spesifikasi XAUUSD dari broker"):
        st.dataframe(specifications, use_container_width=True, hide_index=True)


def render_broker_data(gold_ohlc: pd.DataFrame) -> None:
    st.subheader("Data Broker XAUUSD - Read Only")
    st.caption(
        "Halaman ini memeriksa feed broker tanpa kemampuan mengirim order. Bid/ask tidak dibuat dari Yahoo atau "
        "harga Close; value hanya tampil jika benar-benar tersedia dari feed broker."
    )
    st.warning(
        "Feed broker masih tahap validasi. Jangan gunakan halaman ini sebagai dasar eksekusi uang riil sebelum "
        "sinkronisasi harga, spread, contract size, dan waktu candle selesai diaudit."
    )

    uploaded_bars = st.file_uploader(
        "Upload candle M1 MT5 untuk audit sementara (CSV)",
        type=["csv"],
        help="Kolom minimal: timestamp_utc, open, high, low, close.",
    )
    bars_source, quote_source = _broker_feed_sources()
    bars = load_broker_bars(uploaded_bars if uploaded_bars is not None else bars_source)
    quotes = load_broker_quote(quote_source)
    supabase_url, supabase_read_key, supabase_symbol = _supabase_broker_config()
    terminal_status: dict[str, object] = {}
    if uploaded_bars is None and supabase_url and supabase_read_key:
        try:
            bars, quotes = get_supabase_broker_feed(supabase_url, supabase_read_key, supabase_symbol)
        except Exception as exc:
            st.warning(f"Supabase broker belum dapat dibaca; memakai fallback lokal/CSV. Detail: {exc}")
        try:
            terminal_status = get_supabase_terminal_status(supabase_url, supabase_read_key, supabase_symbol)
        except Exception as exc:
            st.info(f"Status terminal belum dapat dibaca. Jalankan migrasi SQL terbaru. Detail: {exc}")
    bars, quotes = apply_broker_clock_offset(bars, quotes)
    audit = audit_broker_feed(bars, quotes, stale_after_minutes=5)

    latest_quote = audit["latest_quote"]
    latest_bar = audit["latest_bar"]
    source_name = "-"
    if latest_quote is not None:
        source_name = str(latest_quote["source"])
    elif latest_bar is not None:
        source_name = str(latest_bar["source"])

    status = "BELUM TERHUBUNG"
    if audit["connected"]:
        status = "STALE" if audit["stale"] else "AKTIF"
    latest_market_wit = (
        "-"
        if pd.isna(audit["latest_timestamp"])
        else pd.Timestamp(audit["latest_timestamp"]).tz_convert(WIT).strftime("%d %b %Y %H:%M:%S WIT")
    )
    latest_received_wit = (
        latest_market_wit
        if pd.isna(audit["latest_received_at"])
        else pd.Timestamp(audit["latest_received_at"]).tz_convert(WIT).strftime("%d %b %Y %H:%M:%S WIT")
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Status feed", status)
    c2.metric("Sumber", source_name)
    c3.metric("Data diterima", latest_received_wit)
    c4.metric(
        "Usia data",
        "-" if pd.isna(audit["age_minutes"]) else f"{audit['age_minutes']:.1f} menit",
        "Batas stale 5 menit",
    )

    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Bid", "-" if latest_quote is None else f"${float(latest_quote['bid']):,.2f}")
    q2.metric("Ask", "-" if latest_quote is None else f"${float(latest_quote['ask']):,.2f}")
    q3.metric("Spread", "-" if latest_quote is None else f"${float(latest_quote['spread']):,.3f}")
    q4.metric("Close M1 terakhir", "-" if latest_bar is None else f"${float(latest_bar['close']):,.2f}")

    if not audit["connected"]:
        st.info(
            "Feed broker belum tersedia. Tab dapat membaca bridge MT5 lokal, Supabase read-only, atau URL CSV "
            "melalui Streamlit secrets."
        )

    quality = pd.DataFrame(
        [
            {
                "Pemeriksaan": "Feed memiliki timestamp valid",
                "Status": "Lolos" if audit["connected"] else "Menunggu",
                "Detail": latest_market_wit,
            },
            {
                "Pemeriksaan": "Sinkronisasi jam broker",
                "Status": "Lolos" if audit["clock_valid"] else "Perlu perhatian",
                "Detail": f"Offset terdeteksi {audit['clock_offset_hours']:+.0f} jam | diterima {latest_received_wit}",
            },
            {
                "Pemeriksaan": "Usia data maksimal 5 menit",
                "Status": "Lolos" if audit["connected"] and not audit["stale"] else "Menunggu",
                "Detail": "-" if pd.isna(audit["age_minutes"]) else f"{audit['age_minutes']:.1f} menit",
            },
            {
                "Pemeriksaan": "Ask tidak lebih rendah dari bid",
                "Status": "Lolos" if audit["quote_rows"] > 0 and audit["invalid_quotes"] == 0 else "Menunggu",
                "Detail": f"Quote tidak valid: {audit['invalid_quotes']}",
            },
            {
                "Pemeriksaan": "Struktur OHLC valid",
                "Status": "Lolos" if audit["bar_rows"] > 0 and audit["invalid_bars"] == 0 else "Menunggu",
                "Detail": f"Baris {audit['bar_rows']} | Tidak valid {audit['invalid_bars']}",
            },
            {
                "Pemeriksaan": "Celah data M1 lebih dari 5 menit",
                "Status": "Audit" if audit["gaps_over_five_minutes"] > 0 else "Lolos",
                "Detail": f"{audit['gaps_over_five_minutes']} celah; dapat mencakup libur/maintenance broker",
            },
        ]
    )
    st.markdown("**Audit Kualitas Feed**")
    st.dataframe(quality, use_container_width=True, hide_index=True)
    _render_real_trading_readiness(terminal_status, audit)

    if latest_quote is not None and not gold_ohlc.empty:
        broker_mid = float(latest_quote["mid"])
        futures_close = float(gold_ohlc["Close"].iloc[-1])
        comparison = pd.DataFrame(
            [
                {"Sumber": f"Broker {latest_quote['symbol']} mid", "Harga": broker_mid},
                {"Sumber": "Yahoo COMEX GC=F close", "Harga": futures_close},
            ]
        )
        st.markdown("**Perbandingan Broker vs Benchmark**")
        st.dataframe(comparison.style.format({"Harga": "${:,.2f}"}), use_container_width=True, hide_index=True)
        st.caption(
            f"Selisih broker mid terhadap GC=F: ${broker_mid - futures_close:+,.2f}. "
            "Ini bukan error otomatis karena XAUUSD OTC dan futures COMEX adalah instrumen berbeda."
        )

    if not bars.empty:
        preview = bars.tail(100).copy()
        preview["timestamp_wit"] = preview["timestamp_utc"].dt.tz_convert(WIT)
        display_columns = [
            "timestamp_wit",
            "open",
            "high",
            "low",
            "close",
            "tick_volume",
            "spread_points",
            "symbol",
            "source",
        ]
        st.markdown("**100 Candle M1 Terbaru**")
        st.dataframe(
            preview[display_columns].style.format(
                {"open": "${:,.2f}", "high": "${:,.2f}", "low": "${:,.2f}", "close": "${:,.2f}"},
                na_rep="-",
            ),
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("Cara Menjalankan Gold Predictor dari Windows"):
        st.markdown(
            r"""
            Setup satu kali dari root proyek:

            ```powershell
            powershell -NoProfile -ExecutionPolicy Bypass -File scripts\setup_gold_predictor.ps1
            ```

            Setelah setup, gunakan shortcut `START Gold Predictor` dan `STOP Gold Predictor` di Desktop. START membuka
            MT5, bridge, dan dashboard tanpa terminal. STOP hanya menghentikan bridge; MT5 tetap terbuka untuk trading manual.

            Untuk Streamlit Cloud melalui Supabase, tambahkan URL dan publishable/anon key read-only ke secrets:

            ```toml
            [supabase_broker]
            url = "https://PROJECT.supabase.co"
            publishable_key = "sb_publishable_..."
            symbol = "XAUUSD"
            ```

            `service_role` tidak boleh dimasukkan ke Streamlit atau GitHub; key tersebut hanya berada di laptop bridge.
            Gold Predictor tidak memiliki kode pengiriman order dan tidak mempublikasikan nomor akun, nama pemilik,
            balance, equity, atau password broker.
            """
        )


with st.sidebar:
    st.header("Pengaturan")
    auto_refresh = st.checkbox("Auto refresh dashboard", value=True)
    refresh_interval_seconds = st.selectbox(
        "Interval auto refresh",
        options=[60, 300, 900],
        index=1,
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
        get_dashboard_snapshot.clear()
        get_supabase_broker_feed.clear()
        get_supabase_terminal_status.clear()
        get_base_optimizer.clear()
        get_simulations.clear()
        st.rerun()
    page = st.radio(
        "Halaman",
        ["Dashboard", "Simulasi", "Live Trading", "Data Broker", "Audit Intraday"],
        index=0,
    )
    if page == "Dashboard":
        direction_threshold = st.select_slider(
            "Threshold sinyal arah",
            options=[0.50, 0.55, 0.60, 0.65, 0.70],
            value=0.65,
            format_func=lambda value: f"{value:.0%}",
        )
    st.info("Harga emas memakai COMEX `GC=F`, USD per troy ounce.")

if page == "Dashboard":
    try:
        market, data_fetched_at = get_data()
        gold_ohlc = get_gold_ohlc()
        dashboard_snapshot = get_dashboard_snapshot(DASHBOARD_SNAPSHOT_VERSION)
    except Exception as exc:
        st.error(f"Data belum dapat diproses: {exc}")
        st.stop()

    if dashboard_snapshot is None:
        st.warning(
            "Snapshot model harian belum tersedia. GitHub Actions perlu menyelesaikan pembuatan snapshot satu kali. "
            "Dashboard sengaja tidak melatih model saat halaman dibuka agar aplikasi tetap responsif."
        )
        st.stop()

    model_1 = dashboard_snapshot["model_1"]
    model_2 = dashboard_snapshot["model_2"]
    direction_model = dashboard_snapshot["direction_model"]
    v10_leaderboard = dashboard_snapshot.get("v10_leaderboard", pd.DataFrame())

    render_dashboard(
        market,
        data_fetched_at,
        model_1,
        model_2,
        direction_model,
        direction_threshold,
        gold_ohlc,
        v10_leaderboard,
        dashboard_snapshot["generated_at_utc"],
    )

elif page == "Simulasi":
    simulation_payload = load_precomputed_simulations(SIMULATION_CACHE_VERSION)
    if simulation_payload is None:
        st.warning(
            "Hasil simulasi precomputed untuk versi terbaru belum tersedia. "
            "Aplikasi sengaja tidak menghitung v1 dan v10 saat startup agar dashboard bisa dibuka lebih cepat."
        )
        if st.button("Bangun ulang simulasi v1/v10", use_container_width=True):
            with st.spinner("Menghitung simulasi lengkap. Proses ini bisa memakan waktu di Streamlit Cloud."):
                simulation_payload = get_simulations(SIMULATION_CACHE_VERSION)
            st.rerun()
    else:
        st.info(
            "Hasil simulasi precomputed tersedia. Untuk menjaga aplikasi ringan saat dibuka, "
            "visualisasi simulasi tidak dirender otomatis."
        )
        if st.checkbox("Tampilkan hasil simulasi precomputed", value=False):
            try:
                gold_ohlc = get_gold_ohlc()
            except Exception as exc:
                st.error(f"Data OHLC belum dapat diproses: {exc}")
                st.stop()
            render_simulation(*simulation_payload, gold_ohlc)

elif page == "Live Trading":
    try:
        gold_ohlc = get_gold_ohlc()
    except Exception as exc:
        st.error(f"Data OHLC belum dapat diproses: {exc}")
        st.stop()

    broker_quote, broker_feed_error = _latest_broker_quote()
    if broker_feed_error:
        st.warning(f"Feed Supabase gagal dibaca; mencoba quote lokal. Detail: {broker_feed_error}")

    supabase_url, supabase_read_key, supabase_symbol = _supabase_broker_config()
    live_terminal_status: dict[str, object] = {}
    if supabase_url and supabase_read_key:
        try:
            live_terminal_status = get_supabase_terminal_status(
                supabase_url,
                supabase_read_key,
                supabase_symbol,
            )
        except Exception:
            pass
    live_account_mode = str(live_terminal_status.get("account_mode", "BELUM TERVERIFIKASI")).upper()
    live_broker = str(
        live_terminal_status.get("broker_company") or live_terminal_status.get("broker_server") or "-"
    )
    if live_account_mode == "REAL":
        st.error(
            f"Akun MT5 **REAL** terdeteksi | Broker: **{live_broker}** | Gold Predictor tetap read-only. "
            "Order uang riil dilakukan manual di MT5; ledger v1/v10 di bawah tetap paper trading."
        )
    else:
        st.info(
            f"Mode akun MT5: **{live_account_mode}** | Broker: **{live_broker}** | "
            "Periksa panel lengkap di Data Broker sebelum entry manual."
        )

    live_v1_tab, live_v10_tab = st.tabs(["Optimizer v1", "Optimizer v10"])
    st.info("Rencana evaluasi: paper live trading paralel Optimizer v1 dan Optimizer v10 berjalan sampai **30 Agustus 2026**.")
    with live_v1_tab:
        optimization_v1_live_leaderboard = get_v1_leaderboard_for_live(SIMULATION_CACHE_VERSION)
        render_live_trading(
            gold_ohlc,
            optimization_v1_live_leaderboard,
            title="Optimizer v1",
            start_date=LIVE_START_DATE,
            live_path=LIVE_TRADING_PATH,
            manual_path=LIVE_MANUAL_EXIT_PATH,
            key_prefix="v1",
            strategy_note=(
                "Ini adalah Live Trading lama yang sudah berjalan. Entry mengikuti sinyal Optimizer v1, "
                "boleh menambah posisi dari sinyal hari berbeda sampai batas maksimal 8 BUY dan 10 SELL."
            ),
            broker_quote=broker_quote,
        )
    with live_v10_tab:
        optimization_v10_live_leaderboard = get_v10_leaderboard_for_live(SIMULATION_CACHE_VERSION)
        render_live_trading(
            gold_ohlc,
            optimization_v10_live_leaderboard,
            title="Optimizer v10",
            start_date=LIVE_V10_START_DATE,
            live_path=LIVE_TRADING_V10_PATH,
            manual_path=LIVE_MANUAL_EXIT_V10_PATH,
            key_prefix="v10",
            strategy_note=(
                "Paper live v10 mulai 20 Juli 2026. Parameter diambil dari hasil Optimizer v10 precomputed, "
                "sehingga tidak menghitung ulang optimizer berat saat dashboard dibuka."
            ),
            broker_quote=broker_quote,
        )

elif page == "Data Broker":
    try:
        gold_ohlc = get_gold_ohlc()
    except Exception as exc:
        st.error(f"Data benchmark GC=F belum dapat diproses: {exc}")
        gold_ohlc = pd.DataFrame()
    render_broker_data(gold_ohlc)

else:
    try:
        gold_ohlc = get_gold_ohlc()
    except Exception as exc:
        st.error(f"Data OHLC belum dapat diproses: {exc}")
        st.stop()

    render_intraday_audit(gold_ohlc)
