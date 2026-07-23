Exit code: 0
Wall time: 2.2 seconds
Total output lines: 4852
Output:
from __future__ import annotations

import base64
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


SIMULATION_CACHE_VERSION = "optimizer-v1-only-2025q1-2026q2"
PRECOMPUTED_SIMULATION_PATH = Path("data/precomputed/simulations.pkl")
OPTIMIZER_OOS_VERSION = "optimizer-v1-only-train2025-oos2026h1"
OPTIMIZER_OOS_PATH = Path("data/precomputed/optimizer_oos.pkl")
EXACT_BROKER_OOS_VERSION = "optimizer-v1-only-exact-broker-aware-oos-2026h1"
EXACT_BROKER_OOS_PATH = Path("data/precomputed/exact_broker_oos.pkl")
V1_ROBUSTNESS_VERSION = "optimizer-v1-robustness-oos-2026h1"
V1_ROBUSTNESS_PATH = Path("data/precomputed/v1_robustness.pkl")
V1_RISK_CONTROL_VERSION = "optimizer-v1-risk-control-lab-2025-2026h1-v1"
V1_RISK_CONTROL_PATH = Path("data/precomputed/v1_risk_control.pkl")
V1_SIGNAL_QUALITY_VERSION = "optimizer-v1-balanced-entry-2025-2026h1-v3"
V1_SIGNAL_QUALITY_PATH = Path("data/precomputed/v1_signal_quality.pkl.b64")
V1_BALANCED_ROBUSTNESS_VERSION = "optimizer-v1-balanced-entry-robustness-2025-2026h1-v1"
V1_BALANCED_ROBUSTNESS_PATH = Path("data/precomputed/v1_balanced_robustness.pkl.b64")
V1_SIDEWAYS_DEFENSE_VERSION = "optimizer-v1-sideways-defense-2025-2026h1-v1"
V1_SIDEWAYS_DEFENSE_PATH = Path("data/precomputed/v1_sideways_defense.pkl.b64")
V1_REGIME_CLASSIFIER_VERSION = "optimizer-v1-regime-classifier-2025-2026h1-v2"
V1_REGIME_CLASSIFIER_PATH = Path("data/precomputed/v1_regime_classifier.pkl.b64")

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
def load_precomputed_optimizer_oos(backtest_version: str):
    if not OPTIMIZER_OOS_PATH.exists():
        return None
    try:
        with OPTIMIZER_OOS_PATH.open("rb") as file:
            saved = pickle.load(file)
        if saved.get("version") == backtest_version:
            return saved["payload"]
    except Exception:
        return None
    return None


@st.cache_resource
def load_precomputed_exact_broker_oos(backtest_version: str):
    if not EXACT_BROKER_OOS_PATH.exists():
        return None
    try:
        with EXACT_BROKER_OOS_PATH.open("rb") as file:
            saved = pickle.load(file)
        if saved.get("version") == backtest_version:
            return saved["payload"]
    except Exception:
        return None
    return None


@st.cache_resource
def load_precomputed_v1_robustness(backtest_version: str):
    if not V1_ROBUSTNESS_PATH.exists():
        return None
    try:
        with V1_ROBUSTNESS_PATH.open("rb") as file:
            saved = pickle.load(file)
        if saved.get("version") == backtest_version:
            return saved["payload"]
    except Exception:
        return None
    return None


@st.cache_resource
def load_precomputed_v1_risk_control(backtest_version: str):
    if not V1_RISK_CONTROL_PATH.exists():
        return None
    try:
        with V1_RISK_CONTROL_PATH.open("rb") as file:
            saved = pickle.load(file)
        if saved.get("version") == backtest_version:
            return saved["payload"]
    except Exception:
        return None
    return None


@st.cache_resource
def load_precomputed_v1_signal_quality(backtest_version: str):
    if not V1_SIGNAL_QUALITY_PATH.exists():
        return None
    try:
        saved = pickle.loads(base64.b64decode(V1_SIGNAL_QUALITY_PATH.read_text(encoding="ascii")))
        if saved.get("version") == backtest_version:
            return saved["payload"]
    except Exception:
        return None
    return None


@st.cache_resource
def load_precomputed_v1_balanced_robustness(backtest_version: str):
    if not V1_BALANCED_ROBUSTNESS_PATH.exists():
        return None
    try:
        saved = pickle.loads(
            base64.b64decode(V1_BALANCED_ROBUSTNESS_PATH.read_text(encoding="ascii"))
        )
        if saved.get("version") == backtest_version:
            return saved["payload"]
    except Exception:
        return None
    return None


@st.cache_resource
def load_precomputed_v1_sideways_defense(backtest_version: str):
    if not V1_SIDEWAYS_DEFENSE_PATH.exists():
        return None
    try:
        saved = pickle.loads(
            base64.b64decode(V1_SIDEWAYS_DEFENSE_PATH.read_text(encoding="ascii"))
        )
        if saved.get("version") == backtest_version:
            return saved["payload"]
    except Exception:
        return None
    return None


@st.cache_resource
def load_precomputed_v1_regime_classifier(backtest_version: str):
    if not V1_REGIME_CLASSIFIER_PATH.exists():
        return None
    try:
        saved = pickle.loads(
            base64.b64decode(V1_REGIME_CLASSIFIER_PATH.read_text(encoding="ascii"))
        )
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
    payload = (optimized_result, optimization_leaderboard)
    try:
        PRECOMPUTED_SIMULATION_PATH.parent.mkdir(parents=True, exist_ok=True)
        with PRECOMPUTED_SIMULATION_PATH.open("wb") as file:
            pickle.dump({"version": simulation_version, "payload": payload}, file)
    except Exception:
        pass
    return payload


def get_v1_leaderboard_for_live(simulation_version: str) -> pd.DataFrame:
    cached = load_precomputed_simulations(simulation_version)
    if cached is None or len(cached) < 2:
        return pd.DataFrame()
    leaderboard = cached[1]
    if isinstance(leaderboard, pd.DataFrame):
        return leaderboard
    return pd.DataFrame()


def _optimizer_v1_latest_prediction(gold_ohlc: pd.DataFrame, leaderboard: pd.DataFrame) -> dict[str, object]:
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
        note = "Candle terbaru belum memenuhi threshold entry Optimizer v1."
    else:
        prediction = float(signals.iloc[-1])
        expected_change_pct = (prediction / latest_close - 1) * 100
        if expected_change_pct >= threshold:
            direction = "BUY"
            note = "Sinyal BUY v1 lolos pada candle harian terbaru."
        elif expected_change_pct <= -threshold:
            direction = "SELL"
            note = "Sinyal SELL v1 lolos pada candle harian terbaru."
        else:
            direction = "WAIT"
            note = "Expected change belum melewati threshold Optimizer v1."

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


def render_optimizer_v1_dashboard(
    market: pd.DataFrame,
    data_fetched_at: pd.Timestamp,
    gold_ohlc: pd.DataFrame,
    optimization_v1_leaderboard: pd.DataFrame,
    history_years: int,
) -> None:
    gold = market["gold"]
    latest_market = float(gold.iloc[-1])
    previous_market = float(gold.iloc[-2])
    market_last_date = pd.Timestamp(gold.index.max()).strftime("%d %b %Y")
    fetched_label = data_fetched_at.strftime("%d %b %Y %H:%M:%S WIT")
    prediction = _optimizer_v1_latest_prediction(gold_ohlc, optimization_v1_leaderboard)
    params = prediction["params"]

    st.info(f"Data pasar terakhir: **{market_last_date}** | Data diambil dashboard: **{fetched_label}**")
    if optimization_v1_leaderboard.empty:
        st.warning(
            "Parameter precomputed Optimizer v1 belum tersedia. Dashboard memakai parameter fallback dan tidak "
            "menghitung ulang optimizer berat saat halaman dibuka."
        )

    signal_direction = str(prediction["direction"])
    signal_label = signal_direction if signal_direction in {"BUY", "SELL"} else "WAIT"
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Harga terakhir", f"${latest_market:,.2f}", f"{latest_market - previous_market:+,.2f}")
    c2.metric("Sinyal Optimizer v1", signal_label, str(prediction["note"]))
    c3.metric(
        "Prediksi strategi",
        f"${float(prediction['prediction']):,.2f}",
        f"{float(prediction['expected_change_pct']):+.2f}%",
    )
    target_value = prediction["target_price"]
    c4.metric("Target TP v1", "-" if pd.isna(target_value) else f"${float(target_value):,.2f}")

    st.subheader("Prediksi Menurut Optimizer v1")
    if signal_direction == "BUY":
        st.success("Optimizer v1 membaca peluang **BUY** pada candle harian terbaru.")
    elif signal_direction == "SELL":
        st.error("Optimizer v1 membaca peluang **SELL** pada candle harian terbaru.")
    else:
        st.info("Optimizer v1 belum memberi sinyal entry. Posisi yang disarankan: **WAIT / TUNGGU**.")

    detail = pd.DataFrame(
        [
            {"Parameter": "Tanggal candle", "Nilai": _format_date(prediction.get("latest_date")), "Keterangan": "Candle harian terbaru yang dievaluasi."},
            {"Parameter": "Strategi", "Nilai": params["Strategi"], "Keterangan": "Kandidat baseline Optimizer v1 atau fallback jika belum ada precomputed."},
            {"Parameter": "Mode", "Nilai": params["Mode"], "Keterangan": "Jenis rule sinyal: Trend, Breakout, atau Pullback."},
            {"Parameter": "MA cepat/lambat", "Nilai": f"{params['Fast MA']} / {params['Slow MA']}", "Keterangan": "Moving average harian yang dipakai strategi."},
            {"Parameter": "Momentum hari", "Nilai": params["Momentum hari"], "Keterangan": "Jendela momentum harian."},
            {"Parameter": "Threshold entry", "Nilai": f"{params['Threshold entry (%)']:.2f}%", "Keterangan": "Minimal expected change agar entry valid."},
            {"Parameter": "TP / CL", "Nilai": f"USD {params['TP (USD)']:,.2f} / USD {params['SL (USD)']:,.2f}", "Keterangan": "Target profit dan batas rugi per posisi dari strategi."},
            {"Parameter": "Lot backtest", "Nilai": f"{float(params.get('Lot', 0.01)):,.2f}", "Keterangan": "Lot dari kandidat v1. Paper live tetap memakai rule live yang sudah ditetapkan."},
            {"Parameter": "Harga CL risiko", "Nilai": "-" if pd.isna(prediction["risk_price"]) else f"${float(prediction['risk_price']):,.2f}", "Keterangan": "Harga risiko jika sinyal BUY/SELL aktif."},
        ]
    )
    st.dataframe(detail, use_container_width=True, hide_index=True)

    st.caption(
        "Catatan: Model 3 adalah prediksi berbasis strategi Optimizer v1. Ia menilai apakah candle harian terbaru "
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
    optimization_v1_leaderboard: pd.DataFrame,
    snapshot_generated_at: str,
) -> None:
    gold = market["gold"]
    latest = float(gold.iloc[-1])
    previous = float(gold.iloc[-2])
    market_last_date = pd.Timestamp(gold.index.max()).strftime("%d %b %Y")
    fetched_label = data_fetched_at.strftime("%d %b %Y %H:%M:%S WIT")
    v1_prediction = _optimizer_v1_latest_prediction(gold_ohlc, optimization_v1_leaderboard)
    v1_params = v1_prediction["params"]

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
        signal_direction = str(v1_prediction["direction"])
        signal_label = signal_direction if signal_direction in {"BUY", "SELL"} else "WAIT"
        target_value = v1_prediction["target_price"]
        st.markdown("**Model 3 - Optimizer v1 Baseline**")
        st.metric(
            "Prediksi strategi",
            f"${float(v1_prediction['prediction']):,.2f}",
            f"{float(v1_prediction['expected_change_pct']):+.2f}%",
        )
        st.metric("Target TP", "-" if pd.isna(target_value) else f"${float(target_value):,.2f}")
        st.metric("Sinyal", signal_label, str(v1_prediction["note"]))
        st.caption("Prediksi berbasis rule baseline Optimizer v1, bukan forecast probabilistik harga.")

    if optimization_v1_leaderboard.empty:
        st.warning(
            "Parameter precomputed Optimizer v1 belum tersedia. Model 3 memakai parameter fallback dan tidak "
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
                "Model": "Model 3 - Optimizer v1 Baseline",
                "Output utama": f"${float(v1_prediction['prediction']):,.2f}",
                "Arah / Sinyal": signal_label,
                "Perubahan vs terakhir": float(v1_prediction["prediction"]) - latest,
                "Confidence / Expected": f"{float(v1_prediction['expected_change_pct']):+.2f}%",
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
        use…43795 tokens truncated…minal_connected", False))
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
    v1_leaderboard = dashboard_snapshot.get("v1_leaderboard")
    if not isinstance(v1_leaderboard, pd.DataFrame) or v1_leaderboard.empty:
        v1_leaderboard = get_v1_leaderboard_for_live(SIMULATION_CACHE_VERSION)

    render_dashboard(
        market,
        data_fetched_at,
        model_1,
        model_2,
        direction_model,
        direction_threshold,
        gold_ohlc,
        v1_leaderboard,
        dashboard_snapshot["generated_at_utc"],
    )

elif page == "Simulasi":
    simulation_payload = load_precomputed_simulations(SIMULATION_CACHE_VERSION)
    if simulation_payload is None:
        st.warning(
            "Hasil simulasi precomputed untuk versi terbaru belum tersedia. "
            "Aplikasi sengaja tidak menghitung optimizer saat startup agar dashboard bisa dibuka lebih cepat."
        )
        if st.button("Bangun ulang simulasi v1", use_container_width=True):
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
            "Order uang riil dilakukan manual di MT5; ledger v1 di bawah tetap paper trading."
        )
    else:
        st.info(
            f"Mode akun MT5: **{live_account_mode}** | Broker: **{live_broker}** | "
            "Periksa panel lengkap di Data Broker sebelum entry manual."
        )

    st.info(
        "Optimizer v1 adalah satu-satunya baseline paper trading aktif. Strategi v10 telah diarsipkan dan tidak dapat "
        "membuka posisi baru."
    )
    st.warning(
        "Status v1: **KANDIDAT PAPER TRADING, BELUM LAYAK REAL-MONEY**. Exact OOS masih positif, tetapi robustness "
        "belum memenuhi target profit factor 1,30 dan drawdown maksimum 10%."
    )
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
            "Entry mengikuti sinyal Optimizer v1, boleh menambah posisi dari sinyal hari berbeda sampai batas "
            "maksimal 8 BUY dan 10 SELL. Strategi ini masih paper trading."
        ),
        broker_quote=broker_quote,
    )

    archived_v10_ledger = load_live_ledger(LIVE_TRADING_V10_PATH)
    archived_open = archived_v10_ledger[archived_v10_ledger["status"].eq("OPEN")]
    if not archived_open.empty:
        archived_v10 = run_live_trading_update(
            gold_ohlc,
            pd.DataFrame(),
            path=LIVE_TRADING_V10_PATH,
            start_date=LIVE_V10_START_DATE,
            broker_quote=broker_quote,
            allow_new_entries=False,
        )
        archived_open = archived_v10["open_positions"]
        if not archived_open.empty:
            st.info(
                f"Arsip v10 masih memantau **{len(archived_open)} posisi lama** sampai TP/CL. Tidak ada posisi baru yang dibuka."
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

