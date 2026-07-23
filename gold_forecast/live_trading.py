from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from gold_forecast.monitoring import WIT
from gold_forecast.simulation import CONTRACT_OUNCES_PER_LOT
from gold_forecast.strategy_optimizer import _indicator_predictions, _rsi
from gold_forecast.v1_risk_control import _entry_signals_for_period
from gold_forecast.v1_signal_quality import (
    SignalQualityConfig,
    _entry_features,
    _select_signals,
)


LIVE_TRADING_PATH = Path("data/live_trading_optimizer.csv")
LIVE_MANUAL_EXIT_PATH = Path("data/live_trading_manual_exits.csv")
LIVE_TRADING_V10_PATH = Path("data/live_trading_optimizer_v10.csv")
LIVE_MANUAL_EXIT_V10_PATH = Path("data/live_trading_manual_exits_v10.csv")
LIVE_TRADING_FIXED_DELAY_PATH = Path("data/live_trading_fixed_delay_5m.csv")
LIVE_MANUAL_EXIT_FIXED_DELAY_PATH = Path("data/live_trading_manual_exits_fixed_delay_5m.csv")
LIVE_INITIAL_EQUITY = 1000.0
LIVE_START_DATE = pd.Timestamp("2026-07-15")
LIVE_V10_START_DATE = pd.Timestamp("2026-07-20")
LIVE_FIXED_DELAY_START = pd.Timestamp("2026-07-23 22:20:00", tz=WIT)
LIVE_LOT_SIZE = 0.01
LIVE_BUY_SWAP_PER_001_LOT = 0.02
LIVE_SELL_SWAP_PER_001_LOT = 0.0
LIVE_MAX_BUY = 8
LIVE_MAX_SELL = 10
FIXED_DELAY_MINUTES = 5
FIXED_DELAY_SPREAD_LIMIT_POINTS = 20.0
FIXED_DELAY_TP_USD = 25.0
FIXED_DELAY_SL_USD = 10.0

LIVE_COLUMNS = [
    "position_id",
    "signal_date",
    "detected_at_wit",
    "status",
    "arah",
    "lot",
    "prediction",
    "reference_price",
    "expected_change_pct",
    "threshold_entry_pct",
    "tp_usd",
    "cl_usd",
    "entry_time_wit",
    "entry_price",
    "last_swap_date",
    "exit_time_wit",
    "exit_price",
    "exit_reason",
    "gross_pl",
    "swap",
    "net_pl",
    "last_update_wit",
    "catatan",
]
LIVE_TEXT_COLUMNS = [
    "signal_date",
    "detected_at_wit",
    "status",
    "arah",
    "entry_time_wit",
    "last_swap_date",
    "exit_time_wit",
    "exit_reason",
    "last_update_wit",
    "catatan",
]
LIVE_MANUAL_EXIT_COLUMNS = [
    "manual_exit_id",
    "position_id",
    "optimizer_signal_date",
    "optimizer_direction",
    "optimizer_entry_time_wit",
    "optimizer_entry_price",
    "optimizer_tp_usd",
    "optimizer_cl_usd",
    "manual_exit_time_wit",
    "manual_exit_price",
    "manual_gross_pl",
    "manual_swap_at_exit",
    "manual_net_pl",
    "manual_result_label",
    "last_update_wit",
    "catatan",
]
LIVE_MANUAL_EXIT_TEXT_COLUMNS = [
    "optimizer_signal_date",
    "optimizer_direction",
    "optimizer_entry_time_wit",
    "manual_exit_time_wit",
    "manual_result_label",
    "last_update_wit",
    "catatan",
]


def _empty_ledger() -> pd.DataFrame:
    return pd.DataFrame(columns=LIVE_COLUMNS)


def _empty_manual_exit_ledger() -> pd.DataFrame:
    return pd.DataFrame(columns=LIVE_MANUAL_EXIT_COLUMNS)


def load_live_ledger(path: Path = LIVE_TRADING_PATH) -> pd.DataFrame:
    if not path.exists():
        return _empty_ledger()
    frame = pd.read_csv(path)
    for column in LIVE_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
    for column in LIVE_TEXT_COLUMNS:
        frame[column] = frame[column].fillna("").astype(str)
    return frame[LIVE_COLUMNS]


def save_live_ledger(frame: pd.DataFrame, path: Path = LIVE_TRADING_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def load_manual_exit_ledger(path: Path = LIVE_MANUAL_EXIT_PATH) -> pd.DataFrame:
    if not path.exists():
        return _empty_manual_exit_ledger()
    frame = pd.read_csv(path)
    for column in LIVE_MANUAL_EXIT_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
    for column in LIVE_MANUAL_EXIT_TEXT_COLUMNS:
        frame[column] = frame[column].fillna("").astype(str)
    return frame[LIVE_MANUAL_EXIT_COLUMNS]


def save_manual_exit_ledger(frame: pd.DataFrame, path: Path = LIVE_MANUAL_EXIT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _now_wit(now: pd.Timestamp | None = None) -> pd.Timestamp:
    if now is None:
        return pd.Timestamp.now(tz=WIT)
    timestamp = pd.Timestamp(now)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize(WIT)
    return timestamp.tz_convert(WIT)


def _is_live_session_open(now: pd.Timestamp) -> tuple[bool, str]:
    local_time = now.time()
    session_date = now.date()
    if local_time.hour < 6:
        session_date = (now - pd.Timedelta(days=1)).date()

    session_day = pd.Timestamp(session_date)
    if session_day.weekday() >= 5:
        return False, "Libur akhir pekan, tidak membuka posisi baru."
    if local_time.hour == 6:
        return False, "Di luar jam trading 07:00-06:00 WIT."
    return True, f"Jam trading aktif untuk sesi {session_day.strftime('%d %b %Y')}."


def _best_optimizer_params(leaderboard: pd.DataFrame) -> dict[str, object]:
    if leaderboard.empty:
        return {
            "Mode": "Trend",
            "Fast MA": 20,
            "Slow MA": 50,
            "Momentum hari": 10,
            "Threshold entry (%)": 0.15,
            "TP (USD)": 25.0,
            "SL (USD)": 18.0,
            "Max BUY": LIVE_MAX_BUY,
            "Max SELL": LIVE_MAX_SELL,
            "Strategi": "Fallback Optimizer",
        }
    best = leaderboard.iloc[0].to_dict()
    max_buy = pd.to_numeric(best.get("Max BUY", LIVE_MAX_BUY), errors="coerce")
    max_sell = pd.to_numeric(best.get("Max SELL", LIVE_MAX_SELL), errors="coerce")
    return {
        "Mode": best.get("Mode", "Trend"),
        "Fast MA": int(best.get("Fast MA", 20)),
        "Slow MA": int(best.get("Slow MA", 50)),
        "Momentum hari": int(best.get("Momentum hari", 10)),
        "Threshold entry (%)": float(best.get("Threshold entry (%)", 0.15)),
        "TP (USD)": float(best.get("TP (USD)", 25.0)),
        "SL (USD)": float(best.get("SL (USD)", 18.0)),
        "Max BUY": LIVE_MAX_BUY if pd.isna(max_buy) else int(max_buy),
        "Max SELL": LIVE_MAX_SELL if pd.isna(max_sell) else int(max_sell),
        "Strategi": best.get("Strategi", "Strategi Terbaik Optimizer"),
    }


def _current_optimizer_signal(
    gold_ohlc: pd.DataFrame,
    params: dict[str, object],
    now: pd.Timestamp,
    start_date: pd.Timestamp = LIVE_START_DATE,
) -> dict[str, object] | None:
    if gold_ohlc.empty:
        return None
    signals = _indicator_predictions(
        gold_ohlc,
        str(params["Mode"]),
        int(params["Fast MA"]),
        int(params["Slow MA"]),
        int(params["Momentum hari"]),
        float(params["Threshold entry (%)"]),
        test_start=pd.Timestamp(gold_ohlc.index.min()),
        test_end=pd.Timestamp(gold_ohlc.index.max()),
    )
    if signals.empty:
        return None
    signals = signals[(signals.index >= start_date) & (signals.index <= now.tz_localize(None).normalize())]
    if signals.empty:
        return None

    signal_date = pd.Timestamp(signals.index[-1])
    prediction = float(signals.iloc[-1])
    reference_price = float(gold_ohlc.loc[signal_date, "Close"])
    expected_change_pct = (prediction / reference_price - 1) * 100
    if expected_change_pct >= float(params["Threshold entry (%)"]):
        direction = "BUY"
    elif expected_change_pct <= -float(params["Threshold entry (%)"]):
        direction = "SELL"
    else:
        direction = "NETRAL"

    return {
        "signal_date": signal_date,
        "prediction": prediction,
        "reference_price": reference_price,
        "expected_change_pct": expected_change_pct,
        "arah": direction,
    }


def _signal_waiting_state(
    gold_ohlc: pd.DataFrame,
    params: dict[str, object],
    live_price: float | None = None,
    live_timestamp: pd.Timestamp | None = None,
) -> dict[str, object]:
    if gold_ohlc.empty:
        return {
            "Status sinyal": "Data belum tersedia",
            "Yang ditunggu": "Menunggu data harga emas terbaru.",
            "Kondisi BUY": "-",
            "Kondisi SELL": "-",
            "Checklist BUY": [],
            "Checklist SELL": [],
            "Interpretasi": "Data harga belum tersedia untuk membaca kondisi BUY/SELL.",
            "Momentum saat ini": np.nan,
            "Threshold": float(params["Threshold entry (%)"]),
            "MA cepat": np.nan,
            "MA lambat": np.nan,
            "RSI": np.nan,
        }

    mode = str(params["Mode"])
    fast_window = int(params["Fast MA"])
    slow_window = int(params["Slow MA"])
    momentum_days = int(params["Momentum hari"])
    threshold = float(params["Threshold entry (%)"])
    close = gold_ohlc["Close"].astype(float)
    high = gold_ohlc["High"].astype(float)
    low = gold_ohlc["Low"].astype(float)
    latest_date = pd.Timestamp(gold_ohlc.index[-1])
    latest_close = float(close.iloc[-1])
    fast_ma = close.rolling(fast_window).mean()
    slow_ma = close.rolling(slow_window).mean()
    momentum = close.pct_change(momentum_days) * 100
    rsi = _rsi(close)
    previous_high = high.rolling(slow_window).max().shift(1)
    previous_low = low.rolling(slow_window).min().shift(1)

    latest_fast = float(fast_ma.iloc[-1]) if pd.notna(fast_ma.iloc[-1]) else np.nan
    latest_slow = float(slow_ma.iloc[-1]) if pd.notna(slow_ma.iloc[-1]) else np.nan
    latest_momentum = float(momentum.iloc[-1]) if pd.notna(momentum.iloc[-1]) else np.nan
    latest_rsi = float(rsi.iloc[-1]) if pd.notna(rsi.iloc[-1]) else np.nan
    latest_previous_high = float(previous_high.iloc[-1]) if pd.notna(previous_high.iloc[-1]) else np.nan
    latest_previous_low = float(previous_low.iloc[-1]) if pd.notna(previous_low.iloc[-1]) else np.nan

    if mode == "Trend":
        buy_checklist = [
            {
                "Syarat": "Close > MA cepat",
                "Nilai saat ini": f"{latest_close:,.2f} > {latest_fast:,.2f}",
                "Lolos": latest_close > latest_fast,
            },
            {
                "Syarat": "MA cepat > MA lambat",
                "Nilai saat ini": f"{latest_fast:,.2f} > {latest_slow:,.2f}",
                "Lolos": latest_fast > latest_slow,
            },
            {
                "Syarat": f"Momentum {momentum_days} hari > +{threshold:.2f}%",
                "Nilai saat ini": f"{latest_momentum:+.2f}%",
                "Lolos": latest_momentum > threshold,
            },
        ]
        sell_checklist = [
            {
                "Syarat": "Close < MA cepat",
                "Nilai saat ini": f"{latest_close:,.2f} < {latest_fast:,.2f}",
                "Lolos": latest_close < latest_fast,
            },
            {
                "Syarat": "MA cepat < MA lambat",
                "Nilai saat ini": f"{latest_fast:,.2f} < {latest_slow:,.2f}",
                "Lolos": latest_fast < latest_slow,
            },
            {
                "Syarat": f"Momentum {momentum_days} hari < -{threshold:.2f}%",
                "Nilai saat ini": f"{latest_momentum:+.2f}%",
                "Lolos": latest_momentum < -threshold,
            },
        ]
        waiting = "Menunggu alignment tren: BUY jika harga dan MA cepat berada di atas MA lambat; SELL jika kebalikannya."
    elif mode == "Breakout":
        buy_checklist = [
            {
                "Syarat": f"Close > high {slow_window} hari sebelumnya",
                "Nilai saat ini": f"{latest_close:,.2f} > {latest_previous_high:,.2f}",
                "Lolos": latest_close > latest_previous_high,
            },
            {
                "Syarat": f"Momentum {momentum_days} hari > 0",
                "Nilai saat ini": f"{latest_momentum:+.2f}%",
                "Lolos": latest_momentum > 0,
            },
        ]
        sell_checklist = [
            {
                "Syarat": f"Close < low {slow_window} hari sebelumnya",
                "Nilai saat ini": f"{latest_close:,.2f} < {latest_previous_low:,.2f}",
                "Lolos": latest_close < latest_previous_low,
            },
            {
                "Syarat": f"Momentum {momentum_days} hari < 0",
                "Nilai saat ini": f"{latest_momentum:+.2f}%",
                "Lolos": latest_momentum < 0,
            },
        ]
        waiting = "Menunggu breakout: harga menembus high/low periode acuan dengan momentum searah."
    elif mode == "Pullback":
        buy_checklist = [
            {
                "Syarat": "Close > MA lambat",
                "Nilai saat ini": f"{latest_close:,.2f} > {latest_slow:,.2f}",
                "Lolos": latest_close > latest_slow,
            },
            {"Syarat": "RSI < 42", "Nilai saat ini": f"{latest_rsi:.1f}", "Lolos": latest_rsi < 42},
            {
                "Syarat": f"Momentum {momentum_days} hari > -{threshold:.2f}%",
                "Nilai saat ini": f"{latest_momentum:+.2f}%",
                "Lolos": latest_momentum > -threshold,
            },
        ]
        sell_checklist = [
            {
                "Syarat": "Close < MA lambat",
                "Nilai saat ini": f"{latest_close:,.2f} < {latest_slow:,.2f}",
                "Lolos": latest_close < latest_slow,
            },
            {"Syarat": "RSI > 58", "Nilai saat ini": f"{latest_rsi:.1f}", "Lolos": latest_rsi > 58},
            {
                "Syarat": f"Momentum {momentum_days} hari < +{threshold:.2f}%",
                "Nilai saat ini": f"{latest_momentum:+.2f}%",
                "Lolos": latest_momentum < threshold,
            },
        ]
        waiting = "Menunggu pullback: harga tetap di sisi tren utama sambil RSI masuk area koreksi."
    else:
        buy_checklist = [{"Syarat": "Mode strategi valid", "Nilai saat ini": mode, "Lolos": False}]
        sell_checklist = [{"Syarat": "Mode strategi valid", "Nilai saat ini": mode, "Lolos": False}]
        waiting = "Menunggu mode strategi yang valid."

    buy_ready = all(item["Lolos"] for item in buy_checklist)
    sell_ready = all(item["Lolos"] for item in sell_checklist)
    buy_conditions = [f"{item['Syarat']}: {item['Nilai saat ini']}" for item in buy_checklist]
    sell_conditions = [f"{item['Syarat']}: {item['Nilai saat ini']}" for item in sell_checklist]
    buy_passed = sum(item["Lolos"] for item in buy_checklist)
    sell_passed = sum(item["Lolos"] for item in sell_checklist)

    if buy_ready:
        status = "Kondisi BUY siap"
        interpretation = "Semua syarat BUY sudah terpenuhi. Jika jam trading aktif dan limit posisi belum penuh, posisi BUY dapat dibuka."
    elif sell_ready:
        status = "Kondisi SELL siap"
        interpretation = "Semua syarat SELL sudah terpenuhi. Jika jam trading aktif dan limit posisi belum penuh, posisi SELL dapat dibuka."
    else:
        status = "Belum ada sinyal valid"
        if buy_passed > sell_passed:
            interpretation = f"Kondisi lebih dekat ke BUY ({buy_passed}/{len(buy_checklist)} syarat), tetapi belum semua syarat terpenuhi."
        elif sell_passed > buy_passed:
            interpretation = f"Kondisi lebih dekat ke SELL ({sell_passed}/{len(sell_checklist)} syarat), tetapi belum semua syarat terpenuhi."
        else:
            interpretation = "BUY dan SELL sama-sama belum lengkap. Strategi masih menunggu arah yang lebih tegas."

    state = {
        "Status sinyal": status,
        "Yang ditunggu": waiting,
        "Kondisi BUY": " | ".join(buy_conditions),
        "Kondisi SELL": " | ".join(sell_conditions),
        "Checklist BUY": buy_checklist,
        "Checklist SELL": sell_checklist,
        "Interpretasi": interpretation,
        "Skor BUY": buy_passed,
        "Skor SELL": sell_passed,
        "Tanggal evaluasi": latest_date,
        "Harga terakhir": latest_close,
        "Momentum saat ini": latest_momentum,
        "Threshold": threshold,
        "MA cepat": latest_fast,
        "MA lambat": latest_slow,
        "RSI": latest_rsi,
        "High acuan": latest_previous_high,
        "Low acuan": latest_previous_low,
        "Sumber harga": "GC=F harian (candle selesai)",
        "Preview live": None,
    }
    if live_price is not None and np.isfinite(live_price):
        provisional = gold_ohlc.copy()
        preview_timestamp = pd.Timestamp(live_timestamp) if live_timestamp is not None else latest_date
        if preview_timestamp.tzinfo is not None:
            preview_timestamp = preview_timestamp.tz_convert(WIT).tz_localize(None)
        preview_date = preview_timestamp.normalize()
        if preview_date in provisional.index:
            provisional.loc[preview_date, "Close"] = float(live_price)
            provisional.loc[preview_date, "High"] = max(
                float(provisional.loc[preview_date, "High"]), float(live_price)
            )
            provisional.loc[preview_date, "Low"] = min(
                float(provisional.loc[preview_date, "Low"]), float(live_price)
            )
        else:
            provisional.loc[preview_date, ["Open", "High", "Low", "Close"]] = float(live_price)
            provisional = provisional.sort_index()
        preview = _signal_waiting_state(provisional, params)
        preview["Sumber harga"] = "MT5 live mid (provisional)"
        preview["Timestamp live"] = preview_timestamp
        state["Preview live"] = preview
    return state


def _unrealized(direction: str, entry_price: float, current_price: float, lot: float) -> float:
    units = lot * CONTRACT_OUNCES_PER_LOT
    if direction == "BUY":
        return (current_price - entry_price) * units
    return (entry_price - current_price) * units


def record_manual_exit(
    position_id: int,
    latest_price: float,
    now: pd.Timestamp | None = None,
    live_path: Path = LIVE_TRADING_PATH,
    manual_path: Path = LIVE_MANUAL_EXIT_PATH,
) -> tuple[pd.DataFrame, str, bool]:
    """Record a human exit decision without closing the optimizer position."""
    now_wit = _now_wit(now)
    ledger = load_live_ledger(live_path)
    manual_ledger = load_manual_exit_ledger(manual_path)

    if pd.isna(latest_price):
        return manual_ledger, "Harga terbaru belum tersedia, exit manual belum bisa dicatat.", False

    existing_manual = manual_ledger[pd.to_numeric(manual_ledger["position_id"], errors="coerce").eq(position_id)]
    if not existing_manual.empty:
        return manual_ledger, f"Exit manual untuk posisi #{position_id} sudah pernah dicatat.", False

    matches = ledger[pd.to_numeric(ledger["position_id"], errors="coerce").eq(position_id)]
    if matches.empty:
        return manual_ledger, f"Posisi Optimizer #{position_id} tidak ditemukan.", False

    row = matches.iloc[-1]
    if str(row.get("status", "")) != "OPEN":
        return manual_ledger, f"Posisi Optimizer #{position_id} sudah tidak terbuka.", False

    direction = str(row["arah"])
    lot = float(pd.to_numeric(row["lot"], errors="coerce"))
    entry_price = float(pd.to_numeric(row["entry_price"], errors="coerce"))
    manual_price = float(latest_price)
    gross_pl = _unrealized(direction, entry_price, manual_price, lot)
    swap_at_exit = float(pd.to_numeric(row.get("swap", 0.0), errors="coerce") or 0.0)
    net_pl = gross_pl + swap_at_exit
    if net_pl > 0:
        result_label = "TP Manual"
    elif net_pl < 0:
        result_label = "CL Manual"
    else:
        result_label = "BE Manual"

    if manual_ledger.empty:
        next_id = 1
    else:
        max_id = pd.to_numeric(manual_ledger["manual_exit_id"], errors="coerce").max()
        next_id = 1 if pd.isna(max_id) else int(max_id) + 1

    new_row = {
        "manual_exit_id": next_id,
        "position_id": int(position_id),
        "optimizer_signal_date": row.get("signal_date", ""),
        "optimizer_direction": direction,
        "optimizer_entry_time_wit": row.get("entry_time_wit", ""),
        "optimizer_entry_price": entry_price,
        "optimizer_tp_usd": float(pd.to_numeric(row.get("tp_usd", 0.0), errors="coerce") or 0.0),
        "optimizer_cl_usd": float(pd.to_numeric(row.get("cl_usd", 0.0), errors="coerce") or 0.0),
        "manual_exit_time_wit": now_wit.strftime("%Y-%m-%d %H:%M:%S WIT"),
        "manual_exit_price": manual_price,
        "manual_gross_pl": gross_pl,
        "manual_swap_at_exit": swap_at_exit,
        "manual_net_pl": net_pl,
        "manual_result_label": result_label,
        "last_update_wit": now_wit.strftime("%Y-%m-%d %H:%M:%S WIT"),
        "catatan": "Keputusan manusia dicatat paralel; posisi Optimizer tetap mengikuti aturan algoritma.",
    }
    new_frame = pd.DataFrame([new_row], columns=LIVE_MANUAL_EXIT_COLUMNS)
    if manual_ledger.empty:
        manual_ledger = new_frame
    else:
        manual_ledger = pd.concat([manual_ledger, new_frame], ignore_index=True)
    save_manual_exit_ledger(manual_ledger, manual_path)
    return manual_ledger, f"Exit manual posisi #{position_id} dicatat pada ${manual_price:,.2f}.", True


def _open_counts(ledger: pd.DataFrame) -> tuple[int, int]:
    open_rows = ledger[ledger["status"].eq("OPEN")]
    return int(open_rows["arah"].eq("BUY").sum()), int(open_rows["arah"].eq("SELL").sum())


def _apply_swap(ledger: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    if ledger.empty:
        return ledger
    today = now.date()
    for index, row in ledger[ledger["status"].eq("OPEN")].iterrows():
        last_swap_date = pd.to_datetime(row.get("last_swap_date"), errors="coerce")
        if pd.isna(last_swap_date):
            last_swap_date = pd.to_datetime(row.get("entry_time_wit"), errors="coerce")
        if pd.isna(last_swap_date):
            last_swap_date = now
        elapsed_days = max((pd.Timestamp(today) - pd.Timestamp(last_swap_date).normalize()).days, 0)
        if elapsed_days <= 0:
            continue
        swap_per_day = LIVE_BUY_SWAP_PER_001_LOT if row["arah"] == "BUY" else LIVE_SELL_SWAP_PER_001_LOT
        current_swap = float(pd.to_numeric(row.get("swap", 0.0), errors="coerce") or 0.0)
        ledger.loc[index, "swap"] = current_swap - (swap_per_day * elapsed_days * (float(row["lot"]) / 0.01))
        ledger.loc[index, "last_swap_date"] = pd.Timestamp(today).strftime("%Y-%m-%d")
    return ledger


def _close_hit_positions(ledger: pd.DataFrame, candle: pd.Series, now: pd.Timestamp) -> pd.DataFrame:
    if ledger.empty:
        return ledger
    high = float(candle["High"])
    low = float(candle["Low"])
    for index, row in ledger[ledger["status"].eq("OPEN")].iterrows():
        entry_time = pd.to_datetime(str(row.get("entry_time_wit", "")).replace(" WIT", ""), errors="coerce")
        candle_date = pd.Timestamp(candle.name).date() if getattr(candle, "name", None) is not None else None
        if pd.notna(entry_time) and candle_date is not None and entry_time.date() == candle_date:
            continue

        entry_price = float(row["entry_price"])
        lot = float(row["lot"])
        units = lot * CONTRACT_OUNCES_PER_LOT
        tp_points = float(row["tp_usd"]) / units
        cl_points = float(row["cl_usd"]) / units
        direction = str(row["arah"])

        if direction == "BUY":
            tp_price = entry_price + tp_points
            cl_price = entry_price - cl_points
            hit_cl = low <= cl_price
            hit_tp = high >= tp_price
        else:
            tp_price = entry_price - tp_points
            cl_price = entry_price + cl_points
            hit_cl = high >= cl_price
            hit_tp = low <= tp_price

        if not hit_cl and not hit_tp:
            continue

        exit_price = cl_price if hit_cl else tp_price
        exit_reason = "CL tersentuh" if hit_cl else "TP tersentuh"
        gross_pl = _unrealized(direction, entry_price, exit_price, lot)
        swap = float(pd.to_numeric(row.get("swap", 0.0), errors="coerce") or 0.0)
        ledger.loc[index, "status"] = "CLOSED"
        ledger.loc[index, "exit_time_wit"] = now.strftime("%Y-%m-%d %H:%M:%S WIT")
        ledger.loc[index, "exit_price"] = exit_price
        ledger.loc[index, "exit_reason"] = exit_reason
        ledger.loc[index, "gross_pl"] = gross_pl
        ledger.loc[index, "net_pl"] = gross_pl + swap
        ledger.loc[index, "last_update_wit"] = now.strftime("%Y-%m-%d %H:%M:%S WIT")
    return ledger


def _broker_quote_state(quote: pd.Series | None, now: pd.Timestamp) -> dict[str, object]:
    empty = {
        "configured": False,
        "fresh": False,
        "bid": np.nan,
        "ask": np.nan,
        "mid": np.nan,
        "timestamp": pd.NaT,
        "age_minutes": np.nan,
        "source": "GC=F harian",
    }
    if quote is None or len(quote) == 0:
        return empty
    try:
        bid = float(quote["bid"])
        ask = float(quote["ask"])
        market_timestamp = pd.Timestamp(quote["timestamp_utc"])
        timestamp = pd.Timestamp(quote.get("received_at_utc", market_timestamp))
    except (KeyError, TypeError, ValueError):
        return {**empty, "configured": True, "source": "Quote broker tidak valid"}
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    now_utc = now.tz_convert("UTC")
    age_minutes = (now_utc - timestamp).total_seconds() / 60
    valid = bid > 0 and ask >= bid
    clock_valid = bool(quote.get("clock_valid", True))
    return {
        "configured": True,
        "fresh": valid and clock_valid and -1 <= age_minutes <= 5,
        "bid": bid,
        "ask": ask,
        "mid": (bid + ask) / 2,
        "timestamp": timestamp,
        "market_timestamp": market_timestamp,
        "age_minutes": age_minutes,
        "source": str(quote.get("source", "MT5 broker")),
    }


def _close_hit_positions_quote(
    ledger: pd.DataFrame,
    bid: float,
    ask: float,
    now: pd.Timestamp,
) -> pd.DataFrame:
    if ledger.empty:
        return ledger
    for index, row in ledger[ledger["status"].eq("OPEN")].iterrows():
        entry_price = float(row["entry_price"])
        lot = float(row["lot"])
        units = lot * CONTRACT_OUNCES_PER_LOT
        tp_points = float(row["tp_usd"]) / units
        cl_points = float(row["cl_usd"]) / units
        direction = str(row["arah"])

        if direction == "BUY":
            executable_price = bid
            hit_tp = executable_price >= entry_price + tp_points
            hit_cl = executable_price <= entry_price - cl_points
        else:
            executable_price = ask
            hit_tp = executable_price <= entry_price - tp_points
            hit_cl = executable_price >= entry_price + cl_points
        if not hit_tp and not hit_cl:
            continue

        exit_reason = "CL tersentuh" if hit_cl else "TP tersentuh"
        gross_pl = _unrealized(direction, entry_price, executable_price, lot)
        swap = float(pd.to_numeric(row.get("swap", 0.0), errors="coerce") or 0.0)
        ledger.loc[index, "status"] = "CLOSED"
        ledger.loc[index, "exit_time_wit"] = now.strftime("%Y-%m-%d %H:%M:%S WIT")
        ledger.loc[index, "exit_price"] = executable_price
        ledger.loc[index, "exit_reason"] = exit_reason
        ledger.loc[index, "gross_pl"] = gross_pl
        ledger.loc[index, "net_pl"] = gross_pl + swap
        ledger.loc[index, "last_update_wit"] = now.strftime("%Y-%m-%d %H:%M:%S WIT")
    return ledger


def _maybe_open_position(
    ledger: pd.DataFrame,
    signal: dict[str, object] | None,
    params: dict[str, object],
    now: pd.Timestamp,
    can_trade: bool,
    session_note: str,
    broker_bid: float | None = None,
    broker_ask: float | None = None,
) -> pd.DataFrame:
    if signal is None:
        return ledger
    signal_date = pd.Timestamp(signal["signal_date"]).strftime("%Y-%m-%d")
    existing = ledger[
        ledger["signal_date"].astype(str).str.startswith(signal_date)
        & ledger["arah"].astype(str).eq(str(signal["arah"]))
    ]
    if not existing.empty:
        return ledger

    buy_count, sell_count = _open_counts(ledger)
    max_buy = int(params.get("Max BUY", LIVE_MAX_BUY))
    max_sell = int(params.get("Max SELL", LIVE_MAX_SELL))
    max_total = int(params.get("Max Total", max_buy + max_sell))
    open_total = buy_count + sell_count
    direction = str(signal["arah"])
    entry_eligible = bool(signal.get("entry_eligible", True))
    can_open = (
        can_trade
        and entry_eligible
        and direction in {"BUY", "SELL"}
        and open_total < max_total
        and ((direction == "BUY" and buy_count < max_buy) or (direction == "SELL" and sell_count < max_sell))
    )
    status = "OPEN" if can_open else str(signal.get("record_status", "SIGNAL"))
    source = str(signal.get("source", "Optimizer penuh"))
    note = (
        f"Posisi dibuka dari sinyal {source}: seluruh syarat strategi terpenuhi."
        if can_open
        else str(signal.get("event_note", f"Sinyal {source} terdeteksi, belum buka posisi: {session_note}"))
    )
    next_id = int(pd.to_numeric(ledger["position_id"], errors="coerce").max() + 1) if not ledger.empty else 1
    if pd.isna(next_id):
        next_id = 1

    entry_price = float(signal["reference_price"])
    if can_open and direction == "BUY" and broker_ask is not None:
        entry_price = float(broker_ask)
    elif can_open and direction == "SELL" and broker_bid is not None:
        entry_price = float(broker_bid)

    new_row = {
        "position_id": next_id,
        "signal_date": signal_date,
        "detected_at_wit": now.strftime("%Y-%m-%d %H:%M:%S WIT"),
        "status": status,
        "arah": direction,
        "lot": LIVE_LOT_SIZE,
        "prediction": float(signal["prediction"]),
        "reference_price": float(signal["reference_price"]),
        "expected_change_pct": float(signal["expected_change_pct"]),
        "threshold_entry_pct": float(params["Threshold entry (%)"]),
        "tp_usd": float(params["TP (USD)"]),
        "cl_usd": float(params["SL (USD)"]),
        "entry_time_wit": now.strftime("%Y-%m-%d %H:%M:%S WIT") if can_open else "",
        "entry_price": entry_price if can_open else np.nan,
        "last_swap_date": now.strftime("%Y-%m-%d") if can_open else "",
        "exit_time_wit": "",
        "exit_price": np.nan,
        "exit_reason": "",
        "gross_pl": 0.0,
        "swap": 0.0,
        "net_pl": 0.0,
        "last_update_wit": now.strftime("%Y-%m-%d %H:%M:%S WIT"),
        "catatan": note,
    }
    new_frame = pd.DataFrame([new_row], columns=LIVE_COLUMNS)
    if ledger.empty:
        return new_frame
    return pd.concat([ledger, new_frame], ignore_index=True)


def _optimizer_trigger_state(
    ledger: pd.DataFrame,
    signal: dict[str, object] | None,
    params: dict[str, object],
    can_trade: bool,
    session_note: str,
) -> dict[str, object]:
    threshold = float(params["Threshold entry (%)"])
    buy_count, sell_count = _open_counts(ledger)
    max_buy = int(params.get("Max BUY", LIVE_MAX_BUY))
    max_sell = int(params.get("Max SELL", LIVE_MAX_SELL))
    max_total = int(params.get("Max Total", max_buy + max_sell))
    remaining_buy = max(max_buy - buy_count, 0)
    remaining_sell = max(max_sell - sell_count, 0)
    remaining_total = max(max_total - buy_count - sell_count, 0)

    if signal is None:
        checklist = [
            {"Syarat": "Ada sinyal Optimizer terbaru", "Status": "Menunggu", "Detail": "Belum ada prediksi yang melewati threshold."},
            {"Syarat": "Jam trading aktif", "Status": "Lolos" if can_trade else "Menunggu", "Detail": session_note},
            {
                "Syarat": "Slot posisi tersedia",
                "Status": "Lolos" if (remaining_buy > 0 or remaining_sell > 0) else "Menunggu",
                "Detail": f"Sisa slot BUY {remaining_buy}, SELL {remaining_sell}.",
            },
        ]
        return {
            "Status trigger": "Menunggu sinyal Optimizer",
            "Arah sinyal": "-",
            "Tanggal sinyal": pd.NaT,
            "Prediksi": np.nan,
            "Harga referensi": np.nan,
            "Expected change (%)": np.nan,
            "Threshold entry (%)": threshold,
            "Max BUY": max_buy,
            "Max SELL": max_sell,
            "Posisi BUY terbuka": buy_count,
            "Posisi SELL terbuka": sell_count,
            "Sisa slot BUY": remaining_buy,
            "Sisa slot SELL": remaining_sell,
            "Sudah dieksekusi": False,
            "Catatan": "Strategi menunggu candle harian yang memenuhi pola Optimizer.",
            "Checklist": checklist,
        }

    direction = str(signal["arah"])
    signal_date = pd.Timestamp(signal["signal_date"]).strftime("%Y-%m-%d")
    expected_change_pct = float(signal["expected_change_pct"])
    executed_rows = ledger[
        ledger["signal_date"].astype(str).str.startswith(signal_date)
        & ledger["arah"].astype(str).eq(direction)
    ]
    already_executed = not executed_rows.empty
    direction_slot = remaining_buy if direction == "BUY" else remaining_sell if direction == "SELL" else 0
    threshold_ok = (
        (direction == "BUY" and expected_change_pct >= threshold)
        or (direction == "SELL" and expected_change_pct <= -threshold)
    )
    slot_ok = direction in {"BUY", "SELL"} and direction_slot > 0 and remaining_total > 0
    entry_eligible = bool(signal.get("entry_eligible", True))
    can_open_now = direction in {"BUY", "SELL"} and threshold_ok and can_trade and slot_ok and not already_executed

    if not entry_eligible:
        status = str(signal.get("record_status", "Sinyal dibatalkan"))
        note = str(signal.get("event_note", "Sinyal tidak lolos validasi entry."))
    elif can_open_now:
        status = f"Siap buka {direction}"
        note = "Semua syarat eksekusi terpenuhi."
    elif already_executed:
        status = "Sinyal sudah dieksekusi/dicatat"
        note = f"Sinyal {direction} untuk {signal_date} sudah ada di ledger, sehingga tidak dibuka ulang."
    elif direction not in {"BUY", "SELL"} or not threshold_ok:
        status = "Menunggu threshold arah"
        note = f"Expected change {expected_change_pct:+.2f}% belum melewati threshold {threshold:.2f}%."
    elif not can_trade:
        status = "Menunggu jam trading"
        note = session_note
    elif not slot_ok:
        status = f"Slot {direction} penuh"
        note = f"Jumlah posisi {direction} sudah mencapai batas strategi."
    else:
        status = "Menunggu konfirmasi eksekusi"
        note = "Ada syarat eksekusi yang belum terpenuhi."

    checklist = [
        {
            "Syarat": "Ada sinyal Optimizer terbaru",
            "Status": "Lolos",
            "Detail": f"{direction} dari tanggal sinyal {signal_date}.",
        },
        {
            "Syarat": "Expected change melewati threshold",
            "Status": "Lolos" if threshold_ok else "Menunggu",
            "Detail": f"{expected_change_pct:+.2f}% vs threshold +/-{threshold:.2f}%.",
        },
        {"Syarat": "Jam trading aktif", "Status": "Lolos" if can_trade else "Menunggu", "Detail": session_note},
        {
            "Syarat": f"Slot {direction} tersedia",
            "Status": "Lolos" if slot_ok else "Menunggu",
            "Detail": f"Sisa slot BUY {remaining_buy}, SELL {remaining_sell}.",
        },
        {
            "Syarat": "Batas total posisi tersedia",
            "Status": "Lolos" if remaining_total > 0 else "Menunggu",
            "Detail": f"Sisa slot total {remaining_total} dari maksimum {max_total}.",
        },
        {
            "Syarat": "Tanggal dan arah sinyal belum pernah dicatat",
            "Status": "Lolos" if not already_executed else "Sudah tercatat",
            "Detail": "Satu sinyal tanggal/arah hanya dicatat sekali agar tidak over-entry.",
        },
    ]

    return {
        "Status trigger": status,
        "Arah sinyal": direction,
        "Tanggal sinyal": pd.Timestamp(signal["signal_date"]),
        "Prediksi": float(signal["prediction"]),
        "Harga referensi": float(signal["reference_price"]),
        "Expected change (%)": expected_change_pct,
        "Threshold entry (%)": threshold,
        "Max BUY": max_buy,
        "Max SELL": max_sell,
        "Posisi BUY terbuka": buy_count,
        "Posisi SELL terbuka": sell_count,
        "Sisa slot BUY": remaining_buy,
        "Sisa slot SELL": remaining_sell,
        "Sisa slot total": remaining_total,
        "Sudah dieksekusi": already_executed,
        "Catatan": note,
        "Checklist": checklist,
    }


def _start_time_wit(start_date: pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(start_date)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize(WIT)
    return timestamp.tz_convert(WIT)


def _prepare_live_broker_m1(broker_bars: pd.DataFrame | None) -> pd.DataFrame:
    if broker_bars is None or broker_bars.empty:
        return pd.DataFrame()
    frame = broker_bars.copy()
    required = {"timestamp_utc", "open", "high", "low", "close", "spread_points"}
    if not required.issubset(frame.columns):
        return pd.DataFrame()
    timestamps = pd.to_datetime(frame["timestamp_utc"], errors="coerce", utc=True)
    frame = frame.assign(timestamp_utc=timestamps).dropna(subset=["timestamp_utc"])
    frame = frame.set_index("timestamp_utc").sort_index()
    frame.index = frame.index.tz_convert("UTC").tz_localize(None)
    frame = frame.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "spread_points": "SpreadPoints",
        }
    )
    numeric = ["Open", "High", "Low", "Close", "SpreadPoints"]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    return frame[numeric].dropna().loc[~frame.index.duplicated(keep="last")]


def _fixed_delay_live_signal(
    gold_ohlc: pd.DataFrame,
    broker_bars: pd.DataFrame | None,
    params: dict[str, object],
    now: pd.Timestamp,
    start_date: pd.Timestamp,
) -> tuple[dict[str, object] | None, dict[str, object]]:
    empty_state = {
        "Status": "MENUNGGU DATA M1",
        "Detail": "Candle M1 MT5 belum tersedia untuk validasi Fixed Delay.",
        "Waktu sinyal awal": pd.NaT,
        "Waktu konfirmasi": pd.NaT,
        "Spread points": np.nan,
        "Barrier tersentuh": False,
    }
    data = _prepare_live_broker_m1(broker_bars)
    if data.empty or gold_ohlc.empty:
        return None, empty_state

    now_utc = now.tz_convert("UTC").tz_localize(None)
    activation_utc = _start_time_wit(start_date).tz_convert("UTC").tz_localize(None)
    data = data.loc[data.index <= now_utc]
    if data.empty:
        return None, empty_state

    raw = _entry_signals_for_period(
        data,
        gold_ohlc,
        params,
        activation_utc.normalize(),
        now_utc,
    )
    if raw.empty:
        return None, {
            **empty_state,
            "Status": "MENUNGGU SINYAL HARIAN",
            "Detail": "Belum ada sinyal v1 baru setelah aktivasi Fixed Delay.",
        }
    raw = raw.loc[raw.index >= activation_utc]
    if raw.empty:
        return None, {
            **empty_state,
            "Status": "MENUNGGU SINYAL HARIAN",
            "Detail": "Belum ada sinyal v1 baru setelah aktivasi Fixed Delay.",
        }

    features = _entry_features(data)
    balanced_config = SignalQualityConfig(
        "Balanced Entry Frozen",
        "Trend engine",
        conviction_multiplier=1.05,
        require_h1_trend=True,
        wait_hours=2,
    )
    balanced, audit = _select_signals(
        raw,
        features,
        params,
        balanced_config,
        FIXED_DELAY_SPREAD_LIMIT_POINTS,
        now_utc,
    )
    if balanced.empty:
        last_audit = audit.iloc[-1] if not audit.empty else pd.Series(dtype=object)
        return None, {
            **empty_state,
            "Status": "MENUNGGU BALANCED ENTRY",
            "Detail": str(last_audit.get("Alasan", "Alignment H1 atau conviction belum lolos.")),
            "Waktu sinyal awal": last_audit.get("Waktu sinyal awal", pd.NaT),
        }

    signal_time = pd.Timestamp(balanced.index[-1])
    signal = balanced.iloc[-1]
    confirmation_due = signal_time + pd.Timedelta(minutes=FIXED_DELAY_MINUTES)
    location = data.index.searchsorted(confirmation_due, side="left")
    if location >= len(data.index):
        return None, {
            **empty_state,
            "Status": "TUNGGU 5 MENIT",
            "Detail": "Balanced Entry lolos; menunggu candle konfirmasi lima menit.",
            "Waktu sinyal awal": signal_time,
            "Waktu konfirmasi": confirmation_due,
        }

    confirmation_time = pd.Timestamp(data.index[location])
    if confirmation_time > confirmation_due + pd.Timedelta(minutes=5):
        return None, {
            **empty_state,
            "Status": "BATAL DATA M1",
            "Detail": "Candle konfirmasi tidak tersedia dalam toleransi lima menit.",
            "Waktu sinyal awal": signal_time,
            "Waktu konfirmasi": confirmation_due,
        }

    expected = float(signal["expected_change_pct"])
    direction = "BUY" if expected > 0 else "SELL"
    reference = float(data.loc[signal_time, "Close"])
    window = data.loc[(data.index > signal_time) & (data.index <= confirmation_time)]
    units = LIVE_LOT_SIZE * CONTRACT_OUNCES_PER_LOT
    if direction == "BUY":
        adverse = max((reference - float(window["Low"].min())) * units, 0.0)
        favorable = max((float(window["High"].max()) - reference) * units, 0.0)
    else:
        adverse = max((float(window["High"].max()) - reference) * units, 0.0)
        favorable = max((reference - float(window["Low"].min())) * units, 0.0)
    barrier_hit = adverse >= FIXED_DELAY_SL_USD or favorable >= FIXED_DELAY_TP_USD
    spread_points = float(data.loc[confirmation_time, "SpreadPoints"])
    spread_ok = spread_points <= FIXED_DELAY_SPREAD_LIMIT_POINTS
    accepted = not barrier_hit and spread_ok
    state = {
        "Status": "ENTRY" if accepted else "BATAL BARRIER" if barrier_hit else "BATAL SPREAD",
        "Detail": (
            "Delay lima menit selesai; seluruh validasi entry lolos."
            if accepted
            else "TP/SL awal telah tersentuh selama masa tunggu."
            if barrier_hit
            else f"Spread {spread_points:.1f} points melebihi batas 20 points."
        ),
        "Waktu sinyal awal": signal_time,
        "Waktu konfirmasi": confirmation_time,
        "Spread points": spread_points,
        "Barrier tersentuh": barrier_hit,
        "Observed adverse USD": adverse,
        "Observed favorable USD": favorable,
    }
    output = {
        "signal_date": pd.Timestamp(signal["signal_date"]),
        "prediction": float(signal["prediction"]),
        "reference_price": float(data.loc[confirmation_time, "Close"]),
        "expected_change_pct": expected,
        "arah": direction,
        "source": "Fixed Delay 5m",
    }
    if not accepted:
        output.update(
            {
                "entry_eligible": False,
                "record_status": "CANCELLED",
                "event_note": state["Detail"],
            }
        )
    return output, state


def run_live_trading_update(
    gold_ohlc: pd.DataFrame,
    optimizer_leaderboard: pd.DataFrame,
    now: pd.Timestamp | None = None,
    path: Path = LIVE_TRADING_PATH,
    start_date: pd.Timestamp = LIVE_START_DATE,
    broker_quote: pd.Series | None = None,
    allow_new_entries: bool = True,
    entry_strategy: str = "baseline",
    broker_bars: pd.DataFrame | None = None,
) -> dict[str, object]:
    now_wit = _now_wit(now)
    cutoff_date = now_wit.tz_localize(None).normalize()
    usable_gold = gold_ohlc[gold_ohlc.index <= cutoff_date].copy()
    ledger = load_live_ledger(path)
    params = _best_optimizer_params(optimizer_leaderboard)
    can_trade, session_note = _is_live_session_open(now_wit)
    start_time_wit = _start_time_wit(start_date)
    if now_wit < start_time_wit:
        can_trade = False
        session_note = f"Paper live trading strategi ini baru dimulai {start_time_wit.strftime('%d %b %Y %H:%M WIT')}."
    if entry_strategy == "fixed_delay_5m":
        params.update(
            {
                "Strategi": "Fixed Delay 5m",
                "Lot": LIVE_LOT_SIZE,
                "TP (USD)": FIXED_DELAY_TP_USD,
                "SL (USD)": FIXED_DELAY_SL_USD,
                "Max BUY": 1,
                "Max SELL": 1,
                "Max Total": 1,
            }
        )

    daily_data_date = usable_gold.index.max().normalize() if not usable_gold.empty else pd.NaT
    expected_anchor = cutoff_date if now_wit.hour >= 7 else cutoff_date - pd.Timedelta(days=1)
    expected_daily_date = (expected_anchor - pd.offsets.BDay(1)).normalize()
    daily_data_stale = pd.isna(daily_data_date) or daily_data_date < expected_daily_date
    if daily_data_stale:
        can_trade = False
        available_label = (
            "tidak tersedia"
            if pd.isna(daily_data_date)
            else pd.Timestamp(daily_data_date).strftime("%d %b %Y")
        )
        session_note = (
            f"Data harian GC=F stale ({available_label}); entry baru ditahan sampai candle "
            f"{expected_daily_date.strftime('%d %b %Y')} tersedia."
        )

    quote_state = _broker_quote_state(broker_quote, now_wit)
    if quote_state["configured"] and not quote_state["fresh"]:
        can_trade = False
        age = quote_state["age_minutes"]
        age_label = "tidak diketahui" if pd.isna(age) else f"{float(age):.1f} menit"
        session_note = f"Quote broker stale/tidak valid (usia {age_label}); entry dan exit otomatis ditahan."

    ledger = _apply_swap(ledger, now_wit)
    if not usable_gold.empty:
        latest_candle = usable_gold.iloc[-1]
        if quote_state["fresh"]:
            latest_price = float(quote_state["mid"])
            ledger = _close_hit_positions_quote(
                ledger,
                float(quote_state["bid"]),
                float(quote_state["ask"]),
                now_wit,
            )
        else:
            latest_price = float(quote_state["mid"]) if quote_state["configured"] else float(latest_candle["Close"])
            if not quote_state["configured"]:
                ledger = _close_hit_positions(ledger, latest_candle, now_wit)
    else:
        latest_price = float(quote_state["mid"]) if quote_state["configured"] else np.nan

    waiting_state = _signal_waiting_state(
        usable_gold,
        params,
        live_price=float(quote_state["mid"]) if quote_state["fresh"] else None,
        live_timestamp=quote_state.get("market_timestamp") if quote_state["fresh"] else None,
    )
    fixed_delay_state = None
    if entry_strategy == "fixed_delay_5m":
        signal, fixed_delay_state = _fixed_delay_live_signal(
            usable_gold,
            broker_bars,
            params,
            now_wit,
            start_time_wit,
        )
    else:
        signal = _current_optimizer_signal(
            usable_gold,
            params,
            now_wit,
            start_time_wit.tz_localize(None).normalize(),
        )
        if signal is not None:
            signal["source"] = "Optimizer penuh"
    entry_allowed = can_trade and allow_new_entries
    archive_note = session_note if allow_new_entries else "Strategi diarsipkan; posisi baru dinonaktifkan."
    if (
        entry_strategy == "fixed_delay_5m"
        and signal is not None
        and bool(signal.get("entry_eligible", True))
        and not entry_allowed
    ):
        signal["entry_eligible"] = False
        signal["record_status"] = "CANCELLED"
        signal["event_note"] = f"Entry Fixed Delay dibatalkan: {archive_note}"
        if fixed_delay_state is not None:
            fixed_delay_state["Status"] = "BATAL EKSEKUSI"
            fixed_delay_state["Detail"] = signal["event_note"]
    ledger = _maybe_open_position(
        ledger,
        signal,
        params,
        now_wit,
        entry_allowed,
        archive_note,
        broker_bid=float(quote_state["bid"]) if quote_state["fresh"] else None,
        broker_ask=float(quote_state["ask"]) if quote_state["fresh"] else None,
    )
    trigger_state = _optimizer_trigger_state(ledger, signal, params, entry_allowed, archive_note)
    save_live_ledger(ledger, path)

    open_positions = ledger[ledger["status"].eq("OPEN")].copy()
    closed_positions = ledger[ledger["status"].eq("CLOSED")].copy()
    signal_rows = ledger[ledger["status"].isin(["SIGNAL", "OPEN", "CANCELLED"])].copy()

    if open_positions.empty or pd.isna(latest_price):
        floating_pl = 0.0
    else:
        floating_pl = 0.0
        for _, row in open_positions.iterrows():
            direction = str(row["arah"])
            executable_price = latest_price
            if quote_state["fresh"]:
                executable_price = float(quote_state["bid"] if direction == "BUY" else quote_state["ask"])
            floating_pl += _unrealized(direction, float(row["entry_price"]), executable_price, float(row["lot"]))

    closed_net = float(pd.to_numeric(closed_positions["net_pl"], errors="coerce").fillna(0.0).sum()) if not closed_positions.empty else 0.0
    open_swap = float(pd.to_numeric(open_positions["swap"], errors="coerce").fillna(0.0).sum()) if not open_positions.empty else 0.0
    balance = LIVE_INITIAL_EQUITY + closed_net + open_swap
    equity = balance + floating_pl
    open_buy, open_sell = _open_counts(ledger)

    summary = {
        "Equity": equity,
        "Balance": balance,
        "Floating P/L": floating_pl,
        "Closed net P/L": closed_net,
        "Open swap": open_swap,
        "Open BUY": open_buy,
        "Open SELL": open_sell,
        "Latest price": latest_price,
        "Latest bid": quote_state["bid"],
        "Latest ask": quote_state["ask"],
        "Price source": quote_state["source"],
        "Broker quote fresh": quote_state["fresh"],
        "Broker quote age minutes": quote_state["age_minutes"],
        "Latest data date": quote_state["timestamp"] if quote_state["configured"] else (usable_gold.index.max() if not usable_gold.empty else pd.NaT),
        "Daily data date": daily_data_date,
        "Expected daily data date": expected_daily_date,
        "Daily data stale": daily_data_stale,
        "Can trade": entry_allowed,
        "Session note": archive_note,
        "Now WIT": now_wit,
        "Ledger start date": start_date,
    }
    return {
        "summary": summary,
        "params": params,
        "signal": signal,
        "waiting_state": waiting_state,
        "trigger_state": trigger_state,
        "fixed_delay_state": fixed_delay_state,
        "ledger": ledger,
        "signals": signal_rows,
        "open_positions": open_positions,
        "closed_positions": closed_positions,
    }
