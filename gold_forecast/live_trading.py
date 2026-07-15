from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from gold_forecast.monitoring import WIT
from gold_forecast.simulation import CONTRACT_OUNCES_PER_LOT
from gold_forecast.strategy_optimizer import _indicator_predictions


LIVE_TRADING_PATH = Path("data/live_trading_optimizer.csv")
LIVE_INITIAL_EQUITY = 1000.0
LIVE_START_DATE = pd.Timestamp("2026-07-15")
LIVE_LOT_SIZE = 0.01
LIVE_BUY_SWAP_PER_001_LOT = 0.02
LIVE_SELL_SWAP_PER_001_LOT = 0.0
LIVE_MAX_BUY = 8
LIVE_MAX_SELL = 10

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


def _empty_ledger() -> pd.DataFrame:
    return pd.DataFrame(columns=LIVE_COLUMNS)


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
            "Strategi": "Fallback Optimizer",
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
    }


def _current_optimizer_signal(
    gold_ohlc: pd.DataFrame,
    params: dict[str, object],
    now: pd.Timestamp,
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
    signals = signals[(signals.index >= LIVE_START_DATE) & (signals.index <= now.tz_localize(None).normalize())]
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


def _unrealized(direction: str, entry_price: float, current_price: float, lot: float) -> float:
    units = lot * CONTRACT_OUNCES_PER_LOT
    if direction == "BUY":
        return (current_price - entry_price) * units
    return (entry_price - current_price) * units


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


def _maybe_open_position(
    ledger: pd.DataFrame,
    signal: dict[str, object] | None,
    params: dict[str, object],
    now: pd.Timestamp,
    can_trade: bool,
    session_note: str,
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
    direction = str(signal["arah"])
    can_open = (
        can_trade
        and direction in {"BUY", "SELL"}
        and ((direction == "BUY" and buy_count < LIVE_MAX_BUY) or (direction == "SELL" and sell_count < LIVE_MAX_SELL))
    )
    status = "OPEN" if can_open else "SIGNAL"
    note = "Posisi dibuka dari sinyal optimizer." if can_open else f"Sinyal terdeteksi, belum buka posisi: {session_note}"
    next_id = int(pd.to_numeric(ledger["position_id"], errors="coerce").max() + 1) if not ledger.empty else 1
    if pd.isna(next_id):
        next_id = 1

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
        "entry_price": float(signal["reference_price"]) if can_open else np.nan,
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


def run_live_trading_update(
    gold_ohlc: pd.DataFrame,
    optimizer_leaderboard: pd.DataFrame,
    now: pd.Timestamp | None = None,
    path: Path = LIVE_TRADING_PATH,
) -> dict[str, object]:
    now_wit = _now_wit(now)
    cutoff_date = now_wit.tz_localize(None).normalize()
    usable_gold = gold_ohlc[gold_ohlc.index <= cutoff_date].copy()
    ledger = load_live_ledger(path)
    params = _best_optimizer_params(optimizer_leaderboard)
    can_trade, session_note = _is_live_session_open(now_wit)

    if not usable_gold.empty:
        latest_candle = usable_gold.iloc[-1]
        latest_price = float(latest_candle["Close"])
        ledger = _apply_swap(ledger, now_wit)
        ledger = _close_hit_positions(ledger, latest_candle, now_wit)
    else:
        latest_price = np.nan

    signal = _current_optimizer_signal(usable_gold, params, now_wit)
    ledger = _maybe_open_position(ledger, signal, params, now_wit, can_trade, session_note)
    save_live_ledger(ledger, path)

    open_positions = ledger[ledger["status"].eq("OPEN")].copy()
    closed_positions = ledger[ledger["status"].eq("CLOSED")].copy()
    signal_rows = ledger[ledger["status"].isin(["SIGNAL", "OPEN"])].copy()

    if open_positions.empty or pd.isna(latest_price):
        floating_pl = 0.0
    else:
        floating_pl = 0.0
        for _, row in open_positions.iterrows():
            floating_pl += _unrealized(str(row["arah"]), float(row["entry_price"]), latest_price, float(row["lot"]))

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
        "Latest data date": usable_gold.index.max() if not usable_gold.empty else pd.NaT,
        "Can trade": can_trade,
        "Session note": session_note,
        "Now WIT": now_wit,
    }
    return {
        "summary": summary,
        "params": params,
        "signal": signal,
        "ledger": ledger,
        "signals": signal_rows,
        "open_positions": open_positions,
        "closed_positions": closed_positions,
    }
