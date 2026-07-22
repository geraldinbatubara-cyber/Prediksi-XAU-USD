from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from gold_forecast.simulation import CONTRACT_OUNCES_PER_LOT
from gold_forecast.strategy_optimizer import (
    BUY_SWAP_PER_001_LOT,
    INITIAL_EQUITY,
    MultiPhaseSimulationResult,
    _fixed_lot_signals,
    _indicator_predictions,
)


OOS_START = pd.Timestamp("2026-01-01")
OOS_END = pd.Timestamp("2026-06-30 23:59:59")
POINT_SIZE = 0.01
SLIPPAGE_POINTS = 2.0


@dataclass
class BrokerPosition:
    position_id: int
    phase: int
    signal_date: pd.Timestamp
    entry_time: pd.Timestamp
    direction: str
    lot: float
    entry_price: float
    prediction: float
    expected_change_pct: float
    take_profit_usd: float
    stop_loss_usd: float
    protection_activation_usd: float | None
    protection_floor_usd: float | None
    protection_trail_usd: float | None
    max_positions: int
    entry_spread_cost: float
    peak_profit_usd: float = 0.0
    swap_paid: float = 0.0


def run_exact_broker_oos(
    gold_m1: pd.DataFrame,
    signal_daily: pd.DataFrame,
    frozen_payload: dict[str, object],
) -> dict[str, tuple[MultiPhaseSimulationResult, pd.DataFrame, object]]:
    data = _prepare_m1(gold_m1)
    output = {}
    for variant in ("v1", "v10"):
        _, leaderboard, daily_oos = frozen_payload[variant]
        best = leaderboard.iloc[0].to_dict()
        signals = _entry_signals(data, signal_daily, best)
        result = _simulate_exact(data.loc[(data.index >= OOS_START) & (data.index <= OOS_END)], signals, best, variant)
        result.summary.update(
            {
                "Periode train": "01 Jan 2025 - 31 Des 2025 (parameter harian)",
                "Periode test": "01 Jan 2026 - 30 Jun 2026",
                "Candle M1 OOS": float(len(data.loc[(data.index >= OOS_START) & (data.index <= OOS_END)])),
                "Spread historis": True,
                "Slippage points per side": SLIPPAGE_POINTS,
                "Point size": POINT_SIZE,
                "Sumber sinyal": "Sama dengan Optimizer OOS harian (GC=F)",
                "Eksekusi sinyal": "Candle M1 pertama setelah candle harian selesai",
                "Daily OOS equity": daily_oos.summary["Equity akhir"],
                "Daily OOS growth (%)": daily_oos.summary["Growth total"],
                "Daily OOS max drawdown": daily_oos.summary["Max drawdown"],
                "Daily OOS transaksi": daily_oos.summary["Jumlah transaksi"],
                "Parameter dibekukan": True,
            }
        )
        output[variant] = (_compact_curve(result), pd.DataFrame([best]), daily_oos)
    return output


def _prepare_m1(data: pd.DataFrame) -> pd.DataFrame:
    required = {"Open", "High", "Low", "Close", "SpreadPoints"}
    if data.empty or not required.issubset(data.columns):
        raise ValueError("Dataset M1 tidak memiliki OHLC dan SpreadPoints yang lengkap.")
    clean = data.sort_index().copy()
    clean = clean.loc[~clean.index.duplicated(keep="last")]
    for column in required:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
    return clean.dropna(subset=list(required))


def _entry_signals(data: pd.DataFrame, daily: pd.DataFrame, best: dict[str, object]) -> pd.DataFrame:
    predictions = _indicator_predictions(
        daily,
        str(best["Mode"]),
        int(best["Fast MA"]),
        int(best["Slow MA"]),
        int(best["Momentum hari"]),
        float(best["Threshold entry (%)"]),
        test_start=OOS_START,
        test_end=OOS_END,
    )
    daily_signals = _fixed_lot_signals(predictions, float(best["Lot"]))
    rows = []
    daily_groups = {day: group.index[-1] for day, group in data.groupby(data.index.normalize())}
    for signal_date, signal in daily_signals.iterrows():
        last_bar = daily_groups.get(pd.Timestamp(signal_date).normalize())
        if last_bar is None:
            continue
        location = data.index.searchsorted(last_bar, side="right")
        if location >= len(data.index):
            continue
        entry_time = data.index[location]
        if entry_time > OOS_END:
            continue
        daily_close = float(daily.loc[signal_date, "Close"])
        prediction = float(signal["prediction"])
        rows.append(
            {
                "entry_time": entry_time,
                "signal_date": pd.Timestamp(signal_date),
                "prediction": prediction,
                "expected_change_pct": (prediction / daily_close - 1) * 100,
                "lot": float(signal["lot_size"]),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["signal_date", "prediction", "expected_change_pct", "lot"])
    return pd.DataFrame(rows).set_index("entry_time").sort_index()


def _simulate_exact(
    data: pd.DataFrame,
    signals: pd.DataFrame,
    best: dict[str, object],
    variant: str,
    *,
    spread_multiplier: float = 1.0,
    slippage_points: float = SLIPPAGE_POINTS,
) -> MultiPhaseSimulationResult:
    take_profit = float(best["TP (USD)"])
    stop_loss = float(best["SL (USD)"])
    threshold = float(best["Threshold entry (%)"])
    max_buy = int(best.get("Max BUY", 8))
    max_sell = int(best.get("Max SELL", 10))
    risk_cap = _optional_float(best.get("Risk cap floating SL (%)"))
    protection_activation = _optional_float(best.get("Profit protection aktif (USD)"))
    protection_floor = _optional_float(best.get("Profit protection floor (USD)"))
    protection_trail = _optional_float(best.get("Profit protection trail (USD)"))
    close_on_target = bool(best.get("Close-all target equity", True))
    phase_growth = float(best.get("Target fase (%)", 20.0)) / 100

    balance = INITIAL_EQUITY
    phase_start = INITIAL_EQUITY
    target_equity = phase_start * (1 + phase_growth)
    phase = 1
    next_id = 1
    positions: list[BrokerPosition] = []
    trades: list[dict[str, object]] = []
    curve: list[dict[str, object]] = []
    phase_rows: list[dict[str, object]] = []
    phase_curve_start = 0
    phase_trade_start = 0
    previous_trading_day: pd.Timestamp | None = None

    for timestamp, candle in data.iterrows():
        trading_day = pd.Timestamp(timestamp).normalize()
        if previous_trading_day is not None and trading_day != previous_trading_day:
            for position in positions:
                if position.direction == "BUY":
                    swap = BUY_SWAP_PER_001_LOT * (position.lot / 0.01)
                    position.swap_paid += swap
                    balance -= swap
        previous_trading_day = trading_day

        spread = max(0.0, float(candle["SpreadPoints"]) * POINT_SIZE * spread_multiplier)
        bid_high, bid_low, bid_close = float(candle["High"]), float(candle["Low"]), float(candle["Close"])
        ask_high, ask_low, ask_close = bid_high + spread, bid_low + spread, bid_close + spread
        still_open = []
        for position in positions:
            exit_detail = _exit_decision(position, bid_high, bid_low, ask_high, ask_low)
            if exit_detail is None:
                still_open.append(position)
                continue
            raw_exit, reason = exit_detail
            exit_price = raw_exit - POINT_SIZE * slippage_points if position.direction == "BUY" else raw_exit + POINT_SIZE * slippage_points
            balance += _pnl(position, exit_price)
            trades.append(_trade_row(position, timestamp, exit_price, reason, balance, spread, slippage_points))
        positions = still_open

        if timestamp in signals.index:
            signal = signals.loc[timestamp]
            if isinstance(signal, pd.DataFrame):
                signal = signal.iloc[-1]
            expected = float(signal["expected_change_pct"])
            direction = "BUY" if expected >= threshold else "SELL" if expected <= -threshold else None
            if direction is not None:
                direction_count = sum(position.direction == direction for position in positions)
                max_positions = max_buy if direction == "BUY" else max_sell
                can_open = direction_count < max_positions
                if can_open and risk_cap is not None:
                    equity = balance + sum(_mark_pnl(position, bid_close, ask_close) for position in positions)
                    open_risk = sum(position.stop_loss_usd for position in positions)
                    can_open = open_risk + stop_loss <= equity * risk_cap / 100
                if can_open:
                    lot = float(signal["lot"])
                    units = lot * CONTRACT_OUNCES_PER_LOT
                    if direction == "BUY":
                        entry_price = ask_close + POINT_SIZE * slippage_points
                        spread_cost = spread * units
                    else:
                        entry_price = bid_close - POINT_SIZE * slippage_points
                        spread_cost = 0.0
                    positions.append(
                        BrokerPosition(
                            position_id=next_id,
                            phase=phase,
                            signal_date=pd.Timestamp(signal["signal_date"]),
                            entry_time=timestamp,
                            direction=direction,
                            lot=lot,
                            entry_price=entry_price,
                            prediction=float(signal["prediction"]),
                            expected_change_pct=expected,
                            take_profit_usd=take_profit,
                            stop_loss_usd=stop_loss,
                            protection_activation_usd=protection_activation,
                            protection_floor_usd=protection_floor,
                            protection_trail_usd=protection_trail,
                            max_positions=max_positions,
                            entry_spread_cost=spread_cost,
                        )
                    )
                    next_id += 1

        unrealized = sum(_mark_pnl(position, bid_close, ask_close) for position in positions)
        equity = balance + unrealized
        curve.append(
            {
                "Tanggal": timestamp,
                "Fase": phase,
                "Balance": balance,
                "Equity": equity,
                "Unrealized P/L": unrealized,
                "Open BUY": sum(position.direction == "BUY" for position in positions),
                "Open SELL": sum(position.direction == "SELL" for position in positions),
                "Open total": len(positions),
                "Target equity tercapai": equity >= target_equity,
            }
        )
        if close_on_target and equity >= target_equity:
            for position in positions:
                raw_exit = bid_close if position.direction == "BUY" else ask_close
                exit_price = raw_exit - POINT_SIZE * slippage_points if position.direction == "BUY" else raw_exit + POINT_SIZE * slippage_points
                balance += _pnl(position, exit_price)
                trades.append(_trade_row(position, timestamp, exit_price, "Target equity tercapai", balance, spread, slippage_points))
            positions = []
            curve[-1].update({"Balance": balance, "Equity": balance, "Unrealized P/L": 0.0, "Open BUY": 0, "Open SELL": 0, "Open total": 0})
            phase_rows.append(_phase_summary(phase, phase_start, target_equity, trades[phase_trade_start:], curve[phase_curve_start:], True, timestamp))
            phase += 1
            phase_start = balance
            target_equity = phase_start * (1 + phase_growth)
            phase_trade_start = len(trades)
            phase_curve_start = len(curve)

    if positions:
        timestamp = data.index[-1]
        spread = max(0.0, float(data.iloc[-1]["SpreadPoints"]) * POINT_SIZE * spread_multiplier)
        bid_close = float(data.iloc[-1]["Close"])
        ask_close = bid_close + spread
        for position in positions:
            raw_exit = bid_close if position.direction == "BUY" else ask_close
            exit_price = raw_exit - POINT_SIZE * slippage_points if position.direction == "BUY" else raw_exit + POINT_SIZE * slippage_points
            balance += _pnl(position, exit_price)
            trades.append(_trade_row(position, timestamp, exit_price, "Akhir periode data", balance, spread, slippage_points))
        positions = []
        if curve:
            curve[-1].update({"Balance": balance, "Equity": balance, "Unrealized P/L": 0.0, "Open BUY": 0, "Open SELL": 0, "Open total": 0})

    if phase_curve_start < len(curve):
        phase_rows.append(_phase_summary(phase, phase_start, target_equity, trades[phase_trade_start:], curve[phase_curve_start:], False, pd.NaT))
    trades_frame = pd.DataFrame(trades)
    curve_frame = pd.DataFrame(curve).set_index("Tanggal") if curve else pd.DataFrame()
    phases_frame = pd.DataFrame(phase_rows)
    return MultiPhaseSimulationResult(_overall_summary(trades_frame, curve_frame, phases_frame), phases_frame, trades_frame, curve_frame)


def _exit_decision(position: BrokerPosition, bid_high: float, bid_low: float, ask_high: float, ask_low: float):
    units = position.lot * CONTRACT_OUNCES_PER_LOT
    if position.direction == "BUY":
        high, low = bid_high, bid_low
        stop = position.entry_price - position.stop_loss_usd / units
        if low <= stop:
            return stop, "SL tersentuh"
        if position.protection_activation_usd is None:
            target = position.entry_price + position.take_profit_usd / units
            if high >= target:
                return target, "TP tersentuh"
        prior_peak = position.peak_profit_usd
        if prior_peak >= float(position.protection_activation_usd or np.inf):
            locked = max(float(position.protection_floor_usd or 0), prior_peak - float(position.protection_trail_usd or 0))
            lock_price = position.entry_price + locked / units
            if low <= lock_price:
                return lock_price, f"Profit protection lock USD {locked:g}"
        position.peak_profit_usd = max(prior_peak, (high - position.entry_price) * units)
    else:
        high, low = ask_high, ask_low
        stop = position.entry_price + position.stop_loss_usd / units
        if high >= stop:
            return stop, "SL tersentuh"
        if position.protection_activation_usd is None:
            target = position.entry_price - position.take_profit_usd / units
            if low <= target:
                return target, "TP tersentuh"
        prior_peak = position.peak_profit_usd
        if prior_peak >= float(position.protection_activation_usd or np.inf):
            locked = max(float(position.protection_floor_usd or 0), prior_peak - float(position.protection_trail_usd or 0))
            lock_price = position.entry_price - locked / units
            if high >= lock_price:
                return lock_price, f"Profit protection lock USD {locked:g}"
        position.peak_profit_usd = max(prior_peak, (position.entry_price - low) * units)
    return None


def _pnl(position: BrokerPosition, price: float) -> float:
    units = position.lot * CONTRACT_OUNCES_PER_LOT
    return (price - position.entry_price) * units if position.direction == "BUY" else (position.entry_price - price) * units


def _mark_pnl(position: BrokerPosition, bid: float, ask: float) -> float:
    return _pnl(position, bid if position.direction == "BUY" else ask)


def _trade_row(
    position: BrokerPosition,
    timestamp: pd.Timestamp,
    exit_price: float,
    reason: str,
    balance: float,
    exit_spread: float,
    slippage_points: float,
) -> dict[str, object]:
    gross = _pnl(position, exit_price)
    units = position.lot * CONTRACT_OUNCES_PER_LOT
    spread_cost = position.entry_spread_cost + (exit_spread * units if position.direction == "SELL" else 0.0)
    slippage_cost = 2 * slippage_points * POINT_SIZE * units
    return {
        "Fase": position.phase,
        "Position ID": position.position_id,
        "Tanggal sinyal": position.signal_date,
        "Tanggal entry": position.entry_time,
        "Tanggal tutup": timestamp,
        "Arah": position.direction,
        "Lot": position.lot,
        "Prediksi": position.prediction,
        "Expected change (%)": position.expected_change_pct,
        "Entry": position.entry_price,
        "Exit": exit_price,
        "Alasan exit": reason,
        "TP (USD)": position.take_profit_usd,
        "SL (USD)": position.stop_loss_usd,
        "Peak floating profit (USD)": position.peak_profit_usd,
        "Biaya spread": spread_cost,
        "Biaya slippage": slippage_cost,
        "Gross P/L": gross,
        "Swap": -position.swap_paid,
        "Net P/L": gross - position.swap_paid,
        "Balance": balance,
        "Batas posisi": position.max_positions,
    }


def _phase_summary(phase: int, start: float, target: float, trades: list[dict], curve: list[dict], reached: bool, date) -> dict[str, object]:
    equity = pd.Series([row["Equity"] for row in curve], dtype=float)
    net = pd.Series([row["Net P/L"] for row in trades], dtype=float)
    return {
        "Fase": phase,
        "Start equity": start,
        "Target equity": target,
        "Equity close-all": float(curve[-1]["Equity"]) if curve else start,
        "Target tercapai": reached,
        "Tanggal target": date,
        "Equity terendah": float(equity.min()) if not equity.empty else start,
        "Equity tertinggi": float(equity.max()) if not equity.empty else start,
        "Total net P/L": float(net.sum()) if not net.empty else 0.0,
        "Total swap": float(sum(row["Swap"] for row in trades)),
        "Jumlah transaksi": float(len(trades)),
        "Total BUY": float(sum(row["Arah"] == "BUY" for row in trades)),
        "Total SELL": float(sum(row["Arah"] == "SELL" for row in trades)),
        "Status": "Selesai" if reached else "Berjalan sampai akhir periode",
    }


def _overall_summary(trades: pd.DataFrame, curve: pd.DataFrame, phases: pd.DataFrame) -> dict[str, float]:
    final_equity = float(curve["Equity"].iloc[-1]) if not curve.empty else INITIAL_EQUITY
    net = pd.to_numeric(trades.get("Net P/L", pd.Series(dtype=float)), errors="coerce")
    profits, losses = net[net > 0], net[net < 0]
    equity = pd.to_numeric(curve.get("Equity", pd.Series(dtype=float)), errors="coerce")
    return {
        "Modal awal": INITIAL_EQUITY,
        "Balance akhir": final_equity,
        "Equity akhir": final_equity,
        "Growth total": (final_equity / INITIAL_EQUITY - 1) * 100,
        "Equity tertinggi": float(equity.max()) if not equity.empty else final_equity,
        "Equity terendah": float(equity.min()) if not equity.empty else final_equity,
        "Max drawdown": float((equity.cummax() - equity).max()) if not equity.empty else 0.0,
        "Total net P/L": float(net.sum()) if not net.empty else 0.0,
        "Jumlah transaksi": float(len(trades)),
        "Win rate": float((net > 0).mean() * 100) if not net.empty else np.nan,
        "Profit factor": float(profits.sum() / abs(losses.sum())) if not losses.empty else np.nan,
        "Total BUY": float((trades.get("Arah") == "BUY").sum()) if not trades.empty else 0.0,
        "Total SELL": float((trades.get("Arah") == "SELL").sum()) if not trades.empty else 0.0,
        "Max open posisi": float(curve["Open total"].max()) if not curve.empty else 0.0,
        "Total swap": float(trades["Swap"].sum()) if not trades.empty else 0.0,
        "Biaya spread": float(trades["Biaya spread"].sum()) if not trades.empty else 0.0,
        "Biaya slippage": float(trades["Biaya slippage"].sum()) if not trades.empty else 0.0,
        "Fase selesai": float(phases["Target tercapai"].sum()) if not phases.empty else 0.0,
        "Fase total": float(len(phases)),
    }


def _optional_float(value: object) -> float | None:
    return None if value is None or pd.isna(value) else float(value)


def _compact_curve(result: MultiPhaseSimulationResult) -> MultiPhaseSimulationResult:
    curve = result.equity_curve
    if len(curve) <= 6000:
        return result
    important = curve.loc[[curve["Equity"].idxmin(), curve["Equity"].idxmax(), curve.index[-1]]]
    compact = pd.concat([curve.iloc[::30], important]).sort_index()
    compact = compact.loc[~compact.index.duplicated(keep="last")]
    return replace(result, equity_curve=compact)
