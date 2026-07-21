from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd

from gold_forecast.strategy_optimizer import MultiPhaseSimulationResult


INITIAL_EQUITY = 1000.0
CONTRACT_OUNCES_PER_LOT = 100.0
BUY_SWAP_PER_001_LOT = 0.2
TRAIN_END = pd.Timestamp("2026-05-31 23:59:59")
TEST_START = pd.Timestamp("2026-06-01")


@dataclass
class IntradayPosition:
    position_id: int
    direction: str
    entry_time: pd.Timestamp
    entry_index: int
    entry_price: float
    lot: float
    units: float
    stop_price: float
    target_price: float
    confidence: float
    peak_price: float
    initial_risk_price: float


def run_intraday_optimization(
    gold_m1: pd.DataFrame,
    gold_daily: pd.DataFrame,
    *,
    variant: str,
    requested_start: pd.Timestamp,
    requested_end: pd.Timestamp,
) -> tuple[MultiPhaseSimulationResult, pd.DataFrame]:
    if variant not in {"v1", "v10"}:
        raise ValueError("Variant intraday harus v1 atau v10.")
    data = _prepare_m1(gold_m1, requested_start, requested_end)
    if data.empty:
        raise ValueError("Tidak ada candle M1 dalam periode pengujian.")

    train = data.loc[data.index <= TRAIN_END]
    test = data.loc[data.index >= TEST_START]
    if train.empty or test.empty:
        raise ValueError("Dataset M1 belum cukup untuk pembagian train dan test kronologis.")

    daily_regime = _daily_regime(gold_daily, variant)
    mapped_regime = _map_daily_regime(data.index, daily_regime)
    candidates: list[dict[str, object]] = []
    for params in _candidate_params(variant):
        signals = _build_intraday_signals(data, mapped_regime, params, variant)
        train_result = _simulate_intraday(
            train,
            signals.loc[train.index],
            params,
            variant,
            model_name=f"Optimizer {variant} Intraday M1 - Train",
            collect_details=False,
        )
        summary = train_result.summary
        if summary["Jumlah transaksi"] < 8:
            continue
        score = _training_score(summary)
        candidates.append({**params, **_metrics("Train", summary), "_score": score})

    if not candidates:
        raise RuntimeError("Tidak ada kandidat intraday yang menghasilkan transaksi cukup pada data train.")
    candidates.sort(key=lambda row: row["_score"], reverse=True)
    best = candidates[0]
    best_params = {key: value for key, value in best.items() if not key.startswith("_") and not key.startswith("Train ")}
    best_signals = _build_intraday_signals(data, mapped_regime, best_params, variant)
    test_result = _simulate_intraday(
        test,
        best_signals.loc[test.index],
        best_params,
        variant,
        model_name=f"Optimizer {variant} Intraday M1",
        collect_details=True,
    )
    test_result.summary.update(
        {
            "Periode uji": f"{test.index.min():%d %b %Y %H:%M} - {test.index.max():%d %b %Y %H:%M}",
            "Periode train": f"{train.index.min():%d %b %Y %H:%M} - {train.index.max():%d %b %Y %H:%M}",
            "Periode test": f"{test.index.min():%d %b %Y %H:%M} - {test.index.max():%d %b %Y %H:%M}",
            "Periode diminta": f"{requested_start:%d %b %Y} - {requested_end:%d %b %Y}",
            "Jumlah candle": float(len(test)),
            "Jumlah candle train": float(len(train)),
            "Cakupan lengkap": data.index.min() <= requested_start and data.index.max() >= requested_end,
            "Timeframe": "Daily regime + H1/M1 execution" if variant == "v10" else "Daily regime + M1 execution",
            "Status kelayakan": _eligibility(test_result.summary),
        }
    )

    leaderboard_rows = []
    for rank, candidate in enumerate(candidates[:25], start=1):
        row = {key: value for key, value in candidate.items() if not key.startswith("_")}
        row["Peringkat train"] = rank
        if rank == 1:
            row.update(_metrics("Test", test_result.summary))
            row["Status OOS"] = test_result.summary["Status kelayakan"]
        leaderboard_rows.append(row)
    return test_result, pd.DataFrame(leaderboard_rows)


def _prepare_m1(data: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    required = {"Open", "High", "Low", "Close", "SpreadPoints"}
    if data.empty or not required.issubset(data.columns):
        return pd.DataFrame()
    clean = data.sort_index().copy()
    clean = clean.loc[(clean.index >= start) & (clean.index <= end)]
    for column in required:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
    return clean.dropna(subset=list(required))


def _daily_regime(daily: pd.DataFrame, variant: str) -> pd.Series:
    close = daily["Close"].astype(float)
    if variant == "v1":
        fast_window, slow_window, momentum_days, threshold = 10, 50, 10, 0.15
    else:
        fast_window, slow_window, momentum_days, threshold = 20, 50, 14, 0.10
    fast = close.rolling(fast_window).mean()
    slow = close.rolling(slow_window).mean()
    momentum = close.pct_change(momentum_days) * 100
    regime = pd.Series(0, index=daily.index, dtype=int)
    regime[(close > fast) & (fast > slow) & (momentum > threshold)] = 1
    regime[(close < fast) & (fast < slow) & (momentum < -threshold)] = -1
    regime.index = pd.to_datetime(regime.index).normalize()
    return regime.shift(1).fillna(0).astype(int)


def _map_daily_regime(index: pd.DatetimeIndex, regime: pd.Series) -> pd.Series:
    dates = pd.DatetimeIndex(index).normalize()
    mapped = regime.reindex(dates, method="ffill").fillna(0).to_numpy()
    return pd.Series(mapped, index=index, dtype=int)


def _candidate_params(variant: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if variant == "v1":
        signal_presets = [
            (30, 240, 30, 0.01),
            (60, 240, 30, 0.01),
            (60, 480, 60, 0.015),
            (120, 480, 60, 0.02),
        ]
        for (fast, slow, momentum, threshold), risk in product(
            signal_presets, [(2.0, 1.25), (2.5, 1.5)]
        ):
            rows.append(_base_params(fast, slow, momentum, threshold, risk, 45, variant))
    else:
        signal_presets = [(60, 240, 30, 0.01), (120, 480, 60, 0.015)]
        for (fast, slow, momentum, threshold), h1, risk in product(
            signal_presets, [(6, 24), (12, 48)], [(2.5, 1.5), (3.0, 1.8)]
        ):
            row = _base_params(fast, slow, momentum, threshold, risk, 45, variant)
            row.update({"H1 Fast": h1[0], "H1 Slow": h1[1], "Breakout bars": 60})
            rows.append(row)
    return rows


def _base_params(fast, slow, momentum, threshold, risk, cooldown, variant) -> dict[str, object]:
    tp_atr, sl_atr = risk
    return {
        "Strategi": f"Daily {variant} + Intraday confirmation",
        "M1 Fast EMA": fast,
        "M1 Slow EMA": slow,
        "Momentum M1 bars": momentum,
        "Momentum threshold (%)": threshold,
        "ATR bars": 30,
        "TP ATR": tp_atr,
        "SL ATR": sl_atr,
        "Trailing ATR": 1.0 if variant == "v10" else 0.0,
        "Cooldown bars": cooldown,
        "Max holding bars": 360 if variant == "v1" else 480,
        "Max posisi": 1 if variant == "v1" else 2,
        "Max transaksi per hari": 4 if variant == "v1" else 6,
        "Daily loss cap (%)": 2.0 if variant == "v1" else 3.0,
        "Lot minimum": 0.01,
        "Lot maksimum": 0.01 if variant == "v1" else 0.02,
        "Max spread points": 30,
        "Slippage points per side": 2,
        "Point size": 0.01,
    }


def _build_intraday_signals(data, daily_regime, params, variant) -> pd.DataFrame:
    close = data["Close"]
    high = data["High"]
    low = data["Low"]
    fast = close.ewm(span=int(params["M1 Fast EMA"]), adjust=False).mean()
    slow = close.ewm(span=int(params["M1 Slow EMA"]), adjust=False).mean()
    momentum = close.pct_change(int(params["Momentum M1 bars"])) * 100
    previous_close = close.shift(1)
    true_range = pd.concat([high - low, (high - previous_close).abs(), (low - previous_close).abs()], axis=1).max(axis=1)
    atr = true_range.ewm(span=int(params["ATR bars"]), adjust=False, min_periods=int(params["ATR bars"])).mean()
    spread_ok = data["SpreadPoints"] <= float(params["Max spread points"])
    threshold = float(params["Momentum threshold (%)"])
    wit_index = data.index + pd.Timedelta(hours=9)
    session_ok = (wit_index.dayofweek < 5) & (wit_index.hour != 6)

    buy = (daily_regime == 1) & (close > fast) & (fast > slow) & (momentum > threshold)
    sell = (daily_regime == -1) & (close < fast) & (fast < slow) & (momentum < -threshold)
    hold_buy = (daily_regime == 1) & (fast > slow)
    hold_sell = (daily_regime == -1) & (fast < slow)
    strength = (momentum.abs() / max(threshold, 0.001)).clip(0, 4) / 4

    if variant == "v10":
        h1_close = close.resample("1h").last().dropna()
        h1_fast = h1_close.ewm(span=int(params["H1 Fast"]), adjust=False).mean()
        h1_slow = h1_close.ewm(span=int(params["H1 Slow"]), adjust=False).mean()
        h1_direction = pd.Series(np.where(h1_fast > h1_slow, 1, -1), index=h1_close.index).shift(1)
        h1_direction = h1_direction.reindex(data.index, method="ffill").fillna(0)
        breakout = int(params["Breakout bars"])
        previous_high = high.rolling(breakout).max().shift(1)
        previous_low = low.rolling(breakout).min().shift(1)
        buy &= (h1_direction == 1) & (close > previous_high)
        sell &= (h1_direction == -1) & (close < previous_low)
        hold_buy &= h1_direction == 1
        hold_sell &= h1_direction == -1
        strength = ((strength * 0.5) + ((fast - slow).abs() / atr).clip(0, 2) / 4 + (daily_regime.abs() * 0.25)).clip(0, 1)

    direction = pd.Series(0, index=data.index, dtype=int)
    direction[buy & spread_ok & session_ok & atr.notna()] = 1
    direction[sell & spread_ok & session_ok & atr.notna()] = -1
    exit_direction = pd.Series(0, index=data.index, dtype=int)
    exit_direction[hold_buy] = 1
    exit_direction[hold_sell] = -1
    return pd.DataFrame(
        {"Direction": direction, "ExitDirection": exit_direction, "ATR": atr, "Confidence": strength.fillna(0)},
        index=data.index,
    )


def _simulate_intraday(data, signals, params, variant, *, model_name, collect_details):
    balance = INITIAL_EQUITY
    positions: list[IntradayPosition] = []
    trades: list[dict[str, object]] = []
    equity_rows: list[dict[str, object]] = []
    position_id = 1
    last_entry_index = -10_000
    current_day = None
    day_start_balance = balance
    day_trade_count = 0
    min_equity = max_equity = balance
    max_drawdown = 0.0
    equity_peak = balance
    gross_profit = gross_loss = total_swap = 0.0
    wins = buy_count = sell_count = 0
    timestamps = data.index
    high_values = data["High"].to_numpy(dtype=float)
    low_values = data["Low"].to_numpy(dtype=float)
    close_values = data["Close"].to_numpy(dtype=float)
    spread_values = data["SpreadPoints"].to_numpy(dtype=float)
    entry_directions = signals["Direction"].to_numpy(dtype=int)
    holding_directions = signals["ExitDirection"].to_numpy(dtype=int)
    atr_values = signals["ATR"].to_numpy(dtype=float)
    confidence_values = signals["Confidence"].to_numpy(dtype=float)

    for i, timestamp in enumerate(timestamps):
        day = timestamp.normalize()
        if current_day is None or day != current_day:
            current_day = day
            day_start_balance = balance
            day_trade_count = 0
        high = high_values[i]
        low = low_values[i]
        close = close_values[i]
        spread = spread_values[i] * float(params["Point size"])
        slippage = float(params["Slippage points per side"]) * float(params["Point size"])
        entry_direction = int(entry_directions[i])
        holding_direction = int(holding_directions[i])
        bar_atr = atr_values[i]
        bar_confidence = confidence_values[i]

        survivors: list[IntradayPosition] = []
        for position in positions:
            exit_price = None
            reason = None
            if position.direction == "BUY":
                if low <= position.stop_price:
                    exit_price, reason = position.stop_price - slippage, "SL intraday"
                elif high >= position.target_price:
                    exit_price, reason = position.target_price - slippage, "TP intraday"
                elif holding_direction != 1:
                    exit_price, reason = close - slippage, "Daily/M1 regime selesai"
                position.peak_price = max(position.peak_price, high)
                if variant == "v10" and exit_price is None:
                    trail = float(params["Trailing ATR"]) * bar_atr
                    if position.peak_price - position.entry_price >= position.initial_risk_price:
                        position.stop_price = max(position.stop_price, position.peak_price - trail, position.entry_price)
            else:
                ask_high = high + spread
                ask_low = low + spread
                if ask_high >= position.stop_price:
                    exit_price, reason = position.stop_price + slippage, "SL intraday"
                elif ask_low <= position.target_price:
                    exit_price, reason = position.target_price + slippage, "TP intraday"
                elif holding_direction != -1:
                    exit_price, reason = close + spread + slippage, "Daily/M1 regime selesai"
                position.peak_price = min(position.peak_price, ask_low)
                if variant == "v10" and exit_price is None:
                    trail = float(params["Trailing ATR"]) * bar_atr
                    if position.entry_price - position.peak_price >= position.initial_risk_price:
                        position.stop_price = min(position.stop_price, position.peak_price + trail, position.entry_price)

            if exit_price is None and i - position.entry_index >= int(params["Max holding bars"]):
                if position.direction == "BUY":
                    exit_price = close - slippage
                else:
                    exit_price = close + spread + slippage
                reason = "Batas waktu posisi"
            if exit_price is None:
                survivors.append(position)
                continue

            direction_sign = 1 if position.direction == "BUY" else -1
            gross = (float(exit_price) - position.entry_price) * position.units * direction_sign
            overnight_days = max(0, int((timestamp.normalize() - position.entry_time.normalize()).days))
            swap = BUY_SWAP_PER_001_LOT * (position.lot / 0.01) * overnight_days if position.direction == "BUY" else 0.0
            net = gross - swap
            balance += net
            gross_profit += max(net, 0)
            gross_loss += min(net, 0)
            total_swap -= swap
            wins += int(net > 0)
            buy_count += int(position.direction == "BUY")
            sell_count += int(position.direction == "SELL")
            if collect_details:
                trades.append({
                    "Fase": 1, "Model": model_name, "Strategi": str(params["Strategi"]),
                    "Position ID": position.position_id, "Tanggal entry": position.entry_time,
                    "Tanggal tutup": timestamp, "Arah": position.direction, "Lot": position.lot,
                    "Confidence": position.confidence * 100, "Prediksi": np.nan,
                    "Expected change (%)": np.nan, "Entry": position.entry_price, "Exit": float(exit_price),
                    "Alasan exit": reason, "TP (USD)": abs(position.target_price - position.entry_price) * position.units,
                    "SL (USD)": position.initial_risk_price * position.units, "Gross P/L": gross,
                    "Swap": -swap, "Net P/L": net, "Balance": balance,
                })
        positions = survivors

        unrealized = 0.0
        for position in positions:
            mark = close - slippage if position.direction == "BUY" else close + spread + slippage
            unrealized += (mark - position.entry_price) * position.units * (1 if position.direction == "BUY" else -1)
        equity = balance + unrealized
        equity_peak = max(equity_peak, equity)
        min_equity = min(min_equity, equity)
        max_equity = max(max_equity, equity)
        max_drawdown = max(max_drawdown, equity_peak - equity)
        if collect_details:
            equity_rows.append({"Tanggal": timestamp, "Fase": 1, "Balance": balance, "Equity": equity,
                                "Unrealized P/L": unrealized, "Open BUY": sum(p.direction == "BUY" for p in positions),
                                "Open SELL": sum(p.direction == "SELL" for p in positions), "Open total": len(positions),
                                "Target equity tercapai": equity >= 1200})

        daily_loss_blocked = balance <= day_start_balance * (1 - float(params["Daily loss cap (%)"]) / 100)
        can_enter = (
            entry_direction != 0 and len(positions) < int(params["Max posisi"])
            and i - last_entry_index >= int(params["Cooldown bars"])
            and day_trade_count < int(params["Max transaksi per hari"])
            and not daily_loss_blocked and balance > 0 and pd.notna(bar_atr)
        )
        if can_enter:
            confidence = bar_confidence
            lot = float(params["Lot minimum"])
            if variant == "v10" and confidence >= 0.70:
                lot = float(params["Lot maksimum"])
            units = lot * CONTRACT_OUNCES_PER_LOT
            stop_distance = max(bar_atr * float(params["SL ATR"]), spread * 3)
            target_distance = max(bar_atr * float(params["TP ATR"]), spread * 4)
            risk_usd = stop_distance * units
            if risk_usd <= equity * 0.02:
                if entry_direction == 1:
                    entry = close + spread + slippage
                    stop, target, direction = entry - stop_distance, entry + target_distance, "BUY"
                else:
                    entry = close - slippage
                    stop, target, direction = entry + stop_distance, entry - target_distance, "SELL"
                positions.append(IntradayPosition(position_id, direction, timestamp, i, entry, lot, units,
                                                   stop, target, confidence, entry, stop_distance))
                position_id += 1
                last_entry_index = i
                day_trade_count += 1

    if positions:
        timestamp = data.index[-1]
        close = close_values[-1]
        spread = spread_values[-1] * float(params["Point size"])
        slippage = float(params["Slippage points per side"]) * float(params["Point size"])
        for position in positions:
            exit_price = close - slippage if position.direction == "BUY" else close + spread + slippage
            gross = (exit_price - position.entry_price) * position.units * (1 if position.direction == "BUY" else -1)
            days = max(0, int((timestamp.normalize() - position.entry_time.normalize()).days))
            swap = BUY_SWAP_PER_001_LOT * (position.lot / 0.01) * days if position.direction == "BUY" else 0.0
            net = gross - swap
            balance += net
            gross_profit += max(net, 0); gross_loss += min(net, 0); total_swap -= swap
            wins += int(net > 0); buy_count += int(position.direction == "BUY"); sell_count += int(position.direction == "SELL")
            if collect_details:
                trades.append({"Fase": 1, "Model": model_name, "Strategi": str(params["Strategi"]),
                               "Position ID": position.position_id, "Tanggal entry": position.entry_time,
                               "Tanggal tutup": timestamp, "Arah": position.direction, "Lot": position.lot,
                               "Confidence": position.confidence * 100, "Prediksi": np.nan,
                               "Expected change (%)": np.nan, "Entry": position.entry_price, "Exit": exit_price,
                               "Alasan exit": "Akhir periode data", "TP (USD)": abs(position.target_price-position.entry_price)*position.units,
                               "SL (USD)": position.initial_risk_price*position.units, "Gross P/L": gross,
                               "Swap": -swap, "Net P/L": net, "Balance": balance})

    trade_count = buy_count + sell_count
    win_rate = np.nan if trade_count == 0 else wins / trade_count * 100
    profit_factor = np.nan if gross_loss == 0 else gross_profit / abs(gross_loss)
    equity_curve = pd.DataFrame(equity_rows).set_index("Tanggal") if equity_rows else pd.DataFrame()
    summary = {
        "Modal awal": INITIAL_EQUITY, "Balance akhir": balance, "Equity akhir": balance,
        "Target equity": 1200.0, "Target tercapai": max_equity >= 1200,
        "Tanggal target": pd.NaT, "Fase selesai": 0.0, "Fase total": 1.0,
        "Growth total": (balance / INITIAL_EQUITY - 1) * 100, "Equity tertinggi": max_equity,
        "Tanggal equity tertinggi": equity_curve["Equity"].idxmax() if not equity_curve.empty else None,
        "Equity terendah": min_equity, "Tanggal equity terendah": equity_curve["Equity"].idxmin() if not equity_curve.empty else None,
        "Total net P/L": balance - INITIAL_EQUITY, "Jumlah transaksi": float(trade_count),
        "Win rate": win_rate, "Max drawdown": max_drawdown, "Total BUY": float(buy_count),
        "Total SELL": float(sell_count), "Max open posisi": float(params["Max posisi"]),
        "Profit factor": profit_factor, "Avg net P/L": (balance-INITIAL_EQUITY)/trade_count if trade_count else 0.0,
        "Total swap": total_swap,
    }
    trades_frame = pd.DataFrame(trades)
    phases = pd.DataFrame([{
        "Fase": 1, "Start equity": INITIAL_EQUITY, "Target equity": 1200.0,
        "Equity close-all": balance, "Target tercapai": max_equity >= 1200, "Tanggal target": pd.NaT,
        "Equity terendah": min_equity, "Tanggal equity terendah": summary["Tanggal equity terendah"],
        "Equity tertinggi": max_equity, "Tanggal equity tertinggi": summary["Tanggal equity tertinggi"],
        "Total Profit": gross_profit, "Total Loss": gross_loss, "Total net P/L": balance-INITIAL_EQUITY,
        "Total swap": total_swap, "Jumlah transaksi": float(trade_count), "Total BUY": float(buy_count),
        "Total SELL": float(sell_count), "Max open posisi": float(params["Max posisi"]),
        "Win rate": win_rate, "Max drawdown": max_drawdown, "Profit factor": profit_factor,
        "Status": "Out-of-sample test selesai",
    }])
    return MultiPhaseSimulationResult(summary, phases, trades_frame, equity_curve)


def _training_score(summary) -> tuple[float, float, float, float]:
    net = float(summary["Total net P/L"])
    drawdown = float(summary["Max drawdown"])
    profit_factor = float(summary["Profit factor"]) if pd.notna(summary["Profit factor"]) else 0.0
    return (net - 1.5 * drawdown, profit_factor, -drawdown, float(summary["Jumlah transaksi"]))


def _metrics(prefix, summary) -> dict[str, object]:
    return {
        f"{prefix} equity akhir": summary["Equity akhir"], f"{prefix} growth (%)": summary["Growth total"],
        f"{prefix} max drawdown": summary["Max drawdown"], f"{prefix} transaksi": summary["Jumlah transaksi"],
        f"{prefix} win rate": summary["Win rate"], f"{prefix} profit factor": summary["Profit factor"],
    }


def _eligibility(summary) -> str:
    if (summary["Equity akhir"] > INITIAL_EQUITY and summary["Profit factor"] > 1
            and summary["Max drawdown"] <= INITIAL_EQUITY * 0.20 and summary["Jumlah transaksi"] >= 8):
        return "LAYAK KANDIDAT PAPER TEST"
    return "BELUM LAYAK"
