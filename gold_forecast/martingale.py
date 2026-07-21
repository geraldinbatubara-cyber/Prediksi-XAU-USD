from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from gold_forecast.simulation import CONTRACT_OUNCES_PER_LOT
from gold_forecast.strategy_optimizer import MultiPhaseSimulationResult, _indicator_predictions


TEST_START = pd.Timestamp("2025-01-01")
TEST_END = pd.Timestamp("2026-06-30")
INITIAL_EQUITY = 10_000.0
INITIAL_LOT = 0.1
BASE_LOSS_USD = 10.0
LEVERAGE = 100.0
STOP_OUT_LEVEL_PCT = 50.0
BUY_SWAP_PER_001_LOT = 0.2
SELL_SWAP_PER_001_LOT = 0.0
V1_MODE = "Trend"
V1_FAST_MA = 10
V1_SLOW_MA = 50
V1_MOMENTUM_DAYS = 10
V1_THRESHOLD_PCT = 0.15


@dataclass
class MartingalePosition:
    position_id: int
    basket_id: int
    level: int
    direction: str
    lot: float
    entry_date: pd.Timestamp
    entry_price: float
    swap_paid: float = 0.0


def run_martingale_v1(
    gold_ohlc: pd.DataFrame,
) -> tuple[MultiPhaseSimulationResult, pd.DataFrame]:
    data = _prepare_data(gold_ohlc)
    predictions = _indicator_predictions(
        gold_ohlc,
        V1_MODE,
        V1_FAST_MA,
        V1_SLOW_MA,
        V1_MOMENTUM_DAYS,
        V1_THRESHOLD_PCT,
        test_start=TEST_START,
        test_end=TEST_END,
    )
    candidates: list[dict[str, object]] = []
    for max_positions in (3, 4, 5):
        for basket_risk_pct in (10.0, 15.0, 20.0, 25.0, 30.0):
            result = _simulate(
                data,
                predictions,
                max_positions=max_positions,
                basket_risk_pct=basket_risk_pct,
                collect_details=False,
            )
            summary = result.summary
            drawdown_pct = float(summary["Max drawdown"]) / INITIAL_EQUITY * 100
            growth_pct = float(summary["Growth total"])
            risk_adjusted = growth_pct / max(drawdown_pct, 0.01)
            eligible = (
                summary["Stop-out basket"] == 0
                and drawdown_pct <= 20.0
                and summary["Equity akhir"] > 0
            )
            candidates.append(
                {
                    "Strategi": "Sinyal v1 | Martingale dua arah | Close basket di anchor",
                    "Max posisi per basket": max_positions,
                    "Lot maksimum": INITIAL_LOT * (2 ** (max_positions - 1)),
                    "Hard basket loss (%)": basket_risk_pct,
                    "Leverage": LEVERAGE,
                    "Stop-out margin level (%)": STOP_OUT_LEVEL_PCT,
                    "Equity akhir": summary["Equity akhir"],
                    "Growth total": growth_pct,
                    "Max drawdown": summary["Max drawdown"],
                    "Max drawdown (%)": drawdown_pct,
                    "Jumlah basket": summary["Jumlah basket"],
                    "Basket BEP": summary["Basket BEP"],
                    "Basket hard loss": summary["Basket hard loss"],
                    "Stop-out basket": summary["Stop-out basket"],
                    "Margin minimum (%)": summary["Margin level minimum (%)"],
                    "Total swap": summary["Total swap"],
                    "Risk-adjusted score": risk_adjusted,
                    "Lolos batas risiko": eligible,
                    "_score": (
                        1 if eligible else 0,
                        risk_adjusted,
                        growth_pct,
                        -drawdown_pct,
                    ),
                }
            )

    candidates.sort(key=lambda row: row["_score"], reverse=True)
    best = candidates[0]
    result = _simulate(
        data,
        predictions,
        max_positions=int(best["Max posisi per basket"]),
        basket_risk_pct=float(best["Hard basket loss (%)"]),
        collect_details=True,
    )
    leaderboard = pd.DataFrame(
        [{key: value for key, value in row.items() if not key.startswith("_")} for row in candidates]
    )
    return result, leaderboard


def _prepare_data(gold_ohlc: pd.DataFrame) -> pd.DataFrame:
    required = {"Open", "High", "Low", "Close"}
    if gold_ohlc.empty or not required.issubset(gold_ohlc.columns):
        raise ValueError("Data OHLC harian tidak lengkap untuk Martingale v1.")
    data = gold_ohlc.copy()
    data.index = pd.to_datetime(data.index).tz_localize(None)
    data = data.loc[(data.index >= TEST_START) & (data.index <= TEST_END), sorted(required)]
    data = data.astype(float).dropna().sort_index()
    if data.empty:
        raise ValueError("Tidak ada data pada periode Martingale v1.")
    return data


def _simulate(
    data: pd.DataFrame,
    predictions: pd.Series,
    *,
    max_positions: int,
    basket_risk_pct: float,
    collect_details: bool,
) -> MultiPhaseSimulationResult:
    balance = INITIAL_EQUITY
    next_position_id = 1
    next_basket_id = 1
    positions: list[MartingalePosition] = []
    basket: dict[str, object] | None = None
    trade_rows: list[dict[str, object]] = []
    basket_rows: list[dict[str, object]] = []
    equity_rows: list[dict[str, object]] = []
    rejected_additions = 0
    rejected_initial_entries = 0
    margin_minimum = np.inf
    maximum_used_margin = 0.0
    maximum_total_lot = 0.0
    maximum_open_positions = 0

    signal_dates = set(pd.to_datetime(predictions.index))
    for current_date, candle in data.iterrows():
        high = float(candle["High"])
        low = float(candle["Low"])
        close = float(candle["Close"])

        if positions and basket is not None:
            for position in positions:
                if current_date > position.entry_date:
                    swap = _daily_swap(position)
                    position.swap_paid += swap
                    balance -= swap

            adverse_price = low if basket["direction"] == "BUY" else high
            additions_today = 0
            while len(positions) < max_positions:
                level = len(positions)
                loss_threshold = BASE_LOSS_USD * (2 ** (level - 1))
                trigger_distance = loss_threshold / (INITIAL_LOT * CONTRACT_OUNCES_PER_LOT)
                trigger_price = (
                    float(basket["anchor_price"]) - trigger_distance
                    if basket["direction"] == "BUY"
                    else float(basket["anchor_price"]) + trigger_distance
                )
                touched = adverse_price <= trigger_price if basket["direction"] == "BUY" else adverse_price >= trigger_price
                if not touched:
                    break
                lot = INITIAL_LOT * (2 ** level)
                projected = positions + [
                    MartingalePosition(
                        position_id=next_position_id,
                        basket_id=int(basket["basket_id"]),
                        level=level,
                        direction=str(basket["direction"]),
                        lot=lot,
                        entry_date=current_date,
                        entry_price=trigger_price,
                    )
                ]
                projected_equity = balance + _unrealized(projected, trigger_price)
                projected_margin = _used_margin(projected, trigger_price)
                if projected_equity - projected_margin < 0:
                    rejected_additions += 1
                    break
                positions = projected
                next_position_id += 1
                additions_today += 1

            maximum_open_positions = max(maximum_open_positions, len(positions))
            maximum_total_lot = max(maximum_total_lot, sum(position.lot for position in positions))

            adverse_equity = balance + _unrealized(positions, adverse_price)
            adverse_margin = _used_margin(positions, adverse_price)
            adverse_margin_level = _margin_level(adverse_equity, adverse_margin)
            risk_floor = float(basket["start_equity"]) * (1 - basket_risk_pct / 100)
            stop_reason = None
            stop_price = adverse_price
            risk_price = _price_for_equity(positions, balance, risk_floor)
            margin_stop_price = _stopout_price(positions, balance)

            if basket["direction"] == "BUY":
                forced_prices = [
                    (risk_price, "Hard basket loss"),
                    (margin_stop_price, "Margin stop-out"),
                ]
                forced_prices = [(price, reason) for price, reason in forced_prices if adverse_price <= price]
                if forced_prices:
                    stop_price, stop_reason = max(forced_prices, key=lambda item: item[0])
            else:
                forced_prices = [
                    (risk_price, "Hard basket loss"),
                    (margin_stop_price, "Margin stop-out"),
                ]
                forced_prices = [(price, reason) for price, reason in forced_prices if adverse_price >= price]
                if forced_prices:
                    stop_price, stop_reason = min(forced_prices, key=lambda item: item[0])

            if adverse_margin_level <= STOP_OUT_LEVEL_PCT and stop_reason is None:
                stop_reason = "Margin stop-out"
            if adverse_equity <= risk_floor and stop_reason is None:
                stop_reason = "Hard basket loss"

            observed_price = stop_price if stop_reason is not None else adverse_price
            observed_equity = balance + _unrealized(positions, observed_price)
            observed_margin = _used_margin(positions, observed_price)
            margin_minimum = min(margin_minimum, _margin_level(observed_equity, observed_margin))
            maximum_used_margin = max(maximum_used_margin, observed_margin)

            if stop_reason is not None:
                balance, closed = _close_basket(positions, current_date, stop_price, stop_reason, balance)
                trade_rows.extend(closed)
                basket_rows.append(_basket_row(basket, current_date, positions, closed, stop_reason, balance))
                positions = []
                basket = None
            else:
                anchor = float(basket["anchor_price"])
                bep_touched = high >= anchor if basket["direction"] == "BUY" else low <= anchor
                if bep_touched:
                    balance, closed = _close_basket(positions, current_date, anchor, "BEP posisi awal", balance)
                    trade_rows.extend(closed)
                    basket_rows.append(_basket_row(basket, current_date, positions, closed, "BEP posisi awal", balance))
                    positions = []
                    basket = None

        if not positions and current_date in signal_dates and current_date < data.index[-1]:
            prediction = float(predictions.loc[current_date])
            expected_change = (prediction / close - 1) * 100
            direction = "BUY" if expected_change > 0 else "SELL"
            initial = MartingalePosition(
                position_id=next_position_id,
                basket_id=next_basket_id,
                level=0,
                direction=direction,
                lot=INITIAL_LOT,
                entry_date=current_date,
                entry_price=close,
            )
            if balance - _used_margin([initial], close) >= 0:
                positions = [initial]
                basket = {
                    "basket_id": next_basket_id,
                    "direction": direction,
                    "signal_date": current_date,
                    "anchor_price": close,
                    "start_equity": balance,
                    "expected_change_pct": expected_change,
                }
                next_position_id += 1
                next_basket_id += 1
                maximum_open_positions = max(maximum_open_positions, 1)
                maximum_total_lot = max(maximum_total_lot, INITIAL_LOT)
            else:
                rejected_initial_entries += 1

        unrealized = _unrealized(positions, close)
        equity = balance + unrealized
        used_margin = _used_margin(positions, close)
        margin_level = _margin_level(equity, used_margin)
        if positions:
            margin_minimum = min(margin_minimum, margin_level)
        maximum_used_margin = max(maximum_used_margin, used_margin)
        maximum_total_lot = max(maximum_total_lot, sum(position.lot for position in positions))
        equity_rows.append(
            {
                "Tanggal": current_date,
                "Balance": balance,
                "Equity": equity,
                "Unrealized P/L": unrealized,
                "Open BUY": sum(position.direction == "BUY" for position in positions),
                "Open SELL": sum(position.direction == "SELL" for position in positions),
                "Open total": len(positions),
                "Total lot": sum(position.lot for position in positions),
                "Used margin": used_margin,
                "Margin level (%)": margin_level if positions else np.nan,
            }
        )

    if positions and basket is not None:
        final_date = data.index[-1]
        final_price = float(data.iloc[-1]["Close"])
        balance, closed = _close_basket(positions, final_date, final_price, "Akhir periode data", balance)
        trade_rows.extend(closed)
        basket_rows.append(_basket_row(basket, final_date, positions, closed, "Akhir periode data", balance))
        equity_rows[-1].update(
            {
                "Balance": balance,
                "Equity": balance,
                "Unrealized P/L": 0.0,
                "Open BUY": 0,
                "Open SELL": 0,
                "Open total": 0,
                "Total lot": 0.0,
                "Used margin": 0.0,
                "Margin level (%)": np.nan,
            }
        )

    trades = pd.DataFrame(trade_rows)
    baskets = pd.DataFrame(basket_rows)
    equity_curve = pd.DataFrame(equity_rows).set_index("Tanggal")
    equity_series = pd.to_numeric(equity_curve["Equity"], errors="coerce")
    drawdown = equity_series.cummax() - equity_series
    net = pd.to_numeric(trades.get("Net P/L", pd.Series(dtype=float)), errors="coerce")
    gross_profit = float(net[net > 0].sum()) if not net.empty else 0.0
    gross_loss = abs(float(net[net < 0].sum())) if not net.empty else 0.0
    final_equity = float(equity_series.iloc[-1])
    summary = {
        "Modal awal": INITIAL_EQUITY,
        "Balance akhir": final_equity,
        "Equity akhir": final_equity,
        "Target equity": np.nan,
        "Target tercapai": False,
        "Tanggal target": pd.NaT,
        "Fase selesai": 0.0,
        "Fase total": 1.0,
        "Growth total": (final_equity / INITIAL_EQUITY - 1) * 100,
        "Equity tertinggi": float(equity_series.max()),
        "Tanggal equity tertinggi": equity_series.idxmax(),
        "Equity terendah": float(equity_series.min()),
        "Tanggal equity terendah": equity_series.idxmin(),
        "Total net P/L": float(net.sum()) if not net.empty else 0.0,
        "Jumlah transaksi": float(len(trades)),
        "Win rate": float((net > 0).mean() * 100) if not net.empty else np.nan,
        "Max drawdown": float(drawdown.max()),
        "Total BUY": float((trades.get("Arah", pd.Series(dtype=str)) == "BUY").sum()),
        "Total SELL": float((trades.get("Arah", pd.Series(dtype=str)) == "SELL").sum()),
        "Max open posisi": float(maximum_open_positions),
        "Profit factor": np.nan if gross_loss == 0 else gross_profit / gross_loss,
        "Avg net P/L": float(net.mean()) if not net.empty else 0.0,
        "Total swap": float(pd.to_numeric(trades.get("Swap", pd.Series(dtype=float)), errors="coerce").sum()),
        "Jumlah basket": float(len(baskets)),
        "Basket BEP": float((baskets.get("Alasan exit", pd.Series(dtype=str)) == "BEP posisi awal").sum()),
        "Basket hard loss": float((baskets.get("Alasan exit", pd.Series(dtype=str)) == "Hard basket loss").sum()),
        "Stop-out basket": float((baskets.get("Alasan exit", pd.Series(dtype=str)) == "Margin stop-out").sum()),
        "Basket akhir data": float((baskets.get("Alasan exit", pd.Series(dtype=str)) == "Akhir periode data").sum()),
        "Margin level minimum (%)": float(margin_minimum) if np.isfinite(margin_minimum) else np.nan,
        "Used margin maksimum": maximum_used_margin,
        "Total lot maksimum": maximum_total_lot,
        "Lot posisi maksimum": INITIAL_LOT * (2 ** (max_positions - 1)),
        "Max posisi per basket": float(max_positions),
        "Hard basket loss (%)": basket_risk_pct,
        "Leverage": LEVERAGE,
        "Stop-out margin level (%)": STOP_OUT_LEVEL_PCT,
        "Penambahan ditolak margin": float(rejected_additions),
        "Entry awal ditolak margin": float(rejected_initial_entries),
        "Sinyal BUY v1": float(((predictions.reindex(data.index) / data["Close"] - 1) > 0).sum()),
        "Sinyal SELL v1": float(((predictions.reindex(data.index) / data["Close"] - 1) < 0).sum()),
        "Periode uji": f"{data.index.min():%d %b %Y} - {data.index.max():%d %b %Y}",
        "Sumber sinyal": "Optimizer v1 Trend | MA 10/50 | Momentum 10 | threshold 0.15%",
        "Asumsi intrabar": "Adverse-first: averaging/risk/stop-out diperiksa sebelum close BEP",
        "Basket summary": baskets if collect_details else pd.DataFrame(),
    }
    phases = pd.DataFrame(
        [
            {
                "Fase": 1,
                "Start equity": INITIAL_EQUITY,
                "Target equity": np.nan,
                "Equity close-all": final_equity,
                "Target tercapai": False,
                "Tanggal target": pd.NaT,
                "Equity terendah": summary["Equity terendah"],
                "Tanggal equity terendah": summary["Tanggal equity terendah"],
                "Equity tertinggi": summary["Equity tertinggi"],
                "Tanggal equity tertinggi": summary["Tanggal equity tertinggi"],
                "Total net P/L": summary["Total net P/L"],
                "Total swap": summary["Total swap"],
                "Jumlah transaksi": summary["Jumlah transaksi"],
                "Total BUY": summary["Total BUY"],
                "Total SELL": summary["Total SELL"],
                "Max open posisi": summary["Max open posisi"],
                "Win rate": summary["Win rate"],
                "Max drawdown": summary["Max drawdown"],
                "Profit factor": summary["Profit factor"],
                "Status": "Selesai sampai akhir periode",
            }
        ]
    )
    return MultiPhaseSimulationResult(summary, phases, trades, equity_curve)


def _daily_swap(position: MartingalePosition) -> float:
    per_001 = BUY_SWAP_PER_001_LOT if position.direction == "BUY" else SELL_SWAP_PER_001_LOT
    return per_001 * (position.lot / 0.01)


def _unrealized(positions: list[MartingalePosition], price: float) -> float:
    total = 0.0
    for position in positions:
        units = position.lot * CONTRACT_OUNCES_PER_LOT
        if position.direction == "BUY":
            total += (price - position.entry_price) * units
        else:
            total += (position.entry_price - price) * units
    return total


def _used_margin(positions: list[MartingalePosition], price: float) -> float:
    total_units = sum(position.lot * CONTRACT_OUNCES_PER_LOT for position in positions)
    return price * total_units / LEVERAGE


def _margin_level(equity: float, used_margin: float) -> float:
    return np.inf if used_margin <= 0 else equity / used_margin * 100


def _price_for_equity(positions: list[MartingalePosition], balance: float, target_equity: float) -> float:
    total_units = sum(position.lot * CONTRACT_OUNCES_PER_LOT for position in positions)
    entry_notional = sum(position.entry_price * position.lot * CONTRACT_OUNCES_PER_LOT for position in positions)
    if positions[0].direction == "BUY":
        return (target_equity - balance + entry_notional) / total_units
    return (balance + entry_notional - target_equity) / total_units


def _stopout_price(positions: list[MartingalePosition], balance: float) -> float:
    total_units = sum(position.lot * CONTRACT_OUNCES_PER_LOT for position in positions)
    entry_notional = sum(position.entry_price * position.lot * CONTRACT_OUNCES_PER_LOT for position in positions)
    stop_ratio = STOP_OUT_LEVEL_PCT / 100
    if positions[0].direction == "BUY":
        denominator = total_units * (1 - stop_ratio / LEVERAGE)
        return (entry_notional - balance) / denominator
    denominator = total_units * (1 + stop_ratio / LEVERAGE)
    return (balance + entry_notional) / denominator


def _close_basket(
    positions: list[MartingalePosition],
    exit_date: pd.Timestamp,
    exit_price: float,
    reason: str,
    balance: float,
) -> tuple[float, list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    for position in positions:
        units = position.lot * CONTRACT_OUNCES_PER_LOT
        gross = (
            (exit_price - position.entry_price) * units
            if position.direction == "BUY"
            else (position.entry_price - exit_price) * units
        )
        balance += gross
        rows.append(
            {
                "Fase": 1,
                "Model": "Martingale v1",
                "Strategi": "Sinyal Optimizer v1 + martingale dua arah",
                "Basket ID": position.basket_id,
                "Position ID": position.position_id,
                "Level": position.level,
                "Tanggal sinyal": position.entry_date,
                "Tanggal entry": position.entry_date,
                "Tanggal tutup": exit_date,
                "Arah": position.direction,
                "Lot": position.lot,
                "Entry": position.entry_price,
                "Exit": exit_price,
                "Alasan exit": reason,
                "Gross P/L": gross,
                "Swap": -position.swap_paid,
                "Net P/L": gross - position.swap_paid,
                "Balance": balance,
            }
        )
    return balance, rows


def _basket_row(
    basket: dict[str, object],
    exit_date: pd.Timestamp,
    positions: list[MartingalePosition],
    closed: list[dict[str, object]],
    reason: str,
    end_balance: float,
) -> dict[str, object]:
    gross = sum(float(row["Gross P/L"]) for row in closed)
    swap = sum(float(row["Swap"]) for row in closed)
    return {
        "Basket ID": basket["basket_id"],
        "Tanggal entry": basket["signal_date"],
        "Tanggal exit": exit_date,
        "Arah": basket["direction"],
        "Anchor": basket["anchor_price"],
        "Jumlah posisi": len(positions),
        "Total lot": sum(position.lot for position in positions),
        "Lot maksimum": max(position.lot for position in positions),
        "Gross P/L": gross,
        "Swap": swap,
        "Net P/L": gross + swap,
        "Alasan exit": reason,
        "Balance akhir": end_balance,
        "Durasi (hari kalender)": (pd.Timestamp(exit_date) - pd.Timestamp(basket["signal_date"])).days,
    }
