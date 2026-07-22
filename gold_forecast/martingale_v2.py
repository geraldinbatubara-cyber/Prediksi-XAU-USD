from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from gold_forecast.martingale import (
    BASE_LOSS_USD,
    BUY_SWAP_PER_001_LOT,
    INITIAL_EQUITY,
    INITIAL_LOT,
    LEVERAGE,
    SELL_SWAP_PER_001_LOT,
    STOP_OUT_LEVEL_PCT,
    TEST_END,
    TEST_START,
    V1_FAST_MA,
    V1_MODE,
    V1_MOMENTUM_DAYS,
    V1_SLOW_MA,
    V1_THRESHOLD_PCT,
    MartingalePosition,
    _margin_level,
    _prepare_data,
    _price_for_equity,
    _stopout_price,
    _unrealized,
    _used_margin,
)
from gold_forecast.simulation import CONTRACT_OUNCES_PER_LOT
from gold_forecast.strategy_optimizer import MultiPhaseSimulationResult, _indicator_predictions


TRAIN_END = pd.Timestamp("2025-12-31")
OOS_START = pd.Timestamp("2026-01-01")


@dataclass(frozen=True)
class AdaptiveParameters:
    max_positions: int
    lot_multiplier: float
    spacing_atr: float
    basket_risk_pct: float
    target_profit_usd: float
    max_holding_days: int
    minimum_entry_margin_pct: float = 150.0
    direction_mode: str = "BOTH"
    maximum_atr_pct: float = np.inf


def run_martingale_v2(
    gold_ohlc: pd.DataFrame,
) -> tuple[MultiPhaseSimulationResult, pd.DataFrame, MultiPhaseSimulationResult, MultiPhaseSimulationResult]:
    data = _prepare_data(gold_ohlc)
    indicators = _build_indicators(gold_ohlc, data.index)
    train_data = data.loc[data.index <= TRAIN_END]
    train_indicators = indicators.reindex(train_data.index)
    candidates: list[dict[str, object]] = []

    for max_positions in (2, 3):
        for lot_multiplier in (1.25, 1.5):
            for spacing_atr in (0.1, 0.25, 0.5):
                for basket_risk_pct in (0.5, 1.0, 1.5):
                    for target_profit_usd in (50.0, 75.0, 100.0):
                        for max_holding_days in (20, 40):
                            params = AdaptiveParameters(
                                max_positions=max_positions,
                                lot_multiplier=lot_multiplier,
                                spacing_atr=spacing_atr,
                                basket_risk_pct=basket_risk_pct,
                                target_profit_usd=target_profit_usd,
                                max_holding_days=max_holding_days,
                            )
                            result = _simulate(train_data, train_indicators, params, collect_details=False)
                            summary = result.summary
                            drawdown_pct = float(summary["Max drawdown"]) / INITIAL_EQUITY * 100
                            growth = float(summary["Growth total"])
                            eligible = (
                                growth > 0
                                and drawdown_pct <= 10.0
                                and summary["Stop-out basket"] == 0
                                and summary["Jumlah basket"] >= 5
                                and summary["Max open posisi"] >= 2
                            )
                            risk_adjusted = growth / max(drawdown_pct, 0.01)
                            candidates.append(
                                {
                                    "Strategi": "Martingale adaptif ATR + weighted basket exit",
                                    "Max posisi": max_positions,
                                    "Lot multiplier": lot_multiplier,
                                    "Lot maksimum": _level_lot(max_positions - 1, lot_multiplier),
                                    "Jarak entry (ATR)": spacing_atr,
                                    "Hard basket loss (%)": basket_risk_pct,
                                    "Target basket (USD)": target_profit_usd,
                                    "Time stop (hari)": max_holding_days,
                                    "Minimum margin entry (%)": params.minimum_entry_margin_pct,
                                    "Train equity akhir": summary["Equity akhir"],
                                    "Train growth (%)": growth,
                                    "Train max drawdown": summary["Max drawdown"],
                                    "Train max drawdown (%)": drawdown_pct,
                                    "Train basket": summary["Jumlah basket"],
                                    "Train stop-out": summary["Stop-out basket"],
                                    "Train total swap": summary["Total swap"],
                                    "Train risk-adjusted score": risk_adjusted,
                                    "Lolos seleksi train": eligible,
                                    "_params": params,
                                    "_score": (
                                        1 if eligible else 0,
                                        risk_adjusted,
                                        growth,
                                        -drawdown_pct,
                                    ),
                                }
                            )

    candidates.sort(key=lambda row: row["_score"], reverse=True)
    base_params = candidates[0]["_params"]
    filtered_candidates: list[dict[str, object]] = []
    for direction_mode in ("BOTH", "BUY_ONLY"):
        for maximum_atr_pct in (1.5, 2.0, 3.0, np.inf):
            for spacing_atr in (0.1, 0.25, 0.5):
                params = replace(
                    base_params,
                    direction_mode=direction_mode,
                    maximum_atr_pct=maximum_atr_pct,
                    spacing_atr=spacing_atr,
                )
                result = _simulate(train_data, train_indicators, params, collect_details=False)
                summary = result.summary
                drawdown_pct = float(summary["Max drawdown"]) / INITIAL_EQUITY * 100
                growth = float(summary["Growth total"])
                eligible = (
                    growth > 0
                    and drawdown_pct <= 10.0
                    and summary["Stop-out basket"] == 0
                    and summary["Jumlah basket"] >= 5
                    and summary["Max open posisi"] >= 2
                )
                risk_adjusted = growth / max(drawdown_pct, 0.01)
                filtered_candidates.append(
                    {
                        "Strategi": "Martingale adaptif ATR + fresh v1 signal",
                        "Arah diizinkan": direction_mode,
                        "Maks ATR/Close (%)": maximum_atr_pct,
                        "Max posisi": params.max_positions,
                        "Lot multiplier": params.lot_multiplier,
                        "Lot maksimum": _level_lot(params.max_positions - 1, params.lot_multiplier),
                        "Jarak entry (ATR)": params.spacing_atr,
                        "Hard basket loss (%)": params.basket_risk_pct,
                        "Target basket (USD)": params.target_profit_usd,
                        "Time stop (hari)": params.max_holding_days,
                        "Minimum margin entry (%)": params.minimum_entry_margin_pct,
                        "Train equity akhir": summary["Equity akhir"],
                        "Train growth (%)": growth,
                        "Train max drawdown": summary["Max drawdown"],
                        "Train max drawdown (%)": drawdown_pct,
                        "Train basket": summary["Jumlah basket"],
                        "Train stop-out": summary["Stop-out basket"],
                        "Train total swap": summary["Total swap"],
                        "Train risk-adjusted score": risk_adjusted,
                        "Lolos seleksi train": eligible,
                        "_params": params,
                        "_score": (1 if eligible else 0, risk_adjusted, growth, -drawdown_pct),
                    }
                )

    filtered_candidates.sort(key=lambda row: row["_score"], reverse=True)
    selected = filtered_candidates[0]
    params = selected["_params"]
    train_result = _simulate(train_data, train_indicators, params, collect_details=True)
    oos_data = data.loc[data.index >= OOS_START]
    oos_result = _simulate(oos_data, indicators.reindex(oos_data.index), params, collect_details=True)
    full_result = _simulate(data, indicators, params, collect_details=True)

    oos_summary = oos_result.summary
    oos_drawdown_pct = float(oos_summary["Max drawdown"]) / INITIAL_EQUITY * 100
    oos_pass = (
        oos_summary["Growth total"] > 0
        and oos_drawdown_pct <= 10.0
        and oos_summary["Stop-out basket"] == 0
        and oos_summary["Jumlah basket"] >= 3
    )
    full_result.summary.update(
        {
            "Status kelayakan": "LAYAK KANDIDAT PAPER TEST" if oos_pass else "BELUM LAYAK",
            "Periode train": f"{train_data.index.min():%d %b %Y} - {train_data.index.max():%d %b %Y}",
            "Periode OOS": f"{oos_data.index.min():%d %b %Y} - {oos_data.index.max():%d %b %Y}",
            "OOS equity akhir": oos_summary["Equity akhir"],
            "OOS growth (%)": oos_summary["Growth total"],
            "OOS max drawdown": oos_summary["Max drawdown"],
            "OOS max drawdown (%)": oos_drawdown_pct,
            "OOS jumlah basket": oos_summary["Jumlah basket"],
            "OOS total swap": oos_summary["Total swap"],
            "OOS lolos": oos_pass,
        }
    )
    leaderboard = pd.DataFrame(
        [{key: value for key, value in row.items() if not key.startswith("_")} for row in filtered_candidates]
    )
    return full_result, leaderboard, train_result, oos_result


def _build_indicators(gold_ohlc: pd.DataFrame, index: pd.Index) -> pd.DataFrame:
    close = gold_ohlc["Close"].astype(float)
    fast = close.rolling(V1_FAST_MA).mean()
    slow = close.rolling(V1_SLOW_MA).mean()
    momentum = close.pct_change(V1_MOMENTUM_DAYS) * 100
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            gold_ohlc["High"].astype(float) - gold_ohlc["Low"].astype(float),
            (gold_ohlc["High"].astype(float) - previous_close).abs(),
            (gold_ohlc["Low"].astype(float) - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = true_range.rolling(14).mean()
    regime = pd.Series("WAIT", index=gold_ohlc.index, dtype=object)
    regime.loc[(close > fast) & (fast > slow) & (momentum > 0)] = "BUY"
    regime.loc[(close < fast) & (fast < slow) & (momentum < 0)] = "SELL"
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
    entry_signal = pd.Series("WAIT", index=gold_ohlc.index, dtype=object)
    expected_change = pd.Series(0.0, index=gold_ohlc.index)
    common = predictions.index.intersection(gold_ohlc.index)
    expected_change.loc[common] = (predictions.loc[common] / close.loc[common] - 1) * 100
    entry_signal.loc[expected_change > 0] = "BUY"
    entry_signal.loc[expected_change < 0] = "SELL"
    previous_entry = entry_signal.shift(1).fillna("WAIT")
    fresh_entry = entry_signal.where((entry_signal != "WAIT") & (entry_signal != previous_entry), "WAIT")
    return pd.DataFrame(
        {
            "ATR": atr,
            "ATR pct": atr / close * 100,
            "Regime": regime,
            "Previous regime": regime.shift(1).fillna("WAIT"),
            "Entry signal": entry_signal,
            "Fresh entry": fresh_entry,
            "Expected change (%)": expected_change,
        }
    ).reindex(index)


def _simulate(
    data: pd.DataFrame,
    indicators: pd.DataFrame,
    params: AdaptiveParameters,
    *,
    collect_details: bool,
) -> MultiPhaseSimulationResult:
    balance = INITIAL_EQUITY
    positions: list[MartingalePosition] = []
    basket: dict[str, object] | None = None
    next_position_id = 1
    next_basket_id = 1
    trade_rows: list[dict[str, object]] = []
    basket_rows: list[dict[str, object]] = []
    equity_rows: list[dict[str, object]] = []
    maximum_open = 0
    maximum_total_lot = 0.0
    maximum_used_margin = 0.0
    minimum_margin = np.inf
    rejected_additions = 0
    rejected_initial = 0

    for bar_number, (current_date, candle) in enumerate(data.iterrows()):
        high = float(candle["High"])
        low = float(candle["Low"])
        close = float(candle["Close"])
        indicator = indicators.loc[current_date]

        if positions and basket is not None:
            for position in positions:
                if current_date > position.entry_date:
                    swap = _daily_swap(position)
                    position.swap_paid += swap
                    balance -= swap

            direction = str(basket["direction"])
            adverse_price = low if direction == "BUY" else high
            risk_floor = float(basket["start_equity"]) * (1 - params.basket_risk_pct / 100)
            path_forced_price: float | None = None
            path_forced_reason: str | None = None
            while len(positions) < params.max_positions and indicator["Previous regime"] == direction:
                level = len(positions)
                trigger_distance = float(basket["atr_entry"]) * params.spacing_atr * level
                trigger_price = (
                    float(basket["anchor_price"]) - trigger_distance
                    if direction == "BUY"
                    else float(basket["anchor_price"]) + trigger_distance
                )
                touched = adverse_price <= trigger_price if direction == "BUY" else adverse_price >= trigger_price
                if not touched:
                    break
                current_risk_price = _price_for_equity(positions, balance, risk_floor)
                current_stopout_price = _stopout_price(positions, balance)
                forced_price, forced_reason = _first_forced_price(
                    direction,
                    adverse_price,
                    current_risk_price,
                    current_stopout_price,
                )
                forced_before_trigger = (
                    forced_reason is not None
                    and (
                        (direction == "BUY" and forced_price >= trigger_price)
                        or (direction == "SELL" and forced_price <= trigger_price)
                    )
                )
                if forced_before_trigger:
                    path_forced_price = forced_price
                    path_forced_reason = forced_reason
                    break
                lot = _level_lot(level, params.lot_multiplier)
                projected = positions + [
                    MartingalePosition(
                        position_id=next_position_id,
                        basket_id=int(basket["basket_id"]),
                        level=level,
                        direction=direction,
                        lot=lot,
                        entry_date=current_date,
                        entry_price=trigger_price,
                    )
                ]
                projected_equity = balance + _unrealized(projected, trigger_price)
                projected_margin = _used_margin(projected, trigger_price)
                if _margin_level(projected_equity, projected_margin) < params.minimum_entry_margin_pct:
                    rejected_additions += 1
                    break
                positions = projected
                next_position_id += 1

            maximum_open = max(maximum_open, len(positions))
            maximum_total_lot = max(maximum_total_lot, sum(position.lot for position in positions))
            adverse_equity = balance + _unrealized(positions, adverse_price)
            risk_price = _price_for_equity(positions, balance, risk_floor)
            stopout_price = _stopout_price(positions, balance)
            if path_forced_reason is not None:
                forced_price, forced_reason = path_forced_price, path_forced_reason
            else:
                forced_price, forced_reason = _first_forced_price(
                    direction,
                    adverse_price,
                    risk_price,
                    stopout_price,
                )
            if forced_reason is None and adverse_equity <= risk_floor:
                forced_price, forced_reason = adverse_price, "Hard basket loss"

            observed_price = forced_price if forced_reason else adverse_price
            observed_equity = balance + _unrealized(positions, observed_price)
            observed_margin = _used_margin(positions, observed_price)
            minimum_margin = min(minimum_margin, _margin_level(observed_equity, observed_margin))
            maximum_used_margin = max(maximum_used_margin, observed_margin)

            if forced_reason:
                balance, closed = _close_positions(positions, current_date, forced_price, forced_reason, balance)
                trade_rows.extend(closed)
                basket_rows.append(_basket_row(basket, current_date, positions, closed, forced_reason, balance))
                positions = []
                basket = None
            else:
                target_price = _target_price(positions, params.target_profit_usd)
                target_touched = high >= target_price if direction == "BUY" else low <= target_price
                if target_touched:
                    balance, closed = _close_positions(
                        positions,
                        current_date,
                        target_price,
                        "Weighted basket target",
                        balance,
                    )
                    trade_rows.extend(closed)
                    basket_rows.append(
                        _basket_row(basket, current_date, positions, closed, "Weighted basket target", balance)
                    )
                    positions = []
                    basket = None

        if positions and basket is not None:
            held_days = bar_number - int(basket["entry_bar"])
            opposite_regime = indicator["Regime"] not in {"WAIT", basket["direction"]}
            timed_out = held_days >= params.max_holding_days
            if opposite_regime or timed_out:
                reason = "Regime flip" if opposite_regime else "Time stop"
                balance, closed = _close_positions(positions, current_date, close, reason, balance)
                trade_rows.extend(closed)
                basket_rows.append(_basket_row(basket, current_date, positions, closed, reason, balance))
                positions = []
                basket = None

        entry_signal = str(indicator["Fresh entry"])
        atr_value = float(indicator["ATR"]) if pd.notna(indicator["ATR"]) else np.nan
        atr_pct = float(indicator["ATR pct"]) if pd.notna(indicator["ATR pct"]) else np.nan
        if params.direction_mode == "BUY_ONLY" and entry_signal == "SELL":
            entry_signal = "WAIT"
        if np.isfinite(atr_pct) and atr_pct > params.maximum_atr_pct:
            entry_signal = "WAIT"
        if not positions and entry_signal in {"BUY", "SELL"} and np.isfinite(atr_value) and current_date < data.index[-1]:
            initial = MartingalePosition(
                position_id=next_position_id,
                basket_id=next_basket_id,
                level=0,
                direction=entry_signal,
                lot=INITIAL_LOT,
                entry_date=current_date,
                entry_price=close,
            )
            initial_margin = _used_margin([initial], close)
            if _margin_level(balance, initial_margin) >= params.minimum_entry_margin_pct:
                positions = [initial]
                basket = {
                    "basket_id": next_basket_id,
                    "direction": entry_signal,
                    "signal_date": current_date,
                    "entry_bar": bar_number,
                    "anchor_price": close,
                    "atr_entry": atr_value,
                    "start_equity": balance,
                }
                next_position_id += 1
                next_basket_id += 1
                maximum_open = max(maximum_open, 1)
                maximum_total_lot = max(maximum_total_lot, INITIAL_LOT)
            else:
                rejected_initial += 1

        unrealized = _unrealized(positions, close)
        equity = balance + unrealized
        used_margin = _used_margin(positions, close)
        if positions:
            minimum_margin = min(minimum_margin, _margin_level(equity, used_margin))
        maximum_used_margin = max(maximum_used_margin, used_margin)
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
                "Margin level (%)": _margin_level(equity, used_margin) if positions else np.nan,
            }
        )

    if positions and basket is not None:
        final_date = data.index[-1]
        final_price = float(data.iloc[-1]["Close"])
        balance, closed = _close_positions(positions, final_date, final_price, "Akhir periode data", balance)
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
    summary = _summary(
        trades,
        baskets,
        equity_curve,
        params,
        maximum_open,
        maximum_total_lot,
        maximum_used_margin,
        minimum_margin,
        rejected_additions,
        rejected_initial,
        collect_details,
    )
    phases = pd.DataFrame(
        [
            {
                "Fase": 1,
                "Start equity": INITIAL_EQUITY,
                "Target equity": np.nan,
                "Equity close-all": summary["Equity akhir"],
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


def _level_lot(level: int, multiplier: float) -> float:
    return round(INITIAL_LOT * (multiplier ** level), 2)


def _daily_swap(position: MartingalePosition) -> float:
    cost = BUY_SWAP_PER_001_LOT if position.direction == "BUY" else SELL_SWAP_PER_001_LOT
    return cost * (position.lot / 0.01)


def _first_forced_price(
    direction: str,
    adverse_price: float,
    risk_price: float,
    stopout_price: float,
) -> tuple[float, str | None]:
    candidates = [(risk_price, "Hard basket loss"), (stopout_price, "Margin stop-out")]
    if direction == "BUY":
        touched = [(price, reason) for price, reason in candidates if adverse_price <= price]
        return max(touched, key=lambda item: item[0]) if touched else (adverse_price, None)
    touched = [(price, reason) for price, reason in candidates if adverse_price >= price]
    return min(touched, key=lambda item: item[0]) if touched else (adverse_price, None)


def _target_price(positions: list[MartingalePosition], target_profit_usd: float) -> float:
    total_units = sum(position.lot * CONTRACT_OUNCES_PER_LOT for position in positions)
    weighted_entry = sum(
        position.entry_price * position.lot * CONTRACT_OUNCES_PER_LOT for position in positions
    ) / total_units
    distance = target_profit_usd / total_units
    return weighted_entry + distance if positions[0].direction == "BUY" else weighted_entry - distance


def _close_positions(
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
                "Model": "Martingale v2",
                "Strategi": "ATR spacing + capped lot + weighted basket target",
                "Basket ID": position.basket_id,
                "Position ID": position.position_id,
                "Level": position.level,
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
    balance: float,
) -> dict[str, object]:
    gross = sum(float(row["Gross P/L"]) for row in closed)
    swap = sum(float(row["Swap"]) for row in closed)
    return {
        "Basket ID": basket["basket_id"],
        "Tanggal entry": basket["signal_date"],
        "Tanggal exit": exit_date,
        "Arah": basket["direction"],
        "Anchor": basket["anchor_price"],
        "ATR entry": basket["atr_entry"],
        "Jumlah posisi": len(positions),
        "Total lot": sum(position.lot for position in positions),
        "Lot maksimum": max(position.lot for position in positions),
        "Gross P/L": gross,
        "Swap": swap,
        "Net P/L": gross + swap,
        "Alasan exit": reason,
        "Balance akhir": balance,
        "Durasi (hari kalender)": (pd.Timestamp(exit_date) - pd.Timestamp(basket["signal_date"])).days,
    }


def _summary(
    trades: pd.DataFrame,
    baskets: pd.DataFrame,
    equity_curve: pd.DataFrame,
    params: AdaptiveParameters,
    maximum_open: int,
    maximum_total_lot: float,
    maximum_used_margin: float,
    minimum_margin: float,
    rejected_additions: int,
    rejected_initial: int,
    collect_details: bool,
) -> dict[str, object]:
    equity = pd.to_numeric(equity_curve["Equity"], errors="coerce")
    drawdown = equity.cummax() - equity
    net = pd.to_numeric(trades.get("Net P/L", pd.Series(dtype=float)), errors="coerce")
    profit = float(net[net > 0].sum()) if not net.empty else 0.0
    loss = abs(float(net[net < 0].sum())) if not net.empty else 0.0
    final_equity = float(equity.iloc[-1])
    reasons = baskets.get("Alasan exit", pd.Series(dtype=str))
    return {
        "Modal awal": INITIAL_EQUITY,
        "Balance akhir": final_equity,
        "Equity akhir": final_equity,
        "Target equity": np.nan,
        "Target tercapai": False,
        "Tanggal target": pd.NaT,
        "Fase selesai": 0.0,
        "Fase total": 1.0,
        "Growth total": (final_equity / INITIAL_EQUITY - 1) * 100,
        "Equity tertinggi": float(equity.max()),
        "Tanggal equity tertinggi": equity.idxmax(),
        "Equity terendah": float(equity.min()),
        "Tanggal equity terendah": equity.idxmin(),
        "Total net P/L": float(net.sum()) if not net.empty else 0.0,
        "Jumlah transaksi": float(len(trades)),
        "Win rate": float((net > 0).mean() * 100) if not net.empty else np.nan,
        "Max drawdown": float(drawdown.max()),
        "Total BUY": float((trades.get("Arah", pd.Series(dtype=str)) == "BUY").sum()),
        "Total SELL": float((trades.get("Arah", pd.Series(dtype=str)) == "SELL").sum()),
        "Max open posisi": float(maximum_open),
        "Profit factor": np.nan if loss == 0 else profit / loss,
        "Avg net P/L": float(net.mean()) if not net.empty else 0.0,
        "Total swap": float(pd.to_numeric(trades.get("Swap", pd.Series(dtype=float)), errors="coerce").sum()),
        "Jumlah basket": float(len(baskets)),
        "Basket target": float((reasons == "Weighted basket target").sum()),
        "Basket hard loss": float((reasons == "Hard basket loss").sum()),
        "Basket regime flip": float((reasons == "Regime flip").sum()),
        "Basket time stop": float((reasons == "Time stop").sum()),
        "Stop-out basket": float((reasons == "Margin stop-out").sum()),
        "Basket akhir data": float((reasons == "Akhir periode data").sum()),
        "Margin level minimum (%)": float(minimum_margin) if np.isfinite(minimum_margin) else np.nan,
        "Used margin maksimum": maximum_used_margin,
        "Total lot maksimum": maximum_total_lot,
        "Max posisi per basket": float(params.max_positions),
        "Lot multiplier": params.lot_multiplier,
        "Lot posisi maksimum": _level_lot(params.max_positions - 1, params.lot_multiplier),
        "Jarak entry (ATR)": params.spacing_atr,
        "Hard basket loss (%)": params.basket_risk_pct,
        "Target basket (USD)": params.target_profit_usd,
        "Time stop (hari)": float(params.max_holding_days),
        "Minimum margin entry (%)": params.minimum_entry_margin_pct,
        "Arah diizinkan": params.direction_mode,
        "Maks ATR/Close (%)": params.maximum_atr_pct,
        "Leverage": LEVERAGE,
        "Stop-out margin level (%)": STOP_OUT_LEVEL_PCT,
        "Penambahan ditolak margin": float(rejected_additions),
        "Entry awal ditolak margin": float(rejected_initial),
        "Periode uji": f"{equity.index.min():%d %b %Y} - {equity.index.max():%d %b %Y}",
        "Sumber sinyal": "Optimizer v1 Trend | MA 10/50 | Momentum 10 | threshold 0.15%",
        "Asumsi intrabar": "Adverse-first sebelum weighted basket target",
        "Basket summary": baskets if collect_details else pd.DataFrame(),
    }
