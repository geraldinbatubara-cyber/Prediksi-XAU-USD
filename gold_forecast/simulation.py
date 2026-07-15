from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from gold_forecast.model import RidgeRegressor, _features
from gold_forecast.model_v2 import _estimator, _market_features


CONTRACT_OUNCES_PER_LOT = 100
OPTIMIZATION_START = pd.Timestamp("2025-01-01")
OPTIMIZATION_END = pd.Timestamp("2026-06-30")

DEFAULT_SCENARIOS = [
    {
        "Strategi": "Agresif",
        "entry_threshold_pct": 0.0,
        "take_profit_usd": 5.0,
        "stop_loss_usd": 15.0,
    },
    {
        "Strategi": "Moderate",
        "entry_threshold_pct": 0.20,
        "take_profit_usd": 7.0,
        "stop_loss_usd": 14.0,
    },
    {
        "Strategi": "Konservatif",
        "entry_threshold_pct": 0.35,
        "take_profit_usd": 10.0,
        "stop_loss_usd": 15.0,
    },
    {
        "Strategi": "TP Lebar",
        "entry_threshold_pct": 0.20,
        "take_profit_usd": 15.0,
        "stop_loss_usd": 20.0,
    },
]


@dataclass
class SimulationResult:
    summary: dict[str, float]
    trades: pd.DataFrame
    equity_curve: pd.DataFrame


@dataclass
class OpenPosition:
    position_id: int
    model_name: str
    strategy_name: str
    signal_date: pd.Timestamp
    direction: str
    lot_size: float
    entry_price: float
    prediction: float
    expected_change_pct: float
    take_profit_usd: float
    stop_loss_usd: float | None
    entry_threshold_pct: float
    opened_balance: float
    max_positions: int
    swap_paid: float = 0.0


def _position_unrealized(position: OpenPosition, price: float, units: float) -> float:
    if position.direction == "BUY":
        return (price - position.entry_price) * units
    return (position.entry_price - price) * units


def _close_position_row(
    position: OpenPosition,
    exit_date: pd.Timestamp,
    exit_price: float,
    exit_reason: str,
    cash_balance: float,
    units: float,
) -> dict[str, object]:
    gross_pnl = _position_unrealized(position, exit_price, units)
    net_pnl = gross_pnl - position.swap_paid
    return {
        "Model": position.model_name,
        "Strategi": position.strategy_name,
        "Position ID": position.position_id,
        "Tanggal sinyal": position.signal_date,
        "Waktu sinyal": "23:59 WIT",
        "Tanggal entry": position.signal_date,
        "Waktu entry": "23:59 WIT",
        "Tanggal tutup": exit_date,
        "Waktu tutup": "Saat TP/SL tersentuh" if exit_reason in {"TP tersentuh", "SL tersentuh"} else "Close harian",
        "Arah": position.direction,
        "Lot": position.lot_size,
        "Prediksi": position.prediction,
        "Expected change (%)": position.expected_change_pct,
        "Entry": position.entry_price,
        "Exit": exit_price,
        "Alasan exit": exit_reason,
        "TP (USD)": position.take_profit_usd,
        "SL (USD)": np.nan if position.stop_loss_usd is None else position.stop_loss_usd,
        "Threshold entry (%)": position.entry_threshold_pct,
        "Gross P/L": gross_pnl,
        "Swap": -position.swap_paid,
        "Net P/L": net_pnl,
        "Balance": cash_balance,
        "Batas posisi": position.max_positions,
    }


def _simulate_trading(
    predictions: pd.Series,
    gold_ohlc: pd.DataFrame,
    model_name: str,
    initial_balance: float,
    lot_size: float,
    take_profit_usd: float,
    swap_per_position: float,
    max_buy_positions: int,
    max_sell_positions: int,
    entry_threshold_pct: float = 0.0,
    stop_loss_usd: float | None = None,
    strategy_name: str = "Agresif",
    target_equity: float = 1200.0,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    units = lot_size * CONTRACT_OUNCES_PER_LOT
    take_profit_points = take_profit_usd / units
    stop_loss_points = None if stop_loss_usd is None else stop_loss_usd / units
    cash_balance = initial_balance
    next_position_id = 1
    closed_rows: list[dict[str, object]] = []
    equity_rows: list[dict[str, object]] = []
    open_positions: list[OpenPosition] = []
    if predictions.empty:
        return closed_rows, equity_rows

    prediction_dates = set(predictions.index)

    for current_date, candle in gold_ohlc.iterrows():
        if current_date < predictions.index.min():
            continue

        high = float(candle["High"])
        low = float(candle["Low"])
        close = float(candle["Close"])

        still_open: list[OpenPosition] = []
        for position in open_positions:
            if position.direction == "BUY":
                tp_price = position.entry_price + take_profit_points
                sl_price = None if stop_loss_points is None else position.entry_price - stop_loss_points
                hit_tp = high >= tp_price
                hit_sl = sl_price is not None and low <= sl_price
                if hit_sl:
                    exit_price = sl_price
                    exit_reason = "SL tersentuh"
                elif hit_tp:
                    exit_price = tp_price
                    exit_reason = "TP tersentuh"
                else:
                    still_open.append(position)
                    continue
            else:
                tp_price = position.entry_price - take_profit_points
                sl_price = None if stop_loss_points is None else position.entry_price + stop_loss_points
                hit_tp = low <= tp_price
                hit_sl = sl_price is not None and high >= sl_price
                if hit_sl:
                    exit_price = sl_price
                    exit_reason = "SL tersentuh"
                elif hit_tp:
                    exit_price = tp_price
                    exit_reason = "TP tersentuh"
                else:
                    still_open.append(position)
                    continue

            cash_balance += _position_unrealized(position, exit_price, units)
            closed_rows.append(_close_position_row(position, current_date, exit_price, exit_reason, cash_balance, units))
        open_positions = still_open

        for position in open_positions:
            position.swap_paid += swap_per_position
            cash_balance -= swap_per_position

        if current_date in prediction_dates:
            prediction = float(predictions.loc[current_date])
            expected_change_pct = (prediction / close - 1) * 100
            buy_count = sum(position.direction == "BUY" for position in open_positions)
            sell_count = sum(position.direction == "SELL" for position in open_positions)

            if expected_change_pct > 0 and expected_change_pct >= entry_threshold_pct:
                direction = "BUY"
                max_positions = max_buy_positions
                can_open = buy_count < max_buy_positions
            elif expected_change_pct < 0 and abs(expected_change_pct) >= entry_threshold_pct:
                direction = "SELL"
                max_positions = max_sell_positions
                can_open = sell_count < max_sell_positions
            else:
                can_open = False

            if can_open:
                open_positions.append(
                    OpenPosition(
                        position_id=next_position_id,
                        model_name=model_name,
                        strategy_name=strategy_name,
                        signal_date=current_date,
                        direction=direction,
                        lot_size=lot_size,
                        entry_price=close,
                        prediction=prediction,
                        expected_change_pct=expected_change_pct,
                        take_profit_usd=take_profit_usd,
                        stop_loss_usd=stop_loss_usd,
                        entry_threshold_pct=entry_threshold_pct,
                        opened_balance=cash_balance,
                        max_positions=max_positions,
                    )
                )
                next_position_id += 1

        unrealized = sum(_position_unrealized(position, close, units) for position in open_positions)
        equity = cash_balance + unrealized
        equity_rows.append(
            {
                "Tanggal": current_date,
                "Balance": cash_balance,
                "Equity": equity,
                "Unrealized P/L": unrealized,
                "Open BUY": sum(position.direction == "BUY" for position in open_positions),
                "Open SELL": sum(position.direction == "SELL" for position in open_positions),
                "Target equity tercapai": equity >= target_equity,
            }
        )

        if equity >= target_equity:
            for position in open_positions:
                cash_balance += _position_unrealized(position, close, units)
                closed_rows.append(
                    _close_position_row(position, current_date, close, "Target equity tercapai", cash_balance, units)
                )
            open_positions = []
            equity_rows[-1]["Balance"] = cash_balance
            equity_rows[-1]["Equity"] = cash_balance
            equity_rows[-1]["Unrealized P/L"] = 0.0
            equity_rows[-1]["Open BUY"] = 0
            equity_rows[-1]["Open SELL"] = 0
            break

    if open_positions:
        final_date = gold_ohlc.index[-1]
        final_close = float(gold_ohlc.iloc[-1]["Close"])
        for position in open_positions:
            cash_balance += _position_unrealized(position, final_close, units)
            closed_rows.append(_close_position_row(position, final_date, final_close, "Akhir periode data", cash_balance, units))
        if equity_rows and equity_rows[-1]["Tanggal"] == final_date:
            equity_rows[-1]["Balance"] = cash_balance
            equity_rows[-1]["Equity"] = cash_balance
            equity_rows[-1]["Unrealized P/L"] = 0.0
            equity_rows[-1]["Open BUY"] = 0
            equity_rows[-1]["Open SELL"] = 0
        else:
            equity_rows.append(
                {
                    "Tanggal": final_date,
                    "Balance": cash_balance,
                    "Equity": cash_balance,
                    "Unrealized P/L": 0.0,
                    "Open BUY": 0,
                    "Open SELL": 0,
                    "Target equity tercapai": cash_balance >= target_equity,
                }
            )

    return closed_rows, equity_rows


def _summary(
    trades: pd.DataFrame,
    equity_curve: pd.DataFrame,
    initial_balance: float,
    target_equity: float,
) -> dict[str, float]:
    target_reached = False
    target_date = None
    lowest_equity = initial_balance
    lowest_equity_date = None
    highest_equity = initial_balance
    highest_equity_date = None

    if not equity_curve.empty:
        equity = pd.to_numeric(equity_curve["Equity"], errors="coerce")
        lowest_equity = float(equity.min())
        highest_equity = float(equity.max())
        lowest_equity_date = equity.idxmin()
        highest_equity_date = equity.idxmax()
        reached = equity_curve[equity_curve["Target equity tercapai"]]
        if not reached.empty:
            target_reached = True
            target_date = reached.index[0]

    if trades.empty:
        return {
            "Modal awal": initial_balance,
            "Balance akhir": initial_balance,
            "Equity akhir": highest_equity if target_reached else initial_balance,
            "Target equity": target_equity,
            "Target tercapai": target_reached,
            "Tanggal target": target_date,
            "Equity tertinggi": highest_equity,
            "Tanggal equity tertinggi": highest_equity_date,
            "Equity terendah": lowest_equity,
            "Tanggal equity terendah": lowest_equity_date,
            "Total net P/L": 0.0,
            "Jumlah transaksi": 0.0,
            "Win rate": np.nan,
            "Max drawdown": 0.0,
            "Total BUY": 0.0,
            "Total SELL": 0.0,
            "Profit factor": np.nan,
            "Avg net P/L": 0.0,
            "Total swap": 0.0,
        }

    balance = pd.to_numeric(trades["Balance"], errors="coerce")
    equity_for_drawdown = pd.to_numeric(equity_curve["Equity"], errors="coerce") if not equity_curve.empty else balance
    peak = equity_for_drawdown.cummax()
    drawdown = peak - equity_for_drawdown
    net_pnl = pd.to_numeric(trades["Net P/L"], errors="coerce")
    gross_profit = float(net_pnl[net_pnl > 0].sum())
    gross_loss = abs(float(net_pnl[net_pnl < 0].sum()))
    profit_factor = np.nan if gross_loss == 0 else gross_profit / gross_loss
    return {
        "Modal awal": initial_balance,
        "Balance akhir": float(balance.iloc[-1]),
        "Equity akhir": float(equity_curve["Equity"].iloc[-1]) if not equity_curve.empty else float(balance.iloc[-1]),
        "Target equity": target_equity,
        "Target tercapai": target_reached,
        "Tanggal target": target_date,
        "Equity tertinggi": highest_equity,
        "Tanggal equity tertinggi": highest_equity_date,
        "Equity terendah": lowest_equity,
        "Tanggal equity terendah": lowest_equity_date,
        "Total net P/L": float(net_pnl.sum()),
        "Jumlah transaksi": float(len(trades)),
        "Win rate": float((net_pnl > 0).mean() * 100),
        "Max drawdown": float(drawdown.max()),
        "Total BUY": float((trades["Arah"] == "BUY").sum()),
        "Total SELL": float((trades["Arah"] == "SELL").sum()),
        "Profit factor": float(profit_factor) if not pd.isna(profit_factor) else np.nan,
        "Avg net P/L": float(net_pnl.mean()),
        "Total swap": float(pd.to_numeric(trades["Swap"], errors="coerce").sum()),
    }


def _result(
    rows: list[dict[str, object]],
    equity_rows: list[dict[str, object]],
    initial_balance: float,
    target_equity: float,
) -> SimulationResult:
    trades = pd.DataFrame(rows)
    if equity_rows:
        equity_curve = pd.DataFrame(equity_rows).set_index("Tanggal")
    else:
        equity_curve = pd.DataFrame(
            columns=["Tanggal", "Balance", "Equity", "Unrealized P/L", "Open BUY", "Open SELL", "Target equity tercapai"]
        ).set_index("Tanggal")
    return SimulationResult(
        summary=_summary(trades, equity_curve, initial_balance, target_equity),
        trades=trades,
        equity_curve=equity_curve,
    )


def _model_1_predictions(market: pd.DataFrame) -> pd.Series:
    close = market["gold"].dropna()
    features = _features(close)
    dataset = features.copy()
    dataset["target"] = close.shift(-1)
    dataset = dataset.dropna()
    if len(dataset) < 250:
        raise ValueError("Minimal 250 observasi bersih diperlukan untuk simulasi Model 1.")

    split = int(len(dataset) * 0.8)
    train, test = dataset.iloc[:split], dataset.iloc[split:]
    feature_names = list(features.columns)
    estimator = RidgeRegressor(alpha=10.0)
    estimator.fit(train[feature_names], train["target"])
    return pd.Series(estimator.predict(test[feature_names]), index=test.index)


def _model_2_predictions(market: pd.DataFrame) -> pd.Series:
    features = _market_features(market)
    gold = market["gold"]
    clean_features = features.dropna()
    if len(clean_features) < 500:
        raise ValueError("Minimal 500 observasi lintas pasar diperlukan untuk simulasi Model 2.")

    dataset = clean_features.copy()
    dataset["target_return"] = gold.shift(-1) / gold - 1
    dataset = dataset.dropna()
    split = int(len(dataset) * 0.8)
    train, test = dataset.iloc[:split], dataset.iloc[split:]
    feature_names = list(clean_features.columns)
    estimator = _estimator()
    estimator.fit(train[feature_names], train["target_return"])
    predicted_return = pd.Series(estimator.predict(test[feature_names]), index=test.index)
    current = gold.reindex(test.index)
    return current * (1 + predicted_return)


def _simulate_predictions(
    predictions: pd.Series,
    gold_ohlc: pd.DataFrame,
    model_name: str,
    initial_balance: float = 1000.0,
    lot_size: float = 0.01,
    take_profit_usd: float = 5.0,
    swap_per_position: float = 0.2,
    max_buy_positions: int = 8,
    max_sell_positions: int = 10,
    entry_threshold_pct: float = 0.0,
    stop_loss_usd: float | None = None,
    strategy_name: str = "Agresif",
    target_equity: float = 1200.0,
) -> SimulationResult:
    rows, equity_rows = _simulate_trading(
        predictions,
        gold_ohlc,
        model_name,
        initial_balance,
        lot_size,
        take_profit_usd,
        swap_per_position,
        max_buy_positions,
        max_sell_positions,
        entry_threshold_pct,
        stop_loss_usd,
        strategy_name,
        target_equity,
    )
    return _result(rows, equity_rows, initial_balance, target_equity)


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = -delta.clip(upper=0).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr_pct(gold_ohlc: pd.DataFrame, window: int = 14) -> pd.Series:
    high = gold_ohlc["High"].astype(float)
    low = gold_ohlc["Low"].astype(float)
    close = gold_ohlc["Close"].astype(float)
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(window).mean() / close * 100


def _indicator_predictions(
    gold_ohlc: pd.DataFrame,
    mode: str,
    fast_window: int,
    slow_window: int,
    momentum_days: int,
    entry_threshold_pct: float,
    test_start: pd.Timestamp = OPTIMIZATION_START,
    test_end: pd.Timestamp = OPTIMIZATION_END,
) -> pd.Series:
    close = gold_ohlc["Close"].astype(float)
    fast_ma = close.rolling(fast_window).mean()
    slow_ma = close.rolling(slow_window).mean()
    momentum = close.pct_change(momentum_days) * 100
    atr_pct = _atr_pct(gold_ohlc)
    rsi = _rsi(close)
    previous_high = gold_ohlc["High"].astype(float).rolling(slow_window).max().shift(1)
    previous_low = gold_ohlc["Low"].astype(float).rolling(slow_window).min().shift(1)

    expected_change = pd.Series(0.0, index=gold_ohlc.index)
    signal_size = (atr_pct * 0.75).clip(lower=entry_threshold_pct + 0.05, upper=1.8)

    if mode == "Trend":
        buy = (close > fast_ma) & (fast_ma > slow_ma) & (momentum > entry_threshold_pct)
        sell = (close < fast_ma) & (fast_ma < slow_ma) & (momentum < -entry_threshold_pct)
    elif mode == "Breakout":
        buy = (close > previous_high) & (momentum > 0)
        sell = (close < previous_low) & (momentum < 0)
    elif mode == "Pullback":
        buy = (close > slow_ma) & (rsi < 42) & (momentum > -entry_threshold_pct)
        sell = (close < slow_ma) & (rsi > 58) & (momentum < entry_threshold_pct)
    else:
        buy = pd.Series(False, index=gold_ohlc.index)
        sell = pd.Series(False, index=gold_ohlc.index)

    expected_change = expected_change.mask(buy, signal_size)
    expected_change = expected_change.mask(sell, -signal_size)
    predictions = close * (1 + expected_change / 100)
    predictions = predictions.loc[(predictions.index >= test_start) & (predictions.index <= test_end)]
    predictions = predictions[expected_change.loc[predictions.index].abs() >= entry_threshold_pct]
    return predictions.dropna()


def _strategy_score(summary: dict[str, float]) -> tuple[float, float, float, float]:
    target_rank = 1.0 if summary["Target tercapai"] else 0.0
    target_date = summary["Tanggal target"]
    days_to_target = 9999.0
    if target_rank and not pd.isna(target_date):
        days_to_target = float((pd.Timestamp(target_date) - OPTIMIZATION_START).days)
    return (
        target_rank,
        -days_to_target,
        float(summary["Equity akhir"]),
        -float(summary["Max drawdown"]),
    )


def run_optimized_strategy(
    gold_ohlc: pd.DataFrame,
    target_equity: float = 1200.0,
) -> tuple[SimulationResult, pd.DataFrame]:
    candidates: list[dict[str, object]] = []
    modes = ["Trend", "Breakout", "Pullback"]
    fast_windows = [10, 20]
    slow_windows = [50]
    momentum_days_options = [5, 10]
    thresholds = [0.15]
    take_profits = [15.0, 25.0]
    stop_losses = [10.0, 18.0]
    lot_sizes = [0.01, 0.02]

    for mode in modes:
        for fast_window in fast_windows:
            for slow_window in slow_windows:
                if fast_window >= slow_window:
                    continue
                for momentum_days in momentum_days_options:
                    for threshold in thresholds:
                        predictions = _indicator_predictions(
                            gold_ohlc,
                            mode,
                            fast_window,
                            slow_window,
                            momentum_days,
                            threshold,
                        )
                        if predictions.empty:
                            continue
                        for take_profit in take_profits:
                            for stop_loss in stop_losses:
                                if take_profit < stop_loss * 0.5:
                                    continue
                                for lot_size in lot_sizes:
                                    strategy_name = (
                                        f"{mode} | MA {fast_window}/{slow_window} | "
                                        f"Mom {momentum_days} | TP {take_profit:g} SL {stop_loss:g} | "
                                        f"Lot {lot_size:.2f}"
                                    )
                                    result = _simulate_predictions(
                                        predictions,
                                        gold_ohlc,
                                        "Strategi Optimizer",
                                        lot_size=lot_size,
                                        take_profit_usd=take_profit,
                                        stop_loss_usd=stop_loss,
                                        entry_threshold_pct=threshold,
                                        strategy_name=strategy_name,
                                        target_equity=target_equity,
                                    )
                                    summary = result.summary
                                    trade_count = summary["Jumlah transaksi"]
                                    if trade_count < 3:
                                        continue
                                    candidates.append(
                                        {
                                            "Mode": mode,
                                            "Strategi": strategy_name,
                                            "Fast MA": fast_window,
                                            "Slow MA": slow_window,
                                            "Momentum hari": momentum_days,
                                            "Threshold entry (%)": threshold,
                                            "TP (USD)": take_profit,
                                            "SL (USD)": stop_loss,
                                            "Lot": lot_size,
                                            "Target tercapai": summary["Target tercapai"],
                                            "Tanggal target": summary["Tanggal target"],
                                            "Equity akhir": summary["Equity akhir"],
                                            "Equity terendah": summary["Equity terendah"],
                                            "Equity tertinggi": summary["Equity tertinggi"],
                                            "Max drawdown": summary["Max drawdown"],
                                            "Jumlah transaksi": trade_count,
                                            "Win rate": summary["Win rate"],
                                            "Profit factor": summary["Profit factor"],
                                            "Avg net P/L": summary["Avg net P/L"],
                                            "_score": _strategy_score(summary),
                                            "_result": result,
                                        }
                                    )

    if not candidates:
        empty_result = _result([], [], 1000.0, target_equity)
        return empty_result, pd.DataFrame()

    candidates.sort(key=lambda row: row["_score"], reverse=True)
    best_result = candidates[0]["_result"]
    leaderboard = pd.DataFrame([{key: value for key, value in row.items() if not key.startswith("_")} for row in candidates])
    return best_result, leaderboard


def simulate_model_1(
    market: pd.DataFrame,
    gold_ohlc: pd.DataFrame,
    initial_balance: float = 1000.0,
    lot_size: float = 0.01,
    take_profit_usd: float = 5.0,
    swap_per_position: float = 0.2,
    max_buy_positions: int = 8,
    max_sell_positions: int = 10,
    entry_threshold_pct: float = 0.0,
    stop_loss_usd: float | None = None,
    strategy_name: str = "Agresif",
    target_equity: float = 1200.0,
) -> SimulationResult:
    return _simulate_predictions(
        _model_1_predictions(market),
        gold_ohlc,
        "Model 1 - Harga Historis",
        initial_balance,
        lot_size,
        take_profit_usd,
        swap_per_position,
        max_buy_positions,
        max_sell_positions,
        entry_threshold_pct,
        stop_loss_usd,
        strategy_name,
        target_equity,
    )


def simulate_model_2(
    market: pd.DataFrame,
    gold_ohlc: pd.DataFrame,
    initial_balance: float = 1000.0,
    lot_size: float = 0.01,
    take_profit_usd: float = 5.0,
    swap_per_position: float = 0.2,
    max_buy_positions: int = 8,
    max_sell_positions: int = 10,
    entry_threshold_pct: float = 0.0,
    stop_loss_usd: float | None = None,
    strategy_name: str = "Agresif",
    target_equity: float = 1200.0,
) -> SimulationResult:
    return _simulate_predictions(
        _model_2_predictions(market),
        gold_ohlc,
        "Model 2 - Lintas Pasar",
        initial_balance,
        lot_size,
        take_profit_usd,
        swap_per_position,
        max_buy_positions,
        max_sell_positions,
        entry_threshold_pct,
        stop_loss_usd,
        strategy_name,
        target_equity,
    )


def run_simulation_scenarios(
    market: pd.DataFrame,
    gold_ohlc: pd.DataFrame,
    scenarios: list[dict[str, float | str]] | None = None,
    target_equity: float = 1200.0,
) -> tuple[dict[str, dict[str, SimulationResult]], pd.DataFrame]:
    scenario_rows: list[dict[str, object]] = []
    results: dict[str, dict[str, SimulationResult]] = {
        "Model 1 - Harga Historis": {},
        "Model 2 - Lintas Pasar": {},
    }
    model_predictions = {
        "Model 1 - Harga Historis": _model_1_predictions(market),
        "Model 2 - Lintas Pasar": _model_2_predictions(market),
    }

    for scenario in scenarios or DEFAULT_SCENARIOS:
        strategy_name = str(scenario["Strategi"])
        for model_name, predictions in model_predictions.items():
            result = _simulate_predictions(
                predictions,
                gold_ohlc,
                model_name,
                take_profit_usd=float(scenario["take_profit_usd"]),
                entry_threshold_pct=float(scenario["entry_threshold_pct"]),
                stop_loss_usd=float(scenario["stop_loss_usd"]),
                strategy_name=strategy_name,
                target_equity=target_equity,
            )
            results[model_name][strategy_name] = result
            summary = result.summary
            scenario_rows.append(
                {
                    "Model": model_name,
                    "Strategi": strategy_name,
                    "Threshold entry (%)": float(scenario["entry_threshold_pct"]),
                    "TP (USD)": float(scenario["take_profit_usd"]),
                    "SL (USD)": float(scenario["stop_loss_usd"]),
                    "Balance akhir": summary["Balance akhir"],
                    "Equity akhir": summary["Equity akhir"],
                    "Target tercapai": summary["Target tercapai"],
                    "Tanggal target": summary["Tanggal target"],
                    "Equity terendah": summary["Equity terendah"],
                    "Tanggal equity terendah": summary["Tanggal equity terendah"],
                    "Equity tertinggi": summary["Equity tertinggi"],
                    "Tanggal equity tertinggi": summary["Tanggal equity tertinggi"],
                    "Total net P/L": summary["Total net P/L"],
                    "Jumlah transaksi": summary["Jumlah transaksi"],
                    "Win rate": summary["Win rate"],
                    "Max drawdown": summary["Max drawdown"],
                    "Profit factor": summary["Profit factor"],
                    "Avg net P/L": summary["Avg net P/L"],
                }
            )

    return results, pd.DataFrame(scenario_rows)
