from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from gold_forecast.exact_broker_oos import (
    POINT_SIZE,
    SLIPPAGE_POINTS,
    _compact_curve,
    _overall_summary,
    _phase_summary,
    _prepare_m1,
)
from gold_forecast.simulation import CONTRACT_OUNCES_PER_LOT
from gold_forecast.strategy_optimizer import (
    BUY_SWAP_PER_001_LOT,
    INITIAL_EQUITY,
    MultiPhaseSimulationResult,
    _fixed_lot_signals,
    _indicator_predictions,
)
from gold_forecast.v1_robustness import _monte_carlo, _monthly_summary


DEVELOPMENT_START = pd.Timestamp("2025-01-01")
DEVELOPMENT_END = pd.Timestamp("2025-12-31 23:59:59")
VALIDATION_START = pd.Timestamp("2026-01-01")
VALIDATION_END = pd.Timestamp("2026-06-30 23:59:59")
MIN_TRADES = 50
PROFIT_FACTOR_TARGET = 1.30
MAX_DRAWDOWN_PCT = 10.0
MAX_MONTE_CARLO_LOSS_PCT = 10.0


@dataclass(frozen=True)
class RiskControlConfig:
    name: str
    group: str
    max_total_positions: int | None = None
    max_same_direction: int | None = None
    risk_cap_pct: float | None = None
    cooldown_days: int = 0
    daily_loss_limit_pct: float | None = None
    weekly_loss_limit_pct: float | None = None
    time_stop_days: int | None = None
    break_even_activation_usd: float | None = None
    profit_protection_activation_usd: float | None = None
    profit_protection_floor_usd: float | None = None
    spread_limit_points: float | None = None
    soft_drawdown_pct: float | None = None
    hard_drawdown_pct: float | None = None


@dataclass
class RiskPosition:
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
    entry_spread_cost: float
    time_exit_at: pd.Timestamp | None
    peak_profit_usd: float = 0.0
    break_even_active: bool = False
    swap_paid: float = 0.0


def run_v1_risk_control_lab(
    gold_m1: pd.DataFrame,
    signal_daily: pd.DataFrame,
    frozen_payload: dict[str, object],
) -> dict[str, object]:
    data = _prepare_m1(gold_m1)
    _, leaderboard, _ = frozen_payload["v1"]
    best = leaderboard.iloc[0].to_dict()
    development = data.loc[(data.index >= DEVELOPMENT_START) & (data.index <= DEVELOPMENT_END)]
    validation = data.loc[(data.index >= VALIDATION_START) & (data.index <= VALIDATION_END)]
    if development.empty or validation.empty:
        raise ValueError("Dataset M1 development 2025 atau validation 2026H1 belum lengkap.")

    development_signals = _entry_signals_for_period(
        data, signal_daily, best, DEVELOPMENT_START, DEVELOPMENT_END
    )
    validation_signals = _entry_signals_for_period(
        data, signal_daily, best, VALIDATION_START, VALIDATION_END
    )
    spread_p90 = float(development["SpreadPoints"].quantile(0.90))
    spread_twice_median = float(development["SpreadPoints"].median() * 2)
    one_factor, combinations = _candidate_configs(spread_p90, spread_twice_median)
    baseline_config = RiskControlConfig("v1 Exact Baseline", "Baseline")

    baseline_development = _simulate_risk_control(
        development, development_signals, best, baseline_config
    )
    development_rows = [_summary_row(baseline_config, baseline_development, "Development 2025")]
    development_results: dict[str, MultiPhaseSimulationResult] = {
        baseline_config.name: baseline_development
    }
    for config in [*one_factor, *combinations]:
        result = _simulate_risk_control(development, development_signals, best, config)
        development_results[config.name] = result
        development_rows.append(_summary_row(config, result, "Development 2025"))
    development_table = pd.DataFrame(development_rows)
    development_table["Pre-score"] = development_table.apply(
        lambda row: _development_score(row, baseline_development.summary), axis=1
    )

    combination_names = {config.name for config in combinations}
    finalists = (
        development_table[development_table["Kandidat"].isin(combination_names)]
        .sort_values(
            ["Kriteria awal lolos", "Pre-score", "Max drawdown (%)", "Profit factor"],
            ascending=[False, False, True, False],
        )
        .head(3)["Kandidat"]
        .tolist()
    )
    config_by_name = {config.name: config for config in [*one_factor, *combinations]}

    baseline_validation = _simulate_risk_control(
        validation, validation_signals, best, baseline_config
    )
    validation_results: dict[str, MultiPhaseSimulationResult] = {
        baseline_config.name: baseline_validation
    }
    validation_rows = [_summary_row(baseline_config, baseline_validation, "Validation 2026H1")]
    stress_rows: list[dict[str, object]] = []
    monte_carlo_rows: list[dict[str, object]] = []
    monthly_by_candidate: dict[str, pd.DataFrame] = {}
    monte_carlo_frames: dict[str, pd.DataFrame] = {}

    for name in finalists:
        config = config_by_name[name]
        result = _simulate_risk_control(validation, validation_signals, best, config)
        validation_results[name] = result
        validation_rows.append(_summary_row(config, result, "Validation 2026H1"))
        monthly_by_candidate[name] = _monthly_summary(result)
        monte_carlo_frame, monte_carlo_summary = _monte_carlo(result.trades)
        monte_carlo_frames[name] = monte_carlo_frame
        monte_carlo_rows.append({"Kandidat": name, **monte_carlo_summary})
        for spread_multiplier in (1.0, 1.5, 2.0):
            for slippage_points in (2.0, 4.0, 6.0):
                stressed = _simulate_risk_control(
                    validation,
                    validation_signals,
                    best,
                    config,
                    spread_multiplier=spread_multiplier,
                    slippage_points=slippage_points,
                )
                stress_rows.append(
                    {
                        "Kandidat": name,
                        "Spread multiplier": spread_multiplier,
                        "Slippage points/sisi": slippage_points,
                        **_metric_values(stressed),
                    }
                )

    validation_table = pd.DataFrame(validation_rows)
    stress_table = pd.DataFrame(stress_rows)
    monte_carlo_table = pd.DataFrame(monte_carlo_rows)
    decision_table = _decision_table(
        validation_table,
        stress_table,
        monte_carlo_table,
        monthly_by_candidate,
        baseline_validation.summary,
    )
    ranked = decision_table.sort_values(
        ["Lulus seluruh kriteria", "Jumlah kriteria lolos", "Final score"],
        ascending=[False, False, False],
    )
    winner_name = str(ranked.iloc[0]["Kandidat"])
    winner_passed = bool(ranked.iloc[0]["Lulus seluruh kriteria"])
    winner_result = validation_results[winner_name]

    baseline_check = _metric_values(baseline_validation)
    return {
        "methodology": {
            "Baseline lock": "Optimizer v1 live tidak diubah sampai 31 Agustus 2026",
            "Development": "01 Jan 2025 - 31 Des 2025",
            "Validation": "01 Jan 2026 - 30 Jun 2026",
            "Final test": "Forward paper test setelah kandidat dibekukan",
            "Core strategy": str(best["Strategi"]),
            "Spread P90 development": spread_p90,
            "Spread 2x median development": spread_twice_median,
            "Finalists": finalists,
        },
        "criteria": {
            "Growth minimum (%)": 0.0,
            "Max drawdown maksimum (%)": MAX_DRAWDOWN_PCT,
            "Profit factor minimum": PROFIT_FACTOR_TARGET,
            "Monte Carlo rugi maksimum (%)": MAX_MONTE_CARLO_LOSS_PCT,
            "Minimum transaksi": MIN_TRADES,
            "Stress profitable": "9/9",
        },
        "development": development_table,
        "validation": validation_table,
        "stress": stress_table,
        "monte_carlo_summary": monte_carlo_table,
        "decision": decision_table,
        "winner_name": winner_name,
        "winner_status": "LULUS" if winner_passed else "BELUM LULUS",
        "winner_result": _compact_curve(winner_result),
        "winner_monthly": monthly_by_candidate.get(winner_name, _monthly_summary(winner_result)),
        "winner_monte_carlo": monte_carlo_frames.get(winner_name, pd.DataFrame()),
        "winner_config": asdict(config_by_name[winner_name]),
        "baseline_validation": baseline_check,
        "baseline_match_note": (
            "Baseline simulator risk-control direkonsiliasi terhadap Exact v1; perbedaan hanya boleh berasal "
            "dari metadata ringkasan, bukan rule trading."
        ),
    }


def _candidate_configs(
    spread_p90: float,
    spread_twice_median: float,
) -> tuple[list[RiskControlConfig], list[RiskControlConfig]]:
    one_factor = [
        RiskControlConfig("OF-MaxPos-1", "Satu faktor", max_total_positions=1),
        RiskControlConfig("OF-MaxPos-2", "Satu faktor", max_total_positions=2),
        RiskControlConfig("OF-MaxPos-3", "Satu faktor", max_total_positions=3),
        RiskControlConfig("OF-RiskCap-1", "Satu faktor", risk_cap_pct=1.0),
        RiskControlConfig("OF-RiskCap-2", "Satu faktor", risk_cap_pct=2.0),
        RiskControlConfig("OF-RiskCap-3", "Satu faktor", risk_cap_pct=3.0),
        RiskControlConfig("OF-SameSide-1", "Satu faktor", max_same_direction=1),
        RiskControlConfig("OF-SameSide-2", "Satu faktor", max_same_direction=2),
        RiskControlConfig("OF-Cooldown-1", "Satu faktor", cooldown_days=1),
        RiskControlConfig("OF-Cooldown-2", "Satu faktor", cooldown_days=2),
        RiskControlConfig(
            "OF-LossLimit-2-4", "Satu faktor", daily_loss_limit_pct=2.0, weekly_loss_limit_pct=4.0
        ),
        RiskControlConfig(
            "OF-LossLimit-1-3", "Satu faktor", daily_loss_limit_pct=1.0, weekly_loss_limit_pct=3.0
        ),
        RiskControlConfig("OF-TimeStop-3", "Satu faktor", time_stop_days=3),
        RiskControlConfig("OF-TimeStop-5", "Satu faktor", time_stop_days=5),
        RiskControlConfig("OF-TimeStop-7", "Satu faktor", time_stop_days=7),
        RiskControlConfig("OF-BreakEven-10", "Satu faktor", break_even_activation_usd=10.0),
        RiskControlConfig("OF-BreakEven-15", "Satu faktor", break_even_activation_usd=15.0),
        RiskControlConfig(
            "OF-Protect-15-5",
            "Satu faktor",
            profit_protection_activation_usd=15.0,
            profit_protection_floor_usd=5.0,
        ),
        RiskControlConfig("OF-Spread-P90", "Satu faktor", spread_limit_points=spread_p90),
        RiskControlConfig(
            "OF-Spread-2Median", "Satu faktor", spread_limit_points=spread_twice_median
        ),
        RiskControlConfig(
            "OF-Drawdown-8-10", "Satu faktor", soft_drawdown_pct=8.0, hard_drawdown_pct=10.0
        ),
    ]
    combinations = [
        RiskControlConfig(
            "RC-A Defensive",
            "Kombinasi",
            max_total_positions=1,
            risk_cap_pct=1.0,
            cooldown_days=1,
            time_stop_days=5,
            spread_limit_points=spread_p90,
            soft_drawdown_pct=8.0,
            hard_drawdown_pct=10.0,
        ),
        RiskControlConfig(
            "RC-B Balanced",
            "Kombinasi",
            max_total_positions=2,
            max_same_direction=2,
            risk_cap_pct=2.0,
            cooldown_days=1,
            daily_loss_limit_pct=2.0,
            weekly_loss_limit_pct=4.0,
            time_stop_days=5,
            spread_limit_points=spread_p90,
            soft_drawdown_pct=8.0,
            hard_drawdown_pct=10.0,
        ),
        RiskControlConfig(
            "RC-C Break-Even",
            "Kombinasi",
            max_total_positions=2,
            max_same_direction=2,
            risk_cap_pct=2.0,
            cooldown_days=1,
            daily_loss_limit_pct=2.0,
            weekly_loss_limit_pct=4.0,
            time_stop_days=5,
            break_even_activation_usd=15.0,
            spread_limit_points=spread_p90,
            soft_drawdown_pct=8.0,
            hard_drawdown_pct=10.0,
        ),
        RiskControlConfig(
            "RC-D Profit Protection",
            "Kombinasi",
            max_total_positions=2,
            max_same_direction=2,
            risk_cap_pct=2.0,
            cooldown_days=1,
            daily_loss_limit_pct=2.0,
            weekly_loss_limit_pct=4.0,
            time_stop_days=5,
            profit_protection_activation_usd=15.0,
            profit_protection_floor_usd=5.0,
            spread_limit_points=spread_p90,
            soft_drawdown_pct=8.0,
            hard_drawdown_pct=10.0,
        ),
        RiskControlConfig(
            "RC-E Low Frequency",
            "Kombinasi",
            max_total_positions=1,
            cooldown_days=2,
            time_stop_days=7,
            spread_limit_points=spread_p90,
            soft_drawdown_pct=8.0,
            hard_drawdown_pct=10.0,
        ),
    ]
    return one_factor, combinations


def _entry_signals_for_period(
    data: pd.DataFrame,
    daily: pd.DataFrame,
    best: dict[str, object],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    predictions = _indicator_predictions(
        daily,
        str(best["Mode"]),
        int(best["Fast MA"]),
        int(best["Slow MA"]),
        int(best["Momentum hari"]),
        float(best["Threshold entry (%)"]),
        test_start=start,
        test_end=end,
    )
    daily_signals = _fixed_lot_signals(predictions, float(best["Lot"]))
    last_bar_by_day = data.groupby(data.index.normalize()).apply(lambda frame: frame.index[-1])
    rows: list[dict[str, object]] = []
    for signal_date, signal in daily_signals.iterrows():
        day = pd.Timestamp(signal_date).normalize()
        if day not in last_bar_by_day.index:
            continue
        location = data.index.searchsorted(last_bar_by_day.loc[day], side="right")
        if location >= len(data.index):
            continue
        entry_time = data.index[location]
        if entry_time < start or entry_time > end:
            continue
        reference = float(daily.loc[signal_date, "Close"])
        prediction = float(signal["prediction"])
        rows.append(
            {
                "entry_time": entry_time,
                "signal_date": pd.Timestamp(signal_date),
                "prediction": prediction,
                "expected_change_pct": (prediction / reference - 1) * 100,
                "lot": float(signal["lot_size"]),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=["signal_date", "prediction", "expected_change_pct", "lot"]
        )
    return pd.DataFrame(rows).set_index("entry_time").sort_index()


def _simulate_risk_control(
    data: pd.DataFrame,
    signals: pd.DataFrame,
    best: dict[str, object],
    config: RiskControlConfig,
    *,
    spread_multiplier: float = 1.0,
    slippage_points: float = SLIPPAGE_POINTS,
) -> MultiPhaseSimulationResult:
    take_profit = float(best["TP (USD)"])
    stop_loss = float(best["SL (USD)"])
    threshold = float(best["Threshold entry (%)"])
    baseline_max_buy = int(best.get("Max BUY", 8))
    baseline_max_sell = int(best.get("Max SELL", 10))
    close_on_target = bool(best.get("Close-all target equity", True))
    phase_growth = float(best.get("Target fase (%)", 20.0)) / 100

    balance = INITIAL_EQUITY
    phase_start = INITIAL_EQUITY
    target_equity = phase_start * (1 + phase_growth)
    phase = 1
    next_id = 1
    positions: list[RiskPosition] = []
    trades: list[dict[str, object]] = []
    curve: list[dict[str, object]] = []
    phase_rows: list[dict[str, object]] = []
    phase_curve_start = 0
    phase_trade_start = 0
    previous_day: pd.Timestamp | None = None
    current_week: tuple[int, int] | None = None
    day_start_equity = INITIAL_EQUITY
    week_start_equity = INITIAL_EQUITY
    daily_realized = 0.0
    weekly_realized = 0.0
    cooldown_until = pd.Timestamp.min
    soft_pause_until = pd.Timestamp.min
    equity_peak = INITIAL_EQUITY
    hard_stopped = False
    soft_breached = False
    blocked: dict[str, int] = {}
    signal_map = {pd.Timestamp(index): row for index, row in signals.iterrows()}

    for candle in data.itertuples():
        timestamp = pd.Timestamp(candle.Index)
        trading_day = timestamp.normalize()
        iso = trading_day.isocalendar()
        week_key = (int(iso.year), int(iso.week))
        spread_points = float(candle.SpreadPoints) * spread_multiplier
        spread = max(0.0, spread_points * POINT_SIZE)
        bid_high, bid_low, bid_close = (
            float(candle.High),
            float(candle.Low),
            float(candle.Close),
        )
        ask_high, ask_low, ask_close = bid_high + spread, bid_low + spread, bid_close + spread

        if previous_day is None or trading_day != previous_day:
            if previous_day is not None:
                for position in positions:
                    if position.direction == "BUY":
                        swap = BUY_SWAP_PER_001_LOT * (position.lot / 0.01)
                        position.swap_paid += swap
                        balance -= swap
            day_start_equity = balance + sum(
                _mark_pnl(position, bid_close, ask_close) for position in positions
            )
            daily_realized = 0.0
            previous_day = trading_day
        if current_week is None or week_key != current_week:
            week_start_equity = balance + sum(
                _mark_pnl(position, bid_close, ask_close) for position in positions
            )
            weekly_realized = 0.0
            current_week = week_key

        still_open: list[RiskPosition] = []
        for position in positions:
            exit_detail = _risk_exit_decision(
                position, timestamp, bid_high, bid_low, bid_close, ask_high, ask_low, ask_close, config
            )
            if exit_detail is None:
                still_open.append(position)
                continue
            raw_exit, reason = exit_detail
            exit_price = (
                raw_exit - POINT_SIZE * slippage_points
                if position.direction == "BUY"
                else raw_exit + POINT_SIZE * slippage_points
            )
            trade = _risk_trade_row(
                position, timestamp, exit_price, reason, balance, spread, slippage_points
            )
            balance += float(trade["Gross P/L"])
            trade["Balance"] = balance
            trades.append(trade)
            daily_realized += float(trade["Net P/L"])
            weekly_realized += float(trade["Net P/L"])
            if reason == "SL tersentuh" and config.cooldown_days:
                cooldown_until = max(
                    cooldown_until,
                    trading_day + pd.offsets.BDay(config.cooldown_days),
                )
        positions = still_open

        unrealized = sum(_mark_pnl(position, bid_close, ask_close) for position in positions)
        equity = balance + unrealized
        equity_peak = max(equity_peak, equity)
        drawdown_pct = (equity_peak - equity) / equity_peak * 100 if equity_peak else 0.0
        if (
            config.soft_drawdown_pct is not None
            and drawdown_pct >= config.soft_drawdown_pct
            and not soft_breached
        ):
            soft_pause_until = max(soft_pause_until, trading_day + pd.offsets.BDay(5))
            soft_breached = True
        elif config.soft_drawdown_pct is not None and drawdown_pct < config.soft_drawdown_pct:
            soft_breached = False
        if (
            not hard_stopped
            and config.hard_drawdown_pct is not None
            and drawdown_pct >= config.hard_drawdown_pct
        ):
            for position in positions:
                raw_exit = bid_close if position.direction == "BUY" else ask_close
                exit_price = (
                    raw_exit - POINT_SIZE * slippage_points
                    if position.direction == "BUY"
                    else raw_exit + POINT_SIZE * slippage_points
                )
                trade = _risk_trade_row(
                    position,
                    timestamp,
                    exit_price,
                    "Hard drawdown stop",
                    balance,
                    spread,
                    slippage_points,
                )
                balance += float(trade["Gross P/L"])
                trade["Balance"] = balance
                trades.append(trade)
            positions = []
            unrealized = 0.0
            equity = balance
            hard_stopped = True

        signal = signal_map.get(timestamp)
        if signal is not None:
            expected = float(signal["expected_change_pct"])
            direction = "BUY" if expected >= threshold else "SELL" if expected <= -threshold else None
            reason = _entry_block_reason(
                direction,
                positions,
                stop_loss,
                equity,
                trading_day,
                cooldown_until,
                soft_pause_until,
                hard_stopped,
                daily_realized,
                day_start_equity,
                weekly_realized,
                week_start_equity,
                spread_points,
                baseline_max_buy,
                baseline_max_sell,
                config,
            )
            if reason is None and direction is not None:
                lot = float(signal["lot"])
                units = lot * CONTRACT_OUNCES_PER_LOT
                if direction == "BUY":
                    entry_price = ask_close + POINT_SIZE * slippage_points
                    entry_spread_cost = spread * units
                else:
                    entry_price = bid_close - POINT_SIZE * slippage_points
                    entry_spread_cost = 0.0
                time_exit_at = (
                    trading_day + pd.offsets.BDay(config.time_stop_days)
                    if config.time_stop_days is not None
                    else None
                )
                positions.append(
                    RiskPosition(
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
                        entry_spread_cost=entry_spread_cost,
                        time_exit_at=time_exit_at,
                    )
                )
                next_id += 1
            elif reason is not None:
                blocked[reason] = blocked.get(reason, 0) + 1

        unrealized = sum(_mark_pnl(position, bid_close, ask_close) for position in positions)
        equity = balance + unrealized
        equity_peak = max(equity_peak, equity)
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
                exit_price = (
                    raw_exit - POINT_SIZE * slippage_points
                    if position.direction == "BUY"
                    else raw_exit + POINT_SIZE * slippage_points
                )
                trade = _risk_trade_row(
                    position,
                    timestamp,
                    exit_price,
                    "Target equity tercapai",
                    balance,
                    spread,
                    slippage_points,
                )
                balance += float(trade["Gross P/L"])
                trade["Balance"] = balance
                trades.append(trade)
            positions = []
            curve[-1].update(
                {
                    "Balance": balance,
                    "Equity": balance,
                    "Unrealized P/L": 0.0,
                    "Open BUY": 0,
                    "Open SELL": 0,
                    "Open total": 0,
                }
            )
            phase_rows.append(
                _phase_summary(
                    phase,
                    phase_start,
                    target_equity,
                    trades[phase_trade_start:],
                    curve[phase_curve_start:],
                    True,
                    timestamp,
                )
            )
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
            exit_price = (
                raw_exit - POINT_SIZE * slippage_points
                if position.direction == "BUY"
                else raw_exit + POINT_SIZE * slippage_points
            )
            trade = _risk_trade_row(
                position,
                timestamp,
                exit_price,
                "Akhir periode data",
                balance,
                spread,
                slippage_points,
            )
            balance += float(trade["Gross P/L"])
            trade["Balance"] = balance
            trades.append(trade)
        if curve:
            curve[-1].update(
                {
                    "Balance": balance,
                    "Equity": balance,
                    "Unrealized P/L": 0.0,
                    "Open BUY": 0,
                    "Open SELL": 0,
                    "Open total": 0,
                }
            )

    if phase_curve_start < len(curve):
        phase_rows.append(
            _phase_summary(
                phase,
                phase_start,
                target_equity,
                trades[phase_trade_start:],
                curve[phase_curve_start:],
                False,
                pd.NaT,
            )
        )
    trades_frame = pd.DataFrame(trades)
    curve_frame = pd.DataFrame(curve).set_index("Tanggal") if curve else pd.DataFrame()
    phases_frame = pd.DataFrame(phase_rows)
    summary = _overall_summary(trades_frame, curve_frame, phases_frame)
    summary.update(
        {
            "Kandidat": config.name,
            "Hard stop aktif": hard_stopped,
            "Entry diblokir": float(sum(blocked.values())),
            "Detail blokir": blocked,
        }
    )
    return MultiPhaseSimulationResult(summary, phases_frame, trades_frame, curve_frame)


def _entry_block_reason(
    direction: str | None,
    positions: list[RiskPosition],
    stop_loss: float,
    equity: float,
    trading_day: pd.Timestamp,
    cooldown_until: pd.Timestamp,
    soft_pause_until: pd.Timestamp,
    hard_stopped: bool,
    daily_realized: float,
    day_start_equity: float,
    weekly_realized: float,
    week_start_equity: float,
    spread_points: float,
    baseline_max_buy: int,
    baseline_max_sell: int,
    config: RiskControlConfig,
) -> str | None:
    if direction is None:
        return "Sinyal netral"
    if hard_stopped:
        return "Hard drawdown stop"
    if trading_day < cooldown_until:
        return "Cooldown setelah SL"
    if trading_day < soft_pause_until:
        return "Soft drawdown pause"
    if config.spread_limit_points is not None and spread_points > config.spread_limit_points:
        return "Spread ekstrem"
    if config.max_total_positions is not None and len(positions) >= config.max_total_positions:
        return "Batas total posisi"
    same_direction = sum(position.direction == direction for position in positions)
    direction_limit = baseline_max_buy if direction == "BUY" else baseline_max_sell
    if same_direction >= direction_limit:
        return "Batas posisi baseline"
    if config.max_same_direction is not None and same_direction >= config.max_same_direction:
        return "Batas konsentrasi arah"
    if config.risk_cap_pct is not None:
        planned_risk = sum(position.stop_loss_usd for position in positions) + stop_loss
        if planned_risk > equity * config.risk_cap_pct / 100 + 1e-9:
            return "Total risk cap"
    if (
        config.daily_loss_limit_pct is not None
        and daily_realized <= -(day_start_equity * config.daily_loss_limit_pct / 100)
    ):
        return "Daily loss limit"
    if (
        config.weekly_loss_limit_pct is not None
        and weekly_realized <= -(week_start_equity * config.weekly_loss_limit_pct / 100)
    ):
        return "Weekly loss limit"
    return None


def _risk_exit_decision(
    position: RiskPosition,
    timestamp: pd.Timestamp,
    bid_high: float,
    bid_low: float,
    bid_close: float,
    ask_high: float,
    ask_low: float,
    ask_close: float,
    config: RiskControlConfig,
) -> tuple[float, str] | None:
    units = position.lot * CONTRACT_OUNCES_PER_LOT
    if position.direction == "BUY":
        high, low, close = bid_high, bid_low, bid_close
        stop = position.entry_price - position.stop_loss_usd / units
        if position.break_even_active:
            stop = max(stop, position.entry_price)
        if (
            config.profit_protection_activation_usd is not None
            and position.peak_profit_usd >= config.profit_protection_activation_usd
        ):
            stop = max(
                stop,
                position.entry_price + float(config.profit_protection_floor_usd or 0.0) / units,
            )
        if low <= stop:
            reason = "Break-even" if stop == position.entry_price else "SL tersentuh"
            if stop > position.entry_price:
                reason = "Profit protection"
            return stop, reason
        target = position.entry_price + position.take_profit_usd / units
        if high >= target:
            return target, "TP tersentuh"
        position.peak_profit_usd = max(position.peak_profit_usd, (high - position.entry_price) * units)
    else:
        high, low, close = ask_high, ask_low, ask_close
        stop = position.entry_price + position.stop_loss_usd / units
        if position.break_even_active:
            stop = min(stop, position.entry_price)
        if (
            config.profit_protection_activation_usd is not None
            and position.peak_profit_usd >= config.profit_protection_activation_usd
        ):
            stop = min(
                stop,
                position.entry_price - float(config.profit_protection_floor_usd or 0.0) / units,
            )
        if high >= stop:
            reason = "Break-even" if stop == position.entry_price else "SL tersentuh"
            if stop < position.entry_price:
                reason = "Profit protection"
            return stop, reason
        target = position.entry_price - position.take_profit_usd / units
        if low <= target:
            return target, "TP tersentuh"
        position.peak_profit_usd = max(position.peak_profit_usd, (position.entry_price - low) * units)
    if (
        config.break_even_activation_usd is not None
        and position.peak_profit_usd >= config.break_even_activation_usd
    ):
        position.break_even_active = True
    if position.time_exit_at is not None and timestamp >= position.time_exit_at:
        return close, "Time stop"
    return None


def _mark_pnl(position: RiskPosition, bid: float, ask: float) -> float:
    price = bid if position.direction == "BUY" else ask
    units = position.lot * CONTRACT_OUNCES_PER_LOT
    return (
        (price - position.entry_price) * units
        if position.direction == "BUY"
        else (position.entry_price - price) * units
    )


def _risk_trade_row(
    position: RiskPosition,
    timestamp: pd.Timestamp,
    exit_price: float,
    reason: str,
    balance: float,
    exit_spread: float,
    slippage_points: float,
) -> dict[str, object]:
    units = position.lot * CONTRACT_OUNCES_PER_LOT
    gross = (
        (exit_price - position.entry_price) * units
        if position.direction == "BUY"
        else (position.entry_price - exit_price) * units
    )
    spread_cost = position.entry_spread_cost + (
        exit_spread * units if position.direction == "SELL" else 0.0
    )
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
        "Biaya slippage": 2 * slippage_points * POINT_SIZE * units,
        "Gross P/L": gross,
        "Swap": -position.swap_paid,
        "Net P/L": gross - position.swap_paid,
        "Balance": balance,
        "Batas posisi": np.nan,
    }


def _metric_values(result: MultiPhaseSimulationResult) -> dict[str, float]:
    summary = result.summary
    drawdown_pct = float(summary["Max drawdown"]) / float(summary["Modal awal"]) * 100
    return {
        "Equity akhir": float(summary["Equity akhir"]),
        "Growth (%)": float(summary["Growth total"]),
        "Max drawdown": float(summary["Max drawdown"]),
        "Max drawdown (%)": drawdown_pct,
        "Profit factor": float(summary["Profit factor"]),
        "Transaksi": float(summary["Jumlah transaksi"]),
        "Win rate (%)": float(summary["Win rate"]),
        "Max open posisi": float(summary["Max open posisi"]),
        "Total swap": float(summary["Total swap"]),
        "Biaya spread": float(summary["Biaya spread"]),
        "Biaya slippage": float(summary["Biaya slippage"]),
        "Entry diblokir": float(summary.get("Entry diblokir", 0.0)),
    }


def _summary_row(
    config: RiskControlConfig,
    result: MultiPhaseSimulationResult,
    period: str,
) -> dict[str, object]:
    metrics = _metric_values(result)
    initial_pass = bool(
        metrics["Growth (%)"] > 0
        and metrics["Max drawdown (%)"] <= MAX_DRAWDOWN_PCT
        and metrics["Profit factor"] >= PROFIT_FACTOR_TARGET
        and metrics["Transaksi"] >= MIN_TRADES
    )
    return {
        "Kandidat": config.name,
        "Kelompok": config.group,
        "Periode": period,
        **metrics,
        "Kriteria awal lolos": initial_pass,
        "Konfigurasi": _config_label(config),
    }


def _config_label(config: RiskControlConfig) -> str:
    values = []
    for key, value in asdict(config).items():
        if key in {"name", "group"} or value in {None, 0}:
            continue
        values.append(f"{key}={value}")
    return "Baseline tanpa overlay" if not values else " | ".join(values)


def _development_score(row: pd.Series, baseline: dict[str, object]) -> float:
    baseline_trades = max(float(baseline["Jumlah transaksi"]), 1.0)
    dd_score = 30 * max(0.0, 1 - float(row["Max drawdown (%)"]) / 20)
    pf_score = 25 * min(float(row["Profit factor"]) / PROFIT_FACTOR_TARGET, 1.0)
    growth_score = 20 * max(0.0, min(float(row["Growth (%)"]) / 50, 1.0))
    trade_score = 15 * min(float(row["Transaksi"]) / baseline_trades, 1.0)
    cost_score = 10 * max(0.0, 1 - float(row["Biaya spread"]) / 500)
    return dd_score + pf_score + growth_score + trade_score + cost_score


def _decision_table(
    validation: pd.DataFrame,
    stress: pd.DataFrame,
    monte_carlo: pd.DataFrame,
    monthly: dict[str, pd.DataFrame],
    baseline: dict[str, object],
) -> pd.DataFrame:
    rows = []
    finalist_rows = validation[validation["Kelompok"].eq("Kombinasi")]
    for _, candidate in finalist_rows.iterrows():
        name = str(candidate["Kandidat"])
        candidate_stress = stress[stress["Kandidat"].eq(name)]
        mc = monte_carlo[monte_carlo["Kandidat"].eq(name)].iloc[0]
        month = monthly[name]
        growth_pass = bool(candidate["Growth (%)"] > 0)
        drawdown_pass = bool(candidate["Max drawdown (%)"] <= MAX_DRAWDOWN_PCT)
        pf_pass = bool(candidate["Profit factor"] >= PROFIT_FACTOR_TARGET)
        mc_loss = float(mc["Probabilitas equity akhir < modal awal (%)"])
        mc_pass = mc_loss <= MAX_MONTE_CARLO_LOSS_PCT
        stress_pass = bool(len(candidate_stress) == 9 and (candidate_stress["Growth (%)"] > 0).all())
        trades_pass = bool(candidate["Transaksi"] >= MIN_TRADES)
        criteria = [growth_pass, drawdown_pass, pf_pass, mc_pass, stress_pass, trades_pass]
        positive_months = int((month["Net P/L"] > 0).sum())
        baseline_growth = max(float(baseline["Growth total"]), 1.0)
        score = (
            30 * max(0.0, 1 - float(candidate["Max drawdown (%)"]) / MAX_DRAWDOWN_PCT)
            + 25 * min(float(candidate["Profit factor"]) / PROFIT_FACTOR_TARGET, 1.0)
            + 20 * max(0.0, 1 - mc_loss / MAX_MONTE_CARLO_LOSS_PCT)
            + 15 * positive_months / max(len(month), 1)
            + 10 * max(0.0, min(float(candidate["Growth (%)"]) / baseline_growth, 1.0))
        )
        rows.append(
            {
                "Kandidat": name,
                "Growth positif": growth_pass,
                "Drawdown <= 10%": drawdown_pass,
                "Profit factor >= 1.30": pf_pass,
                "Monte Carlo rugi <= 10%": mc_pass,
                "Stress profitable 9/9": stress_pass,
                "Transaksi >= 50": trades_pass,
                "Jumlah kriteria lolos": sum(criteria),
                "Lulus seluruh kriteria": all(criteria),
                "Bulan positif": positive_months,
                "Monte Carlo rugi (%)": mc_loss,
                "Final score": score,
            }
        )
    return pd.DataFrame(rows)
