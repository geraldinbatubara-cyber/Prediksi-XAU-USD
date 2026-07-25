from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from gold_forecast.exact_broker_oos import (
    POINT_SIZE,
    SLIPPAGE_POINTS,
    _overall_summary,
    _prepare_m1,
)
from gold_forecast.simulation import CONTRACT_OUNCES_PER_LOT
from gold_forecast.strategy_optimizer import (
    BUY_SWAP_PER_001_LOT,
    INITIAL_EQUITY,
    MultiPhaseSimulationResult,
)
from gold_forecast.v1_directional_specialization import (
    _ledger_metric_values,
    _monte_carlo_summary,
    _trades_in_period,
)
from gold_forecast.v1_risk_control import _metric_values
from gold_forecast.v1_sell_specialist import (
    DEVELOPMENT_END,
    DEVELOPMENT_START,
    LOCKED_END,
    LOCKED_START,
    REFERENCE_END,
    REFERENCE_START,
)
from gold_forecast.v1_sideways_defense import _regime_features
from gold_forecast.v1_sideways_specialist import (
    _mean_reversion_opportunities,
    _profit_concentration,
    _range_quality_frame,
    _train_outcome_model,
)
from gold_forecast.v1_sideways_specialist_v2 import (
    _augment_opportunities,
    _candidate_signals as _v2_candidate_signals,
    _directional_thresholds,
    _session_selection,
    _train_binary_model,
)
from gold_forecast.v1_sideways_specialist_v3 import (
    _build_position_states,
    _enrich_signals,
    _fold_evaluation,
    _periods,
    _price_levels,
    _result_table,
    _state_audit,
    _state_classification_tables,
    _train_state_model,
)
from gold_forecast.v1_sideways_specialist_v5 import _m15_atr
from gold_forecast.v1_trend_strength_stability import _extended_data_audit
from gold_forecast.v1_unified_benchmark import _unified_best


@dataclass(frozen=True)
class ExitPolicy:
    name: str
    management_interval: int = 15
    break_even_r: float | None = None
    trailing_r: float | None = None
    trailing_tp_fraction: float | None = None
    hazard_exit_score: int | None = None
    hazard_warning_score: int | None = None
    warning_tighten_r: float | None = None
    hazard_confirmations: int = 1


HARD_EXIT_CONTROL = ExitPolicy("Hard Exit Control")

V5B_POLICIES = (
    HARD_EXIT_CONTROL,
    ExitPolicy("M15 BE 1R", break_even_r=1.0),
    ExitPolicy("M1-Close BE 1R", management_interval=1, break_even_r=1.0),
    ExitPolicy("M15 Trail 2R", trailing_r=2.0),
    ExitPolicy("M1-Close Trail 2R", management_interval=1, trailing_r=2.0),
    ExitPolicy(
        "M1-Close BE + Trail",
        management_interval=1,
        break_even_r=1.0,
        trailing_r=2.0,
    ),
)

V5C_POLICIES = (
    HARD_EXIT_CONTROL,
    ExitPolicy("ATR Trail 0.50R", trailing_r=0.50),
    ExitPolicy("ATR Trail 0.75R", trailing_r=0.75),
    ExitPolicy("ATR Trail 1.00R", trailing_r=1.00),
    ExitPolicy("ATR Trail 70% TP", trailing_tp_fraction=0.70),
    ExitPolicy("ATR Trail 2.00R Reference", trailing_r=2.00),
)

V5D_POLICIES = (
    HARD_EXIT_CONTROL,
    ExitPolicy("Immediate Exit Score 5", hazard_exit_score=5),
    ExitPolicy("Immediate Exit Score 6", hazard_exit_score=6),
    ExitPolicy("Immediate Exit Score 7", hazard_exit_score=7),
    ExitPolicy("Warning 5-7 / Exit 8", hazard_warning_score=5, hazard_exit_score=8),
    ExitPolicy(
        "Warning Tighten / Exit 8",
        hazard_warning_score=5,
        warning_tighten_r=0.50,
        hazard_exit_score=8,
    ),
    ExitPolicy(
        "Two M15 Confirm Score 5",
        hazard_exit_score=5,
        hazard_confirmations=2,
    ),
)


def run_v1_sideways_specialist_v5_extensions(
    gold_m1: pd.DataFrame,
    frozen_payload: dict[str, object],
    v5_payload: dict[str, object] | None = None,
) -> dict[str, object]:
    (
        data,
        entry_signals,
        state_frame,
        atr_m15,
        model_1h,
        model_3h,
    ) = _prepare_context(gold_m1, frozen_payload)
    common = {
        "data_audit": _extended_data_audit(data),
        "state_audit": _state_audit(state_frame),
        "model_1h_selection": model_1h["selection"],
        "model_3h_selection": model_3h["selection"],
    }
    classification = _state_classification_tables(
        state_frame, model_1h, model_3h
    )
    common.update(
        {
            "classification_selection": classification["selection"],
            "classification_locked": classification["locked"],
            "classification_reference": classification["reference"],
        }
    )
    threshold_1h = model_1h["threshold"]
    threshold_3h = model_3h["threshold"]
    v5_reference = _v5_reference(v5_payload)

    v5b = _run_lab(
        data,
        entry_signals,
        state_frame,
        atr_m15,
        V5B_POLICIES,
        threshold_1h,
        threshold_3h,
        "v1 Sideways Specialist Lab v5B - Exit Timing Attribution",
        (
            "M1-close adalah proxy paling rapat yang dapat dibuktikan oleh dataset. "
            "Ia bukan tick historis dan setiap perubahan stop baru berlaku pada "
            "candle M1 berikutnya."
        ),
        common,
        v5_reference,
    )
    v5c = _run_lab(
        data,
        entry_signals,
        state_frame,
        atr_m15,
        V5C_POLICIES,
        threshold_1h,
        threshold_3h,
        "v1 Sideways Specialist Lab v5C - Feasible Trailing Activation",
        (
            "Hanya ambang aktivasi ATR trailing yang berubah. Evaluasi tetap pada "
            "close M15 dan jarak trailing tetap 1.5 ATR M15."
        ),
        common,
        v5_reference,
    )
    v5d = _run_lab(
        data,
        entry_signals,
        state_frame,
        atr_m15,
        V5D_POLICIES,
        threshold_1h,
        threshold_3h,
        "v1 Sideways Specialist Lab v5D - Hazard Severity Zones",
        (
            "Hanya keputusan berdasarkan weighted hazard score yang berubah. "
            "Warning tighten memindahkan SL dari -1R ke -0.5R; hazard selalu "
            "dievaluasi pada close M15."
        ),
        common,
        v5_reference,
    )
    return {"v5b": v5b, "v5c": v5c, "v5d": v5d}


def _prepare_context(gold_m1, frozen_payload):
    data = _prepare_m1(gold_m1)
    _, leaderboard, _ = frozen_payload["v1"]
    best = {
        **_unified_best(leaderboard.iloc[0].to_dict()),
        "Close-all target equity": False,
        "Max BUY": 1,
        "Max SELL": 1,
    }
    spread_limit = float(
        data.loc[DEVELOPMENT_START:DEVELOPMENT_END, "SpreadPoints"].quantile(0.90)
    )
    features, h1, m15 = _regime_features(data)
    range_frame = _range_quality_frame(features, h1)
    opportunities = _augment_opportunities(
        data,
        range_frame,
        _mean_reversion_opportunities(data, range_frame, m15, spread_limit),
    )
    outcome_model, _ = _train_outcome_model(opportunities)
    persistence_model = _train_binary_model(
        opportunities, "persistence_12h", "Range Persistence", 101
    )
    entry_hazard_model = _train_binary_model(
        opportunities, "adverse_breakout_6h", "Adverse Breakout Hazard", 111
    )
    directional_thresholds, _ = _directional_thresholds(
        opportunities, outcome_model["probability"]
    )
    selected_sessions, _ = _session_selection(opportunities)
    signals, _ = _v2_candidate_signals(
        opportunities,
        outcome_model,
        persistence_model,
        entry_hazard_model,
        directional_thresholds,
        selected_sessions,
        best,
    )
    entry_signals = signals["Breakout Hazard Gate"].copy()
    source = opportunities.loc[
        ~opportunities.index.duplicated(keep="first")
    ].reindex(entry_signals.index)
    entry_signals = _enrich_signals(entry_signals, source)
    state_frame = _build_position_states(data, range_frame, entry_signals)
    model_1h = _train_state_model(state_frame, "adverse_before_tp_1h", "1 jam", 121)
    model_3h = _train_state_model(state_frame, "adverse_before_tp_3h", "3 jam", 131)
    state_frame["hazard_1h"] = model_1h["probability"]
    state_frame["hazard_3h"] = model_3h["probability"]
    return data, entry_signals, state_frame, _m15_atr(data), model_1h, model_3h


def _run_lab(
    data,
    signals,
    states,
    atr_m15,
    policies,
    threshold_1h,
    threshold_3h,
    name,
    isolation,
    common,
    v5_reference,
):
    development_data = data.loc[DEVELOPMENT_START:DEVELOPMENT_END]
    development_signals = signals.loc[DEVELOPMENT_START:DEVELOPMENT_END]
    reference_data = data.loc[REFERENCE_START:REFERENCE_END]
    reference_signals = signals.loc[REFERENCE_START:REFERENCE_END]
    development_results = {
        policy.name: _simulate_policy(
            development_data,
            development_signals,
            states,
            atr_m15,
            policy,
            threshold_1h,
            threshold_3h,
        )
        for policy in policies
    }
    reference_results = {
        policy.name: _simulate_policy(
            reference_data,
            reference_signals,
            states,
            atr_m15,
            policy,
            threshold_1h,
            threshold_3h,
        )
        for policy in policies
    }
    development = _result_table(development_results, signals)
    reference = _result_table(reference_results, signals)
    periods = _period_validation(development_results, signals)
    folds = _fold_evaluation(development_results)
    monte_carlo = _monte_carlo_summary(development_results)
    concentration = _profit_concentration(development_results)
    attribution = _exit_attribution(
        development_results, HARD_EXIT_CONTROL.name
    )
    exit_reasons = _exit_reason_summary(development_results)
    candidates = tuple(policy.name for policy in policies)
    decisions = _decision_table(
        candidates, development, periods, folds, monte_carlo, concentration
    )
    ranking = _selection_ranking(
        candidates,
        development,
        reference,
        periods,
        decisions,
        attribution,
    )
    eligible = ranking.loc[ranking["Selection eligible"]]
    winner = str(eligible.iloc[0]["Kandidat"]) if not eligible.empty else ""
    policy_map = {policy.name: policy for policy in policies}
    stress = (
        _stress_test(
            development_data,
            development_signals,
            states,
            atr_m15,
            policy_map[winner],
            threshold_1h,
            threshold_3h,
        )
        if winner
        else pd.DataFrame()
    )
    stress_passed = int((stress["Growth (%)"] > 0).sum()) if not stress.empty else 0
    decisions = decisions.copy()
    decisions["Stress profitable"] = (
        decisions["Kandidat"].map({winner: stress_passed}) if winner else np.nan
    )
    winner_passed = False
    if winner:
        decision = decisions.loc[decisions["Kandidat"].eq(winner)].iloc[0]
        winner_passed = bool(decision["Lulus"]) and stress_passed >= 7
        ranking.loc[
            ranking["Kandidat"].eq(winner), "Lulus termasuk stress"
        ] = winner_passed
    return {
        "methodology": {
            "Name": name,
            "Frozen entry": (
                "Breakout Hazard Gate v2, arah, lot 0.01, TP/SL, spread, "
                "slippage, swap, time stop 12 jam, dan satu posisi tetap."
            ),
            "Isolation": isolation,
            "No look-ahead": (
                "Stop yang diubah setelah close M1/M15 baru berlaku pada candle "
                "M1 berikutnya. Jika TP dan SL ada dalam candle sama, SL lebih dulu."
            ),
            "Train": "Position states dari entry 2022",
            "Calibration": "Probability 2023H1 dan threshold 2023H2",
            "Selection": "Trade entry 2024 saja",
            "Locked confirmation": "Trade entry 2025",
            "Historical reference": "Trade entry 2026H1; tidak memilih pemenang",
            "Live trading lock": "Tidak mengubah strategi atau ledger Paper Live Trading.",
        },
        **common,
        "development": development,
        "period_validation": periods,
        "historical_reference": reference,
        "folds": folds,
        "monte_carlo_summary": monte_carlo,
        "profit_concentration": concentration,
        "exit_attribution": attribution,
        "exit_reason_summary": exit_reasons,
        "stress_summary": stress,
        "decisions": decisions,
        "ranking": ranking,
        "winner": winner,
        "control": HARD_EXIT_CONTROL.name,
        "selection_status": (
            f"Eligible: {winner}"
            if winner
            else "Tidak ada kandidat eligible pada model selection 2024"
        ),
        "winner_passed": winner_passed,
        "v5a_reference": v5_reference,
    }


def _simulate_policy(
    data,
    signals,
    states,
    atr_m15,
    policy,
    threshold_1h,
    threshold_3h,
    *,
    spread_multiplier=1.0,
    slippage_points=SLIPPAGE_POINTS,
):
    balance = INITIAL_EQUITY
    trades = []
    curve = []
    busy_until = pd.Timestamp.min
    blocked = 0
    units = 0.01 * CONTRACT_OUNCES_PER_LOT

    for entry_time, signal in signals.sort_index().iterrows():
        entry_time = pd.Timestamp(entry_time)
        if entry_time <= busy_until or entry_time not in data.index:
            blocked += 1
            continue
        path = data.loc[entry_time : entry_time + pd.Timedelta(hours=12)]
        if path.empty:
            continue
        direction = str(signal["direction"])
        first = path.iloc[0]
        spread = float(first["SpreadPoints"]) * POINT_SIZE * spread_multiplier
        bid_entry = float(first["Close"])
        entry = (
            bid_entry + spread + slippage_points * POINT_SIZE
            if direction == "BUY"
            else bid_entry - slippage_points * POINT_SIZE
        )
        tp_price, initial_sl = _price_levels(
            entry, direction, float(signal["tp_usd"]), float(signal["sl_usd"])
        )
        active_sl = initial_sl
        stop_mode = "Initial SL"
        peak = 0.0
        max_adverse = 0.0
        previous_hazard = 0.0
        high_count = 0
        structural_count = 0
        hazard_confirm_count = 0
        max_hazard_score = 0
        break_even_activated = False
        trailing_activated = False
        warning_count = 0
        exit_time = pd.Timestamp(path.index[-1])
        exit_price = None
        reason = "Time stop"
        exit_hazard = np.nan

        for offset, candle in enumerate(path.itertuples()):
            timestamp = pd.Timestamp(candle.Index)
            candle_spread = (
                float(candle.SpreadPoints) * POINT_SIZE * spread_multiplier
            )
            bid_high = float(candle.High)
            bid_low = float(candle.Low)
            bid_close = float(candle.Close)
            ask_high = bid_high + candle_spread
            ask_low = bid_low + candle_spread
            ask_close = bid_close + candle_spread
            if direction == "BUY":
                if bid_low <= active_sl:
                    exit_price = active_sl - slippage_points * POINT_SIZE
                    reason = (
                        f"{stop_mode} tersentuh"
                        if stop_mode != "Initial SL"
                        else "SL tersentuh"
                    )
                elif bid_high >= tp_price:
                    exit_price = tp_price - slippage_points * POINT_SIZE
                    reason = "TP tersentuh"
                mark = bid_close
                peak = max(peak, (bid_high - entry) * units)
                max_adverse = max(max_adverse, (entry - bid_low) * units)
            else:
                if ask_high >= active_sl:
                    exit_price = active_sl + slippage_points * POINT_SIZE
                    reason = (
                        f"{stop_mode} tersentuh"
                        if stop_mode != "Initial SL"
                        else "SL tersentuh"
                    )
                elif ask_low <= tp_price:
                    exit_price = tp_price + slippage_points * POINT_SIZE
                    reason = "TP tersentuh"
                mark = ask_close
                peak = max(peak, (entry - ask_low) * units)
                max_adverse = max(max_adverse, (ask_high - entry) * units)
            floating = (
                (mark - entry) * units
                if direction == "BUY"
                else (entry - mark) * units
            )
            curve.append(
                {
                    "Tanggal": timestamp,
                    "Fase": 1,
                    "Balance": balance,
                    "Equity": balance + floating,
                    "Unrealized P/L": floating,
                    "Open BUY": int(direction == "BUY"),
                    "Open SELL": int(direction == "SELL"),
                    "Open total": 1,
                    "Target equity tercapai": False,
                }
            )
            if exit_price is not None:
                exit_time = timestamp
                break

            if (offset + 1) % policy.management_interval == 0:
                be_trigger = (
                    float(signal["sl_usd"]) * policy.break_even_r
                    if policy.break_even_r is not None
                    else np.inf
                )
                if peak >= be_trigger:
                    candidate_sl = entry
                    if direction == "BUY" and candidate_sl > active_sl:
                        active_sl = candidate_sl
                        stop_mode = "Break-even"
                        break_even_activated = True
                    elif direction == "SELL" and candidate_sl < active_sl:
                        active_sl = candidate_sl
                        stop_mode = "Break-even"
                        break_even_activated = True

                trailing_trigger = np.inf
                if policy.trailing_r is not None:
                    trailing_trigger = float(signal["sl_usd"]) * policy.trailing_r
                elif policy.trailing_tp_fraction is not None:
                    trailing_trigger = (
                        float(signal["tp_usd"]) * policy.trailing_tp_fraction
                    )
                if peak >= trailing_trigger and timestamp >= atr_m15.index.min():
                    atr = float(atr_m15.asof(timestamp))
                    if np.isfinite(atr):
                        candidate_sl = (
                            bid_close - 1.5 * atr
                            if direction == "BUY"
                            else ask_close + 1.5 * atr
                        )
                        if direction == "BUY" and candidate_sl > active_sl:
                            active_sl = candidate_sl
                            stop_mode = "ATR trailing"
                            trailing_activated = True
                        elif direction == "SELL" and candidate_sl < active_sl:
                            active_sl = candidate_sl
                            stop_mode = "ATR trailing"
                            trailing_activated = True

            if offset >= 14 and (offset + 1) % 15 == 0:
                key = (entry_time, timestamp)
                if key not in states.index:
                    continue
                state = states.loc[key]
                hazard_1h = float(state["hazard_1h"])
                hazard_3h = float(state["hazard_3h"])
                range_atr = (
                    float(signal["range_high"] - signal["range_low"])
                    / max(float(signal["range_width_atr"]), 0.01)
                )
                adverse_boundary = (
                    float(signal["range_low"]) - 0.15 * range_atr
                    if direction == "BUY"
                    else float(signal["range_high"]) + 0.15 * range_atr
                )
                outside = (
                    bid_close < adverse_boundary
                    if direction == "BUY"
                    else ask_close > adverse_boundary
                )
                structural_count = structural_count + 1 if outside else 0
                high_count = high_count + 1 if hazard_3h >= threshold_3h else 0
                acceleration = hazard_3h - previous_hazard
                previous_hazard = hazard_3h
                compression = float(state["range_width_change"]) <= -0.10
                hazard_score = (
                    (5 if structural_count >= 2 else 0)
                    + (4 if high_count >= 2 else 0)
                    + (3 if hazard_1h >= threshold_1h else 0)
                    + (
                        2
                        if hazard_3h >= threshold_3h * 0.75
                        and acceleration >= 0.12
                        else 0
                    )
                    + (1 if compression else 0)
                )
                max_hazard_score = max(max_hazard_score, hazard_score)
                qualifies = (
                    policy.hazard_exit_score is not None
                    and hazard_score >= policy.hazard_exit_score
                )
                hazard_confirm_count = (
                    hazard_confirm_count + 1 if qualifies else 0
                )
                should_exit = (
                    qualifies
                    and hazard_confirm_count >= policy.hazard_confirmations
                )
                is_warning = (
                    policy.hazard_warning_score is not None
                    and hazard_score >= policy.hazard_warning_score
                    and not should_exit
                )
                if is_warning:
                    warning_count += 1
                    if policy.warning_tighten_r is not None:
                        risk_usd = (
                            float(signal["sl_usd"]) * policy.warning_tighten_r
                        )
                        candidate_sl = (
                            entry - risk_usd / units
                            if direction == "BUY"
                            else entry + risk_usd / units
                        )
                        if direction == "BUY" and candidate_sl > active_sl:
                            active_sl = candidate_sl
                            stop_mode = "Warning tighten"
                        elif direction == "SELL" and candidate_sl < active_sl:
                            active_sl = candidate_sl
                            stop_mode = "Warning tighten"
                if should_exit:
                    exit_price = (
                        bid_close - slippage_points * POINT_SIZE
                        if direction == "BUY"
                        else ask_close + slippage_points * POINT_SIZE
                    )
                    exit_time = timestamp
                    reason = "Hazard score early exit"
                    exit_hazard = hazard_3h
                    break

        if exit_price is None:
            last = path.iloc[-1]
            last_spread = (
                float(last["SpreadPoints"]) * POINT_SIZE * spread_multiplier
            )
            exit_price = (
                float(last["Close"]) - slippage_points * POINT_SIZE
                if direction == "BUY"
                else float(last["Close"]) + last_spread
                + slippage_points * POINT_SIZE
            )
        gross = (
            (exit_price - entry) * units
            if direction == "BUY"
            else (entry - exit_price) * units
        )
        holding_days = max(
            int((exit_time.normalize() - entry_time.normalize()).days), 0
        )
        swap_paid = (
            BUY_SWAP_PER_001_LOT * holding_days if direction == "BUY" else 0.0
        )
        net = gross - swap_paid
        balance += net
        spread_cost = spread if direction == "BUY" else (
            float(path.loc[:exit_time].iloc[-1]["SpreadPoints"])
            * POINT_SIZE
            * spread_multiplier
        )
        trades.append(
            {
                "Fase": 1,
                "Position ID": len(trades) + 1,
                "Tanggal sinyal": signal["signal_date"],
                "Tanggal entry": entry_time,
                "Tanggal tutup": exit_time,
                "Arah": direction,
                "Lot": 0.01,
                "Prediksi": signal["prediction"],
                "Expected change (%)": signal["expected_change_pct"],
                "Strategi": policy.name,
                "Entry": entry,
                "Exit": exit_price,
                "Alasan exit": reason,
                "TP (USD)": signal["tp_usd"],
                "SL (USD)": signal["sl_usd"],
                "Peak floating profit (USD)": peak,
                "Max adverse excursion (USD)": max_adverse,
                "Biaya spread": spread_cost,
                "Biaya slippage": 2 * slippage_points * POINT_SIZE,
                "Gross P/L": gross,
                "Swap": -swap_paid,
                "Net P/L": net,
                "Balance": balance,
                "Dynamic hazard": exit_hazard,
                "Max hazard score": max_hazard_score,
                "BE activated": break_even_activated,
                "Trailing activated": trailing_activated,
                "Warning count": warning_count,
            }
        )
        curve.append(
            {
                "Tanggal": exit_time,
                "Fase": 1,
                "Balance": balance,
                "Equity": balance,
                "Unrealized P/L": 0.0,
                "Open BUY": 0,
                "Open SELL": 0,
                "Open total": 0,
                "Target equity tercapai": False,
            }
        )
        busy_until = exit_time

    trades_frame = pd.DataFrame(trades)
    if curve:
        curve_frame = pd.DataFrame(curve).set_index("Tanggal").sort_index()
        curve_frame = curve_frame.loc[
            ~curve_frame.index.duplicated(keep="last")
        ]
    else:
        curve_frame = pd.DataFrame(
            [{"Equity": INITIAL_EQUITY, "Balance": INITIAL_EQUITY, "Open total": 0}],
            index=[data.index[0]],
        )
    phases = pd.DataFrame()
    summary = _overall_summary(trades_frame, curve_frame, phases)
    summary.update({"Kandidat": policy.name, "Entry diblokir": float(blocked)})
    return MultiPhaseSimulationResult(summary, phases, trades_frame, curve_frame)


def _period_validation(results, signals):
    rows = []
    for label, start, end in _periods()[:-1]:
        for candidate, result in results.items():
            rows.append(
                {
                    "Periode": label,
                    "Kandidat": candidate,
                    "Sinyal tersedia": len(signals.loc[start:end]),
                    **_ledger_metric_values(
                        _trades_in_period(result.trades, start, end)
                    ),
                }
            )
    return pd.DataFrame(rows)


def _exit_attribution(results, control_name):
    control = results[control_name].trades
    control_map = (
        control.set_index("Tanggal entry")["Net P/L"]
        if not control.empty
        else pd.Series(dtype=float)
    )
    rows = []
    for candidate, result in results.items():
        if candidate == control_name:
            continue
        common = result.trades.loc[
            result.trades["Tanggal entry"].isin(control_map.index)
        ].copy()
        baseline = common["Tanggal entry"].map(control_map)
        delta = common["Net P/L"].to_numpy() - baseline.to_numpy()
        delta_series = pd.Series(delta, dtype=float)
        peak = common["Peak floating profit (USD)"].clip(lower=0)
        captured = common["Net P/L"].clip(lower=0)
        capture = np.where(peak > 0, captured / peak * 100, np.nan)
        rows.append(
            {
                "Kandidat": candidate,
                "Common entry": len(common),
                "BE activations": int(common["BE activated"].sum()),
                "Trailing activations": int(common["Trailing activated"].sum()),
                "Warning evaluations": int(common["Warning count"].sum()),
                "Hazard exits": int(
                    common["Alasan exit"].str.contains("Hazard score").sum()
                ),
                "Saved loss": float(delta_series[delta_series > 0].sum()),
                "Sacrificed profit": float(-delta_series[delta_series < 0].sum()),
                "Net exit benefit": float(delta_series.sum()),
                "Median MFE captured (%)": float(
                    np.nanmedian(capture) if np.isfinite(capture).any() else np.nan
                ),
                "Total profit giveback": float((peak - captured).clip(lower=0).sum()),
            }
        )
    return pd.DataFrame(rows)


def _exit_reason_summary(results):
    rows = []
    for candidate, result in results.items():
        if result.trades.empty:
            continue
        for reason, frame in result.trades.groupby("Alasan exit", dropna=False):
            rows.append(
                {
                    "Kandidat": candidate,
                    "Alasan exit": reason,
                    "Jumlah": len(frame),
                    "Total net P/L": float(frame["Net P/L"].sum()),
                    "Rata-rata net P/L": float(frame["Net P/L"].mean()),
                }
            )
    return pd.DataFrame(rows)


def _decision_table(
    candidates, development, periods, folds, monte_carlo, concentration
):
    dev = development.set_index("Kandidat")
    period = periods.set_index(["Periode", "Kandidat"])
    mc = monte_carlo.set_index("Kandidat")
    concentrated = concentration.set_index("Kandidat")
    rows = []
    for candidate in candidates:
        primary = folds[
            folds["Kandidat"].eq(candidate)
            & folds["Kelompok"].eq("Primary validation")
        ]
        criteria = {
            "Growth positif": float(dev.loc[candidate, "Growth (%)"]) > 0,
            "PF >= 1.30": float(dev.loc[candidate, "Profit factor"]) >= 1.30,
            "DD <= 10%": float(dev.loc[candidate, "Max drawdown (%)"]) <= 10,
            "2024 positif": float(
                period.loc[("Selection 2024", candidate), "Growth (%)"]
            ) > 0,
            "2025 positif": float(
                period.loc[("Locked 2025", candidate), "Growth (%)"]
            ) > 0,
            "Primary fold >= 6/8": int(primary["Profitable"].sum()) >= 6,
            "Monte Carlo rugi <= 10%": float(
                mc.loc[candidate, "Probabilitas equity akhir < modal awal (%)"]
            ) <= 10,
            "Transaksi development >= 30": int(
                dev.loc[candidate, "Transaksi"]
            ) >= 30,
            "Transaksi locked >= 8": int(
                period.loc[("Locked 2025", candidate), "Transaksi"]
            ) >= 8,
            "Konsentrasi 5 profit <= 40%": float(
                concentrated.loc[candidate, "Konsentrasi 5 profit terbesar (%)"]
            ) <= 40,
        }
        rows.append(
            {
                "Kandidat": candidate,
                **criteria,
                "Primary fold profitable": int(primary["Profitable"].sum()),
                "Kriteria lolos": int(sum(criteria.values())),
                "Total kriteria": len(criteria),
                "Lulus": bool(all(criteria.values())),
            }
        )
    return pd.DataFrame(rows)


def _selection_ranking(
    candidates, development, reference, periods, decisions, attribution
):
    dev = development.set_index("Kandidat")
    ref = reference.set_index("Kandidat")
    period = periods.set_index(["Periode", "Kandidat"])
    decision = decisions.set_index("Kandidat")
    attr = attribution.set_index("Kandidat")
    rows = []
    for candidate in candidates:
        selection_growth = float(
            period.loc[("Selection 2024", candidate), "Growth (%)"]
        )
        selection_pf = float(
            period.loc[("Selection 2024", candidate), "Profit factor"]
        )
        selection_dd = float(
            period.loc[("Selection 2024", candidate), "Max drawdown (%)"]
        )
        selection_trades = int(
            period.loc[("Selection 2024", candidate), "Transaksi"]
        )
        safe_pf = 0.0 if not np.isfinite(selection_pf) else selection_pf
        score = (
            selection_growth
            + min(safe_pf, 3.0) * 5
            - selection_dd * 1.5
            + min(selection_trades, 40) * 0.08
        )
        rows.append(
            {
                "Kandidat": candidate,
                "Selection score 2024": score,
                "Growth selection 2024 (%)": selection_growth,
                "PF selection 2024": selection_pf,
                "DD selection 2024 (%)": selection_dd,
                "Transaksi selection 2024": selection_trades,
                "Selection eligible": bool(
                    selection_growth > 0 and selection_trades >= 8
                ),
                "Growth development (%)": float(dev.loc[candidate, "Growth (%)"]),
                "PF development": float(dev.loc[candidate, "Profit factor"]),
                "DD development (%)": float(
                    dev.loc[candidate, "Max drawdown (%)"]
                ),
                "Transaksi development": int(dev.loc[candidate, "Transaksi"]),
                "Growth locked 2025 (%)": float(
                    period.loc[("Locked 2025", candidate), "Growth (%)"]
                ),
                "Growth 2026H1 (%)": float(ref.loc[candidate, "Growth (%)"]),
                "PF 2026H1": float(ref.loc[candidate, "Profit factor"]),
                "DD 2026H1 (%)": float(
                    ref.loc[candidate, "Max drawdown (%)"]
                ),
                "Transaksi 2026H1": int(ref.loc[candidate, "Transaksi"]),
                "Net exit benefit": (
                    float(attr.loc[candidate, "Net exit benefit"])
                    if candidate in attr.index
                    else 0.0
                ),
                "Kriteria lolos": int(decision.loc[candidate, "Kriteria lolos"]),
                "Lulus": bool(decision.loc[candidate, "Lulus"]),
                "Lulus termasuk stress": False,
            }
        )
    ranking = pd.DataFrame(rows).sort_values(
        ["Selection score 2024", "DD selection 2024 (%)"],
        ascending=[False, True],
    ).reset_index(drop=True)
    ranking.insert(0, "Peringkat", np.arange(1, len(ranking) + 1))
    return ranking


def _stress_test(
    data,
    signals,
    states,
    atr_m15,
    policy,
    threshold_1h,
    threshold_3h,
):
    rows = []
    for spread_multiplier in (1.0, 1.25, 1.50):
        for slippage in (2.0, 4.0, 6.0):
            result = _simulate_policy(
                data,
                signals,
                states,
                atr_m15,
                policy,
                threshold_1h,
                threshold_3h,
                spread_multiplier=spread_multiplier,
                slippage_points=slippage,
            )
            rows.append(
                {
                    "Kandidat": policy.name,
                    "Spread multiplier": spread_multiplier,
                    "Slippage points": slippage,
                    **_metric_values(result),
                }
            )
    return pd.DataFrame(rows)


def _v5_reference(payload):
    if not payload:
        return pd.DataFrame()
    ranking = payload.get("ranking")
    if not isinstance(ranking, pd.DataFrame) or ranking.empty:
        return pd.DataFrame()
    return ranking.head(3).copy()
