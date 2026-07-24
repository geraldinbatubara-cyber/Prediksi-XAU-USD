from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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
    _apply_symmetric_calibration,
    _class_weights,
    _ledger_metric_values,
    _monte_carlo_summary,
    _trades_in_period,
)
from gold_forecast.v1_entry_quality_path import FOLDS
from gold_forecast.v1_regime_classifier_v3 import _fit_platt
from gold_forecast.v1_risk_control import _metric_values
from gold_forecast.v1_sell_specialist import (
    CALIBRATION_END,
    CALIBRATION_START,
    DEVELOPMENT_END,
    DEVELOPMENT_START,
    LOCKED_END,
    LOCKED_START,
    REFERENCE_END,
    REFERENCE_START,
    SELECTION_END,
    SELECTION_START,
    THRESHOLD_END,
    THRESHOLD_START,
    TRAIN_END,
    TRAIN_START,
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
from gold_forecast.v1_trend_strength_stability import _extended_data_audit
from gold_forecast.v1_unified_benchmark import _unified_best


CANDIDATES = (
    "Breakout Hazard v2 Control",
    "Structural Exit",
    "Dynamic Hazard 1h",
    "Dynamic Hazard 3h Confirmed",
    "Hazard Acceleration Protection",
    "Adaptive Dynamic Hazard",
)
STATE_FEATURES = (
    "age_hours",
    "floating_pl",
    "peak_profit",
    "max_adverse",
    "distance_tp",
    "distance_sl",
    "midpoint_distance_atr",
    "price_position",
    "adx",
    "adx_change_3h",
    "atr_acceleration",
    "midpoint_drift_atr",
    "range_width_change",
    "touch_imbalance",
    "candle_body_atr",
    "direction_code",
    "session_sin",
    "session_cos",
)


def run_v1_sideways_specialist_v3_lab(
    gold_m1: pd.DataFrame,
    frozen_payload: dict[str, object],
    v2_payload: dict[str, object] | None = None,
) -> dict[str, object]:
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
    v2_signals, _ = _v2_candidate_signals(
        opportunities,
        outcome_model,
        persistence_model,
        entry_hazard_model,
        directional_thresholds,
        selected_sessions,
        best,
    )
    entry_signals = v2_signals["Breakout Hazard Gate"].copy()
    source = opportunities.loc[
        ~opportunities.index.duplicated(keep="first")
    ].reindex(entry_signals.index)
    entry_signals = _enrich_signals(entry_signals, source)

    state_frame = _build_position_states(data, range_frame, entry_signals)
    model_1h = _train_state_model(state_frame, "adverse_before_tp_1h", "1 jam", 121)
    model_3h = _train_state_model(state_frame, "adverse_before_tp_3h", "3 jam", 131)
    state_frame["hazard_1h"] = model_1h["probability"]
    state_frame["hazard_3h"] = model_3h["probability"]

    development_results = {
        candidate: _simulate_dynamic(
            data.loc[DEVELOPMENT_START:DEVELOPMENT_END],
            entry_signals.loc[DEVELOPMENT_START:DEVELOPMENT_END],
            state_frame,
            candidate,
            model_1h["threshold"],
            model_3h["threshold"],
        )
        for candidate in CANDIDATES
    }
    reference_results = {
        candidate: _simulate_dynamic(
            data.loc[REFERENCE_START:REFERENCE_END],
            entry_signals.loc[REFERENCE_START:REFERENCE_END],
            state_frame,
            candidate,
            model_1h["threshold"],
            model_3h["threshold"],
        )
        for candidate in CANDIDATES
    }
    development = _result_table(development_results, entry_signals)
    reference = _result_table(reference_results, entry_signals)
    periods = _period_validation(development_results, entry_signals)
    folds = _fold_evaluation(development_results)
    monte_carlo = _monte_carlo_summary(development_results)
    concentration = _profit_concentration(development_results)
    attribution = _exit_attribution(development_results)
    decisions = _decision_table(
        development, periods, folds, monte_carlo, concentration
    )
    classification = _state_classification_tables(state_frame, model_1h, model_3h)
    ranking = _selection_ranking(
        development, reference, periods, decisions, attribution
    )
    eligible = ranking.loc[ranking["Selection eligible"]]
    winner = str(eligible.iloc[0]["Kandidat"]) if not eligible.empty else ""
    stress = (
        _dynamic_stress(
            data.loc[DEVELOPMENT_START:DEVELOPMENT_END],
            entry_signals.loc[DEVELOPMENT_START:DEVELOPMENT_END],
            state_frame,
            winner,
            model_1h["threshold"],
            model_3h["threshold"],
        )
        if winner
        else pd.DataFrame()
    )
    stress_passed = (
        int((stress["Growth (%)"] > 0).sum()) if not stress.empty else 0
    )
    decisions = decisions.copy()
    decisions["Stress profitable"] = (
        decisions["Kandidat"].map({winner: stress_passed})
        if winner
        else np.nan
    )
    winner_passed = False
    if winner:
        row = decisions.loc[decisions["Kandidat"].eq(winner)].iloc[0]
        winner_passed = bool(row["Lulus"]) and stress_passed >= 7
        ranking.loc[
            ranking["Kandidat"].eq(winner), "Lulus termasuk stress"
        ] = winner_passed

    return {
        "methodology": {
            "Name": "v1 Sideways Specialist Lab v3 - Dynamic Hazard Monitoring & Early Exit",
            "Control": (
                "Entry Breakout Hazard Gate v2 identik; hanya kebijakan exit yang "
                "berubah."
            ),
            "State dataset": (
                "Setiap posisi direkam tiap 15 menit sampai TP, SL, atau 12 jam. "
                "Model tidak melihat candle setelah timestamp state."
            ),
            "Targets": (
                "Adverse boundary/SL sebelum TP dalam 1 jam dan 3 jam."
            ),
            "Monitoring": (
                "LOW hold | MEDIUM observasi | HIGH perlu konfirmasi/protection | "
                "CRITICAL early exit. Structural exit memakai dua evaluasi M15 "
                "berturut-turut di luar adverse boundary."
            ),
            "Train": "Position states dari entry 2022",
            "Probability calibration": "Position states 2023H1",
            "Threshold calibration": (
                "Position states 2023H2; fallback konservatif Q85 dipakai dan "
                "ditandai jika periode hanya memiliki satu kelas target."
            ),
            "Model selection": "Trade entry 2024 saja",
            "Locked confirmation": "Trade entry 2025",
            "Historical reference": "Trade entry 2026H1, tidak menentukan pemenang",
            "Execution": (
                "M1 broker-aware | monitoring tiap 15 menit | lot 0.01 | "
                "spread/slippage/swap BUY dihitung | TP/SL intrabar diperiksa "
                "sebelum dynamic exit pada close."
            ),
            "Baseline lock": (
                "Seluruh strategi dan ledger Paper Live Trading tidak diubah."
            ),
        },
        "data_audit": _extended_data_audit(data),
        "state_audit": _state_audit(state_frame),
        "model_1h_selection": model_1h["selection"],
        "model_3h_selection": model_3h["selection"],
        "classification_selection": classification["selection"],
        "classification_locked": classification["locked"],
        "classification_reference": classification["reference"],
        "development": development,
        "period_validation": periods,
        "historical_reference": reference,
        "folds": folds,
        "monte_carlo_summary": monte_carlo,
        "profit_concentration": concentration,
        "exit_attribution": attribution,
        "stress_summary": stress,
        "decisions": decisions,
        "ranking": ranking,
        "winner": winner,
        "selection_status": (
            f"Eligible: {winner}"
            if winner
            else "Tidak ada kandidat eligible pada model selection 2024"
        ),
        "winner_passed": winner_passed,
        "v2_reference": _v2_reference(v2_payload),
    }


def _enrich_signals(signals, source):
    output = signals.copy()
    for column in (
        "range_low",
        "range_high",
        "range_mid",
        "range_width_atr",
        "direction",
    ):
        output[column] = source[column].to_numpy()
    return output


def _build_position_states(data, range_frame, signals):
    rows = []
    for entry_time, signal in signals.iterrows():
        entry_time = pd.Timestamp(entry_time)
        path = data.loc[entry_time : entry_time + pd.Timedelta(hours=12)]
        if len(path) < 30:
            continue
        direction = str(signal["direction"])
        spread = float(path.iloc[0]["SpreadPoints"]) * POINT_SIZE
        bid_entry = float(path.iloc[0]["Close"])
        entry = (
            bid_entry + spread + SLIPPAGE_POINTS * POINT_SIZE
            if direction == "BUY"
            else bid_entry - SLIPPAGE_POINTS * POINT_SIZE
        )
        tp_price, sl_price = _price_levels(
            entry, direction, float(signal["tp_usd"]), float(signal["sl_usd"])
        )
        baseline_exit = _first_price_exit(path, direction, tp_price, sl_price)
        peak_profit = 0.0
        max_adverse = 0.0
        for offset in range(14, len(path), 15):
            timestamp = pd.Timestamp(path.index[offset])
            if baseline_exit is not None and timestamp >= baseline_exit:
                break
            history = path.iloc[: offset + 1]
            state = _position_state_row(
                timestamp,
                entry_time,
                entry,
                direction,
                tp_price,
                sl_price,
                signal,
                history,
                range_frame,
                peak_profit,
                max_adverse,
            )
            peak_profit = float(state["peak_profit"])
            max_adverse = float(state["max_adverse"])
            future_1h = data.loc[
                timestamp + pd.Timedelta(minutes=1) :
                timestamp + pd.Timedelta(hours=1)
            ]
            future_3h = data.loc[
                timestamp + pd.Timedelta(minutes=1) :
                timestamp + pd.Timedelta(hours=3)
            ]
            state["adverse_before_tp_1h"] = _adverse_before_tp(
                future_1h, direction, tp_price, sl_price
            )
            state["adverse_before_tp_3h"] = _adverse_before_tp(
                future_3h, direction, tp_price, sl_price
            )
            state["entry_time"] = entry_time
            state["state_time"] = timestamp
            rows.append(state)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("Position-state dataset kosong.")
    return frame.set_index(["entry_time", "state_time"]).sort_index().dropna(
        subset=[*STATE_FEATURES, "adverse_before_tp_1h", "adverse_before_tp_3h"]
    )


def _position_state_row(
    timestamp,
    entry_time,
    entry,
    direction,
    tp_price,
    sl_price,
    signal,
    history,
    range_frame,
    previous_peak,
    previous_adverse,
):
    latest = history.iloc[-1]
    spread = float(latest["SpreadPoints"]) * POINT_SIZE
    bid = float(latest["Close"])
    mark = bid if direction == "BUY" else bid + spread
    units = 0.01 * CONTRACT_OUNCES_PER_LOT
    floating = (
        (mark - entry) * units
        if direction == "BUY"
        else (entry - mark) * units
    )
    if direction == "BUY":
        peak = max(previous_peak, (float(history["High"].max()) - entry) * units)
        adverse = max(previous_adverse, (entry - float(history["Low"].min())) * units)
        distance_tp = max(tp_price - mark, 0.0) * units
        distance_sl = max(mark - sl_price, 0.0) * units
    else:
        ask_low = history["Low"] + history["SpreadPoints"] * POINT_SIZE
        ask_high = history["High"] + history["SpreadPoints"] * POINT_SIZE
        peak = max(previous_peak, (entry - float(ask_low.min())) * units)
        adverse = max(previous_adverse, (float(ask_high.max()) - entry) * units)
        distance_tp = max(mark - tp_price, 0.0) * units
        distance_sl = max(sl_price - mark, 0.0) * units
    state = range_frame.loc[:timestamp].iloc[-1]
    width = float(signal["range_high"] - signal["range_low"])
    atr = width / max(float(signal["range_width_atr"]), 0.01)
    midpoint_drift = (float(state["range_mid"]) - float(signal["range_mid"])) / atr
    width_change = (
        float(state["range_high"] - state["range_low"]) / max(width, 0.01) - 1
    )
    touch_imbalance = (
        float(state["touch_upper"] - state["touch_lower"])
        / max(float(state["touch_upper"] + state["touch_lower"]), 1.0)
    )
    candle_body = abs(float(latest["Close"] - latest["Open"])) / max(
        float(state["atr"]), 0.01
    )
    hour = timestamp.hour + timestamp.minute / 60
    return {
        "age_hours": (timestamp - entry_time).total_seconds() / 3600,
        "floating_pl": floating,
        "peak_profit": peak,
        "max_adverse": adverse,
        "distance_tp": distance_tp,
        "distance_sl": distance_sl,
        "midpoint_distance_atr": (mark - float(state["range_mid"])) / atr,
        "price_position": (
            mark - float(signal["range_low"])
        ) / max(width, 0.01),
        "adx": float(state["adx"]),
        "adx_change_3h": float(state["adx"] - range_frame["adx"].loc[:timestamp].iloc[-181]),
        "atr_acceleration": float(
            state["atr"]
            / max(range_frame["atr"].loc[:timestamp].iloc[-361:-181].median(), 0.01)
            - 1
        ),
        "midpoint_drift_atr": midpoint_drift,
        "range_width_change": width_change,
        "touch_imbalance": touch_imbalance,
        "candle_body_atr": candle_body,
        "direction_code": 1.0 if direction == "BUY" else -1.0,
        "session_sin": np.sin(2 * np.pi * hour / 24),
        "session_cos": np.cos(2 * np.pi * hour / 24),
    }


def _price_levels(entry, direction, tp_usd, sl_usd):
    units = 0.01 * CONTRACT_OUNCES_PER_LOT
    if direction == "BUY":
        return entry + tp_usd / units, entry - sl_usd / units
    return entry - tp_usd / units, entry + sl_usd / units


def _first_price_exit(path, direction, tp_price, sl_price):
    for candle in path.itertuples():
        spread = float(candle.SpreadPoints) * POINT_SIZE
        if direction == "BUY":
            if float(candle.Low) <= sl_price or float(candle.High) >= tp_price:
                return pd.Timestamp(candle.Index)
        else:
            ask_high = float(candle.High) + spread
            ask_low = float(candle.Low) + spread
            if ask_high >= sl_price or ask_low <= tp_price:
                return pd.Timestamp(candle.Index)
    return None


def _adverse_before_tp(path, direction, tp_price, sl_price):
    if path.empty:
        return 0.0
    for candle in path.itertuples():
        spread = float(candle.SpreadPoints) * POINT_SIZE
        if direction == "BUY":
            if float(candle.Low) <= sl_price:
                return 1.0
            if float(candle.High) >= tp_price:
                return 0.0
        else:
            ask_high = float(candle.High) + spread
            ask_low = float(candle.Low) + spread
            if ask_high >= sl_price:
                return 1.0
            if ask_low <= tp_price:
                return 0.0
    return 0.0


def _train_state_model(states, target, horizon, seed):
    entry_dates = states.index.get_level_values("entry_time")
    train = states.loc[
        (entry_dates >= TRAIN_START) & (entry_dates <= TRAIN_END)
    ]
    calibration = states.loc[
        (entry_dates >= CALIBRATION_START) & (entry_dates <= CALIBRATION_END)
    ]
    threshold_period = states.loc[
        (entry_dates >= THRESHOLD_START) & (entry_dates <= THRESHOLD_END)
    ]
    if len(train) < 100 or train[target].nunique() < 2:
        raise RuntimeError(f"State train {horizon} tidak cukup.")
    logistic = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            C=0.25,
            random_state=seed,
        ),
    )
    boosting = HistGradientBoostingClassifier(
        learning_rate=0.035,
        max_iter=180,
        max_depth=3,
        min_samples_leaf=25,
        l2_regularization=2.0,
        random_state=seed + 1,
    )
    logistic.fit(train[list(STATE_FEATURES)], train[target].astype(int))
    boosting.fit(
        train[list(STATE_FEATURES)],
        train[target].astype(int),
        sample_weight=_class_weights(train[target]),
    )
    raw = (
        pd.Series(
            logistic.predict_proba(states[list(STATE_FEATURES)])[:, 1],
            index=states.index,
        )
        + pd.Series(
            boosting.predict_proba(states[list(STATE_FEATURES)])[:, 1],
            index=states.index,
        )
    ) / 2
    calibration_index = calibration.index
    calibrator = _fit_platt(
        raw.loc[calibration_index], calibration[target].astype(int)
    )
    probability = _apply_symmetric_calibration(raw, calibrator)
    threshold, audit = _state_threshold(
        threshold_period[target].astype(int),
        probability.loc[threshold_period.index],
    )
    return {
        "target": target,
        "probability": probability,
        "threshold": threshold,
        "selection": pd.DataFrame(
            [{
                "Horizon": horizon,
                "Train states": len(train),
                "Calibration states": len(calibration),
                "Threshold states": len(threshold_period),
                "Threshold": threshold,
                **audit,
            }]
        ),
    }


def _state_threshold(truth, probability):
    if truth.nunique() < 2:
        threshold = float(probability.quantile(0.85))
        return threshold, {
            "Threshold status": (
                "Fallback Q85: periode threshold hanya memiliki satu kelas"
            ),
            "Observed adverse": int(truth.sum()),
            "Precision": np.nan,
            "Recall": np.nan,
            "Balanced accuracy": np.nan,
            "Predicted high hazard": int(probability.ge(threshold).sum()),
        }
    rows = []
    for quantile in (0.45, 0.55, 0.65, 0.75, 0.85):
        threshold = float(probability.quantile(quantile))
        prediction = probability.ge(threshold)
        precision = precision_score(truth, prediction, zero_division=0)
        recall = recall_score(truth, prediction, zero_division=0)
        balanced = balanced_accuracy_score(truth, prediction)
        count = int(prediction.sum())
        eligible = count >= 20
        score = 0.50 * precision + 0.30 * recall + 0.20 * balanced
        rows.append((threshold, precision, recall, balanced, count, score, eligible))
    eligible_rows = [row for row in rows if row[-1]]
    selected = max(eligible_rows or rows, key=lambda row: row[5])
    return selected[0], {
        "Threshold status": "Validated pada 2023H2",
        "Observed adverse": int(truth.sum()),
        "Precision": selected[1],
        "Recall": selected[2],
        "Balanced accuracy": selected[3],
        "Predicted high hazard": selected[4],
    }


def _simulate_dynamic(
    data,
    signals,
    states,
    candidate,
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
        tp_price, sl_price = _price_levels(
            entry, direction, float(signal["tp_usd"]), float(signal["sl_usd"])
        )
        peak = 0.0
        max_adverse = 0.0
        high_count = 0
        structural_count = 0
        previous_hazard = 0.0
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
                if bid_low <= sl_price:
                    exit_price = sl_price - slippage_points * POINT_SIZE
                    reason = "SL tersentuh"
                elif bid_high >= tp_price:
                    exit_price = tp_price - slippage_points * POINT_SIZE
                    reason = "TP tersentuh"
                mark = bid_close
                peak = max(peak, bid_high - entry)
                max_adverse = max(max_adverse, entry - bid_low)
            else:
                if ask_high >= sl_price:
                    exit_price = sl_price + slippage_points * POINT_SIZE
                    reason = "SL tersentuh"
                elif ask_low <= tp_price:
                    exit_price = tp_price + slippage_points * POINT_SIZE
                    reason = "TP tersentuh"
                mark = ask_close
                peak = max(peak, entry - ask_low)
                max_adverse = max(max_adverse, ask_high - entry)
            floating = mark - entry if direction == "BUY" else entry - mark
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
            if offset >= 14 and (offset + 1) % 15 == 0:
                key = (entry_time, timestamp)
                if key in states.index:
                    state = states.loc[key]
                    hazard_1h = float(state["hazard_1h"])
                    hazard_3h = float(state["hazard_3h"])
                    adverse_boundary = (
                        float(signal["range_low"])
                        - 0.15
                        * (
                            float(signal["range_high"] - signal["range_low"])
                            / max(float(signal["range_width_atr"]), 0.01)
                        )
                        if direction == "BUY"
                        else float(signal["range_high"])
                        + 0.15
                        * (
                            float(signal["range_high"] - signal["range_low"])
                            / max(float(signal["range_width_atr"]), 0.01)
                        )
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
                    should_exit = False
                    dynamic_reason = ""
                    if candidate == "Structural Exit":
                        should_exit = structural_count >= 2
                        dynamic_reason = "Structural exit"
                    elif candidate == "Dynamic Hazard 1h":
                        should_exit = hazard_1h >= threshold_1h
                        dynamic_reason = "Dynamic hazard 1h"
                    elif candidate == "Dynamic Hazard 3h Confirmed":
                        should_exit = high_count >= 2
                        dynamic_reason = "Dynamic hazard 3h confirmed"
                    elif candidate == "Hazard Acceleration Protection":
                        should_exit = (
                            hazard_3h >= threshold_3h * 0.75
                            and acceleration >= 0.12
                        )
                        dynamic_reason = "Hazard acceleration"
                    elif candidate == "Adaptive Dynamic Hazard":
                        should_exit = (
                            structural_count >= 2
                            or hazard_1h >= threshold_1h * 1.10
                            or high_count >= 2
                            or (
                                peak >= 5.0
                                and hazard_3h >= threshold_3h * 0.85
                            )
                        )
                        dynamic_reason = "Adaptive dynamic hazard"
                    if should_exit and candidate != "Breakout Hazard v2 Control":
                        exit_price = (
                            bid_close - slippage_points * POINT_SIZE
                            if direction == "BUY"
                            else ask_close + slippage_points * POINT_SIZE
                        )
                        exit_time = timestamp
                        reason = dynamic_reason
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
                else float(last["Close"]) + last_spread + slippage_points * POINT_SIZE
            )
        gross = (
            exit_price - entry if direction == "BUY" else entry - exit_price
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
                "Strategi": candidate,
                "Entry": entry,
                "Exit": exit_price,
                "Alasan exit": reason,
                "TP (USD)": signal["tp_usd"],
                "SL (USD)": signal["sl_usd"],
                "Peak floating profit (USD)": peak,
                "Biaya spread": spread_cost,
                "Biaya slippage": 2 * slippage_points * POINT_SIZE,
                "Gross P/L": gross,
                "Swap": -swap_paid,
                "Net P/L": net,
                "Balance": balance,
                "Dynamic hazard": exit_hazard,
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
            [{
                "Equity": INITIAL_EQUITY,
                "Balance": INITIAL_EQUITY,
                "Open total": 0,
            }],
            index=[data.index[0]],
        )
    phases = pd.DataFrame()
    summary = _overall_summary(trades_frame, curve_frame, phases)
    summary.update({"Kandidat": candidate, "Entry diblokir": float(blocked)})
    return MultiPhaseSimulationResult(summary, phases, trades_frame, curve_frame)


def _result_table(results, signals):
    return pd.DataFrame(
        [
            {
                "Kandidat": candidate,
                "Sinyal tersedia": len(signals.loc[
                    result.equity_curve.index.min() :
                    result.equity_curve.index.max()
                ]),
                **_metric_values(result),
            }
            for candidate, result in results.items()
        ]
    )


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


def _fold_evaluation(results):
    rows = []
    for fold in FOLDS:
        for candidate, result in results.items():
            metrics = _ledger_metric_values(
                _trades_in_period(result.trades, fold.test_start, fold.test_end)
            )
            rows.append(
                {
                    "Fold": fold.name,
                    "Kelompok": (
                        "Calibration diagnostic"
                        if fold.test_start.year == 2023
                        else "Primary validation"
                    ),
                    "Kandidat": candidate,
                    "Test mulai": fold.test_start,
                    "Test akhir": fold.test_end,
                    **metrics,
                    "Profitable": bool(metrics["Growth (%)"] > 0),
                }
            )
    return pd.DataFrame(rows)


def _state_classification_tables(states, model_1h, model_3h):
    output = {}
    entry_dates = states.index.get_level_values("entry_time")
    for key, start, end in (
        ("selection", SELECTION_START, SELECTION_END),
        ("locked", LOCKED_START, LOCKED_END),
        ("reference", REFERENCE_START, REFERENCE_END),
    ):
        mask = (entry_dates >= start) & (entry_dates <= end)
        frame = states.loc[mask]
        rows = []
        for name, model in (("1 jam", model_1h), ("3 jam", model_3h)):
            truth = frame[model["target"]].astype(int)
            probability = model["probability"].loc[frame.index]
            prediction = probability.ge(model["threshold"])
            has_two_classes = truth.nunique() >= 2
            rows.append(
                {
                    "Horizon": name,
                    "States": len(frame),
                    "Base hazard (%)": float(truth.mean() * 100),
                    "Threshold": model["threshold"],
                    "Precision": (
                        precision_score(truth, prediction, zero_division=0)
                        if has_two_classes
                        else np.nan
                    ),
                    "Recall": (
                        recall_score(truth, prediction, zero_division=0)
                        if has_two_classes
                        else np.nan
                    ),
                    "Balanced accuracy": (
                        balanced_accuracy_score(truth, prediction)
                        if has_two_classes
                        else np.nan
                    ),
                    "Coverage (%)": float(prediction.mean() * 100),
                    "Validation status": (
                        "Valid" if has_two_classes else "Single-class; metrik N/A"
                    ),
                }
            )
        output[key] = pd.DataFrame(rows)
    return output


def _exit_attribution(results):
    control = results["Breakout Hazard v2 Control"].trades
    control_map = (
        control.set_index("Tanggal entry")["Net P/L"]
        if not control.empty
        else pd.Series(dtype=float)
    )
    rows = []
    for candidate, result in results.items():
        if candidate == "Breakout Hazard v2 Control":
            continue
        trades = result.trades
        common = trades.loc[trades["Tanggal entry"].isin(control_map.index)].copy()
        baseline = common["Tanggal entry"].map(control_map)
        delta = common["Net P/L"].to_numpy() - baseline.to_numpy()
        dynamic = common["Alasan exit"].str.contains(
            "hazard|Structural", case=False, regex=True
        )
        rows.append(
            {
                "Kandidat": candidate,
                "Common entry": len(common),
                "Dynamic exits": int(dynamic.sum()),
                "Saved loss": float(pd.Series(delta)[pd.Series(delta) > 0].sum()),
                "Sacrificed profit": float(
                    -pd.Series(delta)[pd.Series(delta) < 0].sum()
                ),
                "Net exit benefit": float(np.sum(delta)),
            }
        )
    return pd.DataFrame(rows)


def _decision_table(development, periods, folds, monte_carlo, concentration):
    dev = development.set_index("Kandidat")
    period = periods.set_index(["Periode", "Kandidat"])
    mc = monte_carlo.set_index("Kandidat")
    concentrated = concentration.set_index("Kandidat")
    rows = []
    for candidate in CANDIDATES:
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
            "Transaksi development >= 30": int(dev.loc[candidate, "Transaksi"]) >= 30,
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


def _selection_ranking(development, reference, periods, decisions, attribution):
    dev = development.set_index("Kandidat")
    ref = reference.set_index("Kandidat")
    period = periods.set_index(["Periode", "Kandidat"])
    decision = decisions.set_index("Kandidat")
    attr = attribution.set_index("Kandidat")
    rows = []
    for candidate in CANDIDATES:
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
                "DD development (%)": float(dev.loc[candidate, "Max drawdown (%)"]),
                "Transaksi development": int(dev.loc[candidate, "Transaksi"]),
                "Growth locked 2025 (%)": float(
                    period.loc[("Locked 2025", candidate), "Growth (%)"]
                ),
                "Growth 2026H1 (%)": float(ref.loc[candidate, "Growth (%)"]),
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


def _dynamic_stress(
    data, signals, states, candidate, threshold_1h, threshold_3h
):
    rows = []
    for spread_multiplier in (1.0, 1.25, 1.50):
        for slippage in (2.0, 4.0, 6.0):
            result = _simulate_dynamic(
                data,
                signals,
                states,
                candidate,
                threshold_1h,
                threshold_3h,
                spread_multiplier=spread_multiplier,
                slippage_points=slippage,
            )
            metrics = _metric_values(result)
            rows.append(
                {
                    "Kandidat": candidate,
                    "Spread multiplier": spread_multiplier,
                    "Slippage points": slippage,
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def _state_audit(states):
    entry_dates = states.index.get_level_values("entry_time")
    rows = []
    for label, start, end in _periods():
        frame = states.loc[(entry_dates >= start) & (entry_dates <= end)]
        rows.append(
            {
                "Periode": label,
                "Position states": len(frame),
                "Unique positions": frame.index.get_level_values(
                    "entry_time"
                ).nunique(),
                "Hazard 1h base rate (%)": float(
                    frame["adverse_before_tp_1h"].mean() * 100
                ),
                "Hazard 3h base rate (%)": float(
                    frame["adverse_before_tp_3h"].mean() * 100
                ),
            }
        )
    return pd.DataFrame(rows)


def _periods():
    return (
        ("Train 2022", TRAIN_START, TRAIN_END),
        ("Calibration 2023H1", CALIBRATION_START, CALIBRATION_END),
        ("Threshold 2023H2", THRESHOLD_START, THRESHOLD_END),
        ("Selection 2024", SELECTION_START, SELECTION_END),
        ("Locked 2025", LOCKED_START, LOCKED_END),
        ("Reference 2026H1", REFERENCE_START, REFERENCE_END),
    )


def _v2_reference(payload):
    if not payload:
        return pd.DataFrame()
    ranking = payload.get("ranking")
    if not isinstance(ranking, pd.DataFrame) or ranking.empty:
        return pd.DataFrame()
    return ranking.head(3).copy()
