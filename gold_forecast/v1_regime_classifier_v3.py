from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from gold_forecast.exact_broker_oos import _prepare_m1
from gold_forecast.v1_entry_outcome import _balanced_signals, _safe_monte_carlo
from gold_forecast.v1_entry_quality import _stress_test
from gold_forecast.v1_entry_quality_path import (
    CONFIRMATION_END,
    CONFIRMATION_START,
    DEVELOPMENT_END,
    DEVELOPMENT_START,
    FOLDS,
    _unique_signals,
)
from gold_forecast.v1_fixed_delay import _build_fixed_delay_signals
from gold_forecast.v1_regime_classifier import (
    FEATURE_COLUMNS,
    _classifier_frame,
    _future_labels,
    _ohlc_bars,
    _rule_probabilities,
    _state_machine,
    _timeframe_features,
)
from gold_forecast.v1_risk_control import (
    RiskControlConfig,
    _metric_values,
    _simulate_risk_control,
)
from gold_forecast.v1_signal_quality import SignalQualityConfig, _entry_features
from gold_forecast.v1_trend_strength_stability import _extended_data_audit
from gold_forecast.v1_unified_benchmark import _unified_best


TRAIN_START = pd.Timestamp("2022-01-01")
TRAIN_END = pd.Timestamp("2022-12-31 23:59:59")
CALIBRATION_START = pd.Timestamp("2023-01-01")
CALIBRATION_END = pd.Timestamp("2023-06-30 23:59:59")
THRESHOLD_START = pd.Timestamp("2023-07-01")
THRESHOLD_END = pd.Timestamp("2023-12-31 23:59:59")
VALIDATION_START = pd.Timestamp("2024-01-01")
VALIDATION_END = pd.Timestamp("2024-12-31 23:59:59")
LOCKED_START = pd.Timestamp("2025-01-01")
LOCKED_END = pd.Timestamp("2025-12-31 23:59:59")
HORIZONS = (4, 6, 8)
MODEL_NAMES = ("Hierarchical Logistic", "Hierarchical Boosting", "Hierarchical Ensemble")
CANDIDATES = (
    "Regime Classifier v2 Control",
    "Hierarchical Logistic",
    "Hierarchical Boosting",
    "Hierarchical Ensemble",
    "Ensemble Soft Gate",
    "Ensemble Adaptive Confirmation",
)


@dataclass(frozen=True)
class Thresholds:
    trend: float
    direction: float
    moderate_trend: float
    moderate_direction: float


def run_v1_regime_classifier_v3_lab(
    gold_m1: pd.DataFrame,
    signal_daily: pd.DataFrame,
    frozen_payload: dict[str, object],
) -> dict[str, object]:
    data = _prepare_m1(gold_m1)
    _, leaderboard, _ = frozen_payload["v1"]
    best = _unified_best(leaderboard.iloc[0].to_dict())
    entry_features = _entry_features(data)
    spread_limit = float(
        data.loc[DEVELOPMENT_START:DEVELOPMENT_END, "SpreadPoints"].quantile(0.90)
    )
    balanced_config = SignalQualityConfig(
        "Balanced Entry Frozen",
        "Trend engine",
        conviction_multiplier=1.05,
        require_h1_trend=True,
        wait_hours=2,
    )
    simulation_config = RiskControlConfig(
        "Regime Classifier v3",
        "Hierarchical soft gate",
        max_total_positions=1,
        max_same_direction=1,
    )

    base = _classifier_frame(data).drop(columns=["label"], errors="ignore")
    model_runs, model_selection = _train_hierarchical_candidates(base, data)
    selected_runs = _select_model_horizons(model_runs, model_selection)
    v2_probabilities = _rule_probabilities(base.dropna(subset=FEATURE_COLUMNS))
    v2_states = _state_machine(
        v2_probabilities,
        base.loc[v2_probabilities.index],
    )

    classification = _classification_tables(
        selected_runs, v2_states, base, data
    )
    balanced = _unique_signals(
        _balanced_signals(
            data,
            signal_daily,
            best,
            entry_features,
            balanced_config,
            spread_limit,
            DEVELOPMENT_START,
            CONFIRMATION_END,
        )
    )
    candidate_inputs, input_audit = _candidate_inputs(
        balanced,
        entry_features,
        selected_runs,
        v2_states,
    )
    signals, delay_audit = _delay_candidates(
        data,
        candidate_inputs,
        selected_runs,
        entry_features,
        best,
        spread_limit,
    )
    fixed_delay_reference, reference_events = _build_fixed_delay_signals(
        data, balanced, best, 5, spread_limit
    )
    fixed_delay_reference = _unique_signals(fixed_delay_reference)
    input_audit = pd.concat(
        [
            pd.DataFrame([{
                "Kandidat": "Fixed Delay 5m Reference",
                "Sinyal Balanced": len(balanced),
                "Lolos classifier": len(balanced),
                "Retensi classifier (%)": 100.0,
            }]),
            input_audit,
        ],
        ignore_index=True,
    )
    delay_audit = pd.concat(
        [
            pd.DataFrame([{
                "Kandidat": "Fixed Delay 5m Reference",
                "Sinyal sebelum delay": len(balanced),
                "Lolos barrier/spread": len(fixed_delay_reference),
                "Lolos konfirmasi kedua": len(fixed_delay_reference),
                "Batal barrier": int(reference_events["expired"].sum()),
                "Batal spread": int(
                    (~reference_events["spread_ok"] & ~reference_events["expired"]).sum()
                ),
            }]),
            delay_audit,
        ],
        ignore_index=True,
    )

    development_data = data.loc[DEVELOPMENT_START:DEVELOPMENT_END]
    reference_data = data.loc[CONFIRMATION_START:CONFIRMATION_END]
    development_results = _simulate_period(
        development_data, signals, best, simulation_config,
        DEVELOPMENT_START, DEVELOPMENT_END,
    )
    reference_results = _simulate_period(
        reference_data, signals, best, simulation_config,
        CONFIRMATION_START, CONFIRMATION_END,
    )
    fixed_delay_development_result = _simulate_risk_control(
        development_data,
        fixed_delay_reference.loc[DEVELOPMENT_START:DEVELOPMENT_END],
        best,
        simulation_config,
    )
    fixed_delay_reference_result = _simulate_risk_control(
        reference_data,
        fixed_delay_reference.loc[CONFIRMATION_START:CONFIRMATION_END],
        best,
        simulation_config,
    )
    development = _result_table(
        development_results, signals, DEVELOPMENT_START, DEVELOPMENT_END
    )
    reference = _result_table(
        reference_results, signals, CONFIRMATION_START, CONFIRMATION_END
    )
    period_validation = _period_validation(
        data, signals, best, simulation_config
    )
    folds = _fold_evaluation(data, signals, best, simulation_config)
    retention = _retention_table(signals, fixed_delay_reference)
    monte_carlo = _monte_carlo_summary(development_results)
    direction = _direction_audit(development_results, reference_results)
    calibration_audit = _probability_calibration_audit(selected_runs)
    decisions = _decision_table(
        classification["locked"],
        development,
        period_validation,
        folds,
        retention,
        monte_carlo,
        direction,
    )
    ranking = _ranking_table(
        classification["locked"],
        development,
        reference,
        retention,
        decisions,
    )
    stress_candidates = list(
        ranking.loc[ranking["Kriteria classifier lolos"].ge(4), "Kandidat"].head(2)
    )
    stress = _stress_summary(
        development_data,
        signals,
        best,
        simulation_config,
        stress_candidates,
    )
    rejected = _rejected_trade_audit(
        development_results,
        reference_results,
        fixed_delay_development_result,
        fixed_delay_reference_result,
    )

    return {
        "methodology": {
            "Name": "v1 Regime Classifier Lab v3 - Hierarchical Soft Gate",
            "Architecture": (
                "Stage 1 P(trending) -> Stage 2 P(direction | trending) -> "
                "strong/moderate/weak gate"
            ),
            "Train": "2022",
            "Probability calibration": "2023H1, Platt calibration",
            "Threshold calibration": "2023H2, tanpa label profit",
            "Model selection": "2024, berdasarkan kualitas klasifikasi",
            "Locked confirmation": "2025, tanpa retuning",
            "Historical reference": "2026H1, tidak menentukan pemenang",
            "Horizons": "4, 6, dan 8 jam; dipilih per model hanya pada 2024",
            "Execution contract": (
                "Equity USD 1.000 | lot 0.01 | TP USD 25 | SL USD 10 | "
                "maksimal 1 posisi | Balanced Entry | Fixed Delay 5m"
            ),
            "Baseline lock": (
                "Baseline v1, Fixed Delay paper live, ledger, dan parameter observasi tidak diubah"
            ),
        },
        "data_audit": _extended_data_audit(data),
        "model_selection": model_selection,
        "selected_models": _selected_model_table(selected_runs),
        "classification_validation": classification["validation"],
        "classification_locked": classification["locked"],
        "classification_reference": classification["reference"],
        "calibration_audit": calibration_audit,
        "input_audit": input_audit,
        "delay_audit": delay_audit,
        "development": development,
        "period_validation": period_validation,
        "historical_reference": reference,
        "fixed_delay_reference": pd.DataFrame([
            {
                "Periode": "Development 2022-2025",
                **_metric_values(fixed_delay_development_result),
            },
            {
                "Periode": "Historical reference 2026H1",
                **_metric_values(fixed_delay_reference_result),
            },
        ]),
        "folds": folds,
        "retention": retention,
        "monte_carlo_summary": monte_carlo,
        "direction_audit": direction,
        "stress_summary": stress,
        "rejected_trade_audit": rejected,
        "decisions": decisions,
        "ranking": ranking,
        "winner": str(ranking.iloc[0]["Kandidat"]),
    }


def _label_frame(data: pd.DataFrame, base: pd.DataFrame, horizon: int) -> pd.DataFrame:
    h1 = _ohlc_bars(data, "1h")
    features = _timeframe_features(h1, "h1")
    signed, efficiency = _future_labels(h1["Close"], features["atr"], horizon)
    frame = base.copy()
    frame["signed_atr"] = signed.reindex(frame.index)
    frame["future_efficiency"] = efficiency.reindex(frame.index)
    frame["truth"] = "TRANSITION"
    trend_up = frame["signed_atr"].ge(0.80) & frame["future_efficiency"].ge(0.45)
    trend_down = frame["signed_atr"].le(-0.80) & frame["future_efficiency"].ge(0.45)
    sideways = frame["signed_atr"].abs().le(0.80) & frame["future_efficiency"].le(0.35)
    frame.loc[trend_up, "truth"] = "TREND_UP"
    frame.loc[trend_down, "truth"] = "TREND_DOWN"
    frame.loc[sideways, "truth"] = "SIDEWAYS"
    frame.loc[frame["signed_atr"].isna() | frame["future_efficiency"].isna(), "truth"] = np.nan
    return frame


def _train_hierarchical_candidates(base, data):
    runs = {}
    selection_rows = []
    for horizon in HORIZONS:
        frame = _label_frame(data, base, horizon).dropna(
            subset=[*FEATURE_COLUMNS, "truth"]
        )
        train = frame.loc[TRAIN_START:TRAIN_END]
        calibrate = frame.loc[CALIBRATION_START:CALIBRATION_END]
        threshold_data = frame.loc[THRESHOLD_START:THRESHOLD_END]
        validation = frame.loc[VALIDATION_START:VALIDATION_END]
        estimators = _fit_base_estimators(train)
        raw_all = _raw_model_probabilities(estimators, frame)
        for model_name in MODEL_NAMES:
            calibrated = _calibrate_probabilities(
                raw_all[model_name], calibrate
            )
            probabilities = _apply_calibration(raw_all[model_name], calibrated)
            thresholds = _choose_thresholds(
                threshold_data, probabilities.loc[threshold_data.index]
            )
            states = _states_from_probabilities(probabilities, thresholds, soft=False)
            metrics = _classifier_metrics(
                validation["truth"], states.reindex(validation.index)
            )
            score = _selection_score(metrics)
            runs[(model_name, horizon)] = {
                "frame": frame,
                "probabilities": probabilities,
                "thresholds": thresholds,
                "states": states,
                "calibrators": calibrated,
            }
            selection_rows.append({
                "Model": model_name,
                "Horizon (jam)": horizon,
                **metrics,
                "Selection score": score,
            })
    return runs, pd.DataFrame(selection_rows)


def _fit_base_estimators(train):
    train_binary = train[train["truth"].isin(["TREND_UP", "TREND_DOWN", "SIDEWAYS"])]
    y_trend = train_binary["truth"].isin(["TREND_UP", "TREND_DOWN"]).astype(int)
    train_direction = train[train["truth"].isin(["TREND_UP", "TREND_DOWN"])]
    y_direction = train_direction["truth"].eq("TREND_UP").astype(int)

    logistic_trend = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", C=0.5, random_state=42),
    )
    logistic_direction = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", C=0.5, random_state=42),
    )
    boosting_trend = HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=140, max_depth=3,
        min_samples_leaf=30, l2_regularization=1.0, random_state=42,
    )
    boosting_direction = HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=140, max_depth=3,
        min_samples_leaf=30, l2_regularization=1.0, random_state=43,
    )
    logistic_trend.fit(train_binary[FEATURE_COLUMNS], y_trend)
    logistic_direction.fit(train_direction[FEATURE_COLUMNS], y_direction)
    boosting_trend.fit(
        train_binary[FEATURE_COLUMNS], y_trend,
        sample_weight=_balanced_weights(y_trend),
    )
    boosting_direction.fit(
        train_direction[FEATURE_COLUMNS], y_direction,
        sample_weight=_balanced_weights(y_direction),
    )
    return {
        "logistic_trend": logistic_trend,
        "logistic_direction": logistic_direction,
        "boosting_trend": boosting_trend,
        "boosting_direction": boosting_direction,
    }


def _balanced_weights(y):
    counts = y.value_counts()
    return y.map({label: len(y) / (len(counts) * count) for label, count in counts.items()}).to_numpy()


def _raw_model_probabilities(estimators, frame):
    x = frame[FEATURE_COLUMNS]
    logistic = pd.DataFrame({
        "trend": estimators["logistic_trend"].predict_proba(x)[:, 1],
        "up": estimators["logistic_direction"].predict_proba(x)[:, 1],
    }, index=frame.index)
    boosting = pd.DataFrame({
        "trend": estimators["boosting_trend"].predict_proba(x)[:, 1],
        "up": estimators["boosting_direction"].predict_proba(x)[:, 1],
    }, index=frame.index)
    return {
        "Hierarchical Logistic": logistic,
        "Hierarchical Boosting": boosting,
        "Hierarchical Ensemble": (logistic + boosting) / 2,
    }


def _calibrate_probabilities(probabilities, frame):
    aligned = probabilities.reindex(frame.index)
    trend_rows = frame["truth"].isin(["TREND_UP", "TREND_DOWN", "SIDEWAYS"])
    direction_rows = frame["truth"].isin(["TREND_UP", "TREND_DOWN"])
    trend_calibrator = _fit_platt(
        aligned.loc[trend_rows, "trend"],
        frame.loc[trend_rows, "truth"].isin(["TREND_UP", "TREND_DOWN"]).astype(int),
    )
    direction_calibrator = _fit_platt(
        aligned.loc[direction_rows, "up"],
        frame.loc[direction_rows, "truth"].eq("TREND_UP").astype(int),
    )
    return {"trend": trend_calibrator, "up": direction_calibrator}


def _fit_platt(probability, target):
    probability = probability.clip(1e-6, 1 - 1e-6)
    logit = np.log(probability / (1 - probability)).to_numpy().reshape(-1, 1)
    model = LogisticRegression(max_iter=500, random_state=42)
    model.fit(logit, target)
    return model


def _apply_calibration(probabilities, calibrators):
    output = pd.DataFrame(index=probabilities.index)
    for column in ("trend", "up"):
        probability = probabilities[column].clip(1e-6, 1 - 1e-6)
        logit = np.log(probability / (1 - probability)).to_numpy().reshape(-1, 1)
        output[column] = calibrators[column].predict_proba(logit)[:, 1]
    output["direction_confidence"] = np.maximum(output["up"], 1 - output["up"])
    return output


def _choose_thresholds(frame, probabilities):
    rows = []
    for trend_threshold in (0.50, 0.55, 0.60, 0.65, 0.70):
        for direction_threshold in (0.50, 0.55, 0.60, 0.65, 0.70):
            thresholds = Thresholds(
                trend_threshold,
                direction_threshold,
                max(0.45, trend_threshold - 0.10),
                max(0.50, direction_threshold - 0.05),
            )
            states = _states_from_probabilities(probabilities, thresholds, soft=False)
            metrics = _classifier_metrics(frame["truth"], states)
            rows.append((thresholds, metrics, _threshold_score(metrics)))
    eligible = [
        row for row in rows
        if row[1]["Trend precision"] >= 0.55
        and row[1]["Trend coverage (%)"] >= 35.0
    ]
    pool = eligible if eligible else rows
    return max(pool, key=lambda item: item[2])[0]


def _states_from_probabilities(probabilities, thresholds, *, soft):
    trend_limit = thresholds.moderate_trend if soft else thresholds.trend
    direction_limit = thresholds.moderate_direction if soft else thresholds.direction
    direction_confidence = probabilities["direction_confidence"]
    trending = probabilities["trend"].ge(trend_limit) & direction_confidence.ge(direction_limit)
    sideways = probabilities["trend"].le(1 - trend_limit)
    states = pd.Series("TRANSITION", index=probabilities.index, dtype="object")
    states.loc[sideways] = "SIDEWAYS"
    states.loc[trending & probabilities["up"].ge(0.50)] = "TREND_UP"
    states.loc[trending & probabilities["up"].lt(0.50)] = "TREND_DOWN"
    return states


def _classifier_metrics(truth, prediction):
    aligned = pd.concat(
        [truth.rename("truth"), prediction.rename("prediction")], axis=1
    ).dropna()
    actual_trend = aligned["truth"].isin(["TREND_UP", "TREND_DOWN"])
    predicted_trend = aligned["prediction"].isin(["TREND_UP", "TREND_DOWN"])
    direction_correct = (
        (aligned["truth"].eq("TREND_UP") & aligned["prediction"].eq("TREND_UP"))
        | (aligned["truth"].eq("TREND_DOWN") & aligned["prediction"].eq("TREND_DOWN"))
    )
    false_trend = predicted_trend & ~actual_trend
    trend_precision = precision_score(actual_trend, predicted_trend, zero_division=0)
    trend_recall = recall_score(actual_trend, predicted_trend, zero_division=0)
    direction_precision = (
        float(direction_correct.sum() / max(predicted_trend.sum(), 1))
    )
    binary_truth = actual_trend.astype(int)
    binary_prediction = predicted_trend.astype(int)
    return {
        "Observasi": int(len(aligned)),
        "Trend precision": float(trend_precision),
        "Trend recall": float(trend_recall),
        "Direction precision": direction_precision,
        "Trend F1": float(f1_score(binary_truth, binary_prediction, zero_division=0)),
        "Balanced accuracy": float(
            balanced_accuracy_score(binary_truth, binary_prediction)
        ),
        "Trend coverage (%)": float(predicted_trend.mean() * 100),
        "False trend rate (%)": float(
            false_trend.sum() / max(predicted_trend.sum(), 1) * 100
        ),
        "Median delay (jam)": _median_detection_delay(
            aligned["truth"], aligned["prediction"]
        ),
    }


def _median_detection_delay(truth, prediction):
    episodes = []
    previous = None
    for timestamp, label in truth.items():
        if label in ("TREND_UP", "TREND_DOWN") and label != previous:
            episodes.append((timestamp, label))
        previous = label
    delays = []
    for start, label in episodes:
        window = prediction.loc[start:start + pd.Timedelta(hours=8)]
        hits = window[window.eq(label)]
        if not hits.empty:
            delays.append((hits.index[0] - start).total_seconds() / 3600)
    return float(np.median(delays)) if delays else np.nan


def _threshold_score(metrics):
    return (
        metrics["Trend precision"] * 35
        + metrics["Trend recall"] * 20
        + metrics["Direction precision"] * 25
        + metrics["Balanced accuracy"] * 20
        - metrics["False trend rate (%)"] * 0.20
        - min(metrics["Median delay (jam)"], 8) * 1.5
    )


def _selection_score(metrics):
    return _threshold_score(metrics)


def _select_model_horizons(runs, selection):
    selected = {}
    for model_name in MODEL_NAMES:
        rows = selection[selection["Model"].eq(model_name)]
        best = rows.sort_values(
            ["Selection score", "Trend precision", "Balanced accuracy"],
            ascending=False,
        ).iloc[0]
        selected[model_name] = runs[(model_name, int(best["Horizon (jam)"]))]
        selected[model_name]["horizon"] = int(best["Horizon (jam)"])
    return selected


def _classification_tables(selected_runs, v2_states, base, data):
    labels = {
        model: _label_frame(data, base, run["horizon"])
        for model, run in selected_runs.items()
    }
    v2_labels = _label_frame(data, base, 8)
    periods = {
        "validation": (VALIDATION_START, VALIDATION_END),
        "locked": (LOCKED_START, LOCKED_END),
        "reference": (CONFIRMATION_START, CONFIRMATION_END),
    }
    output = {}
    for key, (start, end) in periods.items():
        rows = [{
            "Kandidat": CANDIDATES[0],
            "Horizon (jam)": 8,
            **_classifier_metrics(
                v2_labels.loc[start:end, "truth"],
                v2_states.reindex(v2_labels.loc[start:end].index),
            ),
        }]
        for model_name in MODEL_NAMES:
            run = selected_runs[model_name]
            frame = labels[model_name].loc[start:end]
            rows.append({
                "Kandidat": model_name,
                "Horizon (jam)": run["horizon"],
                **_classifier_metrics(
                    frame["truth"], run["states"].reindex(frame.index)
                ),
            })
        ensemble = selected_runs["Hierarchical Ensemble"]
        frame = labels["Hierarchical Ensemble"].loc[start:end]
        soft_states = _states_from_probabilities(
            ensemble["probabilities"], ensemble["thresholds"], soft=True
        )
        soft_metrics = _classifier_metrics(
            frame["truth"], soft_states.reindex(frame.index)
        )
        rows.append({
            "Kandidat": "Ensemble Soft Gate",
            "Horizon (jam)": ensemble["horizon"],
            **soft_metrics,
        })
        rows.append({
            "Kandidat": "Ensemble Adaptive Confirmation",
            "Horizon (jam)": ensemble["horizon"],
            **soft_metrics,
        })
        output[key] = pd.DataFrame(rows)
    return output


def _candidate_inputs(balanced, entry_features, selected_runs, v2_states):
    expected = pd.to_numeric(balanced["expected_change_pct"], errors="coerce")
    direction = pd.Series(np.where(expected.gt(0), "BUY", "SELL"), index=balanced.index)
    masks = {}
    v2 = v2_states.reindex(balanced.index, method="ffill").fillna("TRANSITION")
    masks[CANDIDATES[0]] = (
        direction.eq("BUY") & v2.eq("TREND_UP")
    ) | (
        direction.eq("SELL") & v2.eq("TREND_DOWN")
    )
    for model_name in MODEL_NAMES:
        states = selected_runs[model_name]["states"].reindex(
            balanced.index, method="ffill"
        ).fillna("TRANSITION")
        masks[model_name] = _aligned_direction(direction, states)

    ensemble = selected_runs["Hierarchical Ensemble"]
    probabilities = ensemble["probabilities"].reindex(
        balanced.index, method="ffill"
    )
    thresholds = ensemble["thresholds"]
    predicted = pd.Series(
        np.where(probabilities["up"].ge(0.50), "BUY", "SELL"),
        index=balanced.index,
    )
    direction_aligned = direction.eq(predicted)
    strong = (
        probabilities["trend"].ge(thresholds.trend)
        & probabilities["direction_confidence"].ge(thresholds.direction)
        & direction_aligned
    )
    moderate = (
        probabilities["trend"].ge(thresholds.moderate_trend)
        & probabilities["direction_confidence"].ge(thresholds.moderate_direction)
        & direction_aligned
    )
    base = _classifier_frame_from_run(ensemble).reindex(
        balanced.index, method="ffill"
    )
    breakout = (
        direction.eq("BUY") & base["breakout_up"].eq(1)
    ) | (
        direction.eq("SELL") & base["breakout_down"].eq(1)
    )
    momentum = base["adx_change_3"].gt(0)
    m15 = _m15_alignment(entry_features.reindex(balanced.index), direction)
    masks["Ensemble Soft Gate"] = strong | (moderate & breakout & momentum)
    masks["Ensemble Adaptive Confirmation"] = strong | (moderate & m15)
    inputs = {
        candidate: balanced.loc[masks[candidate].fillna(False)].copy()
        for candidate in CANDIDATES
    }
    audit = pd.DataFrame([
        {
            "Kandidat": candidate,
            "Sinyal Balanced": len(balanced),
            "Lolos classifier": len(inputs[candidate]),
            "Retensi classifier (%)": len(inputs[candidate]) / max(len(balanced), 1) * 100,
        }
        for candidate in CANDIDATES
    ])
    return inputs, audit


def _classifier_frame_from_run(run):
    return run["frame"][FEATURE_COLUMNS]


def _aligned_direction(direction, states):
    return (
        direction.eq("BUY") & states.eq("TREND_UP")
    ) | (
        direction.eq("SELL") & states.eq("TREND_DOWN")
    )


def _m15_alignment(features, direction):
    sign = pd.Series(np.where(direction.eq("BUY"), 1.0, -1.0), index=features.index)
    return (
        sign * (features["price"] - features["m15_fast"]) > 0
    ) & (
        sign * (features["m15_fast"] - features["m15_slow"]) > 0
    ) & (
        sign * features["m15_momentum"] > 0
    )


def _delay_candidates(data, inputs, selected_runs, entry_features, best, spread_limit):
    output = {}
    rows = []
    ensemble = selected_runs["Hierarchical Ensemble"]
    probabilities = ensemble["probabilities"]
    thresholds = ensemble["thresholds"]
    for candidate in CANDIDATES:
        delayed, events = _build_fixed_delay_signals(
            data, inputs[candidate], best, 5, spread_limit
        )
        before_second_check = len(delayed)
        if candidate == "Ensemble Adaptive Confirmation" and not delayed.empty:
            expected = pd.to_numeric(delayed["expected_change_pct"], errors="coerce")
            direction = pd.Series(np.where(expected.gt(0), "BUY", "SELL"), index=delayed.index)
            probability = probabilities.reindex(delayed.index, method="ffill")
            predicted = pd.Series(
                np.where(probability["up"].ge(0.50), "BUY", "SELL"),
                index=delayed.index,
            )
            moderate = (
                probability["trend"].ge(thresholds.moderate_trend)
                & probability["direction_confidence"].ge(thresholds.moderate_direction)
                & direction.eq(predicted)
            )
            m15 = _m15_alignment(entry_features.reindex(delayed.index), direction)
            delayed = delayed.loc[(moderate & m15).fillna(False)]
        output[candidate] = _unique_signals(delayed)
        rows.append({
            "Kandidat": candidate,
            "Sinyal sebelum delay": len(inputs[candidate]),
            "Lolos barrier/spread": before_second_check,
            "Lolos konfirmasi kedua": len(output[candidate]),
            "Batal barrier": int(events["expired"].sum()) if not events.empty else 0,
            "Batal spread": int((~events["spread_ok"] & ~events["expired"]).sum()) if not events.empty else 0,
        })
    return output, pd.DataFrame(rows)


def _simulate_period(data, signals, best, config, start, end):
    return {
        candidate: _simulate_risk_control(
            data, signals[candidate].loc[start:end], best, config
        )
        for candidate in CANDIDATES
    }


def _result_table(results, signals, start, end):
    return pd.DataFrame([
        {
            "Kandidat": candidate,
            "Sinyal tersedia": len(signals[candidate].loc[start:end]),
            **_metric_values(results[candidate]),
        }
        for candidate in CANDIDATES
    ])


def _period_validation(data, signals, best, config):
    periods = (
        ("Calibration 2022-2023", DEVELOPMENT_START, THRESHOLD_END),
        ("Model selection 2024", VALIDATION_START, VALIDATION_END),
        ("Locked confirmation 2025", LOCKED_START, LOCKED_END),
    )
    rows = []
    for label, start, end in periods:
        for candidate in CANDIDATES:
            selected = signals[candidate].loc[start:end]
            result = _simulate_risk_control(
                data.loc[start:end], selected, best, config
            )
            rows.append({
                "Periode": label,
                "Kandidat": candidate,
                "Sinyal tersedia": len(selected),
                **_metric_values(result),
            })
    return pd.DataFrame(rows)


def _fold_evaluation(data, signals, best, config):
    rows = []
    for fold in FOLDS:
        for candidate in CANDIDATES:
            result = _simulate_risk_control(
                data.loc[fold.test_start:fold.test_end],
                signals[candidate].loc[fold.test_start:fold.test_end],
                best,
                config,
            )
            metrics = _metric_values(result)
            rows.append({
                "Fold": fold.name,
                "Kelompok": "Calibration diagnostic" if fold.test_start.year == 2023 else "Primary validation",
                "Kandidat": candidate,
                "Test mulai": fold.test_start,
                "Test akhir": fold.test_end,
                **metrics,
                "Profitable": bool(metrics["Growth (%)"] > 0),
            })
    return pd.DataFrame(rows)


def _retention_table(signals, fixed_delay_reference):
    control_dev = max(
        len(fixed_delay_reference.loc[DEVELOPMENT_START:DEVELOPMENT_END]), 1
    )
    control_ref = max(
        len(fixed_delay_reference.loc[CONFIRMATION_START:CONFIRMATION_END]), 1
    )
    return pd.DataFrame([
        {
            "Kandidat": candidate,
            "Sinyal development": len(signals[candidate].loc[DEVELOPMENT_START:DEVELOPMENT_END]),
            "Retensi development (%)": len(signals[candidate].loc[DEVELOPMENT_START:DEVELOPMENT_END]) / control_dev * 100,
            "Sinyal 2026H1": len(signals[candidate].loc[CONFIRMATION_START:CONFIRMATION_END]),
            "Retensi 2026H1 (%)": len(signals[candidate].loc[CONFIRMATION_START:CONFIRMATION_END]) / control_ref * 100,
        }
        for candidate in CANDIDATES
    ])


def _monte_carlo_summary(results):
    rows = []
    for candidate in CANDIDATES:
        _, summary = _safe_monte_carlo(results[candidate].trades)
        rows.append({"Kandidat": candidate, **summary})
    return pd.DataFrame(rows)


def _direction_audit(development_results, reference_results):
    rows = []
    for period, results in (
        ("Development 2022-2025", development_results),
        ("Historical reference 2026H1", reference_results),
    ):
        for candidate in CANDIDATES:
            trades = results[candidate].trades
            for direction in ("BUY", "SELL"):
                selected = trades[
                    trades.get("Arah", pd.Series(index=trades.index, dtype=object)).eq(direction)
                ]
                rows.append({
                    "Periode": period,
                    "Kandidat": candidate,
                    "Arah": direction,
                    **_trade_metrics(selected),
                })
    return pd.DataFrame(rows)


def _probability_calibration_audit(selected_runs):
    rows = []
    for model_name, run in selected_runs.items():
        frame = run["frame"]
        probabilities = run["probabilities"]
        for period, start, end in (
            ("Model selection 2024", VALIDATION_START, VALIDATION_END),
            ("Locked confirmation 2025", LOCKED_START, LOCKED_END),
            ("Historical reference 2026H1", CONFIRMATION_START, CONFIRMATION_END),
        ):
            selected = frame.loc[start:end]
            trend_rows = selected["truth"].isin(["TREND_UP", "TREND_DOWN", "SIDEWAYS"])
            y = selected.loc[trend_rows, "truth"].isin(["TREND_UP", "TREND_DOWN"]).astype(int)
            p = probabilities.reindex(selected.index).loc[trend_rows, "trend"]
            rows.append({
                "Periode": period,
                "Model": model_name,
                "Horizon (jam)": run["horizon"],
                "Brier trend": float(brier_score_loss(y, p)),
                "Rata-rata P(trend)": float(p.mean()),
                "Frekuensi trend aktual": float(y.mean()),
            })
    return pd.DataFrame(rows)


def _decision_table(classification, development, periods, folds, retention, monte_carlo, direction):
    cls = classification.set_index("Kandidat")
    dev = development.set_index("Kandidat")
    period = periods.set_index(["Periode", "Kandidat"])
    retained = retention.set_index("Kandidat")
    mc = monte_carlo.set_index("Kandidat")
    rows = []
    for candidate in CANDIDATES:
        primary = folds[
            folds["Kandidat"].eq(candidate)
            & folds["Kelompok"].eq("Primary validation")
        ]
        candidate_direction = direction[
            direction["Kandidat"].eq(candidate)
            & direction["Periode"].eq("Development 2022-2025")
        ]
        total = max(float(candidate_direction["Transaksi"].sum()), 1)
        minor_share = float(candidate_direction["Transaksi"].min() / total * 100)
        classifier_criteria = {
            "Trend precision >= 60%": float(cls.loc[candidate, "Trend precision"]) >= 0.60,
            "Trend recall >= 50%": float(cls.loc[candidate, "Trend recall"]) >= 0.50,
            "Direction precision >= 60%": float(cls.loc[candidate, "Direction precision"]) >= 0.60,
            "Balanced accuracy >= 55%": float(cls.loc[candidate, "Balanced accuracy"]) >= 0.55,
            "False trend <= 30%": float(cls.loc[candidate, "False trend rate (%)"]) <= 30.0,
            "Coverage >= 55%": float(cls.loc[candidate, "Trend coverage (%)"]) >= 55.0,
        }
        economic_criteria = {
            "Growth development positif": float(dev.loc[candidate, "Growth (%)"]) > 0,
            "PF development >= 1.50": float(dev.loc[candidate, "Profit factor"]) >= 1.50,
            "DD development <= 10%": float(dev.loc[candidate, "Max drawdown (%)"]) <= 10.0,
            "Retensi >= 60%": float(retained.loc[candidate, "Retensi development (%)"]) >= 60.0,
            "2024 positif": float(period.loc[("Model selection 2024", candidate), "Growth (%)"]) > 0,
            "2025 positif": float(period.loc[("Locked confirmation 2025", candidate), "Growth (%)"]) > 0,
            "Primary fold profitable >= 6/8": int(primary["Profitable"].sum()) >= 6,
            "Monte Carlo rugi <= 10%": float(
                mc.loc[candidate, "Probabilitas equity akhir < modal awal (%)"]
            ) <= 10.0,
            "Arah minor >= 15%": minor_share >= 15.0,
        }
        rows.append({
            "Kandidat": candidate,
            **{**classifier_criteria, **economic_criteria},
            "Primary fold profitable": int(primary["Profitable"].sum()),
            "Porsi arah minor (%)": minor_share,
            "Kriteria classifier lolos": int(sum(classifier_criteria.values())),
            "Kriteria ekonomi lolos": int(sum(economic_criteria.values())),
            "Total kriteria lolos": int(sum(classifier_criteria.values()) + sum(economic_criteria.values())),
            "Total kriteria": len(classifier_criteria) + len(economic_criteria),
            "Lulus": bool(all(classifier_criteria.values()) and all(economic_criteria.values())),
        })
    return pd.DataFrame(rows)


def _ranking_table(classification, development, reference, retention, decisions):
    cls = classification.set_index("Kandidat")
    dev = development.set_index("Kandidat")
    ref = reference.set_index("Kandidat")
    retained = retention.set_index("Kandidat")
    decision = decisions.set_index("Kandidat")
    rows = []
    for candidate in CANDIDATES:
        rows.append({
            "Kandidat": candidate,
            "Trend precision 2025": float(cls.loc[candidate, "Trend precision"]),
            "Trend recall 2025": float(cls.loc[candidate, "Trend recall"]),
            "Direction precision 2025": float(cls.loc[candidate, "Direction precision"]),
            "False trend 2025 (%)": float(cls.loc[candidate, "False trend rate (%)"]),
            "Coverage 2025 (%)": float(cls.loc[candidate, "Trend coverage (%)"]),
            "Growth development (%)": float(dev.loc[candidate, "Growth (%)"]),
            "PF development": float(dev.loc[candidate, "Profit factor"]),
            "DD development (%)": float(dev.loc[candidate, "Max drawdown (%)"]),
            "Transaksi development": int(dev.loc[candidate, "Transaksi"]),
            "Retensi development (%)": float(retained.loc[candidate, "Retensi development (%)"]),
            "Growth 2026H1 (%)": float(ref.loc[candidate, "Growth (%)"]),
            "PF 2026H1": float(ref.loc[candidate, "Profit factor"]),
            "DD 2026H1 (%)": float(ref.loc[candidate, "Max drawdown (%)"]),
            "Kriteria classifier lolos": int(decision.loc[candidate, "Kriteria classifier lolos"]),
            "Kriteria ekonomi lolos": int(decision.loc[candidate, "Kriteria ekonomi lolos"]),
            "Total kriteria lolos": int(decision.loc[candidate, "Total kriteria lolos"]),
            "Lulus": bool(decision.loc[candidate, "Lulus"]),
        })
    ranking = pd.DataFrame(rows).sort_values(
        ["Lulus", "Kriteria classifier lolos", "Kriteria ekonomi lolos", "Trend precision 2025", "PF development"],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)
    ranking.insert(0, "Peringkat", np.arange(1, len(ranking) + 1))
    return ranking


def _stress_summary(data, signals, best, config, selected_candidates):
    rows = []
    for candidate in CANDIDATES:
        if candidate not in selected_candidates:
            rows.append({
                "Kandidat": candidate,
                "Skenario profitable": 0,
                "Jumlah skenario": 9,
                "Worst growth (%)": np.nan,
                "Status": "TIDAK DIUJI - bukan shortlist classifier",
            })
            continue
        stress = _stress_test(
            data,
            signals[candidate].loc[DEVELOPMENT_START:DEVELOPMENT_END],
            best,
            config,
        )
        rows.append({
            "Kandidat": candidate,
            "Skenario profitable": int(stress["Growth (%)"].gt(0).sum()),
            "Jumlah skenario": len(stress),
            "Worst growth (%)": float(stress["Growth (%)"].min()),
            "Status": "DIUJI",
        })
    return pd.DataFrame(rows)


def _rejected_trade_audit(
    development_results,
    reference_results,
    fixed_delay_development_result,
    fixed_delay_reference_result,
):
    rows = []
    for period, results, reference_result in (
        (
            "Development 2022-2025",
            development_results,
            fixed_delay_development_result,
        ),
        (
            "Historical reference 2026H1",
            reference_results,
            fixed_delay_reference_result,
        ),
    ):
        control = reference_result.trades
        control_times = pd.to_datetime(control.get("Tanggal entry"), errors="coerce")
        for candidate in CANDIDATES[1:]:
            accepted = results[candidate].trades
            accepted_times = set(
                pd.to_datetime(accepted.get("Tanggal entry"), errors="coerce").dropna()
            )
            rejected = control.loc[~control_times.isin(accepted_times)]
            for status, frame in (("DITERIMA", accepted), ("DITOLAK", rejected)):
                rows.append({
                    "Periode": period,
                    "Kandidat": candidate,
                    "Status": status,
                    **_trade_metrics(frame),
                })
    return pd.DataFrame(rows)


def _trade_metrics(trades):
    if trades.empty:
        return {
            "Transaksi": 0,
            "Net P/L": 0.0,
            "Profit factor": np.nan,
            "Win rate (%)": np.nan,
        }
    net = pd.to_numeric(trades["Net P/L"], errors="coerce").fillna(0.0)
    profit = float(net[net > 0].sum())
    loss = float(-net[net < 0].sum())
    return {
        "Transaksi": int(len(trades)),
        "Net P/L": float(net.sum()),
        "Profit factor": profit / loss if loss > 0 else np.inf,
        "Win rate (%)": float(net.gt(0).mean() * 100),
    }


def _selected_model_table(selected_runs):
    return pd.DataFrame([
        {
            "Model": model,
            "Horizon terpilih (jam)": run["horizon"],
            "Strong P(trend)": run["thresholds"].trend,
            "Strong P(direction)": run["thresholds"].direction,
            "Moderate P(trend)": run["thresholds"].moderate_trend,
            "Moderate P(direction)": run["thresholds"].moderate_direction,
        }
        for model, run in selected_runs.items()
    ])
