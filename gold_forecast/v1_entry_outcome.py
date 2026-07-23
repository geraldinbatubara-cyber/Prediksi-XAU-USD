from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from gold_forecast.exact_broker_oos import POINT_SIZE, SLIPPAGE_POINTS, _compact_curve, _prepare_m1
from gold_forecast.simulation import CONTRACT_OUNCES_PER_LOT
from gold_forecast.v1_risk_control import (
    DEVELOPMENT_END,
    DEVELOPMENT_START,
    MAX_DRAWDOWN_PCT,
    MAX_MONTE_CARLO_LOSS_PCT,
    PROFIT_FACTOR_TARGET,
    VALIDATION_END,
    VALIDATION_START,
    RiskControlConfig,
    _entry_signals_for_period,
    _metric_values,
    _simulate_risk_control,
)
from gold_forecast.v1_robustness import _monte_carlo, _monthly_summary
from gold_forecast.v1_sideways_defense import RegimeConfig, _regime_features, _regime_states
from gold_forecast.v1_signal_quality import SignalQualityConfig, _entry_features, _select_signals


MODEL_NAMES = ("Rule Scorecard", "Logistic Regression", "Gradient Boosting", "Probability Ensemble")
OUTCOME_HORIZON_DAYS = 14
FEATURE_GROUPS = {
    "Conviction": ["conviction_ratio", "expected_abs"],
    "Trend alignment": [
        "price_fast_atr", "fast_slow_atr", "m15_alignment_atr",
        "signed_m15_momentum", "signed_h1_momentum",
    ],
    "Regime": ["adx", "efficiency", "choppiness", "trend_strength", "slope"],
    "Volatility": ["stretch_atr", "atr_percentile", "atr_change"],
    "Location": ["range_position_signed", "rsi_aligned"],
    "Execution": ["spread_points", "hour_sin", "hour_cos", "session_london", "session_new_york"],
}
FEATURE_COLUMNS = [column for columns in FEATURE_GROUPS.values() for column in columns]


@dataclass(frozen=True)
class OutcomeFold:
    name: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


FOLDS = (
    OutcomeFold(
        "Fold 1",
        pd.Timestamp("2025-01-01"),
        pd.Timestamp("2025-06-30 23:59:59"),
        pd.Timestamp("2025-07-01"),
        pd.Timestamp("2025-09-30 23:59:59"),
    ),
    OutcomeFold(
        "Fold 2",
        pd.Timestamp("2025-01-01"),
        pd.Timestamp("2025-09-30 23:59:59"),
        pd.Timestamp("2025-10-01"),
        pd.Timestamp("2025-12-31 23:59:59"),
    ),
)


def run_v1_entry_outcome_lab(
    gold_m1: pd.DataFrame,
    signal_daily: pd.DataFrame,
    frozen_payload: dict[str, object],
) -> dict[str, object]:
    data = _prepare_m1(gold_m1)
    _, leaderboard, _ = frozen_payload["v1"]
    best = leaderboard.iloc[0].to_dict()
    balanced_config = SignalQualityConfig(
        "Balanced Entry Frozen",
        "Trend engine",
        conviction_multiplier=1.05,
        require_h1_trend=True,
        wait_hours=2,
    )
    entry_features = _entry_features(data)
    regime_features, _, m15 = _regime_features(data)
    spread_limit = float(
        data.loc[DEVELOPMENT_START:DEVELOPMENT_END, "SpreadPoints"].quantile(0.90)
    )

    development_signals = _balanced_signals(
        data,
        signal_daily,
        best,
        entry_features,
        balanced_config,
        spread_limit,
        DEVELOPMENT_START,
        DEVELOPMENT_END,
    )
    validation_signals = _balanced_signals(
        data,
        signal_daily,
        best,
        entry_features,
        balanced_config,
        spread_limit,
        VALIDATION_START,
        VALIDATION_END,
    )
    feature_frame = _outcome_features(data, regime_features, m15, best)
    development_events = _event_dataset(data, development_signals, feature_frame, best)
    validation_events = _event_dataset(data, validation_signals, feature_frame, best)
    if len(development_events) < 40 or len(validation_events) < 20:
        raise ValueError("Jumlah event Balanced Entry tidak cukup untuk Entry Outcome Lab.")

    fold_rows: list[dict[str, object]] = []
    oof_frames: list[pd.DataFrame] = []
    for model_name in MODEL_NAMES:
        for fold in FOLDS:
            train = development_events.loc[fold.train_start:fold.train_end]
            test = development_events.loc[fold.test_start:fold.test_end]
            probabilities = _fit_predict_probability(model_name, train, test, FEATURE_COLUMNS)
            metrics = _probability_metrics(test["target"], probabilities)
            fold_rows.append(
                {
                    "Model": model_name,
                    "Fold": fold.name,
                    "Train events": len(train),
                    "Test events": len(test),
                    **metrics,
                }
            )
            oof_frames.append(
                pd.DataFrame(
                    {
                        "Model": model_name,
                        "probability": probabilities,
                        "target": test["target"],
                    },
                    index=test.index,
                )
            )
    folds = pd.DataFrame(fold_rows)
    oof = pd.concat(oof_frames).sort_index()
    model_summary = _model_summary(folds)
    selected_model = str(model_summary.iloc[0]["Model"])
    model_selection_fallback = not bool(
        (model_summary["Mean Brier improvement (%)"] > 0).any()
        and (model_summary["Positive Brier folds"] >= 1).any()
    )
    selected_oof = oof[oof["Model"].eq(selected_model)].copy()

    threshold_table, selected_threshold, threshold_fallback = _select_threshold(
        data,
        development_signals,
        selected_oof,
        best,
    )
    validation_probability = _fit_predict_probability(
        selected_model,
        development_events,
        validation_events,
        FEATURE_COLUMNS,
    )
    validation_events = validation_events.copy()
    validation_events["probability"] = validation_probability
    validation_metrics = _probability_metrics(
        validation_events["target"], validation_events["probability"]
    )
    validation_metrics["Model"] = selected_model
    validation_metrics["Events"] = len(validation_events)

    probability_signals = _probability_gate(
        validation_signals, validation_events["probability"], selected_threshold,
        "v1 Probability-Gated Entry",
    )
    conservative_threshold = min(selected_threshold + 0.10, 0.85)
    conservative_signals = _probability_gate(
        validation_signals, validation_events["probability"], conservative_threshold,
        "v1 Probability-Gated Conservative",
    )
    current_states = _regime_states(
        regime_features,
        RegimeConfig("RG-C Sensitive", 22.0, 0.35, 55.0, 0.45, 0.16, 3),
    )
    rgc_signals = validation_signals.loc[
        current_states.reindex(validation_signals.index, method="ffill").eq("TRENDING").to_numpy()
    ].copy()

    simulation_config = RiskControlConfig(
        "Entry Outcome Lab",
        "Probability gate",
        max_total_positions=1,
        max_same_direction=1,
    )
    validation_data = data.loc[VALIDATION_START:VALIDATION_END]
    economic_signals = {
        "Balanced Entry Frozen": validation_signals,
        "Current RG-C Trend Gate": rgc_signals,
        "v1 Probability-Gated Entry": probability_signals,
        "Probability-Gated Conservative": conservative_signals,
    }
    economic_results = {
        name: _simulate_risk_control(validation_data, signals, best, simulation_config)
        for name, signals in economic_signals.items()
    }
    economic = pd.DataFrame(
        [{"Strategi": name, **_metric_values(result)} for name, result in economic_results.items()]
    )
    selected_result = economic_results["v1 Probability-Gated Entry"]
    selected_metrics = _metric_values(selected_result)
    baseline_net = float(economic_results["Balanced Entry Frozen"].summary["Total net P/L"])
    selected_net = float(selected_result.summary["Total net P/L"])
    profit_retention = selected_net / baseline_net * 100 if baseline_net > 0 else 0.0
    entry_retention = len(probability_signals) / len(validation_signals) * 100

    calibration = _calibration_table(
        validation_events["target"], validation_events["probability"]
    )
    outcome_distribution = _outcome_distribution(development_events, validation_events)
    direction_audit = _direction_audit(validation_events)
    session_audit = _session_audit(validation_events)
    ablation = _ablation_test(selected_model, development_events)
    stress = _stress_test(validation_data, probability_signals, best, simulation_config)
    delay_stress = _delay_stress(
        validation_data, probability_signals, best, simulation_config
    )
    threshold_sensitivity = _threshold_sensitivity(
        validation_data,
        validation_signals,
        validation_events["probability"],
        best,
        simulation_config,
    )
    monte_carlo, monte_carlo_summary = _safe_monte_carlo(selected_result.trades)
    decision = _decision_table(
        validation_metrics,
        selected_metrics,
        entry_retention,
        profit_retention,
        folds,
        stress,
        monte_carlo_summary,
        selected_model,
        model_selection_fallback,
        threshold_fallback,
    )

    return {
        "methodology": {
            "Baseline lock": "Optimizer v1 live tidak diubah sampai 31 Agustus 2026",
            "Unit analisis": "Setiap sinyal Balanced Entry Frozen",
            "Development": "01 Jan 2025 - 31 Des 2025; purged expanding walk-forward",
            "Validation": "01 Jan 2026 - 30 Jun 2026; parameter dan threshold dibekukan",
            "Outcome": "TP_FIRST | SL_FIRST | TIMEOUT; AMBIGUOUS diperlakukan sebagai SL_FIRST",
            "Outcome horizon": f"{OUTCOME_HORIZON_DAYS} hari kalender",
            "Selected model": selected_model,
            "Model selection fallback": model_selection_fallback,
            "Selected threshold": selected_threshold,
            "Conservative threshold": conservative_threshold,
            "Theoretical break-even probability": (
                float(best["SL (USD)"])
                / (float(best["TP (USD)"]) + float(best["SL (USD)"]))
            ),
            "Threshold fallback": threshold_fallback,
            "Direction normalization": (
                "Fitur dinormalisasi terhadap arah sinyal dan tidak memuat identitas BUY/SELL, "
                "karena development memiliki sangat sedikit SELL."
            ),
            "Caveat": (
                "2026H1 sudah pernah diamati. Hasil validation tetap memerlukan forward paper shadow "
                "dan tidak mengubah baseline atau Live Trading."
            ),
        },
        "criteria": {
            "Growth OOS positif": True,
            "Max drawdown maksimum (%)": 10.0,
            "Profit factor minimum": 1.30,
            "Monte Carlo rugi maksimum (%)": 10.0,
            "Retensi entry minimum (%)": 60.0,
            "Retensi profit minimum (%)": 85.0,
            "Calibration error maksimum": 0.10,
        },
        "folds": folds,
        "model_summary": model_summary,
        "threshold_development": threshold_table,
        "validation_metrics": pd.DataFrame([validation_metrics]),
        "calibration": calibration,
        "outcome_distribution": outcome_distribution,
        "direction_audit": direction_audit,
        "session_audit": session_audit,
        "ablation": ablation,
        "economic": economic,
        "stress": stress,
        "delay_stress": delay_stress,
        "threshold_sensitivity": threshold_sensitivity,
        "decision": decision,
        "selected_result": _compact_curve(selected_result),
        "selected_monthly": _safe_monthly_summary(selected_result),
        "selected_monte_carlo": monte_carlo,
        "selected_monte_carlo_summary": monte_carlo_summary,
        "validation_events": _compact_events(validation_events),
        "signal_counts": {name: int(len(signals)) for name, signals in economic_signals.items()},
    }


def _balanced_signals(
    data: pd.DataFrame,
    daily: pd.DataFrame,
    best: dict[str, object],
    features: pd.DataFrame,
    config: SignalQualityConfig,
    spread_limit: float,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    raw = _entry_signals_for_period(data, daily, best, start, end)
    selected, _ = _select_signals(raw, features, best, config, spread_limit, end)
    return selected


def _outcome_features(
    data: pd.DataFrame,
    regime_features: pd.DataFrame,
    m15: pd.DataFrame,
    best: dict[str, object],
) -> pd.DataFrame:
    frame = pd.DataFrame(index=data.index)
    sign_placeholder = pd.Series(1.0, index=data.index)
    atr = regime_features["atr"].replace(0, np.nan)
    frame["price"] = regime_features["price"]
    frame["spread_points"] = regime_features["spread_points"]
    frame["stretch_atr"] = regime_features["stretch_atr"]
    frame["adx"] = regime_features["adx"]
    frame["efficiency"] = regime_features["efficiency"]
    frame["choppiness"] = regime_features["choppiness"]
    frame["trend_strength"] = regime_features["trend_strength"]
    frame["slope"] = regime_features["slope"]
    frame["atr"] = atr
    frame["atr_percentile"] = atr.rolling(24 * 60, min_periods=240).rank(pct=True)
    frame["atr_change"] = atr.pct_change(6).clip(-2, 2)
    frame["raw_price_fast_atr"] = (frame["price"] - regime_features["h1_fast"]) / atr
    frame["raw_fast_slow_atr"] = (
        regime_features["h1_fast"] - regime_features["h1_slow"]
    ) / atr
    m15_atr_scale = atr.reindex(frame.index, method="ffill")
    frame["raw_m15_alignment_atr"] = (
        regime_features["m15_fast"] - regime_features["m15_slow"]
    ) / m15_atr_scale
    frame["raw_m15_momentum"] = regime_features["m15_momentum"]
    frame["raw_h1_momentum"] = regime_features["h1_momentum"]
    range_width = (regime_features["range_high"] - regime_features["range_low"]).replace(0, np.nan)
    frame["raw_range_position"] = (
        (frame["price"] - regime_features["range_mid"]) / range_width
    ).clip(-2, 2)
    rsi = _rsi(m15["Close"], 14).reindex(frame.index, method="ffill")
    frame["raw_rsi"] = rsi
    hour = pd.Series(frame.index.hour, index=frame.index)
    frame["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    frame["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    frame["session_london"] = hour.between(7, 15).astype(float)
    frame["session_new_york"] = hour.between(13, 21).astype(float)
    frame["threshold"] = float(best["Threshold entry (%)"])
    return frame.replace([np.inf, -np.inf], np.nan)


def _event_dataset(
    data: pd.DataFrame,
    signals: pd.DataFrame,
    features: pd.DataFrame,
    best: dict[str, object],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for entry_time, signal in signals.iterrows():
        timestamp = pd.Timestamp(entry_time)
        if timestamp not in features.index or timestamp not in data.index:
            continue
        expected = float(signal["expected_change_pct"])
        direction = "BUY" if expected > 0 else "SELL"
        sign = 1.0 if direction == "BUY" else -1.0
        feature = features.loc[timestamp]
        outcome = _label_outcome(data, timestamp, direction, float(signal["lot"]), best)
        rows.append(
            {
                "entry_time": timestamp,
                "direction": direction,
                "outcome": outcome["outcome"],
                "raw_outcome": outcome["raw_outcome"],
                "target": float(outcome["outcome"] == "TP_FIRST"),
                "outcome_time": outcome["outcome_time"],
                "hours_to_outcome": outcome["hours_to_outcome"],
                "conviction_ratio": abs(expected) / float(feature["threshold"]),
                "expected_abs": abs(expected),
                "price_fast_atr": sign * float(feature["raw_price_fast_atr"]),
                "fast_slow_atr": sign * float(feature["raw_fast_slow_atr"]),
                "m15_alignment_atr": sign * float(feature["raw_m15_alignment_atr"]),
                "signed_m15_momentum": sign * float(feature["raw_m15_momentum"]),
                "signed_h1_momentum": sign * float(feature["raw_h1_momentum"]),
                "stretch_atr": float(feature["stretch_atr"]),
                "adx": float(feature["adx"]),
                "efficiency": float(feature["efficiency"]),
                "choppiness": float(feature["choppiness"]),
                "trend_strength": float(feature["trend_strength"]),
                "slope": float(feature["slope"]),
                "atr_percentile": float(feature["atr_percentile"]),
                "atr_change": float(feature["atr_change"]),
                "range_position_signed": sign * float(feature["raw_range_position"]),
                "rsi_aligned": sign * (float(feature["raw_rsi"]) - 50.0) / 50.0,
                "spread_points": float(feature["spread_points"]),
                "hour_sin": float(feature["hour_sin"]),
                "hour_cos": float(feature["hour_cos"]),
                "session_london": float(feature["session_london"]),
                "session_new_york": float(feature["session_new_york"]),
            }
        )
    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .set_index("entry_time")
        .sort_index()
        .dropna(subset=FEATURE_COLUMNS)
    )


def _label_outcome(
    data: pd.DataFrame,
    entry_time: pd.Timestamp,
    direction: str,
    lot: float,
    best: dict[str, object],
) -> dict[str, object]:
    candle = data.loc[entry_time]
    spread = max(float(candle["SpreadPoints"]) * POINT_SIZE, 0.0)
    units = lot * CONTRACT_OUNCES_PER_LOT
    slippage = SLIPPAGE_POINTS * POINT_SIZE
    if direction == "BUY":
        entry_price = float(candle["Close"]) + spread + slippage
        target = entry_price + float(best["TP (USD)"]) / units
        stop = entry_price - float(best["SL (USD)"]) / units
    else:
        entry_price = float(candle["Close"]) - slippage
        target = entry_price - float(best["TP (USD)"]) / units
        stop = entry_price + float(best["SL (USD)"]) / units
    deadline = entry_time + pd.Timedelta(days=OUTCOME_HORIZON_DAYS)
    path = data.loc[(data.index > entry_time) & (data.index <= deadline)]
    for timestamp, bar in path.iterrows():
        bar_spread = max(float(bar["SpreadPoints"]) * POINT_SIZE, 0.0)
        if direction == "BUY":
            tp_hit = float(bar["High"]) >= target
            sl_hit = float(bar["Low"]) <= stop
        else:
            ask_high = float(bar["High"]) + bar_spread
            ask_low = float(bar["Low"]) + bar_spread
            tp_hit = ask_low <= target
            sl_hit = ask_high >= stop
        if tp_hit or sl_hit:
            ambiguous = tp_hit and sl_hit
            outcome = "SL_FIRST" if sl_hit else "TP_FIRST"
            return {
                "outcome": outcome,
                "raw_outcome": "AMBIGUOUS" if ambiguous else outcome,
                "outcome_time": timestamp,
                "hours_to_outcome": (timestamp - entry_time).total_seconds() / 3600,
            }
    return {
        "outcome": "TIMEOUT",
        "raw_outcome": "TIMEOUT",
        "outcome_time": deadline,
        "hours_to_outcome": OUTCOME_HORIZON_DAYS * 24.0,
    }


def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    relative = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + relative)


def _fit_predict_probability(
    model_name: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: list[str],
) -> pd.Series:
    if test.empty:
        return pd.Series(dtype=float, index=test.index)
    y = train["target"].astype(int)
    if train.empty or y.nunique() < 2:
        probability = float(y.mean()) if not y.empty else 0.5
        return pd.Series(probability, index=test.index, dtype=float)
    if model_name == "Probability Ensemble":
        logistic = _fit_predict_probability("Logistic Regression", train, test, columns)
        boosting = _fit_predict_probability("Gradient Boosting", train, test, columns)
        return ((logistic + boosting) / 2).clip(0.01, 0.99)
    if model_name == "Rule Scorecard":
        raw_train = _rule_score(train)
        raw_test = _rule_score(test)
        return _platt_probability(raw_train, y, raw_test, test.index)

    split = max(int(len(train) * 0.75), 20)
    split = min(split, len(train) - 8)
    core = train.iloc[:split]
    calibration = train.iloc[split:]
    if core["target"].nunique() < 2 or calibration["target"].nunique() < 2:
        core = train
        calibration = train
    model = _logistic_model() if model_name == "Logistic Regression" else _boosting_model()
    weights = _class_weights(core["target"].astype(int))
    fit_kwargs = {"sample_weight": weights}
    if model_name == "Logistic Regression":
        fit_kwargs = {"logisticregression__sample_weight": weights}
    model.fit(core[columns], core["target"].astype(int), **fit_kwargs)
    calibration_raw = pd.Series(
        model.predict_proba(calibration[columns])[:, 1],
        index=calibration.index,
    )
    test_raw = pd.Series(model.predict_proba(test[columns])[:, 1], index=test.index)
    return _platt_probability(
        calibration_raw,
        calibration["target"].astype(int),
        test_raw,
        test.index,
    )


def _logistic_model():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.25,
            class_weight="balanced",
            max_iter=2000,
            random_state=42,
        ),
    )


def _boosting_model():
    return HistGradientBoostingClassifier(
        learning_rate=0.04,
        max_iter=120,
        max_leaf_nodes=7,
        min_samples_leaf=10,
        l2_regularization=2.0,
        random_state=42,
    )


def _class_weights(target: pd.Series) -> np.ndarray:
    counts = target.value_counts()
    return np.array([len(target) / (2 * counts[value]) for value in target], dtype=float)


def _rule_score(frame: pd.DataFrame) -> pd.Series:
    score = (
        0.80 * frame["conviction_ratio"].clip(0, 3)
        + 0.55 * frame["price_fast_atr"].clip(-2, 2)
        + 0.65 * frame["fast_slow_atr"].clip(-2, 2)
        + 0.35 * frame["signed_h1_momentum"].clip(-2, 2)
        + 0.25 * frame["efficiency"].clip(0, 1)
        + 0.20 * frame["adx"].clip(0, 60) / 30
        - 0.35 * frame["stretch_atr"].clip(0, 4)
        - 0.20 * frame["spread_points"].clip(lower=0) / 50
    )
    return 1 / (1 + np.exp(-(score - 1.25)))


def _platt_probability(
    calibration_probability: pd.Series,
    calibration_target: pd.Series,
    test_probability: pd.Series,
    test_index: pd.Index,
) -> pd.Series:
    calibration_probability = calibration_probability.clip(0.01, 0.99)
    test_probability = test_probability.clip(0.01, 0.99)
    if len(calibration_target) < 8 or calibration_target.nunique() < 2:
        return pd.Series(test_probability.to_numpy(), index=test_index).clip(0.01, 0.99)
    calibration_logit = np.log(calibration_probability / (1 - calibration_probability))
    test_logit = np.log(test_probability / (1 - test_probability))
    calibrator = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    calibrator.fit(calibration_logit.to_numpy().reshape(-1, 1), calibration_target)
    probability = calibrator.predict_proba(test_logit.to_numpy().reshape(-1, 1))[:, 1]
    return pd.Series(probability, index=test_index).clip(0.01, 0.99)


def _probability_metrics(target: pd.Series, probability: pd.Series) -> dict[str, float]:
    target = target.astype(int)
    probability = probability.astype(float).clip(0.001, 0.999)
    prediction = probability >= 0.50
    prevalence = float(target.mean())
    baseline_brier = float(np.mean((target - prevalence) ** 2))
    return {
        "Observasi": float(len(target)),
        "TP rate (%)": prevalence * 100,
        "Brier score": float(brier_score_loss(target, probability)),
        "Baseline Brier": baseline_brier,
        "Brier improvement (%)": (
            (baseline_brier - brier_score_loss(target, probability)) / baseline_brier * 100
            if baseline_brier > 0
            else 0.0
        ),
        "Log loss": float(log_loss(target, probability, labels=[0, 1])),
        "ROC-AUC": _safe_auc(target, probability, roc_auc_score),
        "PR-AUC": _safe_auc(target, probability, average_precision_score),
        "Precision TP": float(precision_score(target, prediction, zero_division=0)),
        "Recall TP": float(recall_score(target, prediction, zero_division=0)),
        "Calibration error": _expected_calibration_error(target, probability),
    }


def _safe_auc(target: pd.Series, probability: pd.Series, metric) -> float:
    return float(metric(target, probability)) if target.nunique() > 1 else 0.5


def _expected_calibration_error(target: pd.Series, probability: pd.Series) -> float:
    errors = []
    weights = []
    for lower, upper in ((0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)):
        mask = probability.ge(lower) & probability.lt(upper)
        if not mask.any():
            continue
        errors.append(abs(float(target[mask].mean()) - float(probability[mask].mean())))
        weights.append(int(mask.sum()))
    return float(np.average(errors, weights=weights)) if weights else 1.0


def _model_summary(folds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, group in folds.groupby("Model", sort=False):
        values = {
            "Model": model,
            "Mean Brier": float(group["Brier score"].mean()),
            "Worst Brier": float(group["Brier score"].max()),
            "Mean Brier improvement (%)": float(group["Brier improvement (%)"].mean()),
            "Mean PR-AUC": float(group["PR-AUC"].mean()),
            "Mean ROC-AUC": float(group["ROC-AUC"].mean()),
            "Mean calibration error": float(group["Calibration error"].mean()),
            "Positive Brier folds": int((group["Brier improvement (%)"] > 0).sum()),
        }
        values["Selection score"] = (
            35 * max(values["Mean Brier improvement (%)"], -25) / 100
            + 30 * values["Mean PR-AUC"]
            + 20 * values["Mean ROC-AUC"]
            + 15 * max(0, 1 - values["Mean calibration error"] / 0.20)
        )
        rows.append(values)
    return (
        pd.DataFrame(rows)
        .sort_values(
            ["Positive Brier folds", "Selection score", "Mean Brier"],
            ascending=[False, False, True],
        )
        .reset_index(drop=True)
    )


def _select_threshold(
    data: pd.DataFrame,
    development_signals: pd.DataFrame,
    selected_oof: pd.DataFrame,
    best: dict[str, object],
) -> tuple[pd.DataFrame, float, bool]:
    oof_signals = development_signals.loc[
        development_signals.index.intersection(selected_oof.index)
    ]
    oof_probability = selected_oof["probability"].reindex(oof_signals.index)
    period_data = data.loc[oof_signals.index.min():DEVELOPMENT_END]
    config = RiskControlConfig(
        "Entry Outcome Threshold",
        "Development OOF",
        max_total_positions=1,
        max_same_direction=1,
    )
    baseline = _simulate_risk_control(period_data, oof_signals, best, config)
    baseline_net = float(baseline.summary["Total net P/L"])
    rows = []
    thresholds = (0.30, 0.325, 0.35, 0.375, 0.40, 0.45, 0.50, 0.55, 0.60)
    for threshold in thresholds:
        signals = _probability_gate(
            oof_signals,
            oof_probability,
            float(threshold),
            "Development Probability Gate",
        )
        result = _simulate_risk_control(period_data, signals, best, config)
        metrics = _metric_values(result)
        retention = len(signals) / len(oof_signals) * 100 if len(oof_signals) else 0.0
        net = float(result.summary["Total net P/L"])
        profit_retention = net / baseline_net * 100 if baseline_net > 0 else 0.0
        eligible = bool(
            retention >= 60
            and profit_retention >= 85
            and metrics["Growth (%)"] > 0
            and metrics["Max drawdown (%)"] <= MAX_DRAWDOWN_PCT
            and metrics["Profit factor"] >= 1.30
        )
        rows.append(
            {
                "Threshold": float(threshold),
                **metrics,
                "Entry tersedia": len(oof_signals),
                "Entry diterima": len(signals),
                "Retensi entry (%)": retention,
                "Retensi net profit (%)": profit_retention,
                "Eligible": eligible,
            }
        )
    table = pd.DataFrame(rows)
    eligible = table[table["Eligible"]]
    fallback = eligible.empty
    fallback_pool = table[
        (table["Entry diterima"] > 0) & (table["Retensi entry (%)"] >= 40)
    ]
    pool = eligible if not eligible.empty else fallback_pool
    if pool.empty:
        pool = table[table["Entry diterima"] > 0]
    if pool.empty:
        pool = table
    selected = float(
        pool.sort_values(
            ["Profit factor", "Growth (%)", "Retensi entry (%)"],
            ascending=[False, False, False],
        ).iloc[0]["Threshold"]
    )
    return table, selected, fallback


def _probability_gate(
    signals: pd.DataFrame,
    probability: pd.Series,
    threshold: float,
    strategy: str,
) -> pd.DataFrame:
    aligned = probability.reindex(signals.index)
    selected = signals.loc[aligned.ge(threshold).fillna(False)].copy()
    if not selected.empty:
        selected["outcome_probability"] = aligned.reindex(selected.index)
        selected["strategy"] = strategy
    return selected


def _calibration_table(target: pd.Series, probability: pd.Series) -> pd.DataFrame:
    rows = []
    for lower, upper in ((0.0, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.01)):
        mask = probability.ge(lower) & probability.lt(upper)
        rows.append(
            {
                "Probability bucket": f"{lower:.0%}-{min(upper, 1):.0%}",
                "Events": int(mask.sum()),
                "Mean predicted probability": float(probability[mask].mean()) if mask.any() else np.nan,
                "Actual TP rate": float(target[mask].mean()) if mask.any() else np.nan,
                "Calibration gap": (
                    abs(float(probability[mask].mean()) - float(target[mask].mean()))
                    if mask.any()
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def _outcome_distribution(
    development: pd.DataFrame,
    validation: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for period, frame in (("Development 2025", development), ("Validation 2026H1", validation)):
        for outcome, count in frame["raw_outcome"].value_counts().items():
            rows.append(
                {
                    "Periode": period,
                    "Outcome": outcome,
                    "Events": int(count),
                    "Proporsi (%)": float(count / len(frame) * 100),
                }
            )
    return pd.DataFrame(rows)


def _direction_audit(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for direction, frame in events.groupby("direction"):
        metrics = _probability_metrics(frame["target"], frame["probability"])
        rows.append({"Arah": direction, **metrics})
    return pd.DataFrame(rows)


def _session_audit(events: pd.DataFrame) -> pd.DataFrame:
    hour = pd.Series(events.index.hour, index=events.index)
    session = pd.Series("Asia", index=events.index)
    session.loc[hour.between(7, 12)] = "London"
    session.loc[hour.between(13, 21)] = "New York"
    rows = []
    for name, frame in events.groupby(session):
        rows.append(
            {
                "Sesi UTC": name,
                "Events": len(frame),
                "TP rate (%)": float(frame["target"].mean() * 100),
                "Mean probability (%)": float(frame["probability"].mean() * 100),
            }
        )
    return pd.DataFrame(rows)


def _ablation_test(model_name: str, development: pd.DataFrame) -> pd.DataFrame:
    variants: dict[str, list[str]] = {"Semua fitur": []}
    for group, columns in FEATURE_GROUPS.items():
        variants[f"Tanpa {group}"] = columns
    rows = []
    for variant, removed in variants.items():
        metrics = []
        for fold in FOLDS:
            train = development.loc[fold.train_start:fold.train_end]
            test = development.loc[fold.test_start:fold.test_end].copy()
            if removed:
                medians = train[removed].median()
                for column in removed:
                    test[column] = medians[column]
            probability = _fit_predict_probability(model_name, train, test, FEATURE_COLUMNS)
            metrics.append(_probability_metrics(test["target"], probability))
        rows.append(
            {
                "Ablation": variant,
                "Mean Brier": float(np.mean([item["Brier score"] for item in metrics])),
                "Mean Brier improvement (%)": float(
                    np.mean([item["Brier improvement (%)"] for item in metrics])
                ),
                "Mean PR-AUC": float(np.mean([item["PR-AUC"] for item in metrics])),
                "Mean calibration error": float(
                    np.mean([item["Calibration error"] for item in metrics])
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("Mean Brier").reset_index(drop=True)


def _stress_test(
    data: pd.DataFrame,
    signals: pd.DataFrame,
    best: dict[str, object],
    config: RiskControlConfig,
) -> pd.DataFrame:
    rows = []
    for spread_multiplier in (1.0, 1.25, 1.5):
        for slippage_points in (2.0, 4.0, 6.0):
            result = _simulate_risk_control(
                data,
                signals,
                best,
                config,
                spread_multiplier=spread_multiplier,
                slippage_points=slippage_points,
            )
            rows.append(
                {
                    "Spread multiplier": spread_multiplier,
                    "Slippage points/sisi": slippage_points,
                    **_metric_values(result),
                }
            )
    return pd.DataFrame(rows)


def _delay_stress(
    data: pd.DataFrame,
    signals: pd.DataFrame,
    best: dict[str, object],
    config: RiskControlConfig,
) -> pd.DataFrame:
    rows = []
    for delay_minutes in (0, 1, 5, 15):
        delayed = _delay_signals(signals, data.index, delay_minutes)
        result = _simulate_risk_control(data, delayed, best, config)
        rows.append(
            {
                "Delay entry (menit)": delay_minutes,
                "Entry tersedia": len(delayed),
                **_metric_values(result),
            }
        )
    return pd.DataFrame(rows)


def _delay_signals(
    signals: pd.DataFrame,
    data_index: pd.DatetimeIndex,
    minutes: int,
) -> pd.DataFrame:
    if minutes == 0 or signals.empty:
        return signals.copy()
    rows = []
    for timestamp, signal in signals.iterrows():
        location = data_index.searchsorted(
            pd.Timestamp(timestamp) + pd.Timedelta(minutes=minutes),
            side="left",
        )
        if location >= len(data_index):
            continue
        shifted = signal.copy()
        shifted.name = data_index[location]
        rows.append(shifted)
    if not rows:
        return signals.iloc[0:0].copy()
    output = pd.DataFrame(rows)
    output.index.name = signals.index.name
    return output.sort_index()


def _threshold_sensitivity(
    data: pd.DataFrame,
    signals: pd.DataFrame,
    probability: pd.Series,
    best: dict[str, object],
    config: RiskControlConfig,
) -> pd.DataFrame:
    rows = []
    for threshold in (0.25, 0.30, 0.325, 0.35, 0.375, 0.40, 0.45, 0.50, 0.55, 0.60):
        selected = _probability_gate(
            signals, probability, float(threshold), "Threshold sensitivity"
        )
        result = _simulate_risk_control(data, selected, best, config)
        rows.append(
            {
                "Threshold": float(threshold),
                "Entry diterima": len(selected),
                "Retensi entry (%)": len(selected) / len(signals) * 100 if len(signals) else 0.0,
                **_metric_values(result),
            }
        )
    return pd.DataFrame(rows)


def _decision_table(
    probability: dict[str, float],
    economic: dict[str, float],
    entry_retention: float,
    profit_retention: float,
    folds: pd.DataFrame,
    stress: pd.DataFrame,
    monte_carlo: dict[str, float],
    selected_model: str,
    model_selection_fallback: bool,
    threshold_fallback: bool,
) -> dict[str, object]:
    criteria = {
        "Brier lebih baik dari baseline": bool(probability["Brier improvement (%)"] > 0),
        "Calibration error <= 0.10": bool(probability["Calibration error"] <= 0.10),
        "PR-AUC di atas TP rate": bool(
            probability["PR-AUC"] > probability["TP rate (%)"] / 100
        ),
        "Growth OOS positif": bool(economic["Growth (%)"] > 0),
        "Drawdown <= 10%": bool(economic["Max drawdown (%)"] <= MAX_DRAWDOWN_PCT),
        "Profit factor >= 1.30": bool(economic["Profit factor"] >= PROFIT_FACTOR_TARGET),
        "Retensi entry >= 60%": bool(entry_retention >= 60.0),
        "Retensi net profit >= 85%": bool(profit_retention >= 85.0),
        "Mayoritas fold Brier positif": bool(
            (
                folds.loc[folds["Model"].eq(selected_model), "Brier improvement (%)"]
                > 0
            ).sum()
            >= len(FOLDS) / 2
        ),
        "Stress profitable 9/9": bool(len(stress) == 9 and (stress["Growth (%)"] > 0).all()),
        "Monte Carlo rugi <= 10%": bool(
            monte_carlo["Probabilitas equity akhir < modal awal (%)"]
            <= MAX_MONTE_CARLO_LOSS_PCT
        ),
        "Model tanpa fallback": not model_selection_fallback,
        "Threshold tanpa fallback": not threshold_fallback,
    }
    return {
        **criteria,
        "Jumlah kriteria lolos": sum(criteria.values()),
        "Lulus seluruh kriteria": all(criteria.values()),
        "Retensi entry (%)": entry_retention,
        "Retensi net profit (%)": profit_retention,
    }


def _compact_events(events: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "direction", "raw_outcome", "target", "probability",
        "outcome_time", "hours_to_outcome", "conviction_ratio",
        "spread_points", "adx", "efficiency", "choppiness",
    ]
    return events[[column for column in columns if column in events.columns]].copy()


def _safe_monte_carlo(
    trades: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float]]:
    if not trades.empty and "Net P/L" in trades.columns:
        return _monte_carlo(trades)
    return (
        pd.DataFrame(columns=["Simulasi", "Equity akhir", "Max drawdown"]),
        {
            "Simulasi": 0.0,
            "Median equity akhir": 1000.0,
            "P05 equity akhir": 1000.0,
            "P95 equity akhir": 1000.0,
            "Median max drawdown": 0.0,
            "P95 max drawdown": 0.0,
            "Probabilitas equity akhir < modal awal (%)": 100.0,
        },
    )


def _safe_monthly_summary(result) -> pd.DataFrame:
    if not result.trades.empty and "Tanggal tutup" in result.trades.columns:
        return _monthly_summary(result)
    rows = []
    for month in pd.period_range("2026-01", "2026-06", freq="M"):
        rows.append(
            {
                "Bulan": str(month),
                "Equity awal": 1000.0,
                "Net P/L": 0.0,
                "Growth bulan (%)": 0.0,
                "Equity akhir": 1000.0,
                "Transaksi": 0.0,
                "Win rate (%)": np.nan,
                "Profit factor": np.nan,
                "Spread": 0.0,
                "Slippage": 0.0,
                "Swap": 0.0,
            }
        )
    return pd.DataFrame(rows)
