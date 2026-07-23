Exit code: 0
Wall time: 2.2 seconds
Output:
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from gold_forecast.exact_broker_oos import _compact_curve, _prepare_m1
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
from gold_forecast.v1_signal_quality import SignalQualityConfig, _select_signals


CLASSES = ("TREND_DOWN", "SIDEWAYS", "TRANSITION", "TREND_UP")
MODEL_NAMES = ("Rule-Based v2", "Logistic Multinomial", "Gradient Boosting", "Probability Ensemble")
LABEL_HORIZON_HOURS = 8
FEATURE_GROUPS = {
    "Trend strength": ["ema_gap_atr", "ema_fast_slope_atr", "ema_slow_slope_atr", "adx", "adx_change_3"],
    "Price path": ["return_1", "return_3", "return_6", "efficiency", "choppiness"],
    "Volatility": ["atr_percentile", "bb_width_atr", "range_width_atr"],
    "Breakout": ["donchian_position", "breakout_up", "breakout_down"],
    "Higher timeframe": ["h4_return", "h4_gap_atr", "h4_adx", "d1_return", "d1_gap_atr"],
    "Execution": ["spread_median", "spread_p90"],
}
FEATURE_COLUMNS = [column for columns in FEATURE_GROUPS.values() for column in columns]


@dataclass(frozen=True)
class Fold:
    name: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


FOLDS = (
    Fold("Fold 1", pd.Timestamp("2025-01-01"), pd.Timestamp("2025-03-31 15:59:59"), pd.Timestamp("2025-04-01"), pd.Timestamp("2025-06-30 23:59:59")),
    Fold("Fold 2", pd.Timestamp("2025-01-01"), pd.Timestamp("2025-06-30 15:59:59"), pd.Timestamp("2025-07-01"), pd.Timestamp("2025-09-30 23:59:59")),
    Fold("Fold 3", pd.Timestamp("2025-01-01"), pd.Timestamp("2025-09-30 15:59:59"), pd.Timestamp("2025-10-01"), pd.Timestamp("2025-12-31 23:59:59")),
)


def run_v1_regime_classifier_lab(
    gold_m1: pd.DataFrame,
    signal_daily: pd.DataFrame,
    frozen_payload: dict[str, object],
) -> dict[str, object]:
    data = _prepare_m1(gold_m1)
    frame = _classifier_frame(data)
    usable = frame.dropna(subset=[*FEATURE_COLUMNS, "label"]).copy()
    if usable.empty:
        raise ValueError("Dataset classifier regime tidak memiliki observasi lengkap.")

    fold_rows: list[dict[str, object]] = []
    for model_name in MODEL_NAMES:
        for fold in FOLDS:
            train = usable.loc[fold.train_start : fold.train_end]
            test = usable.loc[fold.test_start : fold.test_end]
            probabilities = _fit_predict(model_name, train, test, FEATURE_COLUMNS)
            prediction = _state_machine(probabilities, test)
            fold_rows.append(
                {
                    "Model": model_name,
                    "Fold": fold.name,
                    "Train": f"{fold.train_start:%d %b %Y} - {fold.train_end:%d %b %Y}",
                    "Test": f"{fold.test_start:%d %b %Y} - {fold.test_end:%d %b %Y}",
                    **_classification_metrics(test["label"], prediction),
                }
            )
    folds = pd.DataFrame(fold_rows)
    summary = _model_summary(folds)
    eligible = summary[
        (summary["Mean Macro F1"] >= 0.50)
        & (summary["Mean balanced accuracy"] >= 0.55)
        & (summary["Mean trend precision"] >= 0.60)
        & (summary["Mean delay (jam)"] <= 2.0)
        & (summary["Mean false trend rate (%)"] <= 25.0)
    ]
    selection_fallback = eligible.empty
    pool = eligible if not eligible.empty else summary
    selected_model = str(
        pool.sort_values(
            ["Selection score", "Mean Macro F1", "Worst Macro F1"],
            ascending=False,
        ).iloc[0]["Model"]
    )

    development_cutoff = pd.Timestamp(DEVELOPMENT_END) - pd.Timedelta(hours=LABEL_HORIZON_HOURS)
    development = usable.loc[DEVELOPMENT_START:development_cutoff]
    validation = usable.loc[VALIDATION_START:VALIDATION_END]
    validation_rows = []
    validation_predictions: dict[str, pd.Series] = {}
    validation_probabilities: dict[str, pd.DataFrame] = {}
    for model_name in MODEL_NAMES:
        probabilities = _fit_predict(model_name, development, validation, FEATURE_COLUMNS)
        prediction = _state_machine(probabilities, validation)
        validation_probabilities[model_name] = probabilities
        validation_predictions[model_name] = prediction
        validation_rows.append(
            {"Model": model_name, **_classification_metrics(validation["label"], prediction)}
        )
    validation_table = pd.DataFrame(validation_rows)
    selected_prediction = validation_predictions[selected_model]
    selected_probabilities = validation_probabilities[selected_model]

    ablation = _ablation_test(selected_model, usable)
    confusion = _confusion_table(validation["label"], selected_prediction)
    probability_audit = _probability_audit(
        validation["label"], selected_prediction, selected_probabilities
    )

    _, leaderboard, _ = frozen_payload["v1"]
    best = leaderboard.iloc[0].to_dict()
    full_features, _, _ = _regime_features(data)
    spread_limit = float(
        data.loc[DEVELOPMENT_START:DEVELOPMENT_END, "SpreadPoints"].quantile(0.90)
    )
    raw_validation = _entry_signals_for_period(
        data, signal_daily, best, VALIDATION_START, VALIDATION_END
    )
    balanced_config = SignalQualityConfig(
        "Balanced Entry Frozen",
        "Trend engine",
        conviction_multiplier=1.05,
        require_h1_trend=True,
        wait_hours=2,
    )
    balanced_signals, _ = _select_signals(
        raw_validation,
        full_features,
        best,
        balanced_config,
        spread_limit,
        VALIDATION_END,
    )
    balanced_signals = _label_signals(balanced_signals, "Balanced Entry Frozen")
    current_states = _regime_states(
        full_features,
        RegimeConfig("RG-C Sensitive", 22.0, 0.35, 55.0, 0.45, 0.16, 3),
    )
    current_gate = _gate_current_classifier(balanced_signals, current_states)
    selected_state_m1 = selected_prediction.reindex(data.index, method="ffill")
    v2_gate = _gate_v2_classifier(balanced_signals, selected_state_m1)
    simulation_config = RiskControlConfig(
        "Regime Classifier v2",
        "Trend gate",
        max_total_positions=1,
        max_same_direction=1,
    )
    validation_data = data.loc[VALIDATION_START:VALIDATION_END]
    economic_signals = {
        "Balanced Entry Frozen": balanced_signals,
        "Current RG-C Trend Gate": current_gate,
        "Classifier v2 Trend Gate": v2_gate,
    }
    economic_results = {
        name: _simulate_risk_control(validation_data, signals, best, simulation_config)
        for name, signals in economic_signals.items()
    }
    economic = pd.DataFrame(
        [{"Strategi": name, **_metric_values(result)} for name, result in economic_results.items()]
    )
    selected_result = economic_results["Classifier v2 Trend Gate"]
    selected_metrics = _metric_values(selected_result)
    selected_classifier_metrics = validation_table[
        validation_table["Model"].eq(selected_model)
    ].iloc[0]
    balanced_net = float(economic_results["Balanced Entry Frozen"].summary["Total net P/L"])
    retained_net = float(selected_result.summary["Total net P/L"])
    profit_retention = retained_net / balanced_net * 100 if balanced_net > 0 else 0.0
    stress = _stress_gate(validation_data, v2_gate, best, simulation_config)
    monte_carlo, monte_carlo_summary = _monte_carlo(selected_result.trades)
    decision = _decision_table(
        selected_classifier_metrics,
        selected_metrics,
        profit_retention,
        stress,
        monte_carlo_summary,
        selection_fallback,
    )

    return {
        "methodology": {
            "Baseline lock": "Optimizer v1 live tidak diubah sampai 31 Agustus 2026",
            "Development": "Purged expanding walk-forward 2025, embargo 8 jam",
            "Validation": "01 Jan 2026 - 30 Jun 2026 (secondary validation)",
            "Label horizon": "8 jam ke depan; fitur hanya memakai candle yang sudah selesai",
            "Classes": "TREND_UP | TREND_DOWN | SIDEWAYS | TRANSITION",
            "Selected model": selected_model,
            "Selection fallback": selection_fallback,
            "State machine": "probability threshold + margin + breakout early-confirmation + hysteresis",
            "Caveat": (
                "2026H1 sudah pernah diamati. Model tidak boleh dipromosikan hanya karena validation bagus; "
                "forward paper shadow tetap menjadi bukti independen."
            ),
        },
        "criteria": {
            "Macro F1 minimum": 0.60,
            "Balanced accuracy minimum": 0.60,
            "Trend precision minimum": 0.65,
            "Median delay maksimum (jam)": 2.0,
            "False trend maksimum (%)": 25.0,
            "False switch maksimum per hari": 0.30,
            "Coverage minimum (%)": 60.0,
            "Profit trend minimum dipertahankan (%)": 85.0,
        },
        "folds": folds,
        "model_summary": summary,
        "validation": validation_table,
        "ablation": ablation,
        "confusion": confusion,
        "probability_audit": probability_audit,
        "class_distribution": _class_distribution(usable),
        "economic": economic,
        "decision": decision,
        "selected_result": _compact_curve(selected_result),
        "selected_monthly": _monthly_summary(selected_result),
        "selected_stress": stress,
        "selected_monte_carlo": monte_carlo,
        "selected_monte_carlo_summary": monte_carlo_summary,
        "signal_counts": {name: int(len(signals)) for name, signals in economic_signals.items()},
    }


def _classifier_frame(data: pd.DataFrame) -> pd.DataFrame:
    h1 = _ohlc_bars(data, "1h")
    h4 = _ohlc_bars(data, "4h")
    d1 = _ohlc_bars(data, "1D")
    h1_features = _timeframe_features(h1, "h1")
    h4_features = _timeframe_features(h4, "h4")
    d1_features = _timeframe_features(d1, "d1")
    frame = pd.DataFrame(index=h1.index)
    frame["return_1"] = h1["Close"].pct_change(1) * 100
    frame["return_3"] = h1["Close"].pct_change(3) * 100
    frame["return_6"] = h1["Close"].pct_change(6) * 100
    for column in (
        "ema_gap_atr", "ema_fast_slope_atr", "ema_slow_slope_atr", "adx",
        "adx_change_3", "efficiency", "choppiness", "atr_percentile",
        "bb_width_atr", "range_width_atr", "donchian_position", "breakout_up", "breakout_down",
    ):
        frame[column] = h1_features[column]
    frame["h4_return"] = h4["Close"].pct_change().reindex(frame.index, method="ffill") * 100
    frame["h4_gap_atr"] = h4_features["ema_gap_atr"].reindex(frame.index, method="ffill")
    frame["h4_adx"] = h4_features["adx"].reindex(frame.index, method="ffill")
    frame["d1_return"] = d1["Close"].pct_change().reindex(frame.index, method="ffill") * 100
    frame["d1_gap_atr"] = d1_features["ema_gap_atr"].reindex(frame.index, method="ffill")
    hourly_spread = data["SpreadPoints"].resample("1h", label="right", closed="left")
    frame["spread_median"] = hourly_spread.median().reindex(frame.index)
    frame["spread_p90"] = hourly_spread.quantile(0.90).reindex(frame.index)
    frame["future_signed_atr"], frame["future_efficiency"] = _future_labels(
        h1["Close"], h1_features["atr"], LABEL_HORIZON_HOURS
    )
    frame["label"] = _label_regime(frame["future_signed_atr"], frame["future_efficiency"])
    return frame.replace([np.inf, -np.inf], np.nan)


def _ohlc_bars(data: pd.DataFrame, rule: str) -> pd.DataFrame:
    return (
        data[["Open", "High", "Low", "Close"]]
        .resample(rule, label="right", closed="left")
        .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"})
        .dropna()
    )


def _timeframe_features(bars: pd.DataFrame, prefix: str) -> pd.DataFrame:
    high, low, close = bars["High"], bars["Low"], bars["Close"]
    previous = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - previous).abs(), (low - previous).abs()], axis=1
    ).max(axis=1)
    atr = true_range.rolling(14, min_periods=14).mean()
    fast = close.ewm(span=10, adjust=False).mean()
    slow = close.ewm(span=30, adjust=False).mean()
    up_move, down_move = high.diff(), -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=bars.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=bars.index)
    tr_sum = true_range.rolling(14).sum()
    plus_di = 100 * plus_dm.rolling(14).sum() / tr_sum
    minus_di = 100 * minus_dm.rolling(14).sum() / tr_sum
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.rolling(14, min_periods=8).mean()
    travel = close.diff().abs().rolling(14).sum()
    range_span = high.rolling(14).max() - low.rolling(14).min()
    donchian_high = high.shift(1).rolling(20).max()
    donchian_low = low.shift(1).rolling(20).min()
    donchian_width = donchian_high - donchian_low
    rolling_std = close.rolling(20).std()
    output = pd.DataFrame(index=bars.index)
    output["atr"] = atr
    output["ema_gap_atr"] = (fast - slow) / atr
    output["ema_fast_slope_atr"] = fast.diff(3) / atr
    output["ema_slow_slope_atr"] = slow.diff(3) / atr
    output["adx"] = adx
    output["adx_change_3"] = adx.diff(3)
    output["efficiency"] = close.diff(14).abs() / travel
    output["choppiness"] = 100 * np.log10(true_range.rolling(14).sum() / range_span) / np.log10(14)
    output["atr_percentile"] = atr.rolling(120, min_periods=40).rank(pct=True)
    output["bb_width_atr"] = 4 * rolling_std / atr
    output["range_width_atr"] = donchian_width / atr
    output["donchian_position"] = (close - donchian_low) / donchian_width
    output["breakout_up"] = (close > donchian_high).astype(float)
    output["breakout_down"] = (close < donchian_low).astype(float)
    return output.replace([np.inf, -np.inf], np.nan)


def _future_labels(
    close: pd.Series,
    atr: pd.Series,
    horizon: int,
) -> tuple[pd.Series, pd.Series]:
    signed = (close.shift(-horizon) - close) / atr
    travel = pd.concat(
        [(close.shift(-offset) - close.shift(-(offset - 1))).abs() for offset in range(1, horizon + 1)],
        axis=1,
    ).sum(axis=1, min_count=horizon)
    efficiency = (close.shift(-horizon) - close).abs() / travel
    return signed, efficiency


def _label_regime(signed_atr: pd.Series, efficiency: pd.Series) -> pd.Series:
    label = pd.Series("TRANSITION", index=signed_atr.index, dtype="object")
    label.loc[(signed_atr.abs() <= 0.80) & (efficiency <= 0.35)] = "SIDEWAYS"
    label.loc[(signed_atr >= 0.80) & (efficiency >= 0.45)] = "TREND_UP"
    label.loc[(signed_atr <= -0.80) & (efficiency >= 0.45)] = "TREND_DOWN"
    label.loc[signed_atr.isna() | efficiency.isna()] = np.nan
    return label


def _fit_predict(
    model_name: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    x_train, y_train = train[columns], train["label"]
    x_test = test[columns]
    if model_name == "Rule-Based v2":
        return _rule_probabilities(x_test)
    if model_name == "Logistic Multinomial":
        model = _logistic_model()
        model.fit(x_train, y_train)
        return _probability_frame(model, x_test)
    if model_name == "Gradient Boosting":
        model = _boosting_model()
        weights = _class_weights(y_train)
        model.fit(x_train, y_train, sample_weight=weights)
        return _probability_frame(model, x_test)
    if model_name == "Probability Ensemble":
        logistic = _logistic_model()
        boosting = _boosting_model()
        logistic.fit(x_train, y_train)
        boosting.fit(x_train, y_train, sample_weight=_class_weights(y_train))
        return (
            _probability_frame(logistic, x_test) + _probability_frame(boosting, x_test)
        ) / 2
    raise ValueError(f"Model classifier tidak dikenal: {model_name}")


def _logistic_model():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            C=0.5,
            random_state=42,
        ),
    )


def _boosting_model():
    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=160,
        max_depth=3,
        min_samples_leaf=30,
        l2_regularization=1.0,
        random_state=42,
    )


def _class_weights(y: pd.Series) -> np.ndarray:
    counts = y.value_counts()
    return y.map({label: len(y) / (len(counts) * count) for label, count in counts.items()}).to_numpy()


def _probability_frame(model, x: pd.DataFrame) -> pd.DataFrame:
    raw = model.predict_proba(x)
    classes = model.classes_ if hasattr(model, "classes_") else model[-1].classes_
    output = pd.DataFrame(0.0, index=x.index, columns=CLASSES)
    for position, label in enumerate(classes):
        output[str(label)] = raw[:, position]
    return output


def _rule_probabilities(x: pd.DataFrame) -> pd.DataFrame:
    output = pd.DataFrame(0.05, index=x.index, columns=CLASSES)
    side_votes = (
        (x["adx"] < 20).astype(int)
        + (x["efficiency"] < 0.30).astype(int)
        + (x["choppiness"] > 58).astype(int)
        + (x["ema_gap_atr"].abs() < 0.35).astype(int)
        + (x["ema_slow_slope_atr"].abs() < 0.12).astype(int)
    )
    up_score = (
        (x["ema_gap_atr"] > 0).astype(float)
        + (x["return_3"] > 0).astype(float)
        + (x["h4_gap_atr"] > 0).astype(float)
        + x["breakout_up"]
    ) / 4
    down_score = (
        (x["ema_gap_atr"] < 0).astype(float)
        + (x["return_3"] < 0).astype(float)
        + (x["h4_gap_atr"] < 0).astype(float)
        + x["breakout_down"]
    ) / 4
    output["SIDEWAYS"] = 0.10 + 0.14 * side_votes
    output["TREND_UP"] = 0.10 + 0.55 * up_score * (1 - side_votes / 5)
    output["TREND_DOWN"] = 0.10 + 0.55 * down_score * (1 - side_votes / 5)
    output["TRANSITION"] = 0.20 + 0.10 * (side_votes.between(2, 3)).astype(float)
    return output.div(output.sum(axis=1), axis=0)


def _state_machine(probabilities: pd.DataFrame, features: pd.DataFrame) -> pd.Series:
    states = []
    current = "UNCERTAIN"
    pending = None
    pending_count = 0
    for timestamp, probability in probabilities.iterrows():
        ordered = probability.sort_values(ascending=False)
        target = str(ordered.index[0])
        confidence = float(ordered.iloc[0])
        margin = confidence - float(ordered.iloc[1])
        row = features.loc[timestamp]
        early_up = target == "TREND_UP" and confidence >= 0.55 and bool(row["breakout_up"]) and row["adx_change_3"] > 0
        early_down = target == "TREND_DOWN" and confidence >= 0.55 and bool(row["breakout_down"]) and row["adx_change_3"] > 0
        if confidence < 0.50 or margin < 0.08:
            target = "UNCERTAIN"
        if current in ("TREND_UP", "TREND_DOWN") and target == "UNCERTAIN":
            if float(probability.get(current, 0.0)) >= 0.40:
                states.append(current)
                continue
        if early_up or early_down:
            current, pending, pending_count = target, None, 0
        elif target == current:
            pending, pending_count = None, 0
        elif target == "UNCERTAIN":
            current, pending, pending_count = "UNCERTAIN", None, 0
        else:
            if target == pending:
                pending_count += 1
            else:
                pending, pending_count = target, 1
            if pending_count >= 2 and confidence >= 0.55:
                current, pending, pending_count = target, None, 0
        states.append(current)
    return pd.Series(states, index=probabilities.index, name="Prediksi")


def _classification_metrics(truth: pd.Series, prediction: pd.Series) -> dict[str, float]:
    aligned = pd.concat([truth.rename("truth"), prediction.rename("prediction")], axis=1).dropna()
    precision, recall, f1, _ = precision_recall_fscore_support(
        aligned["truth"], aligned["prediction"], labels=list(CLASSES), zero_division=0
    )
    metrics = {label: (precision[i], recall[i], f1[i]) for i, label in enumerate(CLASSES)}
    trend_precision = float((metrics["TREND_UP"][0] + metrics["TREND_DOWN"][0]) / 2)
    trend_recall = float((metrics["TREND_UP"][1] + metrics["TREND_DOWN"][1]) / 2)
    delay, detection_rate = _detection_delay(aligned["truth"], aligned["prediction"])
    false_trend = (
        aligned["prediction"].isin(["TREND_UP", "TREND_DOWN"])
        & ~aligned["truth"].isin(["TREND_UP", "TREND_DOWN"])
    )
    predicted_trend = aligned["prediction"].isin(["TREND_UP", "TREND_DOWN"])
    return {
        "Observasi": float(len(aligned)),
        "Macro F1": float(f1_score(aligned["truth"], aligned["prediction"], labels=list(CLASSES), average="macro", zero_division=0)),
        "Balanced accuracy": float(balanced_accuracy_score(aligned["truth"], aligned["prediction"])),
        "Trend precision": trend_precision,
        "Trend recall": trend_recall,
        "Sideways precision": float(metrics["SIDEWAYS"][0]),
        "Sideways recall": float(metrics["SIDEWAYS"][1]),
        "Coverage (%)": float(aligned["prediction"].ne("UNCERTAIN").mean() * 100),
        "False trend rate (%)": float(false_trend.sum() / max(predicted_trend.sum(), 1) * 100),
        "Median delay (jam)": delay,
        "Trend episode detected (%)": detection_rate,
        "False switch/hari": _false_switch_rate(aligned["prediction"]),
    }


def _detection_delay(truth: pd.Series, prediction: pd.Series) -> tuple[float, float]:
    episodes = []
    previous = None
    for timestamp, label in truth.items():
        if label in ("TREND_UP", "TREND_DOWN") and label != previous:
            episodes.append((timestamp, label))
        previous = label
    delays = []
    for start, label in episodes:
        window = prediction.loc[start : start + pd.Timedelta(hours=6)]
        matches = window[window.eq(label)]
        if not matches.empty:
            delays.append((matches.index[0] - start).total_seconds() / 3600)
    return (
        float(np.median(delays)) if delays else 99.0,
        float(len(delays) / max(len(episodes), 1) * 100),
    )


def _false_switch_rate(prediction: pd.Series) -> float:
    decided = prediction[prediction.ne("UNCERTAIN")]
    if decided.empty:
        return 0.0
    switches = decided.ne(decided.shift(1)).sum() - 1
    days = max((decided.index.max() - decided.index.min()).days + 1, 1)
    return float(max(switches, 0) / days)


def _model_summary(folds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, group in folds.groupby("Model", sort=False):
        values = {
            "Model": model,
            "Mean Macro F1": float(group["Macro F1"].mean()),
            "Worst Macro F1": float(group["Macro F1"].min()),
            "Mean balanced accuracy": float(group["Balanced accuracy"].mean()),
            "Mean trend precision": float(group["Trend precision"].mean()),
            "Mean trend recall": float(group["Trend recall"].mean()),
            "Mean delay (jam)": float(group["Median delay (jam)"].mean()),
            "Mean false trend rate (%)": float(group["False trend rate (%)"].mean()),
            "Mean false switch/hari": float(group["False switch/hari"].mean()),
            "Mean coverage (%)": float(group["Coverage (%)"].mean()),
        }
        values["Selection score"] = (
            35 * values["Mean Macro F1"]
            + 25 * values["Mean balanced accuracy"]
            + 20 * values["Mean trend precision"]
            + 10 * max(0, 1 - values["Mean delay (jam)"] / 6)
            + 10 * max(0, 1 - values["Mean false trend rate (%)"] / 50)
        )
        rows.append(values)
    return pd.DataFrame(rows)


def _ablation_test(selected_model: str, usable: pd.DataFrame) -> pd.DataFrame:
    rows = []
    variants: dict[str, list[str]] = {"Semua fitur": []}
    for group, columns in FEATURE_GROUPS.items():
        variants[f"Tanpa {group}"] = columns
    for variant, removed_columns in variants.items():
        metrics = []
        for fold in FOLDS:
            train = usable.loc[fold.train_start : fold.train_end]
            test = usable.loc[fold.test_start : fold.test_end].copy()
            if removed_columns:
                medians = train[removed_columns].median()
                for column in removed_columns:
                    test[column] = medians[column]
            prediction = _state_machine(
                _fit_predict(selected_model, train, test, FEATURE_COLUMNS), test
            )
            metrics.append(_classification_metrics(test["label"], prediction))
        rows.append(
            {
                "Ablation": variant,
                "Mean Macro F1": float(np.mean([item["Macro F1"] for item in metrics])),
                "Mean balanced accuracy": float(np.mean([item["Balanced accuracy"] for item in metrics])),
                "Mean trend precision": float(np.mean([item["Trend precision"] for item in metrics])),
                "Mean delay (jam)": float(np.mean([item["Median delay (jam)"] for item in metrics])),
            }
        )
    return pd.DataFrame(rows).sort_values("Mean Macro F1", ascending=False).reset_index(drop=True)


def _confusion_table(truth: pd.Series, prediction: pd.Series) -> pd.DataFrame:
    labels = [*CLASSES, "UNCERTAIN"]
    matrix = confusion_matrix(truth, prediction, labels=labels)
    output = pd.DataFrame(matrix, index=[f"Aktual {label}" for label in labels], columns=[f"Prediksi {label}" for label in labels])
    return output.reset_index(names="Aktual")


def _probability_audit(
    truth: pd.Series,
    prediction: pd.Series,
    probabilities: pd.DataFrame,
) -> pd.DataFrame:
    confidence = probabilities.max(axis=1)
    rows = []
    for lower, upper in ((0.0, 0.50), (0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 1.01)):
        mask = (confidence >= lower) & (confidence < upper)
        if not mask.any():
            continue
        rows.append(
            {
                "Confidence bin": f"{lower:.1f}-{min(upper, 1.0):.1f}",
                "Observasi": int(mask.sum()),
                "Rata-rata confidence": float(confidence[mask].mean()),
                "Akurasi aktual": float(prediction[mask].eq(truth[mask]).mean()),
            }
        )
    return pd.DataFrame(rows)


def _class_distribution(usable: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for period, start, end in (
        ("Development 2025", DEVELOPMENT_START, DEVELOPMENT_END),
        ("Validation 2026H1", VALIDATION_START, VALIDATION_END),
    ):
        subset = usable.loc[start:end, "label"]
        for label, count in subset.value_counts().items():
            rows.append(
                {"Periode": period, "Class": label, "Observasi": int(count), "Proporsi (%)": float(count / len(subset) * 100)}
            )
    return pd.DataFrame(rows)


def _label_signals(signals: pd.DataFrame, strategy: str) -> pd.DataFrame:
    output = signals.copy()
    output["strategy"] = strategy
    return output


def _gate_current_classifier(signals: pd.DataFrame, states: pd.Series) -> pd.DataFrame:
    keep = [states.get(pd.Timestamp(timestamp), "UNCERTAIN") == "TRENDING" for timestamp in signals.index]
    return _label_signals(signals.loc[keep].copy(), "Current RG-C Trend Gate")


def _gate_v2_classifier(signals: pd.DataFrame, states: pd.Series) -> pd.DataFrame:
    keep = []
    for timestamp, signal in signals.iterrows():
        expected = float(signal["expected_change_pct"])
        required = "TREND_UP" if expected > 0 else "TREND_DOWN"
        keep.append(states.get(pd.Timestamp(timestamp), "UNCERTAIN") == required)
    return _label_signals(signals.loc[keep].copy(), "Classifier v2 Trend Gate")


def _stress_gate(validation, signals, best, config) -> pd.DataFrame:
    rows = []
    for spread_multiplier in (1.0, 1.5, 2.0):
        for slippage_points in (2.0, 4.0, 6.0):
            result = _simulate_risk_control(
                validation,
                signals,
                best,
                config,
                spread_multiplier=spread_multiplier,
                slippage_points=slippage_points,
            )
            rows.append(
                {"Spread multiplier": spread_multiplier, "Slippage points/sisi": slippage_points, **_metric_values(result)}
            )
    return pd.DataFrame(rows)


def _decision_table(
    classifier: pd.Series,
    economic: dict[str, float],
    profit_retention: float,
    stress: pd.DataFrame,
    monte_carlo: dict[str, float],
    selection_fallback: bool,
) -> dict[str, object]:
    criteria = {
        "Macro F1 >= 0.60": bool(classifier["Macro F1"] >= 0.60),
        "Balanced accuracy >= 0.60": bool(classifier["Balanced accuracy"] >= 0.60),
        "Trend precision >= 0.65": bool(classifier["Trend precision"] >= 0.65),
        "Median delay <= 2 jam": bool(classifier["Median delay (jam)"] <= 2.0),
        "False trend <= 25%": bool(classifier["False trend rate (%)"] <= 25.0),
        "False switch <= 0.30/hari": bool(classifier["False switch/hari"] <= 0.30),
        "Coverage >= 60%": bool(classifier["Coverage (%)"] >= 60.0),
        "Growth positif": bool(economic["Growth (%)"] > 0),
        "Drawdown <= 10%": bool(economic["Max drawdown (%)"] <= MAX_DRAWDOWN_PCT),
        "Profit factor >= 1.30": bool(economic["Profit factor"] >= PROFIT_FACTOR_TARGET),
        "Profit trend dipertahankan >= 85%": bool(profit_retention >= 85.0),
        "Stress profitable 9/9": bool(len(stress) == 9 and (stress["Growth (%)"] > 0).all()),
        "Monte Carlo rugi <= 10%": bool(monte_carlo["Probabilitas equity akhir < modal awal (%)"] <= MAX_MONTE_CARLO_LOSS_PCT),
        "Tidak memakai selection fallback": not selection_fallback,
    }
    return {
        **criteria,
        "Jumlah kriteria lolos": sum(criteria.values()),
        "Lulus seluruh kriteria": all(criteria.values()),
        "Profit trend dipertahankan (%)": profit_retention,
    }

