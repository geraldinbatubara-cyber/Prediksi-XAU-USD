from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from gold_forecast.v1_entry_outcome import (
    FEATURE_COLUMNS,
    OUTCOME_HORIZON_DAYS,
    _balanced_signals,
    _compact_events,
    _delay_signals,
    _event_dataset,
    _outcome_features,
    _safe_monte_carlo,
    _session_audit,
    _stress_test,
)
from gold_forecast.exact_broker_oos import _compact_curve
from gold_forecast.v1_risk_control import (
    MAX_DRAWDOWN_PCT,
    MAX_MONTE_CARLO_LOSS_PCT,
    PROFIT_FACTOR_TARGET,
    RiskControlConfig,
    _metric_values,
    _simulate_risk_control,
)
from gold_forecast.v1_sideways_defense import _regime_features
from gold_forecast.v1_signal_quality import SignalQualityConfig, _entry_features


DEVELOPMENT_START = pd.Timestamp("2024-01-01")
DEVELOPMENT_END = pd.Timestamp("2025-12-31 23:59:59")
CONFIRMATION_START = pd.Timestamp("2026-01-01")
CONFIRMATION_END = pd.Timestamp("2026-06-30 23:59:59")
MODEL_NAMES = ("Rule Scorecard", "Logistic Regression", "Gradient Boosting", "Probability Ensemble")
OUTCOME_CLASSES = ("SL_FIRST", "TIMEOUT", "TP_FIRST")


@dataclass(frozen=True)
class QualityFold:
    name: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


FOLDS = (
    QualityFold("Fold 1", pd.Timestamp("2024-01-01"), pd.Timestamp("2024-06-30 23:59:59"), pd.Timestamp("2024-07-01"), pd.Timestamp("2024-09-30 23:59:59")),
    QualityFold("Fold 2", pd.Timestamp("2024-01-01"), pd.Timestamp("2024-09-30 23:59:59"), pd.Timestamp("2024-10-01"), pd.Timestamp("2024-12-31 23:59:59")),
    QualityFold("Fold 3", pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31 23:59:59"), pd.Timestamp("2025-01-01"), pd.Timestamp("2025-03-31 23:59:59")),
    QualityFold("Fold 4", pd.Timestamp("2024-01-01"), pd.Timestamp("2025-03-31 23:59:59"), pd.Timestamp("2025-04-01"), pd.Timestamp("2025-06-30 23:59:59")),
    QualityFold("Fold 5", pd.Timestamp("2024-01-01"), pd.Timestamp("2025-06-30 23:59:59"), pd.Timestamp("2025-07-01"), pd.Timestamp("2025-09-30 23:59:59")),
    QualityFold("Fold 6", pd.Timestamp("2024-01-01"), pd.Timestamp("2025-09-30 23:59:59"), pd.Timestamp("2025-10-01"), pd.Timestamp("2025-12-31 23:59:59")),
)


def run_v1_entry_quality_lab(
    gold_m1: pd.DataFrame,
    signal_daily: pd.DataFrame,
    frozen_payload: dict[str, object],
) -> dict[str, object]:
    from gold_forecast.exact_broker_oos import _prepare_m1

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
    spread_limit = float(data.loc[DEVELOPMENT_START:DEVELOPMENT_END, "SpreadPoints"].quantile(0.90))

    development_signals = _balanced_signals(
        data, signal_daily, best, entry_features, balanced_config, spread_limit,
        DEVELOPMENT_START, DEVELOPMENT_END,
    )
    confirmation_signals = _balanced_signals(
        data, signal_daily, best, entry_features, balanced_config, spread_limit,
        CONFIRMATION_START, CONFIRMATION_END,
    )
    development_signals = development_signals.loc[
        ~development_signals.index.duplicated(keep="last")
    ]
    confirmation_signals = confirmation_signals.loc[
        ~confirmation_signals.index.duplicated(keep="last")
    ]
    feature_frame = _outcome_features(data, regime_features, m15, best)
    development_events = _add_excursions(
        data, _event_dataset(data, development_signals, feature_frame, best), best
    )
    confirmation_events = _add_excursions(
        data, _event_dataset(data, confirmation_signals, feature_frame, best), best
    )
    if len(development_events) < 80 or len(confirmation_events) < 20:
        raise ValueError(
            "Jumlah event Balanced Entry tidak cukup untuk Entry Quality Lab v2: "
            f"development={len(development_events)}, confirmation={len(confirmation_events)}."
        )

    fold_rows: list[dict[str, object]] = []
    oof_frames: list[pd.DataFrame] = []
    for model_name in MODEL_NAMES:
        for fold in FOLDS:
            train = development_events.loc[fold.train_start:fold.train_end]
            test = development_events.loc[fold.test_start:fold.test_end]
            probabilities = _fit_multiclass(model_name, train, test)
            metrics = _multiclass_metrics(test["raw_outcome"], probabilities)
            fold_rows.append({
                "Model": model_name,
                "Fold": fold.name,
                "Train events": len(train),
                "Test events": len(test),
                **metrics,
            })
            tagged = probabilities.copy()
            tagged["Model"] = model_name
            tagged["actual"] = test["raw_outcome"]
            oof_frames.append(tagged)

    folds = pd.DataFrame(fold_rows)
    oof = pd.concat(oof_frames).sort_index()
    model_summary = _model_summary(folds)
    selected_model = str(model_summary.iloc[0]["Model"])
    selected_folds = folds[folds["Model"].eq(selected_model)]
    model_fallback = not bool(
        (selected_folds["Brier improvement (%)"] > 0).sum() >= len(FOLDS) / 2
    )
    selected_oof = oof[oof["Model"].eq(selected_model)].copy()
    selected_oof = selected_oof.loc[~selected_oof.index.duplicated(keep="last")]
    selected_oof["timeout_mark_to_market_usd"] = development_events[
        "timeout_mark_to_market_usd"
    ].reindex(selected_oof.index)
    threshold_table, selected_rule, threshold_fallback = _select_rule(
        data, development_signals, selected_oof, best
    )

    confirmation_probability = _fit_multiclass(
        selected_model, development_events, confirmation_events
    )
    confirmation_events = confirmation_events.copy()
    for column in OUTCOME_CLASSES:
        confirmation_events[f"p_{column.lower()}"] = confirmation_probability[column]
    confirmation_events["expected_value"] = _expected_value(
        confirmation_probability, best, confirmation_events["spread_points"]
    )
    confirmation_metrics = _multiclass_metrics(
        confirmation_events["raw_outcome"], confirmation_probability
    )

    probability_signals = _quality_gate(
        confirmation_signals, confirmation_events, selected_rule, "v1 Entry Quality v2"
    )
    simulation_config = RiskControlConfig(
        "Entry Quality Lab v2", "Three-outcome EV gate",
        max_total_positions=1, max_same_direction=1,
    )
    confirmation_data = data.loc[CONFIRMATION_START:CONFIRMATION_END]
    baseline_result = _simulate_risk_control(
        confirmation_data, confirmation_signals, best, simulation_config
    )
    selected_result = _simulate_risk_control(
        confirmation_data, probability_signals, best, simulation_config
    )
    economic = pd.DataFrame([
        {"Strategi": "Balanced Entry Frozen", **_metric_values(baseline_result)},
        {"Strategi": "v1 Entry Quality v2", **_metric_values(selected_result)},
    ])
    selected_metrics = _metric_values(selected_result)
    monte_carlo, monte_carlo_summary = _safe_monte_carlo(selected_result.trades)
    stress = _stress_test(
        confirmation_data, probability_signals, best, simulation_config
    )
    delay = _delay_test(
        confirmation_data, probability_signals, best, simulation_config
    )
    data_audit = _data_audit(data)
    direction = _direction_report(development_events, confirmation_events)
    outcome = _outcome_report(development_events, confirmation_events)
    mfe_mae = _excursion_report(development_events, confirmation_events)
    event_audit = _compact_quality_events(confirmation_events)
    decision = _decision(
        confirmation_metrics,
        selected_metrics,
        selected_result,
        confirmation_signals,
        probability_signals,
        selected_folds,
        stress,
        monte_carlo_summary,
        model_fallback,
        threshold_fallback,
    )

    return {
        "methodology": {
            "Baseline lock": "v1 Exact Baseline, Balanced Entry, ledger, dan Live Trading tidak diubah",
            "Development": "01 Jan 2024 - 31 Des 2025; enam expanding walk-forward fold",
            "Historical confirmation": "01 Jan 2026 - 30 Jun 2026; sudah pernah dilihat, bukan true OOS",
            "True prospective OOS": "Paper shadow setelah model dibekukan sampai 31 Agustus 2026",
            "Outcome": "TP_FIRST | SL_FIRST | TIMEOUT; TIMEOUT dipertahankan sebagai kelas tersendiri",
            "Outcome horizon": f"{OUTCOME_HORIZON_DAYS} hari kalender",
            "Selected model": selected_model,
            "Selected EV minimum": selected_rule["ev_min"],
            "Selected TP probability minimum": selected_rule["tp_min"],
            "Model fallback": model_fallback,
            "Threshold fallback": threshold_fallback,
            "Caveat": (
                "Hasil 2026H1 hanya historical confirmation. Kandidat tidak boleh menggantikan "
                "baseline sebelum lulus prospective paper shadow."
            ),
        },
        "data_audit": data_audit,
        "folds": folds,
        "model_summary": model_summary,
        "threshold_development": threshold_table,
        "confirmation_metrics": pd.DataFrame([confirmation_metrics]),
        "economic": economic,
        "decision": decision,
        "outcome_distribution": outcome,
        "direction_audit": direction,
        "session_audit": _session_audit(_binary_compatibility_frame(confirmation_events)),
        "mfe_mae": mfe_mae,
        "stress": stress,
        "delay_stress": delay,
        "selected_result": _compact_curve(selected_result),
        "selected_monte_carlo": monte_carlo,
        "selected_monte_carlo_summary": monte_carlo_summary,
        "confirmation_events": event_audit,
        "signal_counts": {
            "Balanced Entry Frozen": int(len(confirmation_signals)),
            "v1 Entry Quality v2": int(len(probability_signals)),
        },
    }


def _add_excursions(
    data: pd.DataFrame,
    events: pd.DataFrame,
    best: dict[str, object],
) -> pd.DataFrame:
    if events.empty:
        return events
    output = events.copy()
    mfe_values = []
    mae_values = []
    timeout_net = []
    for timestamp, event in output.iterrows():
        direction = str(event["direction"])
        end = min(
            pd.Timestamp(timestamp) + pd.Timedelta(days=OUTCOME_HORIZON_DAYS),
            data.index.max(),
        )
        path = data.loc[(data.index > timestamp) & (data.index <= end)]
        entry = float(data.loc[timestamp, "Close"])
        lot = float(best.get("Lot", 0.01))
        units = lot * 100.0
        if path.empty:
            favorable = adverse = final_net = 0.0
        elif direction == "BUY":
            favorable = max((float(path["High"].max()) - entry) * units, 0.0)
            adverse = max((entry - float(path["Low"].min())) * units, 0.0)
            final_net = (float(path["Close"].iloc[-1]) - entry) * units
        else:
            favorable = max((entry - float(path["Low"].min())) * units, 0.0)
            adverse = max((float(path["High"].max()) - entry) * units, 0.0)
            final_net = (entry - float(path["Close"].iloc[-1])) * units
        mfe_values.append(favorable)
        mae_values.append(adverse)
        timeout_net.append(final_net)
    output["mfe_usd"] = mfe_values
    output["mae_usd"] = mae_values
    output["timeout_mark_to_market_usd"] = timeout_net
    return output


def _fit_multiclass(
    model_name: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> pd.DataFrame:
    if test.empty:
        return pd.DataFrame(index=test.index, columns=OUTCOME_CLASSES, dtype=float)
    if model_name == "Rule Scorecard":
        return _rule_probability(train, test)
    if model_name == "Probability Ensemble":
        logistic = _fit_multiclass("Logistic Regression", train, test)
        boosting = _fit_multiclass("Gradient Boosting", train, test)
        return (logistic + boosting) / 2

    x_train = train[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    x_test = test[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    medians = x_train.median()
    x_train = x_train.fillna(medians).fillna(0.0)
    x_test = x_test.fillna(medians).fillna(0.0)
    y_train = train["raw_outcome"].astype(str)
    if y_train.nunique() < 2:
        probabilities = pd.DataFrame(0.0, index=test.index, columns=OUTCOME_CLASSES)
        probabilities[y_train.iloc[0]] = 1.0
        return probabilities
    if model_name == "Logistic Regression":
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=0.35, max_iter=2000, class_weight="balanced", random_state=42
            ),
        )
    else:
        weights = y_train.map(
            {name: len(y_train) / max((y_train == name).sum(), 1) for name in y_train.unique()}
        )
        model = HistGradientBoostingClassifier(
            learning_rate=0.04,
            max_iter=90,
            max_leaf_nodes=7,
            min_samples_leaf=max(8, len(train) // 15),
            l2_regularization=2.0,
            random_state=42,
        )
        model.fit(x_train, y_train, sample_weight=weights)
        return _align_probability(model.predict_proba(x_test), model.classes_, test.index)
    model.fit(x_train, y_train)
    classifier = model[-1]
    return _align_probability(model.predict_proba(x_test), classifier.classes_, test.index)


def _rule_probability(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    score = (
        0.50 * test["conviction_ratio"].clip(0, 3)
        + 0.35 * test["trend_strength"].clip(0, 3)
        + 0.25 * test["efficiency"].clip(0, 1)
        - 0.25 * test["choppiness"].clip(0, 100) / 100
        - 0.20 * test["spread_points"].clip(lower=0) / max(train["spread_points"].quantile(0.9), 1)
    )
    center = float(train["target"].mean())
    tp = (center + 0.16 * np.tanh(score - score.median())).clip(0.05, 0.85)
    timeout_rate = float((train["raw_outcome"] == "TIMEOUT").mean())
    timeout = pd.Series(timeout_rate, index=test.index).clip(0.01, 0.50)
    sl = (1.0 - tp - timeout).clip(0.01)
    total = tp + timeout + sl
    return pd.DataFrame({
        "SL_FIRST": sl / total,
        "TIMEOUT": timeout / total,
        "TP_FIRST": tp / total,
    }, index=test.index)


def _align_probability(
    values: np.ndarray,
    classes: np.ndarray,
    index: pd.Index,
) -> pd.DataFrame:
    output = pd.DataFrame(0.0, index=index, columns=OUTCOME_CLASSES)
    for position, name in enumerate(classes):
        if str(name) in output.columns:
            output[str(name)] = values[:, position]
    output = output.clip(0.001)
    return output.div(output.sum(axis=1), axis=0)


def _multiclass_metrics(
    actual: pd.Series,
    probability: pd.DataFrame,
) -> dict[str, float]:
    y = actual.astype(str)
    probability = probability[list(OUTCOME_CLASSES)].clip(0.001, 0.999)
    probability = probability.div(probability.sum(axis=1), axis=0)
    encoded = pd.get_dummies(y).reindex(columns=OUTCOME_CLASSES, fill_value=0).astype(float)
    prevalence = encoded.mean()
    baseline = pd.DataFrame(
        np.tile(prevalence.to_numpy(), (len(encoded), 1)),
        index=encoded.index,
        columns=encoded.columns,
    )
    brier = float(np.mean(np.sum((probability.to_numpy() - encoded.to_numpy()) ** 2, axis=1)))
    baseline_brier = float(np.mean(np.sum((baseline.to_numpy() - encoded.to_numpy()) ** 2, axis=1)))
    prediction = probability.idxmax(axis=1)
    tp_target = y.eq("TP_FIRST").astype(int)
    tp_auc = (
        float(roc_auc_score(tp_target, probability["TP_FIRST"]))
        if tp_target.nunique() > 1 else 0.5
    )
    return {
        "Observasi": float(len(y)),
        "TP rate (%)": float(tp_target.mean() * 100),
        "TIMEOUT rate (%)": float(y.eq("TIMEOUT").mean() * 100),
        "Multiclass Brier": brier,
        "Baseline Brier": baseline_brier,
        "Brier improvement (%)": (
            (baseline_brier - brier) / baseline_brier * 100 if baseline_brier > 0 else 0.0
        ),
        "Log loss": float(log_loss(y, probability, labels=list(OUTCOME_CLASSES))),
        "Macro F1": float(f1_score(y, prediction, labels=list(OUTCOME_CLASSES), average="macro", zero_division=0)),
        "TP ROC-AUC": tp_auc,
    }


def _model_summary(folds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, group in folds.groupby("Model", sort=False):
        rows.append({
            "Model": model,
            "Mean Brier": float(group["Multiclass Brier"].mean()),
            "Worst Brier": float(group["Multiclass Brier"].max()),
            "Mean Brier improvement (%)": float(group["Brier improvement (%)"].mean()),
            "Positive Brier folds": int((group["Brier improvement (%)"] > 0).sum()),
            "Mean Macro F1": float(group["Macro F1"].mean()),
            "Mean TP ROC-AUC": float(group["TP ROC-AUC"].mean()),
        })
    frame = pd.DataFrame(rows)
    return frame.sort_values(
        ["Positive Brier folds", "Mean Brier improvement (%)", "Mean TP ROC-AUC"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def _expected_value(
    probabilities: pd.DataFrame,
    best: dict[str, object],
    spread_points: pd.Series,
) -> pd.Series:
    tp = float(best["TP (USD)"])
    sl = float(best["SL (USD)"])
    timeout_penalty = sl * 0.25
    execution_cost = spread_points.reindex(probabilities.index).fillna(0.0) * 0.01
    return (
        probabilities["TP_FIRST"] * tp
        - probabilities["SL_FIRST"] * sl
        - probabilities["TIMEOUT"] * timeout_penalty
        - execution_cost
    )


def _select_rule(
    data: pd.DataFrame,
    signals: pd.DataFrame,
    oof: pd.DataFrame,
    best: dict[str, object],
) -> tuple[pd.DataFrame, dict[str, float], bool]:
    event_index = signals.index.intersection(oof.index)
    oof = oof.reindex(event_index)
    for outcome in OUTCOME_CLASSES:
        oof[f"p_{outcome.lower()}"] = oof[outcome]
    oof["expected_value"] = _expected_value(
        oof, best, data["SpreadPoints"].reindex(event_index)
    )
    baseline_values = _realized_event_values(oof, best)
    baseline_net = float(baseline_values.sum())
    rows = []
    for ev_min in (-2.0, 0.0, 1.0, 2.0, 3.0, 5.0):
        for tp_min in (0.25, 0.30, 0.35, 0.40):
            rule = {"ev_min": ev_min, "tp_min": tp_min}
            mask = (
                oof["expected_value"].ge(ev_min)
                & oof["p_tp_first"].ge(tp_min)
            )
            selected_events = oof.loc[mask]
            values = _realized_event_values(selected_events, best)
            event_metrics = _event_economic_metrics(values)
            net = float(values.sum())
            retention = len(selected_events) / len(event_index) * 100 if len(event_index) else 0.0
            profit_retention = net / baseline_net * 100 if baseline_net > 0 else 0.0
            eligible = bool(
                len(selected_events) >= 20
                and event_metrics["Growth (%)"] > 0
                and event_metrics["Max drawdown (%)"] <= MAX_DRAWDOWN_PCT
                and event_metrics["Profit factor"] >= PROFIT_FACTOR_TARGET
                and retention >= 25
            )
            rows.append({
                "EV minimum": ev_min,
                "TP probability minimum": tp_min,
                **event_metrics,
                "Entry tersedia": len(event_index),
                "Entry diterima": len(selected_events),
                "Retensi entry (%)": retention,
                "Retensi net profit (%)": profit_retention,
                "Eligible": eligible,
            })
    table = pd.DataFrame(rows)
    eligible = table[table["Eligible"]]
    fallback = eligible.empty
    pool = eligible if not eligible.empty else table[table["Entry diterima"] >= 20]
    if pool.empty:
        pool = table[table["Entry diterima"] > 0]
    chosen = pool.sort_values(
        ["Profit factor", "Growth (%)", "Retensi entry (%)"],
        ascending=[False, False, False],
    ).iloc[0]
    return table, {
        "ev_min": float(chosen["EV minimum"]),
        "tp_min": float(chosen["TP probability minimum"]),
    }, fallback


def _realized_event_values(
    events: pd.DataFrame,
    best: dict[str, object],
) -> pd.Series:
    tp = float(best["TP (USD)"])
    sl = float(best["SL (USD)"])
    values = pd.Series(0.0, index=events.index)
    values.loc[events["actual"].eq("TP_FIRST")] = tp
    values.loc[events["actual"].eq("SL_FIRST")] = -sl
    timeout_mask = events["actual"].eq("TIMEOUT")
    timeout_values = events.get(
        "timeout_mark_to_market_usd", pd.Series(0.0, index=events.index)
    ).reindex(events.index).fillna(0.0).clip(-sl, tp)
    values.loc[timeout_mask] = timeout_values.loc[timeout_mask]
    return values


def _event_economic_metrics(values: pd.Series) -> dict[str, float]:
    if values.empty:
        return {
            "Equity akhir": 1000.0,
            "Growth (%)": 0.0,
            "Max drawdown": 0.0,
            "Max drawdown (%)": 0.0,
            "Profit factor": 0.0,
            "Transaksi": 0.0,
            "Win rate (%)": 0.0,
        }
    equity = 1000.0 + values.cumsum()
    running_peak = pd.concat(
        [pd.Series([1000.0]), equity.reset_index(drop=True)]
    ).cummax().iloc[1:].to_numpy()
    drawdown = running_peak - equity.to_numpy()
    drawdown_pct = np.divide(
        drawdown, running_peak, out=np.zeros_like(drawdown), where=running_peak > 0
    ) * 100
    gross_profit = float(values[values > 0].sum())
    gross_loss = float(-values[values < 0].sum())
    return {
        "Equity akhir": float(equity.iloc[-1]),
        "Growth (%)": float((equity.iloc[-1] / 1000.0 - 1) * 100),
        "Max drawdown": float(drawdown.max(initial=0.0)),
        "Max drawdown (%)": float(drawdown_pct.max(initial=0.0)),
        "Profit factor": gross_profit / gross_loss if gross_loss > 0 else np.inf,
        "Transaksi": float(len(values)),
        "Win rate (%)": float((values > 0).mean() * 100),
    }


def _quality_gate(
    signals: pd.DataFrame,
    events: pd.DataFrame,
    rule: dict[str, float],
    strategy: str,
) -> pd.DataFrame:
    aligned = events.reindex(signals.index)
    mask = (
        aligned["expected_value"].ge(rule["ev_min"])
        & aligned["p_tp_first"].ge(rule["tp_min"])
    )
    selected = signals.loc[mask.fillna(False)].copy()
    if not selected.empty:
        selected["outcome_probability"] = aligned.loc[selected.index, "p_tp_first"]
        selected["expected_value"] = aligned.loc[selected.index, "expected_value"]
        selected["strategy"] = strategy
    return selected


def _delay_test(
    data: pd.DataFrame,
    signals: pd.DataFrame,
    best: dict[str, object],
    config: RiskControlConfig,
) -> pd.DataFrame:
    rows = []
    for minutes in (0, 1, 5, 15):
        delayed = _delay_signals(signals, data.index, minutes)
        result = _simulate_risk_control(data, delayed, best, config)
        rows.append({
            "Delay entry (menit)": minutes,
            "Entry tersedia": len(delayed),
            **_metric_values(result),
        })
    return pd.DataFrame(rows)


def _data_audit(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    expected = set(pd.period_range("2024-01", "2026-06", freq="M"))
    actual = set(data.loc[DEVELOPMENT_START:CONFIRMATION_END].index.to_period("M").unique())
    rows.append({
        "Pemeriksaan": "Cakupan bulan 2024-01 sampai 2026-06",
        "Status": "LOLOS" if expected.issubset(actual) else "BELUM",
        "Detail": f"{len(expected & actual)}/{len(expected)} bulan tersedia",
    })
    duplicates = int(data.index.duplicated().sum())
    rows.append({
        "Pemeriksaan": "Timestamp unik",
        "Status": "LOLOS" if duplicates == 0 else "BELUM",
        "Detail": f"Duplikat: {duplicates}",
    })
    invalid = int(
        (
            (data["Low"] > data[["Open", "Close"]].min(axis=1))
            | (data["High"] < data[["Open", "Close"]].max(axis=1))
            | (data["High"] < data["Low"])
        ).sum()
    )
    rows.append({
        "Pemeriksaan": "Struktur OHLC",
        "Status": "LOLOS" if invalid == 0 else "BELUM",
        "Detail": f"Baris tidak valid: {invalid}",
    })
    return pd.DataFrame(rows)


def _outcome_report(
    development: pd.DataFrame,
    confirmation: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for period, frame in (
        ("Development 2024-2025", development),
        ("Historical confirmation 2026H1", confirmation),
    ):
        for outcome, count in frame["raw_outcome"].value_counts().items():
            rows.append({
                "Periode": period,
                "Outcome": outcome,
                "Events": int(count),
                "Proporsi (%)": float(count / len(frame) * 100),
            })
    return pd.DataFrame(rows)


def _direction_report(
    development: pd.DataFrame,
    confirmation: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for period, source in (
        ("Development 2024-2025", development),
        ("Historical confirmation 2026H1", confirmation),
    ):
        for direction, frame in source.groupby("direction"):
            rows.append({
                "Periode": period,
                "Arah": direction,
                "Events": len(frame),
                "TP rate (%)": float(frame["raw_outcome"].eq("TP_FIRST").mean() * 100),
                "TIMEOUT rate (%)": float(frame["raw_outcome"].eq("TIMEOUT").mean() * 100),
                "Median MFE": float(frame["mfe_usd"].median()),
                "Median MAE": float(frame["mae_usd"].median()),
            })
    return pd.DataFrame(rows)


def _excursion_report(
    development: pd.DataFrame,
    confirmation: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for period, source in (
        ("Development 2024-2025", development),
        ("Historical confirmation 2026H1", confirmation),
    ):
        for outcome, frame in source.groupby("raw_outcome"):
            rows.append({
                "Periode": period,
                "Outcome": outcome,
                "Events": len(frame),
                "Median MFE": float(frame["mfe_usd"].median()),
                "P90 MFE": float(frame["mfe_usd"].quantile(0.9)),
                "Median MAE": float(frame["mae_usd"].median()),
                "P90 MAE": float(frame["mae_usd"].quantile(0.9)),
                "Median jam outcome": float(frame["hours_to_outcome"].median()),
            })
    return pd.DataFrame(rows)


def _binary_compatibility_frame(events: pd.DataFrame) -> pd.DataFrame:
    output = events.copy()
    output["target"] = output["raw_outcome"].eq("TP_FIRST").astype(float)
    output["probability"] = output["p_tp_first"]
    return output


def _decision(
    probability: dict[str, float],
    economic: dict[str, float],
    result,
    available_signals: pd.DataFrame,
    selected_signals: pd.DataFrame,
    folds: pd.DataFrame,
    stress: pd.DataFrame,
    monte_carlo: dict[str, float],
    model_fallback: bool,
    threshold_fallback: bool,
) -> dict[str, object]:
    criteria = {
        "Multiclass Brier lebih baik dari baseline": probability["Brier improvement (%)"] > 0,
        "Mayoritas fold Brier positif": int((folds["Brier improvement (%)"] > 0).sum()) >= len(FOLDS) / 2,
        "Growth historical confirmation positif": economic["Growth (%)"] > 0,
        "Max drawdown <= 10%": economic["Max drawdown (%)"] <= MAX_DRAWDOWN_PCT,
        "Profit factor >= 1.30": economic["Profit factor"] >= PROFIT_FACTOR_TARGET,
        "Minimal 20 transaksi confirmation": economic["Transaksi"] >= 20,
        "Retensi entry >= 25%": len(selected_signals) >= max(1, int(len(available_signals) * 0.25)),
        "Stress profitable 9/9": len(stress) == 9 and bool((stress["Growth (%)"] > 0).all()),
        "Monte Carlo rugi <= 10%": monte_carlo["Probabilitas equity akhir < modal awal (%)"] <= MAX_MONTE_CARLO_LOSS_PCT,
        "Model tanpa fallback": not model_fallback,
        "Threshold tanpa fallback": not threshold_fallback,
    }
    return {
        **{key: bool(value) for key, value in criteria.items()},
        "Jumlah kriteria lolos": int(sum(bool(value) for value in criteria.values())),
        "Jumlah kriteria": len(criteria),
        "Lulus seluruh kriteria": bool(all(criteria.values())),
    }


def _compact_quality_events(events: pd.DataFrame) -> pd.DataFrame:
    compact = _compact_events(events)
    extra = [
        "p_sl_first", "p_timeout", "p_tp_first", "expected_value",
        "mfe_usd", "mae_usd", "timeout_mark_to_market_usd",
    ]
    for column in extra:
        if column in events.columns:
            compact[column] = events[column]
    return compact
