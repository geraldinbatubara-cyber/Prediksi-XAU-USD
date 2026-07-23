from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from gold_forecast.exact_broker_oos import (
    POINT_SIZE,
    SLIPPAGE_POINTS,
    _compact_curve,
    _prepare_m1,
)
from gold_forecast.simulation import CONTRACT_OUNCES_PER_LOT
from gold_forecast.v1_entry_outcome import (
    FEATURE_COLUMNS,
    OUTCOME_HORIZON_DAYS,
    _balanced_signals,
    _delay_signals,
    _event_dataset,
    _outcome_features,
    _safe_monte_carlo,
    _session_audit,
)
from gold_forecast.v1_entry_quality import (
    _data_audit,
    _event_economic_metrics,
    _stress_test,
)
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


DEVELOPMENT_START = pd.Timestamp("2022-01-01")
DEVELOPMENT_END = pd.Timestamp("2025-12-31 23:59:59")
CONFIRMATION_START = pd.Timestamp("2026-01-01")
CONFIRMATION_END = pd.Timestamp("2026-06-30 23:59:59")
MODEL_NAMES = ("Rule Scorecard", "Logistic Regression", "Gradient Boosting", "Probability Ensemble")


@dataclass(frozen=True)
class PathFold:
    name: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def _build_folds() -> tuple[PathFold, ...]:
    rows = []
    for number, quarter in enumerate(pd.period_range("2023Q1", "2025Q4", freq="Q"), start=1):
        test_start = quarter.start_time
        rows.append(
            PathFold(
                f"Fold {number}",
                DEVELOPMENT_START,
                test_start - pd.Timedelta(days=OUTCOME_HORIZON_DAYS, seconds=1),
                test_start,
                quarter.end_time,
            )
        )
    return tuple(rows)


FOLDS = _build_folds()


def run_v1_entry_quality_path_lab(
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

    development_signals = _unique_signals(
        _balanced_signals(
            data, signal_daily, best, entry_features, balanced_config, spread_limit,
            DEVELOPMENT_START, DEVELOPMENT_END,
        )
    )
    confirmation_signals = _unique_signals(
        _balanced_signals(
            data, signal_daily, best, entry_features, balanced_config, spread_limit,
            CONFIRMATION_START, CONFIRMATION_END,
        )
    )
    feature_frame = _outcome_features(data, regime_features, m15, best)
    development_events = _path_aware_events(
        data, development_signals, feature_frame, best
    )
    confirmation_events = _path_aware_events(
        data, confirmation_signals, feature_frame, best
    )
    if len(development_events) < 250 or len(confirmation_events) < 20:
        raise ValueError(
            "Jumlah event tidak cukup untuk Path-Aware Lab: "
            f"development={len(development_events)}, confirmation={len(confirmation_events)}."
        )

    fold_rows: list[dict[str, object]] = []
    oof_rows: list[pd.DataFrame] = []
    for model_name in MODEL_NAMES:
        for fold in FOLDS:
            train = development_events.loc[fold.train_start:fold.train_end]
            test = development_events.loc[fold.test_start:fold.test_end]
            probability = _fit_binary(model_name, train, test)
            metrics = _binary_metrics(test["target"], probability)
            fold_rows.append({
                "Model": model_name,
                "Fold": fold.name,
                "Train events": len(train),
                "Test events": len(test),
                **metrics,
            })
            oof_rows.append(pd.DataFrame({
                "Model": model_name,
                "probability": probability,
                "target": test["target"],
                "direction": test["direction"],
                "mfe_usd": test["mfe_usd"],
                "mae_usd": test["mae_usd"],
                "hours_to_outcome": test["hours_to_outcome"],
            }, index=test.index))

    folds = pd.DataFrame(fold_rows)
    oof = pd.concat(oof_rows).sort_index()
    model_summary = _model_summary(folds)
    selected_model = str(model_summary.iloc[0]["Model"])
    selected_folds = folds[folds["Model"].eq(selected_model)]
    model_fallback = int((selected_folds["Brier improvement (%)"] > 0).sum()) < 7
    selected_oof = oof[oof["Model"].eq(selected_model)].copy()
    selected_oof = selected_oof.loc[~selected_oof.index.duplicated(keep="last")]
    threshold_table, selected_rule, threshold_fallback = _select_rule(
        selected_oof, best
    )

    confirmation_probability = _fit_binary(
        selected_model, development_events, confirmation_events
    )
    confirmation_events = confirmation_events.copy()
    confirmation_events["probability"] = confirmation_probability
    confirmation_events["expected_value"] = _expected_value(
        confirmation_probability,
        best,
        confirmation_events["spread_points"],
    )
    confirmation_metrics = _binary_metrics(
        confirmation_events["target"], confirmation_probability
    )
    selected_signals = _quality_gate(
        confirmation_signals,
        confirmation_events,
        selected_rule,
        "v1 Entry Quality Path-Aware",
    )
    simulation_config = RiskControlConfig(
        "Entry Quality Path-Aware v3",
        "Binary path-aware EV gate",
        max_total_positions=1,
        max_same_direction=1,
    )
    confirmation_data = data.loc[CONFIRMATION_START:CONFIRMATION_END]
    baseline_result = _simulate_risk_control(
        confirmation_data, confirmation_signals, best, simulation_config
    )
    selected_result = _simulate_risk_control(
        confirmation_data, selected_signals, best, simulation_config
    )
    economic = pd.DataFrame([
        {"Strategi": "Balanced Entry Frozen", **_metric_values(baseline_result)},
        {"Strategi": "v1 Entry Quality Path-Aware", **_metric_values(selected_result)},
    ])
    selected_metrics = _metric_values(selected_result)
    baseline_net = float(baseline_result.summary["Total net P/L"])
    selected_net = float(selected_result.summary["Total net P/L"])
    profit_retention = selected_net / baseline_net * 100 if baseline_net > 0 else 0.0
    monte_carlo, monte_carlo_summary = _safe_monte_carlo(selected_result.trades)
    stress = _stress_test(
        confirmation_data, selected_signals, best, simulation_config
    )
    delay = _delay_test(
        confirmation_data, selected_signals, best, simulation_config
    )
    classification_audit = _classification_audit(
        confirmation_events, selected_rule
    )
    decision = _decision(
        confirmation_metrics,
        selected_metrics,
        selected_folds,
        stress,
        monte_carlo_summary,
        len(selected_signals),
        len(confirmation_signals),
        profit_retention,
        model_fallback,
        threshold_fallback,
    )

    return {
        "methodology": {
            "Baseline lock": "v1 Exact Baseline, Balanced Entry, ledger, dan Live Trading tidak diubah",
            "Development": "01 Jan 2022 - 31 Des 2025; 12 expanding quarterly folds",
            "Purge": f"{OUTCOME_HORIZON_DAYS} hari antara akhir train dan awal test",
            "Historical confirmation": "01 Jan 2026 - 30 Jun 2026; bukan true OOS baru",
            "True prospective OOS": "Paper shadow setelah kandidat dibekukan sampai 31 Agustus 2026",
            "Outcome": "TP_FIRST vs SL_FIRST; barrier pertama menentukan label",
            "Path-aware excursion": "MFE/MAE berhenti tepat pada candle barrier pertama",
            "Selected model": selected_model,
            "Selected EV minimum": selected_rule["ev_min"],
            "Selected TP probability minimum": selected_rule["tp_min"],
            "Model fallback": model_fallback,
            "Threshold fallback": threshold_fallback,
            "Caveat": (
                "2026H1 sudah pernah diamati. Hasil ini hanya historical confirmation dan "
                "tidak mengubah baseline atau paper live trading."
            ),
        },
        "data_audit": _extended_data_audit(data),
        "folds": folds,
        "model_summary": model_summary,
        "threshold_development": threshold_table,
        "confirmation_metrics": pd.DataFrame([confirmation_metrics]),
        "economic": economic,
        "decision": decision,
        "classification_audit": classification_audit,
        "outcome_distribution": _outcome_report(
            development_events, confirmation_events
        ),
        "direction_audit": _direction_report(
            development_events, confirmation_events
        ),
        "path_excursions": _path_report(
            development_events, confirmation_events
        ),
        "session_audit": _session_audit(confirmation_events),
        "stress": stress,
        "delay_stress": delay,
        "selected_result": _compact_curve(selected_result),
        "selected_monte_carlo": monte_carlo,
        "selected_monte_carlo_summary": monte_carlo_summary,
        "confirmation_events": _compact_events(confirmation_events, selected_rule),
        "signal_counts": {
            "Balanced Entry Frozen": int(len(confirmation_signals)),
            "v1 Entry Quality Path-Aware": int(len(selected_signals)),
        },
        "profit_retention": profit_retention,
    }


def _unique_signals(signals: pd.DataFrame) -> pd.DataFrame:
    return signals.loc[~signals.index.duplicated(keep="last")].sort_index()


def _path_aware_events(
    data: pd.DataFrame,
    signals: pd.DataFrame,
    features: pd.DataFrame,
    best: dict[str, object],
) -> pd.DataFrame:
    events = _event_dataset(data, signals, features, best)
    if events.empty:
        return events
    rows = []
    for timestamp, event in events.iterrows():
        signal = signals.loc[timestamp]
        lot = float(signal.get("lot", best.get("Lot", 0.01)) or 0.01)
        path = _first_barrier_path(
            data, pd.Timestamp(timestamp), str(event["direction"]), lot, best
        )
        updated = event.to_dict()
        updated.update(path)
        updated["target"] = float(path["raw_outcome"] == "TP_FIRST")
        rows.append({"entry_time": timestamp, **updated})
    return pd.DataFrame(rows).set_index("entry_time").sort_index()


def _first_barrier_path(
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
    tp_usd = float(best["TP (USD)"])
    sl_usd = float(best["SL (USD)"])
    if direction == "BUY":
        entry_price = float(candle["Close"]) + spread + slippage
        target = entry_price + tp_usd / units
        stop = entry_price - sl_usd / units
    else:
        entry_price = float(candle["Close"]) - slippage
        target = entry_price - tp_usd / units
        stop = entry_price + sl_usd / units

    deadline = entry_time + pd.Timedelta(days=OUTCOME_HORIZON_DAYS)
    path = data.loc[(data.index > entry_time) & (data.index <= deadline)]
    mfe = 0.0
    mae = 0.0
    for timestamp, bar in path.iterrows():
        bar_spread = max(float(bar["SpreadPoints"]) * POINT_SIZE, 0.0)
        if direction == "BUY":
            favorable = max((float(bar["High"]) - entry_price) * units, 0.0)
            adverse = max((entry_price - float(bar["Low"])) * units, 0.0)
            tp_hit = float(bar["High"]) >= target
            sl_hit = float(bar["Low"]) <= stop
        else:
            ask_high = float(bar["High"]) + bar_spread
            ask_low = float(bar["Low"]) + bar_spread
            favorable = max((entry_price - ask_low) * units, 0.0)
            adverse = max((ask_high - entry_price) * units, 0.0)
            tp_hit = ask_low <= target
            sl_hit = ask_high >= stop
        mfe = max(mfe, min(favorable, tp_usd))
        mae = max(mae, min(adverse, sl_usd))
        if tp_hit or sl_hit:
            ambiguous = tp_hit and sl_hit
            outcome = "SL_FIRST" if sl_hit else "TP_FIRST"
            return {
                "raw_outcome": outcome,
                "ambiguous": ambiguous,
                "outcome_time": timestamp,
                "hours_to_outcome": (timestamp - entry_time).total_seconds() / 3600,
                "mfe_usd": mfe,
                "mae_usd": mae,
            }

    if path.empty:
        final_net = 0.0
    elif direction == "BUY":
        final_net = (float(path["Close"].iloc[-1]) - entry_price) * units
    else:
        final_net = (entry_price - float(path["Close"].iloc[-1])) * units
    fallback_outcome = "TP_FIRST" if final_net > 0 else "SL_FIRST"
    return {
        "raw_outcome": fallback_outcome,
        "ambiguous": False,
        "outcome_time": deadline,
        "hours_to_outcome": OUTCOME_HORIZON_DAYS * 24.0,
        "mfe_usd": mfe,
        "mae_usd": mae,
    }


def _fit_binary(
    model_name: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> pd.Series:
    if test.empty:
        return pd.Series(dtype=float, index=test.index)
    if model_name == "Probability Ensemble":
        logistic = _fit_binary("Logistic Regression", train, test)
        boosting = _fit_binary("Gradient Boosting", train, test)
        return ((logistic + boosting) / 2).clip(0.01, 0.99)

    x_train = train[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    x_test = test[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    medians = x_train.median()
    x_train = x_train.fillna(medians).fillna(0.0)
    x_test = x_test.fillna(medians).fillna(0.0)
    target = train["target"].astype(int)
    if target.nunique() < 2:
        return pd.Series(float(target.iloc[0]), index=test.index)

    if model_name == "Rule Scorecard":
        train_score = _rule_score(train)
        test_score = _rule_score(test)
        model = LogisticRegression(C=0.25, max_iter=1000, random_state=42)
        model.fit(train_score.to_numpy().reshape(-1, 1), target)
        values = model.predict_proba(test_score.to_numpy().reshape(-1, 1))[:, 1]
    elif model_name == "Logistic Regression":
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.20, max_iter=2000, random_state=42),
        )
        model.fit(x_train, target)
        values = model.predict_proba(x_test)[:, 1]
    else:
        model = HistGradientBoostingClassifier(
            learning_rate=0.035,
            max_iter=80,
            max_leaf_nodes=7,
            min_samples_leaf=max(12, len(train) // 20),
            l2_regularization=3.0,
            random_state=42,
        )
        model.fit(x_train, target)
        values = model.predict_proba(x_test)[:, 1]
    return pd.Series(values, index=test.index).clip(0.01, 0.99)


def _rule_score(frame: pd.DataFrame) -> pd.Series:
    spread_scale = max(float(frame["spread_points"].quantile(0.90)), 1.0)
    return (
        0.45 * frame["conviction_ratio"].clip(0, 3)
        + 0.30 * frame["trend_strength"].clip(0, 3)
        + 0.25 * frame["efficiency"].clip(0, 1)
        + 0.15 * frame["rsi_aligned"].clip(-1, 1)
        - 0.20 * frame["choppiness"].clip(0, 100) / 100
        - 0.15 * frame["spread_points"].clip(lower=0) / spread_scale
    )


def _binary_metrics(target: pd.Series, probability: pd.Series) -> dict[str, float]:
    target = target.astype(int)
    probability = probability.astype(float).clip(0.001, 0.999)
    prevalence = float(target.mean())
    baseline_brier = float(np.mean((target - prevalence) ** 2))
    brier = float(brier_score_loss(target, probability))
    auc = (
        float(roc_auc_score(target, probability))
        if target.nunique() > 1 else 0.5
    )
    pr_auc = (
        float(average_precision_score(target, probability))
        if target.nunique() > 1 else prevalence
    )
    return {
        "Observasi": float(len(target)),
        "TP rate (%)": prevalence * 100,
        "Brier score": brier,
        "Baseline Brier": baseline_brier,
        "Brier improvement (%)": (
            (baseline_brier - brier) / baseline_brier * 100
            if baseline_brier > 0 else 0.0
        ),
        "ROC-AUC": auc,
        "PR-AUC": pr_auc,
    }


def _model_summary(folds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, group in folds.groupby("Model", sort=False):
        rows.append({
            "Model": model,
            "Mean Brier": float(group["Brier score"].mean()),
            "Worst Brier": float(group["Brier score"].max()),
            "Mean Brier improvement (%)": float(group["Brier improvement (%)"].mean()),
            "Positive Brier folds": int((group["Brier improvement (%)"] > 0).sum()),
            "Mean ROC-AUC": float(group["ROC-AUC"].mean()),
            "Mean PR-AUC": float(group["PR-AUC"].mean()),
        })
    return pd.DataFrame(rows).sort_values(
        ["Positive Brier folds", "Mean Brier improvement (%)", "Mean ROC-AUC"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def _expected_value(
    probability: pd.Series,
    best: dict[str, object],
    spread_points: pd.Series,
) -> pd.Series:
    tp = float(best["TP (USD)"])
    sl = float(best["SL (USD)"])
    execution_cost = spread_points.reindex(probability.index).fillna(0.0) * 0.01
    return probability * tp - (1 - probability) * sl - execution_cost


def _select_rule(
    oof: pd.DataFrame,
    best: dict[str, object],
) -> tuple[pd.DataFrame, dict[str, float], bool]:
    events = oof.copy()
    events["expected_value"] = _expected_value(
        events["probability"], best, pd.Series(0.0, index=events.index)
    )
    tp = float(best["TP (USD)"])
    sl = float(best["SL (USD)"])
    realized = pd.Series(
        np.where(events["target"].eq(1), tp, -sl),
        index=events.index,
    )
    baseline_net = float(realized.sum())
    rows = []
    for ev_min in (0.0, 1.0, 2.0, 3.0, 4.0, 5.0):
        for tp_min in (0.30, 0.35, 0.40, 0.45, 0.50):
            mask = (
                events["expected_value"].ge(ev_min)
                & events["probability"].ge(tp_min)
            )
            values = realized.loc[mask]
            metrics = _event_economic_metrics(values)
            net = float(values.sum())
            retention = len(values) / len(events) * 100 if len(events) else 0.0
            profit_retention = net / baseline_net * 100 if baseline_net > 0 else 0.0
            eligible = bool(
                len(values) >= 80
                and metrics["Growth (%)"] > 0
                and metrics["Max drawdown (%)"] <= MAX_DRAWDOWN_PCT
                and metrics["Profit factor"] >= PROFIT_FACTOR_TARGET
                and profit_retention >= 70
            )
            rows.append({
                "EV minimum": ev_min,
                "TP probability minimum": tp_min,
                **metrics,
                "Entry tersedia": len(events),
                "Entry diterima": len(values),
                "Retensi entry (%)": retention,
                "Retensi net profit (%)": profit_retention,
                "Eligible": eligible,
            })
    table = pd.DataFrame(rows)
    eligible = table[table["Eligible"]]
    fallback = eligible.empty
    pool = eligible if not eligible.empty else table[table["Entry diterima"] >= 80]
    if pool.empty:
        pool = table[table["Entry diterima"] > 0]
    selected = pool.sort_values(
        ["Profit factor", "Growth (%)", "Retensi net profit (%)"],
        ascending=[False, False, False],
    ).iloc[0]
    return table, {
        "ev_min": float(selected["EV minimum"]),
        "tp_min": float(selected["TP probability minimum"]),
    }, fallback


def _quality_gate(
    signals: pd.DataFrame,
    events: pd.DataFrame,
    rule: dict[str, float],
    strategy: str,
) -> pd.DataFrame:
    aligned = events.reindex(signals.index)
    mask = (
        aligned["expected_value"].ge(rule["ev_min"])
        & aligned["probability"].ge(rule["tp_min"])
    )
    selected = signals.loc[mask.fillna(False)].copy()
    if not selected.empty:
        selected["outcome_probability"] = aligned.loc[selected.index, "probability"].to_numpy()
        selected["expected_value"] = aligned.loc[selected.index, "expected_value"].to_numpy()
        selected["strategy"] = strategy
    return selected


def _classification_audit(
    events: pd.DataFrame,
    rule: dict[str, float],
) -> pd.DataFrame:
    accepted = (
        events["expected_value"].ge(rule["ev_min"])
        & events["probability"].ge(rule["tp_min"])
    )
    winner = events["target"].eq(1)
    rows = [
        ("Winner diterima", accepted & winner),
        ("Winner salah ditolak", ~accepted & winner),
        ("Loser berhasil ditolak", ~accepted & ~winner),
        ("Loser tetap diterima", accepted & ~winner),
    ]
    output = []
    for label, mask in rows:
        output.append({
            "Kelompok": label,
            "Events": int(mask.sum()),
            "Proporsi dari seluruh event (%)": float(mask.mean() * 100),
            "Median probability (%)": (
                float(events.loc[mask, "probability"].median() * 100)
                if mask.any() else np.nan
            ),
            "Median MFE": (
                float(events.loc[mask, "mfe_usd"].median())
                if mask.any() else np.nan
            ),
            "Median MAE": (
                float(events.loc[mask, "mae_usd"].median())
                if mask.any() else np.nan
            ),
        })
    return pd.DataFrame(output)


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


def _extended_data_audit(data: pd.DataFrame) -> pd.DataFrame:
    audit = _data_audit(data)
    expected = set(pd.period_range("2022-01", "2026-06", freq="M"))
    actual = set(data.loc[DEVELOPMENT_START:CONFIRMATION_END].index.to_period("M").unique())
    audit.loc[audit["Pemeriksaan"].str.startswith("Cakupan bulan"), "Pemeriksaan"] = (
        "Cakupan bulan 2022-01 sampai 2026-06"
    )
    audit.loc[audit["Pemeriksaan"].str.startswith("Cakupan bulan"), "Status"] = (
        "LOLOS" if expected.issubset(actual) else "BELUM"
    )
    audit.loc[audit["Pemeriksaan"].str.startswith("Cakupan bulan"), "Detail"] = (
        f"{len(expected & actual)}/{len(expected)} bulan tersedia"
    )
    return audit


def _outcome_report(
    development: pd.DataFrame,
    confirmation: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for period, frame in (
        ("Development 2022-2025", development),
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
        ("Development 2022-2025", development),
        ("Historical confirmation 2026H1", confirmation),
    ):
        for direction, frame in source.groupby("direction"):
            rows.append({
                "Periode": period,
                "Arah": direction,
                "Events": len(frame),
                "TP rate (%)": float(frame["target"].mean() * 100),
                "Median MFE": float(frame["mfe_usd"].median()),
                "Median MAE": float(frame["mae_usd"].median()),
                "Median jam outcome": float(frame["hours_to_outcome"].median()),
            })
    return pd.DataFrame(rows)


def _path_report(
    development: pd.DataFrame,
    confirmation: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for period, source in (
        ("Development 2022-2025", development),
        ("Historical confirmation 2026H1", confirmation),
    ):
        for outcome, frame in source.groupby("raw_outcome"):
            rows.append({
                "Periode": period,
                "Outcome": outcome,
                "Events": len(frame),
                "Ambiguous": int(frame["ambiguous"].sum()),
                "Median MFE": float(frame["mfe_usd"].median()),
                "P90 MFE": float(frame["mfe_usd"].quantile(0.90)),
                "Median MAE": float(frame["mae_usd"].median()),
                "P90 MAE": float(frame["mae_usd"].quantile(0.90)),
                "Median jam outcome": float(frame["hours_to_outcome"].median()),
            })
    return pd.DataFrame(rows)


def _decision(
    probability: dict[str, float],
    economic: dict[str, float],
    folds: pd.DataFrame,
    stress: pd.DataFrame,
    monte_carlo: dict[str, float],
    selected_count: int,
    available_count: int,
    profit_retention: float,
    model_fallback: bool,
    threshold_fallback: bool,
) -> dict[str, object]:
    criteria = {
        "Brier lebih baik dari baseline": probability["Brier improvement (%)"] > 0,
        "Minimal 7 dari 12 fold Brier positif": int((folds["Brier improvement (%)"] > 0).sum()) >= 7,
        "TP ROC-AUC >= 0.58": probability["ROC-AUC"] >= 0.58,
        "Growth historical confirmation positif": economic["Growth (%)"] > 0,
        "Max drawdown <= 10%": economic["Max drawdown (%)"] <= MAX_DRAWDOWN_PCT,
        "Profit factor >= 1.30": economic["Profit factor"] >= PROFIT_FACTOR_TARGET,
        "Minimal 30 transaksi confirmation": economic["Transaksi"] >= 30,
        "Retensi entry >= 25%": selected_count >= max(1, int(available_count * 0.25)),
        "Retensi net profit >= 70%": profit_retention >= 70,
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
        "Retensi net profit (%)": float(profit_retention),
    }


def _compact_events(
    events: pd.DataFrame,
    rule: dict[str, float],
) -> pd.DataFrame:
    accepted = (
        events["expected_value"].ge(rule["ev_min"])
        & events["probability"].ge(rule["tp_min"])
    )
    columns = [
        "direction", "raw_outcome", "ambiguous", "probability",
        "expected_value", "outcome_time", "hours_to_outcome",
        "mfe_usd", "mae_usd", "conviction_ratio", "spread_points",
        "adx", "efficiency", "choppiness",
    ]
    output = events[[column for column in columns if column in events.columns]].copy()
    output["gate_status"] = np.where(accepted, "DITERIMA", "DITOLAK")
    return output
