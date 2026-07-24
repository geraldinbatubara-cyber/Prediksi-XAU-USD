from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    precision_score,
    recall_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from gold_forecast.exact_broker_oos import _prepare_m1
from gold_forecast.v1_directional_specialization import (
    _apply_symmetric_calibration,
    _class_weights,
    _ledger_metric_values,
    _monte_carlo_summary,
    _stress_summary,
    _trades_in_period,
)
from gold_forecast.v1_entry_quality_path import FOLDS, _unique_signals
from gold_forecast.v1_regime_classifier_v3 import _fit_platt
from gold_forecast.v1_risk_control import (
    RiskControlConfig,
    _metric_values,
    _simulate_risk_control,
)
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
    MODEL_FEATURES,
    _mean_reversion_opportunities,
    _opportunities_to_signals,
    _profit_concentration,
    _range_quality_frame,
    _train_outcome_model,
)
from gold_forecast.v1_trend_strength_stability import _extended_data_audit
from gold_forecast.v1_unified_benchmark import _unified_best


CANDIDATES = (
    "Sideways v1 Control",
    "Persistence Gate",
    "Breakout Hazard Gate",
    "Directional Calibration",
    "Session-Aware Persistence",
    "Adaptive Persistence Ensemble",
)
PERSISTENCE_FEATURES = (
    *MODEL_FEATURES,
    "range_age_hours",
    "midpoint_drift_atr",
    "range_width_change",
    "adx_change_3h",
    "atr_acceleration",
    "touch_imbalance",
    "session_sin",
    "session_cos",
    "direction_code",
)


def run_v1_sideways_specialist_v2_lab(
    gold_m1: pd.DataFrame,
    frozen_payload: dict[str, object],
    v1_payload: dict[str, object] | None = None,
) -> dict[str, object]:
    data = _prepare_m1(gold_m1)
    _, leaderboard, _ = frozen_payload["v1"]
    best = {
        **_unified_best(leaderboard.iloc[0].to_dict()),
        "Close-all target equity": False,
        "Max BUY": 1,
        "Max SELL": 1,
    }
    config = RiskControlConfig(
        "Sideways Specialist v2",
        "Range persistence and breakout hazard",
        max_total_positions=1,
        max_same_direction=1,
    )
    spread_limit = float(
        data.loc[DEVELOPMENT_START:DEVELOPMENT_END, "SpreadPoints"].quantile(0.90)
    )
    features, h1, m15 = _regime_features(data)
    range_frame = _range_quality_frame(features, h1)
    opportunities = _mean_reversion_opportunities(
        data, range_frame, m15, spread_limit
    )
    opportunities = _augment_opportunities(data, range_frame, opportunities)
    outcome_model, outcome_selection = _train_outcome_model(opportunities)
    persistence_model = _train_binary_model(
        opportunities, "persistence_12h", "Range Persistence", 101
    )
    hazard_model = _train_binary_model(
        opportunities, "adverse_breakout_6h", "Adverse Breakout Hazard", 111
    )
    directional_thresholds, directional_audit = _directional_thresholds(
        opportunities, outcome_model["probability"]
    )
    selected_sessions, session_audit = _session_selection(opportunities)
    signals, funnel = _candidate_signals(
        opportunities,
        outcome_model,
        persistence_model,
        hazard_model,
        directional_thresholds,
        selected_sessions,
        best,
    )

    development_data = data.loc[DEVELOPMENT_START:DEVELOPMENT_END]
    reference_data = data.loc[REFERENCE_START:REFERENCE_END]
    development_results = _simulate_all(
        development_data, signals, best, config, DEVELOPMENT_START, DEVELOPMENT_END
    )
    reference_results = _simulate_all(
        reference_data, signals, best, config, REFERENCE_START, REFERENCE_END
    )
    development = _result_table(
        development_results, signals, DEVELOPMENT_START, DEVELOPMENT_END
    )
    reference = _result_table(
        reference_results, signals, REFERENCE_START, REFERENCE_END
    )
    periods = _period_validation(development_results, signals)
    folds = _fold_evaluation(development_results)
    classification = _classification_tables(
        opportunities, outcome_model, persistence_model, hazard_model
    )
    monte_carlo = _monte_carlo_summary(development_results)
    concentration = _profit_concentration(development_results)
    decisions = _decision_table(
        development, periods, folds, monte_carlo, concentration
    )
    ranking = _selection_ranking(
        development,
        reference,
        periods,
        classification["locked"],
        decisions,
    )
    eligible = ranking.loc[ranking["Selection eligible"]]
    winner = str(eligible.iloc[0]["Kandidat"]) if not eligible.empty else ""
    stress = (
        _stress_summary(development_data, signals, best, config, [winner])
        if winner
        else pd.DataFrame()
    )
    stress_passed = (
        int(stress.iloc[0]["Skenario profitable"]) if not stress.empty else 0
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
            "Name": "v1 Sideways Specialist Lab v2 - Range Persistence & Breakout Hazard",
            "Mandat": (
                "Sideways v1 menjadi control. V2 hanya membuka mean-reversion "
                "ketika outcome dan persistence cukup tinggi serta adverse-breakout "
                "hazard cukup rendah."
            ),
            "Persistence": (
                "Range dinilai bertahan bila 12 jam berikutnya tidak menembus "
                "boundary, midpoint tidak bergeser >0.5 ATR, dan ekspansi path "
                "tetap terkendali."
            ),
            "Hazard": (
                "Risiko adverse breakout 6 jam dipelajari terpisah untuk entry BUY "
                "dan SELL menggunakan dinamika ADX, ATR, midpoint, touch imbalance, "
                "usia range, sesi, dan arah."
            ),
            "Directional": (
                "Threshold outcome BUY dan SELL dikalibrasi terpisah hanya pada "
                "2023H2."
            ),
            "Session": (
                f"Sesi eligible hasil threshold 2023H2: {', '.join(selected_sessions)}."
            ),
            "Train": "2022",
            "Probability calibration": "2023H1",
            "Threshold calibration": "2023H2",
            "Model selection": "2024 saja",
            "Locked confirmation": "2025",
            "Historical reference": "2026H1, tidak menentukan pemenang",
            "Execution": (
                "Equity USD 1.000 | lot 0.01 | maksimal satu posisi | TP menuju "
                "midpoint | adaptive time stop 6/9/12 jam | biaya broker MT5."
            ),
            "Baseline lock": (
                "Baseline v1, BUY Specialist v4, SELL Specialist, Sideways v1, "
                "dan seluruh ledger paper live tidak diubah."
            ),
        },
        "data_audit": _extended_data_audit(data),
        "label_audit": _label_audit(opportunities),
        "outcome_model_selection": outcome_selection,
        "persistence_model_selection": persistence_model["selection"],
        "hazard_model_selection": hazard_model["selection"],
        "directional_thresholds": directional_audit,
        "session_audit": session_audit,
        "classification_selection": classification["selection"],
        "classification_locked": classification["locked"],
        "classification_reference": classification["reference"],
        "funnel": funnel,
        "development": development,
        "period_validation": periods,
        "historical_reference": reference,
        "folds": folds,
        "monte_carlo_summary": monte_carlo,
        "profit_concentration": concentration,
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
        "v1_reference": _v1_reference(v1_payload),
    }


def _augment_opportunities(data, frame, opportunities):
    augmented = opportunities.copy()
    confirmed = frame["range_confirmed"].fillna(False)
    group = (~confirmed).cumsum()
    range_age = confirmed.groupby(group).cumcount() / 60
    atr = frame["atr"].clip(lower=0.01)
    width = frame["range_high"] - frame["range_low"]
    midpoint_drift = (frame["range_mid"] - frame["range_mid"].shift(180)) / atr
    width_change = width / width.shift(180) - 1
    adx_change = frame["adx"] - frame["adx"].shift(180)
    atr_acceleration = atr / atr.shift(360).rolling(180, min_periods=60).median() - 1
    touch_imbalance = (
        (frame["touch_upper"] - frame["touch_lower"])
        / (frame["touch_upper"] + frame["touch_lower"]).replace(0, np.nan)
    )
    setup_times = pd.DatetimeIndex(pd.to_datetime(augmented["setup_time"]))
    mappings = {
        "range_age_hours": range_age,
        "midpoint_drift_atr": midpoint_drift,
        "range_width_change": width_change,
        "adx_change_3h": adx_change,
        "atr_acceleration": atr_acceleration,
        "touch_imbalance": touch_imbalance,
    }
    for column, series in mappings.items():
        augmented[column] = series.reindex(setup_times).to_numpy()
    hour = setup_times.hour + setup_times.minute / 60
    augmented["session_sin"] = np.sin(2 * np.pi * hour / 24)
    augmented["session_cos"] = np.cos(2 * np.pi * hour / 24)
    augmented["direction_code"] = np.where(
        augmented["direction"].eq("BUY"), 1.0, -1.0
    )
    augmented["session"] = pd.Series(
        [_session_name(timestamp) for timestamp in setup_times],
        index=augmented.index,
    )
    augmented = _attach_persistence_labels(data, augmented)
    return augmented.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[*PERSISTENCE_FEATURES, "persistence_12h", "adverse_breakout_6h"]
    )


def _attach_persistence_labels(data, opportunities):
    output = opportunities.copy()
    labels = []
    for timestamp, row in output.iterrows():
        location = data.index.searchsorted(timestamp, side="left")
        future_6h = data.iloc[location + 1 : location + 6 * 60 + 1]
        future_12h = data.iloc[location + 1 : location + 12 * 60 + 1]
        if len(future_12h) < 12 * 60:
            labels.append((np.nan, np.nan))
            continue
        width = float(row["range_high"] - row["range_low"])
        atr = width / max(float(row["range_width_atr"]), 0.01)
        upper = float(row["range_high"]) + 0.15 * atr
        lower = float(row["range_low"]) - 0.15 * atr
        breakout_up_12 = bool(future_12h["High"].max() > upper)
        breakout_down_12 = bool(future_12h["Low"].min() < lower)
        future_mid = (
            float(future_12h["High"].iloc[-60:].max())
            + float(future_12h["Low"].iloc[-60:].min())
        ) / 2
        midpoint_stable = abs(future_mid - float(row["range_mid"])) <= 0.50 * atr
        path_span = float(future_12h["High"].max() - future_12h["Low"].min())
        persistence = (
            not breakout_up_12
            and not breakout_down_12
            and midpoint_stable
            and path_span <= width + 0.50 * atr
        )
        if row["direction"] == "BUY":
            adverse_breakout = bool(future_6h["Low"].min() < lower)
        else:
            adverse_breakout = bool(future_6h["High"].max() > upper)
        labels.append((float(persistence), float(adverse_breakout)))
    output[["persistence_12h", "adverse_breakout_6h"]] = labels
    return output


def _train_binary_model(opportunities, target, name, seed):
    train = opportunities.loc[TRAIN_START:TRAIN_END]
    calibration = opportunities.loc[CALIBRATION_START:CALIBRATION_END]
    threshold_period = opportunities.loc[THRESHOLD_START:THRESHOLD_END]
    if len(train) < 50 or train[target].nunique() < 2:
        raise RuntimeError(f"Data train {name} tidak cukup.")
    logistic = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            C=0.30,
            random_state=seed,
        ),
    )
    boosting = HistGradientBoostingClassifier(
        learning_rate=0.035,
        max_iter=180,
        max_depth=3,
        min_samples_leaf=20,
        l2_regularization=2.0,
        random_state=seed + 1,
    )
    logistic.fit(train[list(PERSISTENCE_FEATURES)], train[target].astype(int))
    boosting.fit(
        train[list(PERSISTENCE_FEATURES)],
        train[target].astype(int),
        sample_weight=_class_weights(train[target]),
    )
    raw = (
        pd.Series(
            logistic.predict_proba(
                opportunities[list(PERSISTENCE_FEATURES)]
            )[:, 1],
            index=opportunities.index,
        )
        + pd.Series(
            boosting.predict_proba(
                opportunities[list(PERSISTENCE_FEATURES)]
            )[:, 1],
            index=opportunities.index,
        )
    ) / 2
    calibrator = _fit_platt(
        raw.loc[CALIBRATION_START:CALIBRATION_END],
        calibration[target].astype(int),
    )
    probability = _apply_symmetric_calibration(raw, calibrator)
    threshold, audit = _classification_threshold(
        threshold_period[target].astype(int),
        probability.loc[THRESHOLD_START:THRESHOLD_END],
    )
    return {
        "name": name,
        "target": target,
        "probability": probability,
        "threshold": threshold,
        "selection": pd.DataFrame(
            [
                {
                    "Model": name,
                    "Train observations": len(train),
                    "Calibration observations": len(calibration),
                    "Threshold observations": len(threshold_period),
                    "Threshold": threshold,
                    **audit,
                }
            ]
        ),
    }


def _classification_threshold(truth, probability):
    rows = []
    for quantile in (0.35, 0.45, 0.55, 0.65, 0.75):
        threshold = float(probability.quantile(quantile))
        prediction = probability.ge(threshold)
        precision = precision_score(truth, prediction, zero_division=0)
        recall = recall_score(truth, prediction, zero_division=0)
        balanced = balanced_accuracy_score(truth, prediction)
        count = int(prediction.sum())
        eligible = count >= 8
        score = 0.45 * precision + 0.35 * recall + 0.20 * balanced
        rows.append((threshold, precision, recall, balanced, count, score, eligible))
    eligible_rows = [row for row in rows if row[-1]]
    selected = max(eligible_rows or rows, key=lambda row: row[5])
    return selected[0], {
        "Precision threshold": selected[1],
        "Recall threshold": selected[2],
        "Balanced accuracy": selected[3],
        "Sinyal threshold": selected[4],
    }


def _directional_thresholds(opportunities, probability):
    thresholds = {}
    rows = []
    period = opportunities.loc[THRESHOLD_START:THRESHOLD_END]
    period_probability = probability.loc[THRESHOLD_START:THRESHOLD_END]
    for direction in ("BUY", "SELL"):
        mask = period["direction"].eq(direction)
        threshold, audit = _outcome_threshold(
            period.loc[mask, "target_12h"].astype(int),
            period_probability.loc[mask],
        )
        thresholds[direction] = threshold
        rows.append({"Arah": direction, "Threshold": threshold, **audit})
    return thresholds, pd.DataFrame(rows)


def _outcome_threshold(truth, probability):
    rows = []
    for quantile in (0.30, 0.40, 0.50, 0.60, 0.70):
        threshold = float(probability.quantile(quantile))
        prediction = probability.ge(threshold)
        selected = truth.loc[prediction]
        precision = float(selected.mean()) if len(selected) else 0.0
        recall = float(
            (prediction & truth.eq(1)).sum() / max(int(truth.eq(1).sum()), 1)
        )
        eligible = len(selected) >= 5
        score = precision * 20 - (1 - precision) * 10 + recall * 3
        rows.append((threshold, precision, recall, len(selected), score, eligible))
    eligible_rows = [row for row in rows if row[-1]]
    selected = max(eligible_rows or rows, key=lambda row: row[4])
    return selected[0], {
        "Precision": selected[1],
        "Recall": selected[2],
        "Selected": selected[3],
        "Expected value proxy": selected[1] * 20 - (1 - selected[1]) * 10,
    }


def _session_selection(opportunities):
    threshold_period = opportunities.loc[THRESHOLD_START:THRESHOLD_END]
    overall = float(threshold_period["target_12h"].mean())
    rows = []
    for session, frame in threshold_period.groupby("session"):
        rate = float(frame["target_12h"].mean())
        eligible = len(frame) >= 5 and rate >= max(overall, 0.40)
        rows.append(
            {
                "Sesi": session,
                "Opportunity": len(frame),
                "TP-before-SL (%)": rate * 100,
                "Eligible": eligible,
            }
        )
    audit = pd.DataFrame(rows).sort_values("Sesi").reset_index(drop=True)
    selected = audit.loc[audit["Eligible"], "Sesi"].tolist()
    if not selected:
        selected = [str(audit.sort_values(
            ["TP-before-SL (%)", "Opportunity"], ascending=False
        ).iloc[0]["Sesi"])]
    return selected, audit


def _session_name(timestamp):
    hour = pd.Timestamp(timestamp).hour
    if hour < 7:
        return "Asia"
    if hour < 13:
        return "London"
    if hour < 21:
        return "New York"
    return "Rollover"


def _candidate_signals(
    opportunities,
    outcome_model,
    persistence_model,
    hazard_model,
    directional_thresholds,
    selected_sessions,
    best,
):
    outcome = outcome_model["probability"]
    persistence = persistence_model["probability"]
    hazard = hazard_model["probability"]
    v1_selected = outcome.ge(outcome_model["threshold"])
    v1_control = (
        v1_selected
        & opportunities["breakout_guard"]
        & (
            opportunities["strong_rejection"]
            | opportunities["reward_risk"].ge(1.20)
        )
    )
    directional = pd.Series(False, index=opportunities.index)
    for direction, threshold in directional_thresholds.items():
        directional |= opportunities["direction"].eq(direction) & outcome.ge(threshold)
    persistence_gate = persistence.ge(persistence_model["threshold"])
    hazard_gate = hazard.lt(hazard_model["threshold"])
    session_gate = opportunities["session"].isin(selected_sessions)
    masks = {
        "Sideways v1 Control": v1_control,
        "Persistence Gate": v1_control & persistence_gate,
        "Breakout Hazard Gate": v1_control & hazard_gate,
        "Directional Calibration": (
            directional & opportunities["breakout_guard"]
        ),
        "Session-Aware Persistence": (
            directional & persistence_gate & hazard_gate & session_gate
        ),
        "Adaptive Persistence Ensemble": (
            directional
            & persistence_gate
            & hazard_gate
            & session_gate
            & (
                opportunities["strong_rejection"]
                | opportunities["reward_risk"].ge(1.10)
            )
        ),
    }
    output = {}
    rows = []
    for candidate, mask in masks.items():
        selected = opportunities.loc[mask].copy()
        signals = _opportunities_to_signals(selected, best, candidate)
        if candidate == "Sideways v1 Control" and not signals.empty:
            source = selected.loc[
                ~selected.index.duplicated(keep="first")
            ].reindex(signals.index)
            high_quality = (
                source["reward_risk"].ge(1.30)
                & source["atr_percentile"].le(0.65)
            )
            signals.loc[high_quality, "tp_usd"] = np.minimum(
                signals.loc[high_quality, "tp_usd"] * 1.15, 17.0
            )
        if candidate == "Adaptive Persistence Ensemble" and not signals.empty:
            source = selected.loc[
                ~selected.index.duplicated(keep="first")
            ].reindex(signals.index)
            source_persistence = persistence.loc[
                ~persistence.index.duplicated(keep="first")
            ].reindex(signals.index)
            source_hazard = hazard.loc[
                ~hazard.index.duplicated(keep="first")
            ].reindex(signals.index)
            signals["time_stop_hours"] = np.select(
                [
                    source_hazard.ge(hazard_model["threshold"] * 0.75),
                    source_persistence.ge(
                        max(persistence_model["threshold"] * 1.25, 0.65)
                    ),
                ],
                [6.0, 12.0],
                default=9.0,
            )
            cautious = source["reward_risk"].lt(1.10)
            signals.loc[cautious, "tp_usd"] = np.maximum(
                signals.loc[cautious, "tp_usd"] * 0.80, 5.0
            )
        output[candidate] = _unique_signals(signals)
        unique = selected.loc[~selected.index.duplicated(keep="first")]
        rows.append(
            {
                "Kandidat": candidate,
                "Opportunity": len(opportunities),
                "Lolos": len(output[candidate]),
                "BUY": int(unique["direction"].eq("BUY").sum()),
                "SELL": int(unique["direction"].eq("SELL").sum()),
            }
        )
    return output, pd.DataFrame(rows)


def _simulate_all(data, signals, best, config, start, end):
    return {
        candidate: _simulate_risk_control(
            data, signal_frame.loc[start:end], best, config
        )
        for candidate, signal_frame in signals.items()
    }


def _result_table(results, signals, start, end):
    return pd.DataFrame(
        [
            {
                "Kandidat": candidate,
                "Sinyal tersedia": len(signals[candidate].loc[start:end]),
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
                    "Sinyal tersedia": len(signals[candidate].loc[start:end]),
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


def _classification_tables(
    opportunities, outcome_model, persistence_model, hazard_model
):
    output = {}
    for key, start, end in (
        ("selection", SELECTION_START, SELECTION_END),
        ("locked", LOCKED_START, LOCKED_END),
        ("reference", REFERENCE_START, REFERENCE_END),
    ):
        frame = opportunities.loc[start:end]
        rows = []
        for name, model, target in (
            ("Outcome", outcome_model, "target_12h"),
            ("Persistence", persistence_model, "persistence_12h"),
            ("Breakout Hazard", hazard_model, "adverse_breakout_6h"),
        ):
            probability = model["probability"].loc[start:end]
            prediction = probability.ge(model["threshold"])
            truth = frame[target].astype(int)
            rows.append(
                {
                    "Model": name,
                    "Observasi": len(frame),
                    "Threshold": model["threshold"],
                    "Base rate (%)": float(truth.mean() * 100),
                    "Precision": precision_score(
                        truth, prediction, zero_division=0
                    ),
                    "Recall": recall_score(truth, prediction, zero_division=0),
                    "Balanced accuracy": balanced_accuracy_score(
                        truth, prediction
                    ),
                    "Coverage (%)": float(prediction.mean() * 100),
                    "Brier": float(brier_score_loss(truth, probability)),
                }
            )
        output[key] = pd.DataFrame(rows)
    return output


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


def _selection_ranking(development, reference, periods, classification, decisions):
    dev = development.set_index("Kandidat")
    ref = reference.set_index("Kandidat")
    period = periods.set_index(["Periode", "Kandidat"])
    decision = decisions.set_index("Kandidat")
    persistence_precision = float(
        classification.loc[
            classification["Model"].eq("Persistence"), "Precision"
        ].iloc[0]
    )
    hazard_precision = float(
        classification.loc[
            classification["Model"].eq("Breakout Hazard"), "Precision"
        ].iloc[0]
    )
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
                "Persistence precision locked": persistence_precision,
                "Hazard precision locked": hazard_precision,
                "Growth 2026H1 (%)": float(ref.loc[candidate, "Growth (%)"]),
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


def _label_audit(opportunities):
    rows = []
    for label, start, end in _periods():
        frame = opportunities.loc[start:end]
        rows.append(
            {
                "Periode": label,
                "Opportunity": len(frame),
                "Outcome TP-first (%)": float(frame["target_12h"].mean() * 100),
                "Range persistent (%)": float(
                    frame["persistence_12h"].mean() * 100
                ),
                "Adverse breakout (%)": float(
                    frame["adverse_breakout_6h"].mean() * 100
                ),
                "BUY": int(frame["direction"].eq("BUY").sum()),
                "SELL": int(frame["direction"].eq("SELL").sum()),
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


def _v1_reference(payload):
    if not payload:
        return pd.DataFrame()
    ranking = payload.get("ranking")
    if not isinstance(ranking, pd.DataFrame) or ranking.empty:
        return pd.DataFrame()
    return ranking.head(3).copy()
