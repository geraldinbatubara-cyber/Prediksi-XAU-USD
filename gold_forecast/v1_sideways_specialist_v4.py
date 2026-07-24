from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from gold_forecast.exact_broker_oos import (
    SLIPPAGE_POINTS,
    _prepare_m1,
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
    _opportunities_to_signals,
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
    STATE_FEATURES,
    _build_position_states,
    _enrich_signals,
    _simulate_dynamic,
)
from gold_forecast.v1_trend_strength_stability import _extended_data_audit
from gold_forecast.v1_unified_benchmark import _unified_best


CANDIDATES = (
    "Breakout Hazard v2 Control",
    "Structural + Path Confirmation",
    "Event Hazard Confirmed",
    "Selective Protection",
    "Recovery Veto Protection",
    "Two-Layer Protection",
)
EVENT_FEATURES = (
    *STATE_FEATURES,
    "structural_score",
    "path_pressure",
    "recovery_score",
    "floating_change",
    "midpoint_progress",
    "mae_change",
    "episode_pressure",
    "episode_breakout",
    "episode_recovery",
)


def run_v1_sideways_specialist_v4_lab(
    gold_m1: pd.DataFrame,
    frozen_payload: dict[str, object],
    v3_payload: dict[str, object] | None = None,
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
        opportunities, "persistence_12h", "Range Persistence", 141
    )
    entry_hazard_model = _train_binary_model(
        opportunities, "adverse_breakout_6h", "Adverse Breakout Hazard", 151
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
    unique_opportunities = opportunities.loc[
        ~opportunities.index.duplicated(keep="first")
    ]
    entry_source = unique_opportunities.reindex(entry_signals.index)
    entry_signals = _enrich_signals(entry_signals, entry_source)

    counterfactual_signals = _opportunities_to_signals(
        unique_opportunities, best, "Counterfactual Episode"
    )
    counterfactual_signals = _enrich_signals(
        counterfactual_signals,
        unique_opportunities.reindex(counterfactual_signals.index),
    )
    all_states = _build_position_states(
        data, range_frame, counterfactual_signals
    )
    all_states, episodes = _eventize_states(all_states)
    event_model = _train_event_model(episodes, all_states)
    all_states["event_hazard"] = event_model["probability"]
    actual_entries = all_states.index.get_level_values("entry_time").isin(
        entry_signals.index
    )
    actual_states = all_states.loc[actual_entries].copy()
    actual_states["hazard_1h"] = actual_states["event_hazard"]
    actual_states["hazard_3h"] = actual_states["event_hazard"]

    development_results = _simulate_candidates(
        data.loc[DEVELOPMENT_START:DEVELOPMENT_END],
        entry_signals.loc[DEVELOPMENT_START:DEVELOPMENT_END],
        actual_states,
        event_model["threshold"],
    )
    reference_results = _simulate_candidates(
        data.loc[REFERENCE_START:REFERENCE_END],
        entry_signals.loc[REFERENCE_START:REFERENCE_END],
        actual_states,
        event_model["threshold"],
    )
    development = _result_table(
        development_results,
        entry_signals,
        DEVELOPMENT_START,
        DEVELOPMENT_END,
    )
    reference = _result_table(
        reference_results,
        entry_signals,
        REFERENCE_START,
        REFERENCE_END,
    )
    periods = _period_validation(development_results, entry_signals)
    folds = _fold_evaluation(development_results)
    monte_carlo = _monte_carlo_summary(development_results)
    concentration = _profit_concentration(development_results)
    attribution = _exit_attribution(development_results)
    classification = _classification_tables(all_states, event_model)
    decisions = _decision_table(
        development,
        reference,
        periods,
        folds,
        monte_carlo,
        concentration,
    )
    ranking = _selection_ranking(
        development, reference, periods, decisions, attribution
    )
    eligible = ranking.loc[ranking["Selection eligible"]]
    winner = str(eligible.iloc[0]["Kandidat"]) if not eligible.empty else ""
    stress = (
        _stress_test(
            data.loc[DEVELOPMENT_START:DEVELOPMENT_END],
            entry_signals.loc[DEVELOPMENT_START:DEVELOPMENT_END],
            actual_states,
            winner,
            event_model["threshold"],
        )
        if winner
        else pd.DataFrame()
    )
    stress_passed = int((stress["Growth (%)"] > 0).sum()) if not stress.empty else 0
    decisions = decisions.copy()
    decisions["Stress profitable"] = decisions["Kandidat"].map(
        {winner: stress_passed}
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
            "Name": (
                "v1 Sideways Specialist Lab v4 - Event-Based Hazard & "
                "Selective Protection"
            ),
            "Control": (
                "Entry Breakout Hazard Gate v2 identik. Hanya kebijakan exit "
                "yang diuji."
            ),
            "Counterfactual dataset": (
                "Seluruh opportunity range yang unik disimulasikan untuk "
                "membangun episode; backtest ekonomi tetap memakai entry v2."
            ),
            "Episodes": (
                "State M15 dikelompokkan menjadi HEALTHY, PRESSURE, BREAKOUT, "
                "dan RECOVERY berdasarkan transisi informasi yang tersedia saat itu."
            ),
            "Hazard target": (
                "Adverse boundary/SL sebelum TP dalam tiga jam setelah episode."
            ),
            "Recovery veto": (
                "Early exit diblokir saat floating, jarak midpoint, dan tekanan "
                "path menunjukkan pemulihan."
            ),
            "Two layers": (
                "Protection di-arm saat risiko awal meningkat; close hanya saat "
                "structural dan price-path hazard menjadi emergency."
            ),
            "Train": "Episode 2022",
            "Calibration": "Episode 2023H1",
            "Threshold": "Episode 2023H2",
            "Selection": "Trade entry 2024",
            "Locked": "Trade entry 2025",
            "Reference": "Trade entry 2026H1; wajib positif untuk lulus",
            "Execution": (
                "M1 broker-aware, evaluasi M15, TP/SL intrabar lebih dulu, lot "
                "0.01, spread/slippage/swap BUY dihitung."
            ),
            "Baseline lock": (
                "Semua strategi dan ledger Paper Live Trading tidak diubah."
            ),
        },
        "data_audit": _extended_data_audit(data),
        "episode_audit": _episode_audit(episodes),
        "event_model_selection": event_model["selection"],
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
        "v3_reference": _v3_reference(v3_payload),
    }


def _eventize_states(states: pd.DataFrame):
    frame = states.copy()
    groups = frame.groupby(level="entry_time", sort=False)
    previous_float = groups["floating_pl"].shift(1)
    previous_midpoint = groups["midpoint_distance_atr"].shift(1)
    previous_mae = groups["max_adverse"].shift(1)
    frame["floating_change"] = (
        frame["floating_pl"] - previous_float
    ).fillna(0.0)
    frame["midpoint_progress"] = (
        previous_midpoint.abs() - frame["midpoint_distance_atr"].abs()
    ).fillna(0.0)
    frame["mae_change"] = (
        frame["max_adverse"] - previous_mae
    ).fillna(0.0).clip(lower=0.0)
    adverse_distance = (
        -frame["direction_code"] * frame["midpoint_distance_atr"]
    )
    frame["structural_score"] = pd.concat(
        [
            adverse_distance.gt(0.50),
            frame["adx_change_3h"].gt(2.0),
            frame["atr_acceleration"].gt(0.12),
            frame["range_width_change"].gt(0.12),
        ],
        axis=1,
    ).mean(axis=1)
    reward_giveback = frame["peak_profit"] - frame["floating_pl"]
    distance_total = (frame["distance_tp"] + frame["distance_sl"]).clip(lower=0.01)
    frame["path_pressure"] = pd.concat(
        [
            frame["floating_pl"].lt(-2.0),
            frame["mae_change"].gt(0.50),
            reward_giveback.gt(5.0),
            frame["distance_sl"].div(distance_total).lt(0.35),
        ],
        axis=1,
    ).mean(axis=1)
    frame["recovery_score"] = pd.concat(
        [
            frame["floating_change"].gt(0.40),
            frame["midpoint_progress"].gt(0.05),
            frame["mae_change"].le(0.05),
        ],
        axis=1,
    ).mean(axis=1)
    frame["episode"] = np.select(
        [
            frame["recovery_score"].ge(2 / 3),
            frame["structural_score"].ge(0.50)
            & frame["path_pressure"].ge(0.50),
            frame["structural_score"].ge(0.25)
            | frame["path_pressure"].ge(0.50),
        ],
        ["RECOVERY", "BREAKOUT", "PRESSURE"],
        default="HEALTHY",
    )
    frame["episode_pressure"] = frame["episode"].eq("PRESSURE").astype(float)
    frame["episode_breakout"] = frame["episode"].eq("BREAKOUT").astype(float)
    frame["episode_recovery"] = frame["episode"].eq("RECOVERY").astype(float)
    episode_change = groups["episode"].shift(1).ne(frame["episode"])
    age_bucket = np.floor(frame["age_hours"] / 2)
    bucket_change = age_bucket.ne(age_bucket.groupby(level="entry_time").shift(1))
    episodes = frame.loc[episode_change | bucket_change].copy()
    episodes["target_event_3h"] = episodes["adverse_before_tp_3h"].astype(float)
    return frame, episodes


def _train_event_model(episodes: pd.DataFrame, all_states: pd.DataFrame):
    entry_dates = episodes.index.get_level_values("entry_time")
    train = episodes.loc[(entry_dates >= TRAIN_START) & (entry_dates <= TRAIN_END)]
    calibration = episodes.loc[
        (entry_dates >= CALIBRATION_START) & (entry_dates <= CALIBRATION_END)
    ]
    threshold_period = episodes.loc[
        (entry_dates >= THRESHOLD_START) & (entry_dates <= THRESHOLD_END)
    ]
    target = "target_event_3h"
    if len(train) < 100 or train[target].nunique() < 2:
        raise RuntimeError("Episode train tidak cukup untuk event hazard model.")
    logistic = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            C=0.20,
            random_state=161,
        ),
    )
    boosting = HistGradientBoostingClassifier(
        learning_rate=0.03,
        max_iter=180,
        max_depth=3,
        min_samples_leaf=20,
        l2_regularization=3.0,
        random_state=162,
    )
    logistic.fit(train[list(EVENT_FEATURES)], train[target].astype(int))
    boosting.fit(
        train[list(EVENT_FEATURES)],
        train[target].astype(int),
        sample_weight=_class_weights(train[target]),
    )

    def raw_probability(frame):
        return (
            logistic.predict_proba(frame[list(EVENT_FEATURES)])[:, 1]
            + boosting.predict_proba(frame[list(EVENT_FEATURES)])[:, 1]
        ) / 2

    raw_events = pd.Series(raw_probability(episodes), index=episodes.index)
    calibrator = _fit_platt(
        raw_events.loc[calibration.index], calibration[target].astype(int)
    )
    raw_states = pd.Series(raw_probability(all_states), index=all_states.index)
    probability = _apply_symmetric_calibration(raw_states, calibrator)
    threshold_probability = _apply_symmetric_calibration(
        raw_events.loc[threshold_period.index], calibrator
    )
    threshold, audit = _event_threshold(
        threshold_period[target].astype(int), threshold_probability
    )
    return {
        "target": "adverse_before_tp_3h",
        "probability": probability,
        "threshold": threshold,
        "selection": pd.DataFrame(
            [{
                "Model": "Event hazard 3h",
                "Train episodes": len(train),
                "Calibration episodes": len(calibration),
                "Threshold episodes": len(threshold_period),
                "Threshold": threshold,
                **audit,
            }]
        ),
    }


def _event_threshold(truth: pd.Series, probability: pd.Series):
    if truth.nunique() < 2:
        threshold = float(probability.quantile(0.85))
        return threshold, {
            "Threshold status": "Fallback Q85: threshold period single-class",
            "Observed adverse": int(truth.sum()),
            "Precision": np.nan,
            "Recall": np.nan,
            "Balanced accuracy": np.nan,
            "Predicted high hazard": int(probability.ge(threshold).sum()),
        }
    rows = []
    for quantile in (0.50, 0.60, 0.70, 0.80, 0.90):
        threshold = float(probability.quantile(quantile))
        prediction = probability.ge(threshold)
        precision = precision_score(truth, prediction, zero_division=0)
        recall = recall_score(truth, prediction, zero_division=0)
        balanced = balanced_accuracy_score(truth, prediction)
        count = int(prediction.sum())
        eligible = count >= 12
        score = 0.50 * precision + 0.25 * recall + 0.25 * balanced
        rows.append((threshold, precision, recall, balanced, count, score, eligible))
    selected = max(
        [row for row in rows if row[-1]] or rows,
        key=lambda row: row[5],
    )
    return selected[0], {
        "Threshold status": "Validated pada episode 2023H2",
        "Observed adverse": int(truth.sum()),
        "Precision": selected[1],
        "Recall": selected[2],
        "Balanced accuracy": selected[3],
        "Predicted high hazard": selected[4],
    }


def _simulate_candidates(data, signals, states, threshold):
    return {
        candidate: _simulate_dynamic(
            data, signals, states, candidate, threshold, threshold
        )
        for candidate in CANDIDATES
    }


def _result_table(results, signals, start, end):
    return pd.DataFrame(
        [{
            "Kandidat": candidate,
            "Sinyal tersedia": len(signals.loc[start:end]),
            **_metric_values(result),
        } for candidate, result in results.items()]
    )


def _period_validation(results, signals):
    rows = []
    for label, start, end in _periods()[:-1]:
        for candidate, result in results.items():
            rows.append({
                "Periode": label,
                "Kandidat": candidate,
                "Sinyal tersedia": len(signals.loc[start:end]),
                **_ledger_metric_values(
                    _trades_in_period(result.trades, start, end)
                ),
            })
    return pd.DataFrame(rows)


def _fold_evaluation(results):
    rows = []
    for fold in FOLDS:
        for candidate, result in results.items():
            metrics = _ledger_metric_values(
                _trades_in_period(result.trades, fold.test_start, fold.test_end)
            )
            rows.append({
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
            })
    return pd.DataFrame(rows)


def _classification_tables(states, model):
    output = {}
    entry_dates = states.index.get_level_values("entry_time")
    for key, start, end in (
        ("selection", SELECTION_START, SELECTION_END),
        ("locked", LOCKED_START, LOCKED_END),
        ("reference", REFERENCE_START, REFERENCE_END),
    ):
        frame = states.loc[(entry_dates >= start) & (entry_dates <= end)]
        truth = frame[model["target"]].astype(int)
        probability = model["probability"].loc[frame.index]
        prediction = probability.ge(model["threshold"])
        valid = truth.nunique() >= 2
        output[key] = pd.DataFrame([{
            "Model": "Event hazard 3h",
            "States": len(frame),
            "Unique positions": frame.index.get_level_values(
                "entry_time"
            ).nunique(),
            "Base hazard (%)": float(truth.mean() * 100),
            "Threshold": model["threshold"],
            "Precision": (
                precision_score(truth, prediction, zero_division=0)
                if valid else np.nan
            ),
            "Recall": (
                recall_score(truth, prediction, zero_division=0)
                if valid else np.nan
            ),
            "Balanced accuracy": (
                balanced_accuracy_score(truth, prediction)
                if valid else np.nan
            ),
            "Coverage (%)": float(prediction.mean() * 100),
            "Validation status": "Valid" if valid else "Single-class; metric N/A",
        }])
    return output


def _exit_attribution(results):
    control = results["Breakout Hazard v2 Control"].trades
    control_map = (
        control.set_index("Tanggal entry")["Net P/L"]
        if not control.empty else pd.Series(dtype=float)
    )
    rows = []
    for candidate, result in results.items():
        if candidate == "Breakout Hazard v2 Control":
            continue
        common = result.trades.loc[
            result.trades["Tanggal entry"].isin(control_map.index)
        ].copy()
        baseline = common["Tanggal entry"].map(control_map)
        delta = common["Net P/L"].to_numpy() - baseline.to_numpy()
        dynamic = common["Alasan exit"].str.contains(
            "structural|event|protection", case=False, regex=True
        )
        rows.append({
            "Kandidat": candidate,
            "Common entry": len(common),
            "Dynamic exits": int(dynamic.sum()),
            "Saved loss": float(pd.Series(delta)[pd.Series(delta) > 0].sum()),
            "Sacrificed profit": float(
                -pd.Series(delta)[pd.Series(delta) < 0].sum()
            ),
            "Net exit benefit": float(np.sum(delta)),
        })
    return pd.DataFrame(rows)


def _decision_table(
    development, reference, periods, folds, monte_carlo, concentration
):
    dev = development.set_index("Kandidat")
    ref = reference.set_index("Kandidat")
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
            "2026H1 positif": float(ref.loc[candidate, "Growth (%)"]) > 0,
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
        rows.append({
            "Kandidat": candidate,
            **criteria,
            "Primary fold profitable": int(primary["Profitable"].sum()),
            "Kriteria lolos": int(sum(criteria.values())),
            "Total kriteria": len(criteria),
            "Lulus": bool(all(criteria.values())),
        })
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
        rows.append({
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
                if candidate in attr.index else 0.0
            ),
            "Kriteria lolos": int(decision.loc[candidate, "Kriteria lolos"]),
            "Lulus": bool(decision.loc[candidate, "Lulus"]),
            "Lulus termasuk stress": False,
        })
    ranking = pd.DataFrame(rows).sort_values(
        ["Selection score 2024", "DD selection 2024 (%)"],
        ascending=[False, True],
    ).reset_index(drop=True)
    ranking.insert(0, "Peringkat", np.arange(1, len(ranking) + 1))
    return ranking


def _stress_test(data, signals, states, candidate, threshold):
    rows = []
    for spread_multiplier in (1.0, 1.25, 1.50):
        for slippage in (2.0, 4.0, 6.0):
            result = _simulate_dynamic(
                data,
                signals,
                states,
                candidate,
                threshold,
                threshold,
                spread_multiplier=spread_multiplier,
                slippage_points=slippage,
            )
            rows.append({
                "Kandidat": candidate,
                "Spread multiplier": spread_multiplier,
                "Slippage points": slippage,
                **_metric_values(result),
            })
    return pd.DataFrame(rows)


def _episode_audit(episodes):
    entry_dates = episodes.index.get_level_values("entry_time")
    rows = []
    for label, start, end in _periods():
        frame = episodes.loc[(entry_dates >= start) & (entry_dates <= end)]
        rows.append({
            "Periode": label,
            "Episodes": len(frame),
            "Unique positions": frame.index.get_level_values(
                "entry_time"
            ).nunique(),
            "Healthy": int(frame["episode"].eq("HEALTHY").sum()),
            "Pressure": int(frame["episode"].eq("PRESSURE").sum()),
            "Breakout": int(frame["episode"].eq("BREAKOUT").sum()),
            "Recovery": int(frame["episode"].eq("RECOVERY").sum()),
            "Adverse 3h (%)": float(frame["target_event_3h"].mean() * 100),
        })
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


def _v3_reference(payload):
    if not payload:
        return pd.DataFrame()
    ranking = payload.get("ranking")
    if not isinstance(ranking, pd.DataFrame) or ranking.empty:
        return pd.DataFrame()
    return ranking.head(3).copy()
