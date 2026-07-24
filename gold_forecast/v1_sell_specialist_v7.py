from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from gold_forecast.exact_broker_oos import _prepare_m1
from gold_forecast.v1_directional_specialization import (
    SYMMETRIC_FEATURES,
    _apply_symmetric_calibration,
    _class_weights,
    _ledger_metric_values,
    _monte_carlo_summary,
    _stress_summary,
    _trades_in_period,
)
from gold_forecast.v1_entry_quality_path import FOLDS, _unique_signals
from gold_forecast.v1_regime_classifier import _ohlc_bars
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
    _predict,
    _raw_sell_signals,
    _sell_outcome_frame,
)
from gold_forecast.v1_sell_specialist_v6 import (
    _profit_concentration,
    _regime_and_setup_frame,
    _timed_signals,
)
from gold_forecast.v1_trend_strength_stability import _extended_data_audit
from gold_forecast.v1_unified_benchmark import _unified_best


CANDIDATES = (
    "Bear Event Control",
    "Breakdown Acceptance 5m",
    "Breakdown Acceptance 15m",
    "Failed Recovery 5m",
    "Failed Recovery 15m",
    "Adaptive Event Ensemble",
)
EVENT_FEATURES = (
    "event_age_hours",
    "initial_impulse_atr",
    "cumulative_drop_atr",
    "distance_support_atr",
    "rebound_atr",
    "body_atr",
    "volume_z",
)
MODEL_FEATURES = (*SYMMETRIC_FEATURES, *EVENT_FEATURES)


def run_v1_sell_specialist_v7_lab(
    gold_m1: pd.DataFrame,
    frozen_payload: dict[str, object],
    v6_payload: dict[str, object] | None = None,
) -> dict[str, object]:
    data = _prepare_m1(gold_m1)
    _, leaderboard, _ = frozen_payload["v1"]
    best = {
        **_unified_best(leaderboard.iloc[0].to_dict()),
        "Close-all target equity": False,
        "Max BUY": 0,
        "Max SELL": 1,
    }
    config = RiskControlConfig(
        "SELL Specialist v7",
        "Bear event and exhaustion timing",
        max_total_positions=1,
        max_same_direction=1,
    )
    spread_limit = float(
        data.loc[DEVELOPMENT_START:DEVELOPMENT_END, "SpreadPoints"].quantile(0.90)
    )
    base = _regime_and_setup_frame(data, _sell_outcome_frame(data))
    events, opportunities = _bear_event_dataset(data, base)
    model_runs, model_selection = _train_event_models(opportunities)
    signals, funnel = _candidate_signals(
        data,
        opportunities,
        model_runs,
        best,
        spread_limit,
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
    classification = _classification_tables(opportunities, model_runs)
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
        winner_row = decisions.loc[decisions["Kandidat"].eq(winner)].iloc[0]
        winner_passed = bool(winner_row["Lulus"]) and stress_passed >= 4
        ranking.loc[
            ranking["Kandidat"].eq(winner), "Lulus termasuk stress"
        ] = winner_passed

    return {
        "methodology": {
            "Name": "v1 SELL Specialist Lab v7 - Bear Event & Exhaustion Timing",
            "Mandat": "SELL atau ABSTAIN; tidak pernah membuka BUY.",
            "Event detector": (
                "Satu episode dimulai oleh breakdown support atau downside shock "
                "dalam hard bear regime. Event berakhir ketika recovery terkonfirmasi "
                "atau setelah 36 jam; candle dalam event tidak menjadi event baru."
            ),
            "Phase": (
                "Initiation -> Continuation -> Exhaustion/Recovery. Entry hanya "
                "diizinkan sebelum cumulative drop mencapai 2.5 ATR."
            ),
            "Setup": "Breakdown Acceptance dan Failed Recovery dipelajari terpisah.",
            "Label": (
                "Path-aware TP USD 20 sebelum SL USD 10 dalam 24 jam, ditambah "
                "MFE, MAE, dan time-to-TP untuk audit."
            ),
            "Train": "2022",
            "Probability calibration": "2023H1",
            "Threshold calibration": "2023H2",
            "Model selection": "2024 saja",
            "Locked confirmation": "2025",
            "Historical reference": "2026H1, tidak menentukan pemenang",
            "Execution": (
                "Equity USD 1.000 | lot 0.01 | maksimal 1 SELL | delay 5m/15m | "
                "TP 15/20/25 | SL 10 | time stop maksimal 24 jam."
            ),
            "Baseline lock": (
                "Baseline v1, BUY Specialist v4, dan ledger paper live tidak dibaca "
                "atau ditulis oleh eksperimen."
            ),
        },
        "data_audit": _extended_data_audit(data),
        "event_audit": _event_audit(events, opportunities),
        "phase_audit": _phase_audit(events),
        "model_selection": model_selection,
        "classification_selection": classification["selection"],
        "classification_locked": classification["locked"],
        "classification_reference": classification["reference"],
        "path_audit": _path_audit(opportunities),
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
        "v6_reference": _v6_reference(v6_payload),
    }


def _bear_event_dataset(data, frame):
    h1 = _ohlc_bars(data, "1h").reindex(frame.index)
    close = h1["Close"]
    atr = frame["raw_atr"].clip(lower=0.01)
    fast = close.ewm(span=10, adjust=False).mean()
    support = h1["Low"].rolling(20, min_periods=10).min().shift(1)
    body_atr = (h1["Open"] - close).clip(lower=0) / atr
    return_3_atr = (close - close.shift(3)) / atr
    volume_source = (
        h1["Volume"]
        if "Volume" in h1
        else pd.Series(0.0, index=h1.index)
    )
    volume = pd.to_numeric(volume_source, errors="coerce").fillna(0)
    volume_mean = volume.rolling(48, min_periods=12).mean()
    volume_std = volume.rolling(48, min_periods=12).std().replace(0, np.nan)
    volume_z = ((volume - volume_mean) / volume_std).fillna(0).clip(-5, 5)
    initiation = (
        frame["bear_regime"]
        & (
            (
                close.lt(support)
                & body_atr.ge(0.25)
                & frame["raw_momentum_6"].lt(0)
            )
            | (
                return_3_atr.le(-0.80)
                & body_atr.ge(0.35)
                & frame["h4_bear_evidence"].ge(2)
            )
        )
    )

    event_rows = []
    opportunity_rows = []
    index = frame.index
    position = 0
    event_id = 0
    while position < len(index):
        timestamp = index[position]
        if not bool(initiation.iloc[position]):
            position += 1
            continue
        event_id += 1
        start_position = position
        start_close = float(close.iloc[position])
        start_atr = float(atr.iloc[position])
        fixed_support = float(support.iloc[position])
        initial_impulse = max(
            0.0,
            float((close.shift(3).iloc[position] - close.iloc[position]) / start_atr),
        )
        end_position = min(position + 36, len(index) - 1)
        recovery_position = None
        for cursor in range(position + 3, end_position + 1):
            recovered = (
                close.iloc[cursor] > fast.iloc[cursor]
                and close.pct_change(3).iloc[cursor] > 0
            )
            if recovered:
                recovery_position = cursor
                end_position = cursor
                break

        event_slice = h1.iloc[start_position : end_position + 1]
        event_low = event_slice["Low"].cummin()
        cumulative_drop = (start_close - event_slice["Close"]) / start_atr
        for offset, event_time in enumerate(event_slice.index):
            cursor = start_position + offset
            drop = float(cumulative_drop.iloc[offset])
            phase = "Initiation"
            if recovery_position is not None and cursor == recovery_position:
                phase = "Recovery"
            elif drop >= 2.5:
                phase = "Exhaustion"
            elif offset >= 1:
                phase = "Continuation"
            event_rows.append(
                {
                    "timestamp": event_time,
                    "event_id": event_id,
                    "phase": phase,
                    "event_age_hours": float(offset),
                    "cumulative_drop_atr": drop,
                }
            )

        acceptance = None
        for cursor in range(position + 1, min(position + 5, end_position + 1)):
            drop = (start_close - float(close.iloc[cursor])) / start_atr
            if (
                close.iloc[cursor] < fixed_support - 0.05 * start_atr
                and h1["High"].iloc[cursor] <= fixed_support + 0.25 * start_atr
                and drop < 2.5
            ):
                acceptance = cursor
                break
        if acceptance is not None:
            opportunity_rows.append(
                _opportunity_row(
                    "Breakdown Acceptance",
                    event_id,
                    acceptance,
                    position,
                    frame,
                    h1,
                    support,
                    volume_z,
                    start_close,
                    start_atr,
                    initial_impulse,
                    0.0,
                )
            )

        failed_recovery = None
        rebound_atr = 0.0
        running_low = float(h1["Low"].iloc[position])
        for cursor in range(position + 2, min(position + 13, end_position + 1)):
            running_low = min(running_low, float(h1["Low"].iloc[cursor - 1]))
            rebound = (float(h1["High"].iloc[cursor]) - running_low) / start_atr
            drop = (start_close - float(close.iloc[cursor])) / start_atr
            bearish_rejection = (
                close.iloc[cursor] < h1["Open"].iloc[cursor]
                and close.iloc[cursor] < fast.iloc[cursor]
                and close.iloc[cursor] < close.iloc[cursor - 1]
            )
            if rebound >= 0.30 and bearish_rejection and drop < 2.5:
                failed_recovery = cursor
                rebound_atr = rebound
                break
        if failed_recovery is not None:
            opportunity_rows.append(
                _opportunity_row(
                    "Failed Recovery",
                    event_id,
                    failed_recovery,
                    position,
                    frame,
                    h1,
                    support,
                    volume_z,
                    start_close,
                    start_atr,
                    initial_impulse,
                    rebound_atr,
                )
            )
        position = end_position + 1

    events = pd.DataFrame(event_rows)
    if events.empty:
        raise RuntimeError("Bear event detector tidak menghasilkan event.")
    events = events.set_index("timestamp").sort_index()
    opportunities = pd.DataFrame(opportunity_rows)
    if opportunities.empty:
        raise RuntimeError("Bear event detector tidak menghasilkan setup entry.")
    opportunities = opportunities.set_index("timestamp").sort_index()
    opportunities = _attach_path_labels(data, opportunities)
    return events, opportunities.dropna(subset=["target_24h", *MODEL_FEATURES])


def _opportunity_row(
    setup,
    event_id,
    cursor,
    start_position,
    frame,
    h1,
    support,
    volume_z,
    start_close,
    start_atr,
    initial_impulse,
    rebound_atr,
):
    timestamp = frame.index[cursor]
    row = frame.iloc[cursor]
    close = float(h1["Close"].iloc[cursor])
    output = {feature: float(row[feature]) for feature in SYMMETRIC_FEATURES}
    output.update(
        {
            "timestamp": timestamp,
            "setup": setup,
            "event_id": event_id,
            "event_age_hours": float(cursor - start_position),
            "initial_impulse_atr": initial_impulse,
            "cumulative_drop_atr": (start_close - close) / start_atr,
            "distance_support_atr": (float(support.iloc[cursor]) - close) / start_atr,
            "rebound_atr": rebound_atr,
            "body_atr": max(
                0.0, float(h1["Open"].iloc[cursor] - close) / start_atr
            ),
            "volume_z": float(volume_z.iloc[cursor]),
            "raw_close": close,
            "raw_atr": float(row["raw_atr"]),
        }
    )
    return output


def _attach_path_labels(data, opportunities):
    h1 = _ohlc_bars(data, "1h")
    spread_price = float(data["SpreadPoints"].median()) * 0.01
    output = opportunities.copy()
    labels = []
    for timestamp, row in output.iterrows():
        if timestamp not in h1.index:
            labels.append((np.nan, np.nan, np.nan, np.nan))
            continue
        location = h1.index.get_loc(timestamp)
        future = h1.iloc[location + 1 : location + 25]
        if len(future) < 24:
            labels.append((np.nan, np.nan, np.nan, np.nan))
            continue
        entry = float(row["raw_close"])
        favorable = entry - (future["Low"] + spread_price)
        adverse = (future["High"] + spread_price) - entry
        tp_steps = np.flatnonzero(favorable.to_numpy() >= 20.0)
        sl_steps = np.flatnonzero(adverse.to_numpy() >= 10.0)
        first_tp = int(tp_steps[0] + 1) if len(tp_steps) else 10_000
        first_sl = int(sl_steps[0] + 1) if len(sl_steps) else 10_000
        labels.append(
            (
                float(first_tp < first_sl),
                float(favorable.max()),
                float(adverse.max()),
                float(first_tp) if first_tp < 10_000 else np.nan,
            )
        )
    output[
        ["target_24h", "mfe_24h_usd", "mae_24h_usd", "time_to_tp_hours"]
    ] = labels
    return output


def _train_event_models(opportunities):
    rows = []
    runs = {}
    for setup in ("Breakdown Acceptance", "Failed Recovery"):
        setup_frame = opportunities.loc[opportunities["setup"].eq(setup)].copy()
        train = setup_frame.loc[TRAIN_START:TRAIN_END]
        calibration = setup_frame.loc[CALIBRATION_START:CALIBRATION_END]
        threshold_period = setup_frame.loc[THRESHOLD_START:THRESHOLD_END]
        if len(train) < 20 or train["target_24h"].nunique() < 2:
            raise RuntimeError(f"Data train {setup} tidak cukup untuk klasifikasi.")
        logistic = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                C=0.30,
                random_state=80,
            ),
        )
        boosting = HistGradientBoostingClassifier(
            learning_rate=0.035,
            max_iter=160,
            max_depth=2,
            min_samples_leaf=12,
            l2_regularization=2.0,
            random_state=81,
        )
        logistic.fit(train[list(MODEL_FEATURES)], train["target_24h"].astype(int))
        boosting.fit(
            train[list(MODEL_FEATURES)],
            train["target_24h"].astype(int),
            sample_weight=_class_weights(train["target_24h"]),
        )
        raw = (
            _predict_event(logistic, setup_frame)
            + _predict_event(boosting, setup_frame)
        ) / 2
        if len(calibration) >= 10 and calibration["target_24h"].nunique() == 2:
            calibrator = _fit_platt(
                raw.reindex(calibration.index),
                calibration["target_24h"].astype(int),
            )
            probability = _apply_symmetric_calibration(raw, calibrator)
        else:
            probability = raw
        threshold, audit = _select_event_threshold(
            threshold_period["target_24h"].astype(int),
            probability.reindex(threshold_period.index),
        )
        full_probability = pd.Series(np.nan, index=opportunities.index)
        full_probability.loc[setup_frame.index] = probability
        runs[setup] = {
            "probability": full_probability,
            "threshold": threshold,
        }
        rows.append(
            {
                "Setup": setup,
                "Train events": len(train),
                "Calibration events": len(calibration),
                "Threshold events": len(threshold_period),
                "Threshold": threshold,
                **audit,
            }
        )
    return runs, pd.DataFrame(rows)


def _predict_event(model, frame):
    return pd.Series(
        model.predict_proba(frame[list(MODEL_FEATURES)])[:, 1],
        index=frame.index,
    )


def _select_event_threshold(truth, probability):
    rows = []
    for quantile in (0.40, 0.50, 0.60, 0.70, 0.80):
        threshold = float(probability.quantile(quantile))
        prediction = probability.ge(threshold)
        selected = truth.loc[prediction]
        precision = float(selected.mean()) if len(selected) else 0.0
        recall = float(
            (prediction & truth.eq(1)).sum() / max(int(truth.eq(1).sum()), 1)
        )
        expected_value = precision * 20 - (1 - precision) * 10
        eligible = len(selected) >= 6
        score = expected_value + recall * 3 + min(len(selected), 20) * 0.03
        rows.append((threshold, precision, recall, len(selected), score, eligible))
    eligible_rows = [row for row in rows if row[-1]]
    selected = max(eligible_rows or rows, key=lambda row: row[4])
    return selected[0], {
        "Precision threshold": selected[1],
        "Recall threshold": selected[2],
        "Sinyal threshold": selected[3],
        "Expected value proxy": selected[1] * 20 - (1 - selected[1]) * 10,
    }


def _candidate_signals(data, opportunities, runs, best, spread_limit):
    breakdown = opportunities["setup"].eq("Breakdown Acceptance")
    recovery = opportunities["setup"].eq("Failed Recovery")
    breakdown_selected = breakdown & runs["Breakdown Acceptance"][
        "probability"
    ].ge(runs["Breakdown Acceptance"]["threshold"])
    recovery_selected = recovery & runs["Failed Recovery"]["probability"].ge(
        runs["Failed Recovery"]["threshold"]
    )
    specs = {
        "Bear Event Control": (breakdown | recovery, 5, "none", 20.0),
        "Breakdown Acceptance 5m": (breakdown_selected, 5, "none", 20.0),
        "Breakdown Acceptance 15m": (
            breakdown_selected,
            15,
            "rejection",
            20.0,
        ),
        "Failed Recovery 5m": (recovery_selected, 5, "none", 15.0),
        "Failed Recovery 15m": (recovery_selected, 15, "rejection", 15.0),
    }
    output = {}
    funnel = []
    for candidate, (mask, delay, confirmation, tp) in specs.items():
        selected = opportunities.loc[mask].copy()
        selected = selected.loc[~selected.index.duplicated(keep="first")]
        before = _raw_sell_signals(selected, best)
        delayed, event_frame = _timed_signals(
            data, before, best, delay, spread_limit, confirmation
        )
        delayed["tp_usd"] = tp
        delayed["sl_usd"] = 10.0
        delayed["time_stop_hours"] = 24.0
        output[candidate] = _unique_signals(delayed)
        funnel.append(
            {
                "Kandidat": candidate,
                "Event setup": len(before),
                "Lolos timing": len(output[candidate]),
                "Batal barrier": (
                    int(event_frame["expired"].sum())
                    if not event_frame.empty
                    else 0
                ),
                "Batal spread": (
                    int((~event_frame["spread_ok"] & ~event_frame["expired"]).sum())
                    if not event_frame.empty
                    else 0
                ),
            }
        )
    adaptive = pd.concat(
        [
            output["Breakdown Acceptance 5m"],
            output["Breakdown Acceptance 15m"],
            output["Failed Recovery 5m"],
            output["Failed Recovery 15m"],
        ]
    ).sort_index()
    adaptive = adaptive.loc[~adaptive.index.duplicated(keep="first")].copy()
    if not adaptive.empty:
        original = pd.to_datetime(
            adaptive.get("original_signal_time", adaptive.index), errors="coerce"
        )
        unique_opportunities = opportunities.loc[
            ~opportunities.index.duplicated(keep="first")
        ]
        source = unique_opportunities.reindex(pd.DatetimeIndex(original))
        strong = (
            source["initial_impulse_atr"].ge(1.0)
            & source["cumulative_drop_atr"].lt(1.5)
        ).fillna(False).to_numpy()
        adaptive["tp_usd"] = np.where(strong, 25.0, adaptive["tp_usd"])
    adaptive["sl_usd"] = 10.0
    adaptive["time_stop_hours"] = 24.0
    output["Adaptive Event Ensemble"] = _unique_signals(adaptive)
    funnel.append(
        {
            "Kandidat": "Adaptive Event Ensemble",
            "Event setup": sum(len(output[name]) for name in specs if name != "Bear Event Control"),
            "Lolos timing": len(adaptive),
            "Batal barrier": np.nan,
            "Batal spread": np.nan,
        }
    )
    return output, pd.DataFrame(funnel)


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
    periods = (
        ("Train 2022", TRAIN_START, TRAIN_END),
        ("Calibration 2023H1", CALIBRATION_START, CALIBRATION_END),
        ("Threshold 2023H2", THRESHOLD_START, THRESHOLD_END),
        ("Model selection 2024", SELECTION_START, SELECTION_END),
        ("Locked confirmation 2025", LOCKED_START, LOCKED_END),
    )
    rows = []
    for label, start, end in periods:
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


def _classification_tables(opportunities, runs):
    periods = {
        "selection": (SELECTION_START, SELECTION_END),
        "locked": (LOCKED_START, LOCKED_END),
        "reference": (REFERENCE_START, REFERENCE_END),
    }
    output = {}
    for key, (start, end) in periods.items():
        rows = []
        for setup, run in runs.items():
            period_frame = opportunities.loc[start:end]
            period_probability = run["probability"].loc[start:end]
            setup_mask = period_frame["setup"].eq(setup).to_numpy()
            selected = period_frame.iloc[np.flatnonzero(setup_mask)].copy()
            probability = period_probability.iloc[
                np.flatnonzero(setup_mask)
            ].copy()
            prediction = probability.ge(run["threshold"])
            truth = selected["target_24h"].astype(int)
            rows.append(
                {
                    "Setup": setup,
                    "Threshold": run["threshold"],
                    "Event": len(selected),
                    "Precision": precision_score(
                        truth, prediction, zero_division=0
                    ),
                    "Recall": recall_score(truth, prediction, zero_division=0),
                    "Coverage (%)": float(prediction.mean() * 100),
                    "Brier": (
                        float(brier_score_loss(truth, probability))
                        if len(selected)
                        else np.nan
                    ),
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
                period.loc[("Model selection 2024", candidate), "Growth (%)"]
            ) > 0,
            "2025 positif": float(
                period.loc[("Locked confirmation 2025", candidate), "Growth (%)"]
            ) > 0,
            "Primary fold >= 6/8": int(primary["Profitable"].sum()) >= 6,
            "Monte Carlo rugi <= 10%": float(
                mc.loc[candidate, "Probabilitas equity akhir < modal awal (%)"]
            ) <= 10,
            "Transaksi development >= 20": int(dev.loc[candidate, "Transaksi"]) >= 20,
            "Transaksi locked >= 6": int(
                period.loc[("Locked confirmation 2025", candidate), "Transaksi"]
            ) >= 6,
            "Konsentrasi 5 profit <= 50%": float(
                concentrated.loc[candidate, "Konsentrasi 5 profit terbesar (%)"]
            ) <= 50,
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
    locked_precision = (
        float(classification["Precision"].mean()) if not classification.empty else 0.0
    )
    rows = []
    for candidate in CANDIDATES:
        selection_growth = float(
            period.loc[("Model selection 2024", candidate), "Growth (%)"]
        )
        selection_pf = float(
            period.loc[("Model selection 2024", candidate), "Profit factor"]
        )
        selection_dd = float(
            period.loc[("Model selection 2024", candidate), "Max drawdown (%)"]
        )
        selection_trades = int(
            period.loc[("Model selection 2024", candidate), "Transaksi"]
        )
        safe_pf = 0.0 if not np.isfinite(selection_pf) else selection_pf
        score = (
            selection_growth
            + min(safe_pf, 3.0) * 5
            - selection_dd * 1.5
            + min(selection_trades, 30) * 0.10
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
                    selection_growth > 0 and selection_trades >= 6
                ),
                "Growth development (%)": float(dev.loc[candidate, "Growth (%)"]),
                "PF development": float(dev.loc[candidate, "Profit factor"]),
                "DD development (%)": float(dev.loc[candidate, "Max drawdown (%)"]),
                "Transaksi development": int(dev.loc[candidate, "Transaksi"]),
                "Growth locked 2025 (%)": float(
                    period.loc[("Locked confirmation 2025", candidate), "Growth (%)"]
                ),
                "Precision locked setup": locked_precision,
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


def _event_audit(events, opportunities):
    rows = []
    for label, start, end in _periods_with_reference():
        event_count = events.loc[start:end, "event_id"].nunique()
        selected = opportunities.loc[start:end]
        breakdown = selected.loc[selected["setup"].eq("Breakdown Acceptance")]
        recovery = selected.loc[selected["setup"].eq("Failed Recovery")]
        rows.append(
            {
                "Periode": label,
                "Bear events": event_count,
                "Breakdown Acceptance": len(breakdown),
                "Breakdown TP-first 24h (%)": (
                    float(breakdown["target_24h"].mean() * 100)
                    if len(breakdown)
                    else np.nan
                ),
                "Failed Recovery": len(recovery),
                "Failed Recovery TP-first 24h (%)": (
                    float(recovery["target_24h"].mean() * 100)
                    if len(recovery)
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def _phase_audit(events):
    rows = []
    for label, start, end in _periods_with_reference():
        counts = events.loc[start:end, "phase"].value_counts()
        rows.append(
            {
                "Periode": label,
                "Initiation": int(counts.get("Initiation", 0)),
                "Continuation": int(counts.get("Continuation", 0)),
                "Exhaustion": int(counts.get("Exhaustion", 0)),
                "Recovery": int(counts.get("Recovery", 0)),
            }
        )
    return pd.DataFrame(rows)


def _path_audit(opportunities):
    rows = []
    for label, start, end in _periods_with_reference():
        for setup in ("Breakdown Acceptance", "Failed Recovery"):
            selected = opportunities.loc[start:end]
            selected = selected.loc[selected["setup"].eq(setup)]
            winners = selected.loc[selected["target_24h"].eq(1)]
            rows.append(
                {
                    "Periode": label,
                    "Setup": setup,
                    "Event setup": len(selected),
                    "TP-before-SL (%)": (
                        float(selected["target_24h"].mean() * 100)
                        if len(selected)
                        else np.nan
                    ),
                    "Median MFE": (
                        float(selected["mfe_24h_usd"].median())
                        if len(selected)
                        else np.nan
                    ),
                    "Median MAE": (
                        float(selected["mae_24h_usd"].median())
                        if len(selected)
                        else np.nan
                    ),
                    "Median time-to-TP": (
                        float(winners["time_to_tp_hours"].median())
                        if len(winners)
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def _periods_with_reference():
    return (
        ("Train 2022", TRAIN_START, TRAIN_END),
        ("Calibration 2023H1", CALIBRATION_START, CALIBRATION_END),
        ("Threshold 2023H2", THRESHOLD_START, THRESHOLD_END),
        ("Selection 2024", SELECTION_START, SELECTION_END),
        ("Locked 2025", LOCKED_START, LOCKED_END),
        ("Reference 2026H1", REFERENCE_START, REFERENCE_END),
    )


def _v6_reference(v6_payload):
    if not v6_payload:
        return pd.DataFrame()
    ranking = v6_payload.get("ranking")
    if not isinstance(ranking, pd.DataFrame) or ranking.empty:
        return pd.DataFrame()
    return ranking.head(1).copy()
