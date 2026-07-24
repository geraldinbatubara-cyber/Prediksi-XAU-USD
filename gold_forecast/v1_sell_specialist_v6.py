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
from gold_forecast.v1_entry_timing import _micro_event_frame
from gold_forecast.v1_fixed_delay import _build_fixed_delay_signals
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
    _select_threshold,
    _sell_outcome_frame,
)
from gold_forecast.v1_trend_strength_stability import _extended_data_audit
from gold_forecast.v1_unified_benchmark import _unified_best


CANDIDATES = (
    "Bear Regime Control",
    "Breakdown Outcome 5m",
    "Breakdown Retest 15m",
    "Failed Rally Outcome 5m",
    "Failed Rally Rejection 15m",
    "Adaptive Setup Ensemble",
)


def run_v1_sell_specialist_v6_lab(
    gold_m1: pd.DataFrame,
    frozen_payload: dict[str, object],
    v5_payload: dict[str, object] | None = None,
) -> dict[str, object]:
    data = _prepare_m1(gold_m1)
    _, leaderboard, _ = frozen_payload["v1"]
    best = _unified_best(leaderboard.iloc[0].to_dict())
    best = {
        **best,
        "Close-all target equity": False,
        "Max BUY": 0,
        "Max SELL": 1,
    }
    config = RiskControlConfig(
        "SELL Specialist v6",
        "Bear regime and setup separation",
        max_total_positions=1,
        max_same_direction=1,
    )
    spread_limit = float(
        data.loc[DEVELOPMENT_START:DEVELOPMENT_END, "SpreadPoints"].quantile(0.90)
    )
    frame = _sell_outcome_frame(data)
    frame = _regime_and_setup_frame(data, frame)
    model_runs, model_selection = _train_setup_models(frame)
    signals, funnel = _candidate_signals(
        data,
        frame,
        model_runs,
        best,
        spread_limit,
    )

    development_data = data.loc[DEVELOPMENT_START:DEVELOPMENT_END]
    reference_data = data.loc[REFERENCE_START:REFERENCE_END]
    development_results = _simulate_all(
        development_data,
        signals,
        best,
        config,
        DEVELOPMENT_START,
        DEVELOPMENT_END,
    )
    reference_results = _simulate_all(
        reference_data,
        signals,
        best,
        config,
        REFERENCE_START,
        REFERENCE_END,
    )
    development = _result_table(
        development_results,
        signals,
        DEVELOPMENT_START,
        DEVELOPMENT_END,
    )
    reference = _result_table(
        reference_results,
        signals,
        REFERENCE_START,
        REFERENCE_END,
    )
    periods = _period_validation(development_results, signals)
    folds = _fold_evaluation(development_results)
    classification = _classification_tables(frame, model_runs)
    monte_carlo = _monte_carlo_summary(development_results)
    concentration = _profit_concentration(development_results)
    decisions = _decision_table(
        development,
        periods,
        folds,
        monte_carlo,
        concentration,
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
        _stress_summary(
            development_data,
            signals,
            best,
            config,
            [winner],
        )
        if winner
        else pd.DataFrame()
    )
    stress_passed = int(stress.iloc[0]["Skenario profitable"]) if not stress.empty else 0
    decisions = decisions.copy()
    decisions["Stress profitable"] = (
        decisions["Kandidat"].map({winner: stress_passed})
        if winner
        else np.nan
    )
    winner_passed = False
    if winner:
        winner_decision = decisions.loc[
            decisions["Kandidat"].eq(winner)
        ].iloc[0]
        winner_passed = bool(winner_decision["Lulus"]) and stress_passed >= 4
        ranking.loc[
            ranking["Kandidat"].eq(winner), "Lulus termasuk stress"
        ] = winner_passed

    return {
        "methodology": {
            "Name": "v1 SELL Specialist Lab v6 - Bear Regime & Setup Separation",
            "Mandat": "SELL atau ABSTAIN; mesin tidak pernah membuka BUY.",
            "Bear gate": (
                "Minimal 2/3 bukti bearish pada D1 dan H4, serta salah satu timeframe "
                "harus memenuhi 3/3: close di bawah MA cepat, MA cepat di bawah MA "
                "lambat, dan momentum negatif."
            ),
            "Setup": (
                "Breakdown Continuation dan Failed Rally dilatih serta dieksekusi "
                "sebagai pola yang berbeda."
            ),
            "Train": "2022",
            "Probability calibration": "2023H1",
            "Threshold calibration": "2023H2",
            "Model selection": "2024 saja",
            "Locked confirmation": "2025",
            "Historical reference": "2026H1, tidak menentukan pemenang",
            "Execution": (
                "Equity USD 1.000 | lot 0.01 | maksimal 1 SELL | delay 5m/15m | "
                "TP 15/20/25 | SL 10 | time stop 24 jam pada kandidat adaptive"
            ),
            "Baseline lock": (
                "Baseline v1, BUY Specialist v4, dan seluruh ledger paper live "
                "tidak dibaca atau ditulis oleh eksperimen."
            ),
        },
        "data_audit": _extended_data_audit(data),
        "regime_setup_audit": _regime_setup_audit(frame),
        "model_selection": model_selection,
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
        "v5_reference": _v5_reference(v5_payload),
    }


def _regime_and_setup_frame(data, frame):
    daily = _ohlc_bars(data, "1D")
    daily_close = daily["Close"]
    daily_fast = daily_close.ewm(span=20, adjust=False).mean()
    daily_slow = daily_close.ewm(span=50, adjust=False).mean()
    daily_momentum = daily_close.pct_change(20)
    daily_evidence = pd.concat(
        [
            daily_close.lt(daily_fast),
            daily_fast.lt(daily_slow),
            daily_momentum.lt(0),
        ],
        axis=1,
    ).sum(axis=1)

    h4 = _ohlc_bars(data, "4h")
    h4_close = h4["Close"]
    h4_fast = h4_close.ewm(span=10, adjust=False).mean()
    h4_slow = h4_close.ewm(span=30, adjust=False).mean()
    h4_momentum = h4_close.pct_change(6)
    h4_evidence = pd.concat(
        [
            h4_close.lt(h4_fast),
            h4_fast.lt(h4_slow),
            h4_momentum.lt(0),
        ],
        axis=1,
    ).sum(axis=1)

    output = frame.copy()
    output["d1_bear_evidence"] = daily_evidence.reindex(
        output.index, method="ffill"
    )
    output["h4_bear_evidence"] = h4_evidence.reindex(
        output.index, method="ffill"
    )
    output["bear_regime"] = (
        output["d1_bear_evidence"].ge(2)
        & output["h4_bear_evidence"].ge(2)
        & (
            output["d1_bear_evidence"].eq(3)
            | output["h4_bear_evidence"].eq(3)
        )
    )

    h1 = _ohlc_bars(data, "1h").reindex(output.index)
    h1_fast = h1["Close"].ewm(span=10, adjust=False).mean()
    previous_support = h1["Low"].rolling(20, min_periods=10).min().shift(1)
    output["breakdown_setup"] = (
        h1["Close"].lt(previous_support)
        & output["raw_momentum_6"].lt(0)
        & h1["Close"].lt(h1["Open"])
    )
    output["failed_rally_setup"] = (
        h1["High"].ge(h1_fast * 0.999)
        & h1["Close"].lt(h1_fast)
        & h1["Close"].lt(h1["Open"])
        & h1["Close"].shift(1).ge(h1_fast.shift(1) * 0.998)
    )
    output["strong_bear"] = (
        output["d1_bear_evidence"].eq(3)
        & output["h4_bear_evidence"].eq(3)
        & output["adx"].ge(24)
        & output["efficiency"].ge(0.30)
    )
    return output.dropna(
        subset=[
            "d1_bear_evidence",
            "h4_bear_evidence",
            "target_12h",
            *SYMMETRIC_FEATURES,
        ]
    )


def _train_setup_models(frame):
    rows = []
    runs = {}
    for setup_name, setup_column in (
        ("Breakdown", "breakdown_setup"),
        ("Failed Rally", "failed_rally_setup"),
    ):
        setup_frame = frame.loc[frame[setup_column]].copy()
        train = setup_frame.loc[TRAIN_START:TRAIN_END]
        calibration = setup_frame.loc[CALIBRATION_START:CALIBRATION_END]
        threshold_period = setup_frame.loc[THRESHOLD_START:THRESHOLD_END]
        if train["target_12h"].nunique() < 2:
            raise RuntimeError(f"Label train {setup_name} tidak memiliki dua kelas.")
        logistic = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                C=0.35,
                random_state=70,
            ),
        )
        boosting = HistGradientBoostingClassifier(
            learning_rate=0.04,
            max_iter=180,
            max_depth=3,
            min_samples_leaf=20,
            l2_regularization=1.5,
            random_state=71,
        )
        logistic.fit(
            train[list(SYMMETRIC_FEATURES)],
            train["target_12h"].astype(int),
        )
        boosting.fit(
            train[list(SYMMETRIC_FEATURES)],
            train["target_12h"].astype(int),
            sample_weight=_class_weights(train["target_12h"]),
        )
        raw = (
            _predict(logistic, setup_frame)
            + _predict(boosting, setup_frame)
        ) / 2
        calibrator = _fit_platt(
            raw.reindex(calibration.index),
            calibration["target_12h"].astype(int),
        )
        probability = _apply_symmetric_calibration(raw, calibrator)
        threshold, audit = _select_threshold(
            threshold_period["target_12h"].astype(int),
            probability.reindex(threshold_period.index),
        )
        full_probability = pd.Series(np.nan, index=frame.index)
        full_probability.loc[setup_frame.index] = probability
        runs[setup_name] = {
            "probability": full_probability,
            "threshold": threshold,
            "setup_column": setup_column,
        }
        rows.append(
            {
                "Setup": setup_name,
                "Train observations": len(train),
                "Calibration observations": len(calibration),
                "Threshold observations": len(threshold_period),
                "Threshold": threshold,
                **audit,
            }
        )
    return runs, pd.DataFrame(rows)


def _candidate_signals(data, frame, runs, best, spread_limit):
    raw = _raw_sell_signals(frame, best)
    gate = frame["bear_regime"]
    breakdown = frame["breakdown_setup"]
    failed_rally = frame["failed_rally_setup"]
    breakdown_probability = runs["Breakdown"]["probability"]
    failed_probability = runs["Failed Rally"]["probability"]
    breakdown_selected = (
        gate
        & breakdown
        & breakdown_probability.ge(runs["Breakdown"]["threshold"])
    )
    failed_selected = (
        gate
        & failed_rally
        & failed_probability.ge(runs["Failed Rally"]["threshold"])
    )

    specs = {
        "Bear Regime Control": {
            "mask": gate & (breakdown | failed_rally),
            "delay": 5,
            "confirmation": "none",
            "tp": 25.0,
            "sl": 10.0,
        },
        "Breakdown Outcome 5m": {
            "mask": breakdown_selected,
            "delay": 5,
            "confirmation": "none",
            "tp": 20.0,
            "sl": 10.0,
        },
        "Breakdown Retest 15m": {
            "mask": breakdown_selected,
            "delay": 15,
            "confirmation": "retest",
            "tp": 20.0,
            "sl": 10.0,
        },
        "Failed Rally Outcome 5m": {
            "mask": failed_selected,
            "delay": 5,
            "confirmation": "none",
            "tp": 20.0,
            "sl": 10.0,
        },
        "Failed Rally Rejection 15m": {
            "mask": failed_selected,
            "delay": 15,
            "confirmation": "rejection",
            "tp": 15.0,
            "sl": 10.0,
        },
    }
    output = {}
    funnel_rows = []
    for candidate, spec in specs.items():
        before = raw.loc[spec["mask"].reindex(raw.index).fillna(False)].copy()
        delayed, events = _timed_signals(
            data,
            before,
            best,
            int(spec["delay"]),
            spread_limit,
            str(spec["confirmation"]),
        )
        delayed["tp_usd"] = float(spec["tp"])
        delayed["sl_usd"] = float(spec["sl"])
        output[candidate] = _unique_signals(delayed)
        funnel_rows.append(
            _funnel_row(candidate, before, output[candidate], events)
        )

    adaptive_parts = [
        output["Breakdown Outcome 5m"],
        output["Breakdown Retest 15m"],
        output["Failed Rally Outcome 5m"],
        output["Failed Rally Rejection 15m"],
    ]
    adaptive = pd.concat(adaptive_parts).sort_index()
    adaptive = adaptive.loc[~adaptive.index.duplicated(keep="first")].copy()
    original_times = pd.to_datetime(
        adaptive.get("original_signal_time", adaptive.index),
        errors="coerce",
    )
    strength = frame["strong_bear"].reindex(
        pd.DatetimeIndex(original_times), method="ffill"
    ).fillna(False).to_numpy()
    adaptive["tp_usd"] = np.where(strength, 25.0, 20.0)
    adaptive["sl_usd"] = 10.0
    adaptive["time_stop_hours"] = 24.0
    output["Adaptive Setup Ensemble"] = _unique_signals(adaptive)
    funnel_rows.append(
        {
            "Kandidat": "Adaptive Setup Ensemble",
            "Sinyal setup": sum(len(part) for part in adaptive_parts),
            "Lolos konfirmasi": len(adaptive),
            "Batal barrier": np.nan,
            "Batal spread": np.nan,
        }
    )
    return output, pd.DataFrame(funnel_rows)


def _timed_signals(
    data,
    signals,
    best,
    delay,
    spread_limit,
    confirmation,
):
    delayed, events = _build_fixed_delay_signals(
        data,
        signals,
        best,
        delay,
        spread_limit,
    )
    if delayed.empty or confirmation == "none" or events.empty:
        return delayed, events
    accepted_events = events.loc[~events["expired"] & events["spread_ok"]].copy()
    if confirmation == "retest":
        confirmed = (
            accepted_events["observed_adverse_usd"].between(0.25, 8.0)
            & accepted_events["signed_momentum_atr"].ge(0)
        )
    else:
        confirmed = (
            accepted_events["signed_return_atr"].ge(0)
            & accepted_events["signed_momentum_atr"].ge(0)
            & accepted_events["signed_ema_gap_atr"].ge(-0.10)
        )
    confirmed_times = set(accepted_events.index[confirmed])
    original = pd.to_datetime(
        delayed.get("original_signal_time", delayed.index),
        errors="coerce",
    )
    delayed = delayed.loc[
        pd.Series(original.isin(confirmed_times), index=delayed.index)
    ].copy()
    return delayed, events


def _funnel_row(candidate, before, delayed, events):
    return {
        "Kandidat": candidate,
        "Sinyal setup": len(before),
        "Lolos konfirmasi": len(delayed),
        "Batal barrier": int(events["expired"].sum()) if not events.empty else 0,
        "Batal spread": int(
            (~events["spread_ok"] & ~events["expired"]).sum()
        ) if not events.empty else 0,
    }


def _simulate_all(data, signals, best, config, start, end):
    return {
        candidate: _simulate_risk_control(
            data,
            frame.loc[start:end],
            best,
            config,
        )
        for candidate, frame in signals.items()
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


def _classification_tables(frame, runs):
    periods = {
        "selection": (SELECTION_START, SELECTION_END),
        "locked": (LOCKED_START, LOCKED_END),
        "reference": (REFERENCE_START, REFERENCE_END),
    }
    output = {}
    for key, (start, end) in periods.items():
        rows = []
        for setup, run in runs.items():
            selected = frame.loc[start:end]
            selected = selected.loc[selected[run["setup_column"]]].copy()
            probability = run["probability"].reindex(selected.index)
            prediction = probability.ge(run["threshold"])
            truth = selected["target_12h"].astype(int)
            rows.append(
                {
                    "Setup": setup,
                    "Threshold": run["threshold"],
                    "Observasi": len(selected),
                    "Precision": precision_score(
                        truth, prediction, zero_division=0
                    ),
                    "Recall": recall_score(truth, prediction, zero_division=0),
                    "Coverage (%)": float(prediction.mean() * 100),
                    "Brier": float(brier_score_loss(truth, probability)),
                }
            )
        output[key] = pd.DataFrame(rows)
    return output


def _profit_concentration(results):
    rows = []
    for candidate, result in results.items():
        positive = (
            pd.to_numeric(result.trades.get("Net P/L"), errors="coerce")
            if not result.trades.empty
            else pd.Series(dtype=float)
        )
        positive = positive[positive > 0].sort_values(ascending=False)
        concentration = (
            float(positive.head(5).sum() / positive.sum() * 100)
            if positive.sum() > 0
            else 100.0
        )
        rows.append(
            {
                "Kandidat": candidate,
                "Konsentrasi 5 profit terbesar (%)": concentration,
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
            "PF >= 1.50": float(dev.loc[candidate, "Profit factor"]) >= 1.50,
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
            "Transaksi development >= 30": int(dev.loc[candidate, "Transaksi"]) >= 30,
            "Transaksi locked >= 8": int(
                period.loc[("Locked confirmation 2025", candidate), "Transaksi"]
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
    locked_precision = float(classification["Precision"].mean())
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
        selection_score = (
            selection_growth
            + min(safe_pf, 3.0) * 5
            - selection_dd * 1.5
            + min(selection_trades, 30) * 0.10
        )
        rows.append(
            {
                "Kandidat": candidate,
                "Selection score 2024": selection_score,
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


def _regime_setup_audit(frame):
    rows = []
    for label, start, end in (
        ("Train 2022", TRAIN_START, TRAIN_END),
        ("Calibration 2023H1", CALIBRATION_START, CALIBRATION_END),
        ("Threshold 2023H2", THRESHOLD_START, THRESHOLD_END),
        ("Selection 2024", SELECTION_START, SELECTION_END),
        ("Locked 2025", LOCKED_START, LOCKED_END),
        ("Reference 2026H1", REFERENCE_START, REFERENCE_END),
    ):
        selected = frame.loc[start:end]
        regime = selected.loc[selected["bear_regime"]]
        breakdown = regime.loc[regime["breakdown_setup"]]
        failed = regime.loc[regime["failed_rally_setup"]]
        rows.append(
            {
                "Periode": label,
                "Observasi H1": len(selected),
                "Bear regime": len(regime),
                "Breakdown": len(breakdown),
                "Breakdown TP-first 12h (%)": (
                    float(breakdown["target_12h"].mean() * 100)
                    if len(breakdown)
                    else np.nan
                ),
                "Failed rally": len(failed),
                "Failed rally TP-first 12h (%)": (
                    float(failed["target_12h"].mean() * 100)
                    if len(failed)
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def _v5_reference(v5_payload):
    if not v5_payload:
        return pd.DataFrame()
    ranking = v5_payload.get("ranking")
    if not isinstance(ranking, pd.DataFrame) or ranking.empty:
        return pd.DataFrame()
    return ranking.head(1).copy()
