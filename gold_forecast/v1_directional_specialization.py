from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, precision_score, recall_score
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
    _ohlc_bars,
    _timeframe_features,
)
from gold_forecast.v1_regime_classifier_v3 import (
    CALIBRATION_END,
    CALIBRATION_START,
    LOCKED_END,
    LOCKED_START,
    THRESHOLD_END,
    THRESHOLD_START,
    TRAIN_END,
    TRAIN_START,
    VALIDATION_END,
    VALIDATION_START,
    _candidate_inputs as _v3_candidate_inputs,
    _delay_candidates as _v3_delay_candidates,
    _fit_platt,
    _m15_alignment,
    _select_model_horizons,
    _train_hierarchical_candidates,
)
from gold_forecast.v1_risk_control import (
    RiskControlConfig,
    _metric_values,
    _simulate_risk_control,
)
from gold_forecast.v1_signal_quality import SignalQualityConfig, _entry_features
from gold_forecast.v1_trend_strength_stability import _extended_data_audit
from gold_forecast.v1_unified_benchmark import _unified_best


LONG_CANDIDATES = (
    "Adaptive v3 Frozen",
    "Adaptive + Bear Defense",
    "Adaptive + Bear/Sideways Defense",
)
SYMMETRIC_CANDIDATES = (
    "Symmetric Logistic",
    "Symmetric Boosting",
    "Symmetric Ensemble",
    "Direction-Balanced Boosting",
    "Dual Expert BUY/SELL",
    "Symmetric + M15 Confirmation",
)
SIGNED_FEATURES = (
    "ema_gap_atr",
    "ema_fast_slope_atr",
    "ema_slow_slope_atr",
    "return_1",
    "return_3",
    "return_6",
    "h4_return",
    "h4_gap_atr",
    "d1_return",
    "d1_gap_atr",
    "donchian_position_centered",
)
SYMMETRIC_FEATURES = (
    *SIGNED_FEATURES,
    "adx",
    "adx_change_3",
    "efficiency",
    "choppiness",
    "atr_percentile",
    "bb_width_atr",
    "range_width_atr",
    "breakout_support",
    "h4_adx",
    "spread_median",
    "spread_p90",
)


def run_v1_directional_specialization_lab(
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
    config = RiskControlConfig(
        "Directional Specialization v4",
        "Two-track experiment",
        max_total_positions=1,
        max_same_direction=1,
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
    fixed_reference, reference_events = _build_fixed_delay_signals(
        data, balanced, best, 5, spread_limit
    )
    fixed_reference = _unique_signals(fixed_reference)

    base = _classifier_frame(data).drop(columns=["label"], errors="ignore")
    v3_runs, v3_selection = _train_hierarchical_candidates(base, data)
    v3_selected = _select_model_horizons(v3_runs, v3_selection)
    v3_inputs, v3_inputs_audit = _v3_candidate_inputs_with_placeholder(
        balanced, entry_features, v3_selected
    )
    v3_signals, _ = _v3_delay_candidates(
        data,
        v3_inputs,
        v3_selected,
        entry_features,
        best,
        spread_limit,
    )
    adaptive = v3_signals["Ensemble Adaptive Confirmation"]

    regime_state, regime_audit = _market_regime_state(data, base)
    long_signals = _long_track_signals(
        data, adaptive, regime_state, best, config
    )

    symmetric_frame = _symmetric_training_frame(data, base)
    symmetric_runs, symmetric_selection = _train_symmetric_models(
        symmetric_frame
    )
    symmetric_signals, symmetric_funnel = _symmetric_candidate_signals(
        data,
        balanced,
        entry_features,
        symmetric_runs,
        best,
        spread_limit,
    )

    all_signals = {**long_signals, **symmetric_signals}
    development_data = data.loc[DEVELOPMENT_START:DEVELOPMENT_END]
    reference_data = data.loc[CONFIRMATION_START:CONFIRMATION_END]
    development_results = _simulate_all(
        development_data, all_signals, best, config,
        DEVELOPMENT_START, DEVELOPMENT_END,
    )
    reference_results = _simulate_all(
        reference_data, all_signals, best, config,
        CONFIRMATION_START, CONFIRMATION_END,
    )
    fixed_dev_result = _simulate_risk_control(
        development_data,
        fixed_reference.loc[DEVELOPMENT_START:DEVELOPMENT_END],
        best,
        config,
    )
    fixed_ref_result = _simulate_risk_control(
        reference_data,
        fixed_reference.loc[CONFIRMATION_START:CONFIRMATION_END],
        best,
        config,
    )
    development = _result_table(
        development_results, all_signals, DEVELOPMENT_START, DEVELOPMENT_END
    )
    reference = _result_table(
        reference_results, all_signals, CONFIRMATION_START, CONFIRMATION_END
    )
    periods = _period_validation(development_results, all_signals)
    folds = _fold_evaluation(development_results)
    monte_carlo = _monte_carlo_summary(development_results)
    direction = _direction_audit(development_results, reference_results)
    regime_economic = _regime_economic_audit(
        development_results, long_signals, regime_state
    )
    symmetric_classification = _symmetric_classification_tables(
        symmetric_frame, symmetric_runs
    )
    retention = _retention_table(all_signals, fixed_reference)
    long_decisions = _long_decision_table(
        development, periods, folds, monte_carlo, regime_economic
    )
    symmetric_decisions = _symmetric_decision_table(
        symmetric_classification["locked"],
        development,
        periods,
        folds,
        monte_carlo,
        direction,
        retention,
    )
    long_ranking = _long_ranking(
        development, reference, long_decisions, regime_economic
    )
    symmetric_ranking = _symmetric_ranking(
        symmetric_classification["locked"],
        development,
        reference,
        symmetric_decisions,
        retention,
    )
    stress_candidates = [str(long_ranking.iloc[0]["Kandidat"])]
    if bool(symmetric_ranking.iloc[0]["Lulus"]):
        stress_candidates.append(str(symmetric_ranking.iloc[0]["Kandidat"]))
    stress = _stress_summary(
        development_data, all_signals, best, config, stress_candidates
    )

    return {
        "methodology": {
            "Name": "v1 Directional Specialization Lab v4",
            "Long track": (
                "Adaptive v3 diperlakukan sebagai BUY-only specialist; tidak diwajibkan membuka SELL"
            ),
            "Symmetric track": (
                "Setiap timestamp dilihat sebagai observasi BUY dan SELL dengan fitur relatif arah"
            ),
            "Train": "2022",
            "Probability calibration": "2023H1",
            "Threshold calibration": "2023H2",
            "Model selection": "2024",
            "Locked confirmation": "2025",
            "Historical reference": "2026H1, tidak menentukan pemenang",
            "Fold diagnostic": (
                "Fold, subperiode, dan regime dihitung dari ledger development yang "
                "sudah disimulasikan; tidak menjalankan ulang candle M1 untuk setiap irisan"
            ),
            "Directional label": (
                "4 jam; favorable displacement >= 0.8 ATR, path efficiency >= 0.45, "
                "adverse excursion <= 0.6 ATR"
            ),
            "Execution contract": (
                "Equity USD 1.000 | lot 0.01 | TP USD 25 | SL USD 10 | "
                "maksimal 1 posisi | Balanced Entry | Fixed Delay 5m"
            ),
            "Baseline lock": (
                "Baseline v1, Fixed Delay paper live, ledger, dan parameter observasi tidak diubah"
            ),
        },
        "data_audit": _extended_data_audit(data),
        "fixed_delay_reference": pd.DataFrame([
            {"Periode": "Development 2022-2025", **_metric_values(fixed_dev_result)},
            {"Periode": "Historical reference 2026H1", **_metric_values(fixed_ref_result)},
        ]),
        "regime_definition_audit": regime_audit,
        "symmetric_model_selection": symmetric_selection,
        "symmetric_selected_models": _selected_symmetric_table(symmetric_runs),
        "symmetric_classification_validation": symmetric_classification["validation"],
        "symmetric_classification_locked": symmetric_classification["locked"],
        "symmetric_classification_reference": symmetric_classification["reference"],
        "symmetric_funnel": symmetric_funnel,
        "development": development,
        "period_validation": periods,
        "historical_reference": reference,
        "folds": folds,
        "retention": retention,
        "monte_carlo_summary": monte_carlo,
        "direction_audit": direction,
        "regime_economic_audit": regime_economic,
        "stress_summary": stress,
        "long_decisions": long_decisions,
        "symmetric_decisions": symmetric_decisions,
        "long_ranking": long_ranking,
        "symmetric_ranking": symmetric_ranking,
        "long_winner": str(long_ranking.iloc[0]["Kandidat"]),
        "symmetric_winner": str(symmetric_ranking.iloc[0]["Kandidat"]),
        "reference_cancellation": pd.DataFrame([{
            "Sinyal Balanced": len(balanced),
            "Lolos Fixed Delay": len(fixed_reference),
            "Batal barrier": int(reference_events["expired"].sum()),
            "Batal spread": int(
                (~reference_events["spread_ok"] & ~reference_events["expired"]).sum()
            ),
        }]),
        "v3_input_reference": v3_inputs_audit,
    }


def _v3_candidate_inputs_with_placeholder(balanced, entry_features, selected_runs):
    placeholder = pd.Series("TRANSITION", index=entry_features.index, dtype=object)
    return _v3_candidate_inputs(
        balanced, entry_features, selected_runs, placeholder
    )


def _market_regime_state(data, base):
    daily = _ohlc_bars(data, "1D")
    close = daily["Close"]
    fast = close.ewm(span=20, adjust=False).mean()
    slow = close.ewm(span=50, adjust=False).mean()
    momentum = close.pct_change(20) * 100
    daily_state = pd.Series("TRANSITION", index=daily.index, dtype="object")
    daily_state.loc[(close > fast) & (fast > slow) & momentum.gt(0)] = "BULLISH"
    daily_state.loc[(close < fast) & (fast < slow) & momentum.lt(0)] = "BEARISH"
    hourly = base[["adx", "efficiency", "choppiness"]].copy()
    sideways = (
        hourly["adx"].lt(20)
        & hourly["efficiency"].lt(0.30)
        & hourly["choppiness"].gt(58)
    )
    state = daily_state.reindex(data.index, method="ffill").fillna("TRANSITION")
    state.loc[sideways.reindex(data.index, method="ffill").fillna(False)] = "SIDEWAYS"
    audit = state.loc[DEVELOPMENT_START:CONFIRMATION_END].value_counts().rename_axis(
        "Regime"
    ).reset_index(name="Candle M1")
    audit["Proporsi (%)"] = audit["Candle M1"] / audit["Candle M1"].sum() * 100
    return state, audit


def _long_track_signals(data, adaptive, regime_state, best, config):
    adaptive_buy = adaptive.loc[
        pd.to_numeric(adaptive["expected_change_pct"], errors="coerce").gt(0)
    ].copy()
    signal_state = regime_state.reindex(
        adaptive_buy.index, method="ffill"
    ).fillna("TRANSITION")
    bear_defended = adaptive_buy.loc[~signal_state.eq("BEARISH")].copy()
    defended = adaptive_buy.loc[
        ~signal_state.isin(["BEARISH", "SIDEWAYS"])
    ].copy()
    preliminary = _simulate_risk_control(data, defended, best, config)
    paused = _loss_pause_filter(defended, preliminary.trades)
    return {
        "Adaptive v3 Frozen": adaptive_buy,
        "Adaptive + Bear Defense": bear_defended,
        "Adaptive + Bear/Sideways Defense": paused,
    }


def _loss_pause_filter(signals, trades):
    if trades.empty:
        return signals
    ordered = trades.copy()
    ordered["entry"] = pd.to_datetime(ordered["Tanggal entry"], errors="coerce")
    ordered["close"] = pd.to_datetime(ordered["Tanggal tutup"], errors="coerce")
    ordered["net"] = pd.to_numeric(ordered["Net P/L"], errors="coerce").fillna(0)
    ordered = ordered.dropna(subset=["entry", "close"]).sort_values("close")
    pauses = []
    consecutive_losses = 0
    for _, trade in ordered.iterrows():
        consecutive_losses = consecutive_losses + 1 if trade["net"] < 0 else 0
        if consecutive_losses >= 2:
            pauses.append((trade["close"], trade["close"] + pd.Timedelta(hours=48)))
            consecutive_losses = 0
    keep = pd.Series(True, index=signals.index)
    for start, end in pauses:
        keep &= ~signals.index.to_series().between(start, end, inclusive="both")
    return signals.loc[keep].copy()


def _symmetric_training_frame(data, base):
    h1 = _ohlc_bars(data, "1h")
    h1_features = _timeframe_features(h1, "h1")
    atr = h1_features["atr"]
    close = h1["Close"]
    future_close = close.shift(-4)
    future_high = pd.concat([h1["High"].shift(-step) for step in range(1, 5)], axis=1).max(axis=1)
    future_low = pd.concat([h1["Low"].shift(-step) for step in range(1, 5)], axis=1).min(axis=1)
    steps = pd.concat(
        [(close.shift(-step) - close.shift(-(step - 1))).abs() for step in range(1, 5)],
        axis=1,
    ).sum(axis=1, min_count=4)
    efficiency = (future_close - close).abs() / steps
    frames = []
    for direction, sign in (("BUY", 1.0), ("SELL", -1.0)):
        frame = base.copy()
        frame["direction"] = direction
        frame["sign"] = sign
        frame["donchian_position_centered"] = (frame["donchian_position"] - 0.5) * 2
        for feature in SIGNED_FEATURES:
            frame[feature] = pd.to_numeric(frame[feature], errors="coerce") * sign
        frame["breakout_support"] = (
            frame["breakout_up"] if direction == "BUY" else frame["breakout_down"]
        )
        favorable = (future_close - close) * sign / atr
        adverse = (
            (close - future_low) / atr
            if direction == "BUY"
            else (future_high - close) / atr
        )
        frame["favorable_atr"] = favorable.reindex(frame.index)
        frame["adverse_atr"] = adverse.reindex(frame.index)
        frame["path_efficiency"] = efficiency.reindex(frame.index)
        frame["target"] = (
            frame["favorable_atr"].ge(0.80)
            & frame["path_efficiency"].ge(0.45)
            & frame["adverse_atr"].le(0.60)
        ).astype(int)
        missing = (
            frame["favorable_atr"].isna()
            | frame["path_efficiency"].isna()
            | frame["adverse_atr"].isna()
        )
        frame.loc[missing, "target"] = np.nan
        frame["timestamp"] = frame.index
        frame["scenario"] = direction
        frames.append(frame)
    symmetric = (
        pd.concat(frames)
        .set_index(["timestamp", "scenario"])
        .sort_index()
    )
    return symmetric.dropna(subset=[*SYMMETRIC_FEATURES, "target"])


def _train_symmetric_models(frame):
    train = frame.loc[TRAIN_START:TRAIN_END]
    calibrate = frame.loc[CALIBRATION_START:CALIBRATION_END]
    threshold_data = frame.loc[THRESHOLD_START:THRESHOLD_END]
    logistic = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", C=0.5, random_state=42),
    )
    boosting = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=160,
        max_depth=3,
        min_samples_leaf=30,
        l2_regularization=1.0,
        random_state=42,
    )
    balanced_boosting = HistGradientBoostingClassifier(
        learning_rate=0.04,
        max_iter=180,
        max_depth=3,
        min_samples_leaf=25,
        l2_regularization=1.5,
        random_state=43,
    )
    logistic.fit(train[list(SYMMETRIC_FEATURES)], train["target"].astype(int))
    boosting.fit(
        train[list(SYMMETRIC_FEATURES)],
        train["target"].astype(int),
        sample_weight=_class_weights(train["target"]),
    )
    balanced_boosting.fit(
        train[list(SYMMETRIC_FEATURES)],
        train["target"].astype(int),
        sample_weight=_direction_class_weights(train),
    )
    dual = {}
    for direction in ("BUY", "SELL"):
        selected = train[train["direction"].eq(direction)]
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", C=0.35, random_state=44),
        )
        model.fit(selected[list(SYMMETRIC_FEATURES)], selected["target"].astype(int))
        dual[direction] = model

    raw = {
        "Symmetric Logistic": _predict_probability(logistic, frame),
        "Symmetric Boosting": _predict_probability(boosting, frame),
        "Direction-Balanced Boosting": _predict_probability(balanced_boosting, frame),
        "Dual Expert BUY/SELL": _dual_probability(dual, frame),
    }
    raw["Symmetric Ensemble"] = (
        raw["Symmetric Logistic"] + raw["Symmetric Boosting"]
    ) / 2
    raw["Symmetric + M15 Confirmation"] = raw["Symmetric Ensemble"].copy()

    runs = {}
    selection_rows = []
    for candidate in SYMMETRIC_CANDIDATES:
        calibrated = _calibrate_symmetric(raw[candidate], calibrate)
        probability = _apply_symmetric_calibration(raw[candidate], calibrated)
        threshold = _choose_symmetric_threshold(
            threshold_data, probability.reindex(threshold_data.index)
        )
        runs[candidate] = {
            "probability": probability,
            "threshold": threshold,
            "moderate_threshold": max(0.05, threshold - 0.05),
            "calibrator": calibrated,
        }
        metrics = _symmetric_metrics(
            frame.loc[VALIDATION_START:VALIDATION_END],
            probability,
            threshold,
        )
        selection_rows.append({
            "Kandidat": candidate,
            "Threshold": threshold,
            **metrics,
            "Selection score": _symmetric_score(metrics),
        })
    return runs, pd.DataFrame(selection_rows)


def _predict_probability(model, frame):
    return pd.Series(
        model.predict_proba(frame[list(SYMMETRIC_FEATURES)])[:, 1],
        index=frame.index,
    )


def _dual_probability(models, frame):
    output = pd.Series(index=frame.index, dtype=float)
    for direction, model in models.items():
        mask = frame["direction"].eq(direction)
        output.loc[mask] = model.predict_proba(
            frame.loc[mask, list(SYMMETRIC_FEATURES)]
        )[:, 1]
    return output


def _class_weights(target):
    target = target.astype(int)
    counts = target.value_counts()
    return target.map({label: len(target) / (len(counts) * count) for label, count in counts.items()})


def _direction_class_weights(frame):
    groups = frame.groupby(["direction", "target"]).size()
    return pd.Series(
        [
            len(frame) / (len(groups) * groups.loc[(row.direction, row.target)])
            for row in frame[["direction", "target"]].itertuples(index=False)
        ],
        index=frame.index,
    )


def _calibrate_symmetric(probability, frame):
    aligned = probability.reindex(frame.index)
    return _fit_platt(aligned, frame["target"].astype(int))


def _apply_symmetric_calibration(probability, calibrator):
    clipped = probability.clip(1e-6, 1 - 1e-6)
    logit = np.log(clipped / (1 - clipped)).to_numpy().reshape(-1, 1)
    return pd.Series(calibrator.predict_proba(logit)[:, 1], index=probability.index)


def _choose_symmetric_threshold(frame, probability):
    rows = []
    for threshold in (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45):
        metrics = _symmetric_metrics(frame, probability, threshold)
        eligible = (
            metrics["Worst precision"] >= 0.50
            and metrics["Worst recall"] >= 0.25
            and metrics["Coverage (%)"] >= 20
        )
        rows.append((threshold, metrics, eligible, _symmetric_score(metrics)))
    eligible_rows = [row for row in rows if row[2]]
    return max(eligible_rows or rows, key=lambda row: row[3])[0]


def _symmetric_metrics(frame, probability, threshold):
    selected = frame.copy()
    selected["probability"] = probability.reindex(selected.index)
    rows = {}
    for direction in ("BUY", "SELL"):
        direction_frame = selected[selected["direction"].eq(direction)]
        prediction = direction_frame["probability"].ge(threshold)
        truth = direction_frame["target"].astype(int)
        rows[direction] = {
            "precision": precision_score(truth, prediction, zero_division=0),
            "recall": recall_score(truth, prediction, zero_division=0),
            "coverage": float(prediction.mean() * 100),
            "brier": float(brier_score_loss(truth, direction_frame["probability"])),
        }
    return {
        "BUY precision": float(rows["BUY"]["precision"]),
        "SELL precision": float(rows["SELL"]["precision"]),
        "BUY recall": float(rows["BUY"]["recall"]),
        "SELL recall": float(rows["SELL"]["recall"]),
        "Worst precision": float(min(rows["BUY"]["precision"], rows["SELL"]["precision"])),
        "Worst recall": float(min(rows["BUY"]["recall"], rows["SELL"]["recall"])),
        "Precision gap (pp)": float(abs(rows["BUY"]["precision"] - rows["SELL"]["precision"]) * 100),
        "Coverage (%)": float((rows["BUY"]["coverage"] + rows["SELL"]["coverage"]) / 2),
        "Brier": float((rows["BUY"]["brier"] + rows["SELL"]["brier"]) / 2),
    }


def _symmetric_score(metrics):
    return (
        metrics["Worst precision"] * 40
        + metrics["Worst recall"] * 25
        - metrics["Precision gap (pp)"] * 0.5
        + min(metrics["Coverage (%)"], 60) * 0.25
        - metrics["Brier"] * 20
    )


def _symmetric_candidate_signals(data, balanced, entry_features, runs, best, spread_limit):
    rows = []
    output = {}
    expected = pd.to_numeric(balanced["expected_change_pct"], errors="coerce")
    direction = pd.Series(np.where(expected.gt(0), "BUY", "SELL"), index=balanced.index)
    for candidate in SYMMETRIC_CANDIDATES:
        run = runs[candidate]
        probability = _probability_for_signals(
            run["probability"], direction, balanced.index
        )
        if candidate == "Symmetric + M15 Confirmation":
            strong = probability.ge(run["threshold"])
            moderate = probability.ge(run["moderate_threshold"])
            m15 = _m15_alignment(entry_features.reindex(balanced.index), direction)
            accepted = strong | (moderate & m15)
        else:
            accepted = probability.ge(run["threshold"])
        before_delay = balanced.loc[accepted.fillna(False)].copy()
        delayed, events = _build_fixed_delay_signals(
            data, before_delay, best, 5, spread_limit
        )
        output[candidate] = _unique_signals(delayed)
        rows.append({
            "Kandidat": candidate,
            "Threshold": run["threshold"],
            "Moderate threshold": run["moderate_threshold"],
            "Sinyal Balanced": len(balanced),
            "Lolos directional model": len(before_delay),
            "Lolos Fixed Delay": len(output[candidate]),
            "Batal barrier": int(events["expired"].sum()) if not events.empty else 0,
            "Batal spread": int(
                (~events["spread_ok"] & ~events["expired"]).sum()
            ) if not events.empty else 0,
        })
    return output, pd.DataFrame(rows)


def _probability_for_signals(probability, direction, timestamps):
    frame = probability.rename("probability").reset_index()
    pivot = frame.pivot_table(
        index="timestamp", columns="scenario", values="probability", aggfunc="first"
    )
    selected = pd.Series(index=timestamps, dtype=float)
    for trade_direction in ("BUY", "SELL"):
        mask = direction.eq(trade_direction)
        selected.loc[mask] = pivot[trade_direction].reindex(
            timestamps[mask], method="ffill"
        ).to_numpy()
    return selected


def _symmetric_classification_tables(frame, runs):
    output = {}
    for key, start, end in (
        ("validation", VALIDATION_START, VALIDATION_END),
        ("locked", LOCKED_START, LOCKED_END),
        ("reference", CONFIRMATION_START, CONFIRMATION_END),
    ):
        period = frame.loc[start:end]
        output[key] = pd.DataFrame([
            {
                "Kandidat": candidate,
                "Threshold": run["threshold"],
                **_symmetric_metrics(
                    period, run["probability"], run["threshold"]
                ),
            }
            for candidate, run in runs.items()
        ])
    return output


def _simulate_all(data, signals, best, config, start, end):
    return {
        candidate: _simulate_risk_control(
            data, frame.loc[start:end], best, config
        )
        for candidate, frame in signals.items()
    }


def _result_table(results, signals, start, end):
    return pd.DataFrame([
        {
            "Kandidat": candidate,
            "Sinyal tersedia": len(signals[candidate].loc[start:end]),
            **_metric_values(result),
        }
        for candidate, result in results.items()
    ])


def _period_validation(results, signals):
    periods = (
        ("Calibration 2022-2023", DEVELOPMENT_START, THRESHOLD_END),
        ("Model selection 2024", VALIDATION_START, VALIDATION_END),
        ("Locked confirmation 2025", LOCKED_START, LOCKED_END),
    )
    rows = []
    for label, start, end in periods:
        for candidate, result in results.items():
            selected = _trades_in_period(result.trades, start, end)
            rows.append({
                "Periode": label,
                "Kandidat": candidate,
                "Sinyal tersedia": len(signals[candidate].loc[start:end]),
                **_ledger_metric_values(selected),
            })
    return pd.DataFrame(rows)


def _fold_evaluation(results):
    rows = []
    for fold in FOLDS:
        for candidate, result in results.items():
            selected = _trades_in_period(
                result.trades, fold.test_start, fold.test_end
            )
            metrics = _ledger_metric_values(selected)
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


def _monte_carlo_summary(results):
    rows = []
    for candidate, result in results.items():
        _, summary = _safe_monte_carlo(result.trades)
        rows.append({"Kandidat": candidate, **summary})
    return pd.DataFrame(rows)


def _direction_audit(development_results, reference_results):
    rows = []
    for period, results in (
        ("Development 2022-2025", development_results),
        ("Historical reference 2026H1", reference_results),
    ):
        for candidate, result in results.items():
            for direction in ("BUY", "SELL"):
                trades = result.trades
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


def _regime_economic_audit(results, signals, regime_state):
    rows = []
    for candidate in LONG_CANDIDATES:
        frame = signals[candidate].loc[DEVELOPMENT_START:DEVELOPMENT_END]
        states = regime_state.reindex(frame.index, method="ffill").fillna("TRANSITION")
        trades = results[candidate].trades
        entry_times = _trade_entry_times(trades)
        if regime_state.index.tz is None:
            entry_times = entry_times.tz_convert(None)
        elif entry_times.tz is None:
            entry_times = entry_times.tz_localize(regime_state.index.tz)
        else:
            entry_times = entry_times.tz_convert(regime_state.index.tz)
        trade_states = regime_state.reindex(entry_times, method="ffill").fillna(
            "TRANSITION"
        )
        for regime in ("BULLISH", "BEARISH", "SIDEWAYS", "TRANSITION"):
            selected = frame.loc[states.eq(regime)]
            regime_trades = trades.loc[
                trade_states.eq(regime).to_numpy()
            ] if not trades.empty else trades
            rows.append({
                "Kandidat": candidate,
                "Regime": regime,
                "Sinyal": len(selected),
                **_ledger_metric_values(regime_trades),
            })
    return pd.DataFrame(rows)


def _trade_entry_times(trades):
    if trades.empty or "Tanggal entry" not in trades:
        return pd.DatetimeIndex([], tz="UTC")
    timestamps = pd.to_datetime(trades["Tanggal entry"], utc=True, errors="coerce")
    return pd.DatetimeIndex(timestamps)


def _trades_in_period(trades, start, end):
    if trades.empty:
        return trades.copy()
    timestamps = _trade_entry_times(trades)
    start_utc = pd.Timestamp(start)
    end_utc = pd.Timestamp(end)
    start_utc = start_utc.tz_localize("UTC") if start_utc.tzinfo is None else start_utc.tz_convert("UTC")
    end_utc = end_utc.tz_localize("UTC") if end_utc.tzinfo is None else end_utc.tz_convert("UTC")
    valid = timestamps.notna() & (timestamps >= start_utc) & (timestamps <= end_utc)
    return trades.loc[valid].copy()


def _ledger_metric_values(trades):
    if trades.empty:
        return {
            "Equity akhir": 1000.0,
            "Growth (%)": 0.0,
            "Max drawdown": 0.0,
            "Max drawdown (%)": 0.0,
            "Profit factor": np.nan,
            "Transaksi": 0.0,
            "Win rate (%)": np.nan,
            "Max open posisi": 0.0,
            "Total swap": 0.0,
            "Biaya spread": 0.0,
            "Biaya slippage": 0.0,
            "Entry diblokir": 0.0,
        }
    ordered = trades.copy()
    if "Tanggal tutup" in ordered:
        ordered = ordered.assign(
            _closed_at=pd.to_datetime(
                ordered["Tanggal tutup"], utc=True, errors="coerce"
            )
        ).sort_values("_closed_at")
    net = pd.to_numeric(ordered["Net P/L"], errors="coerce").fillna(0.0)
    equity = pd.concat(
        [pd.Series([1000.0]), 1000.0 + net.cumsum().reset_index(drop=True)],
        ignore_index=True,
    )
    drawdown = equity.cummax() - equity
    profit = float(net[net > 0].sum())
    loss = float(-net[net < 0].sum())
    total_net = float(net.sum())
    return {
        "Equity akhir": 1000.0 + total_net,
        "Growth (%)": total_net / 1000.0 * 100,
        "Max drawdown": float(drawdown.max()),
        "Max drawdown (%)": float(drawdown.max()) / 1000.0 * 100,
        "Profit factor": profit / loss if loss > 0 else np.inf,
        "Transaksi": float(len(ordered)),
        "Win rate (%)": float(net.gt(0).mean() * 100),
        "Max open posisi": 1.0,
        "Total swap": float(
            pd.to_numeric(ordered.get("Swap", 0.0), errors="coerce").fillna(0).sum()
        ),
        "Biaya spread": float(
            pd.to_numeric(
                ordered.get("Biaya spread", 0.0), errors="coerce"
            ).fillna(0).sum()
        ),
        "Biaya slippage": float(
            pd.to_numeric(
                ordered.get("Biaya slippage", 0.0), errors="coerce"
            ).fillna(0).sum()
        ),
        "Entry diblokir": 0.0,
    }


def _retention_table(signals, reference):
    dev_total = max(len(reference.loc[DEVELOPMENT_START:DEVELOPMENT_END]), 1)
    ref_total = max(len(reference.loc[CONFIRMATION_START:CONFIRMATION_END]), 1)
    return pd.DataFrame([
        {
            "Kandidat": candidate,
            "Retensi development (%)": len(frame.loc[DEVELOPMENT_START:DEVELOPMENT_END]) / dev_total * 100,
            "Retensi 2026H1 (%)": len(frame.loc[CONFIRMATION_START:CONFIRMATION_END]) / ref_total * 100,
        }
        for candidate, frame in signals.items()
    ])


def _long_decision_table(development, periods, folds, monte_carlo, regime_economic):
    dev = development.set_index("Kandidat")
    period = periods.set_index(["Periode", "Kandidat"])
    mc = monte_carlo.set_index("Kandidat")
    regime = regime_economic.set_index(["Kandidat", "Regime"])
    rows = []
    for candidate in LONG_CANDIDATES:
        primary = folds[
            folds["Kandidat"].eq(candidate)
            & folds["Kelompok"].eq("Primary validation")
        ]
        criteria = {
            "Growth positif": float(dev.loc[candidate, "Growth (%)"]) > 0,
            "PF >= 1.50": float(dev.loc[candidate, "Profit factor"]) >= 1.50,
            "DD <= 10%": float(dev.loc[candidate, "Max drawdown (%)"]) <= 10,
            "Bearish DD <= 5%": float(regime.loc[(candidate, "BEARISH"), "Max drawdown (%)"]) <= 5,
            "Sideways growth >= -2%": float(regime.loc[(candidate, "SIDEWAYS"), "Growth (%)"]) >= -2,
            "2024 positif": float(period.loc[("Model selection 2024", candidate), "Growth (%)"]) > 0,
            "2025 positif": float(period.loc[("Locked confirmation 2025", candidate), "Growth (%)"]) > 0,
            "Primary fold >= 6/8": int(primary["Profitable"].sum()) >= 6,
            "Monte Carlo rugi <= 10%": float(
                mc.loc[candidate, "Probabilitas equity akhir < modal awal (%)"]
            ) <= 10,
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


def _symmetric_decision_table(classification, development, periods, folds, monte_carlo, direction, retention):
    cls = classification.set_index("Kandidat")
    dev = development.set_index("Kandidat")
    period = periods.set_index(["Periode", "Kandidat"])
    mc = monte_carlo.set_index("Kandidat")
    retained = retention.set_index("Kandidat")
    rows = []
    for candidate in SYMMETRIC_CANDIDATES:
        primary = folds[
            folds["Kandidat"].eq(candidate)
            & folds["Kelompok"].eq("Primary validation")
        ]
        candidate_direction = direction[
            direction["Kandidat"].eq(candidate)
            & direction["Periode"].eq("Development 2022-2025")
        ].set_index("Arah")
        total_profit = candidate_direction["Net P/L"].clip(lower=0).sum()
        concentration = (
            candidate_direction["Net P/L"].clip(lower=0).max() / total_profit * 100
            if total_profit > 0 else 100.0
        )
        criteria = {
            "BUY precision >= 55%": float(cls.loc[candidate, "BUY precision"]) >= 0.55,
            "SELL precision >= 55%": float(cls.loc[candidate, "SELL precision"]) >= 0.55,
            "BUY recall >= 40%": float(cls.loc[candidate, "BUY recall"]) >= 0.40,
            "SELL recall >= 40%": float(cls.loc[candidate, "SELL recall"]) >= 0.40,
            "Precision gap <= 10pp": float(cls.loc[candidate, "Precision gap (pp)"]) <= 10,
            "Growth positif": float(dev.loc[candidate, "Growth (%)"]) > 0,
            "PF >= 1.50": float(dev.loc[candidate, "Profit factor"]) >= 1.50,
            "DD <= 10%": float(dev.loc[candidate, "Max drawdown (%)"]) <= 10,
            "Retensi >= 50%": float(retained.loc[candidate, "Retensi development (%)"]) >= 50,
            "2024 dan 2025 positif": (
                float(period.loc[("Model selection 2024", candidate), "Growth (%)"]) > 0
                and float(period.loc[("Locked confirmation 2025", candidate), "Growth (%)"]) > 0
            ),
            "Primary fold >= 6/8": int(primary["Profitable"].sum()) >= 6,
            "Monte Carlo rugi <= 10%": float(
                mc.loc[candidate, "Probabilitas equity akhir < modal awal (%)"]
            ) <= 10,
            "Profit satu arah <= 80%": concentration <= 80,
        }
        rows.append({
            "Kandidat": candidate,
            **criteria,
            "Primary fold profitable": int(primary["Profitable"].sum()),
            "Konsentrasi profit arah terbesar (%)": concentration,
            "Kriteria lolos": int(sum(criteria.values())),
            "Total kriteria": len(criteria),
            "Lulus": bool(all(criteria.values())),
        })
    return pd.DataFrame(rows)


def _long_ranking(development, reference, decisions, regime):
    dev = development.set_index("Kandidat")
    ref = reference.set_index("Kandidat")
    decision = decisions.set_index("Kandidat")
    regime_indexed = regime.set_index(["Kandidat", "Regime"])
    rows = []
    for candidate in LONG_CANDIDATES:
        rows.append({
            "Kandidat": candidate,
            "Growth development (%)": float(dev.loc[candidate, "Growth (%)"]),
            "PF development": float(dev.loc[candidate, "Profit factor"]),
            "DD development (%)": float(dev.loc[candidate, "Max drawdown (%)"]),
            "Transaksi": int(dev.loc[candidate, "Transaksi"]),
            "Bearish growth (%)": float(regime_indexed.loc[(candidate, "BEARISH"), "Growth (%)"]),
            "Bearish DD (%)": float(regime_indexed.loc[(candidate, "BEARISH"), "Max drawdown (%)"]),
            "Sideways growth (%)": float(regime_indexed.loc[(candidate, "SIDEWAYS"), "Growth (%)"]),
            "Growth 2026H1 (%)": float(ref.loc[candidate, "Growth (%)"]),
            "Kriteria lolos": int(decision.loc[candidate, "Kriteria lolos"]),
            "Lulus": bool(decision.loc[candidate, "Lulus"]),
        })
    ranking = pd.DataFrame(rows).sort_values(
        ["Lulus", "Kriteria lolos", "PF development", "DD development (%)"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    ranking.insert(0, "Peringkat", np.arange(1, len(ranking) + 1))
    return ranking


def _symmetric_ranking(classification, development, reference, decisions, retention):
    cls = classification.set_index("Kandidat")
    dev = development.set_index("Kandidat")
    ref = reference.set_index("Kandidat")
    decision = decisions.set_index("Kandidat")
    retained = retention.set_index("Kandidat")
    rows = []
    for candidate in SYMMETRIC_CANDIDATES:
        rows.append({
            "Kandidat": candidate,
            "BUY precision": float(cls.loc[candidate, "BUY precision"]),
            "SELL precision": float(cls.loc[candidate, "SELL precision"]),
            "Worst precision": float(cls.loc[candidate, "Worst precision"]),
            "Worst recall": float(cls.loc[candidate, "Worst recall"]),
            "Precision gap (pp)": float(cls.loc[candidate, "Precision gap (pp)"]),
            "Growth development (%)": float(dev.loc[candidate, "Growth (%)"]),
            "PF development": float(dev.loc[candidate, "Profit factor"]),
            "DD development (%)": float(dev.loc[candidate, "Max drawdown (%)"]),
            "Transaksi": int(dev.loc[candidate, "Transaksi"]),
            "Retensi (%)": float(retained.loc[candidate, "Retensi development (%)"]),
            "Growth 2026H1 (%)": float(ref.loc[candidate, "Growth (%)"]),
            "Kriteria lolos": int(decision.loc[candidate, "Kriteria lolos"]),
            "Lulus": bool(decision.loc[candidate, "Lulus"]),
        })
    ranking = pd.DataFrame(rows).sort_values(
        ["Lulus", "Kriteria lolos", "Worst precision", "PF development", "DD development (%)"],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)
    ranking.insert(0, "Peringkat", np.arange(1, len(ranking) + 1))
    return ranking


def _stress_summary(data, signals, best, config, candidates):
    rows = []
    for candidate in candidates:
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
            "Worst DD (%)": float(stress["Max drawdown (%)"].max()),
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
    net = pd.to_numeric(trades["Net P/L"], errors="coerce").fillna(0)
    profit = float(net[net > 0].sum())
    loss = float(-net[net < 0].sum())
    return {
        "Transaksi": int(len(trades)),
        "Net P/L": float(net.sum()),
        "Profit factor": profit / loss if loss > 0 else np.inf,
        "Win rate (%)": float(net.gt(0).mean() * 100),
    }


def _selected_symmetric_table(runs):
    return pd.DataFrame([
        {
            "Kandidat": candidate,
            "Threshold": run["threshold"],
            "Moderate threshold": run["moderate_threshold"],
        }
        for candidate, run in runs.items()
    ])
