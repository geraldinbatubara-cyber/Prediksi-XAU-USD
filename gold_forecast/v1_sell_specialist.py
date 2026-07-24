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
    SIGNED_FEATURES,
    SYMMETRIC_FEATURES,
    _apply_symmetric_calibration,
    _class_weights,
    _ledger_metric_values,
    _monte_carlo_summary,
    _stress_summary,
    _trades_in_period,
)
from gold_forecast.v1_entry_quality_path import FOLDS, _unique_signals
from gold_forecast.v1_fixed_delay import _build_fixed_delay_signals
from gold_forecast.v1_regime_classifier import (
    _classifier_frame,
    _ohlc_bars,
    _timeframe_features,
)
from gold_forecast.v1_regime_classifier_v3 import _fit_platt, _m15_alignment
from gold_forecast.v1_risk_control import (
    RiskControlConfig,
    _metric_values,
    _simulate_risk_control,
)
from gold_forecast.v1_signal_quality import _entry_features
from gold_forecast.v1_trend_strength_stability import _extended_data_audit
from gold_forecast.v1_unified_benchmark import _unified_best


TRAIN_START = pd.Timestamp("2022-01-01")
TRAIN_END = pd.Timestamp("2022-12-31 23:59:59")
CALIBRATION_START = pd.Timestamp("2023-01-01")
CALIBRATION_END = pd.Timestamp("2023-06-30 23:59:59")
THRESHOLD_START = pd.Timestamp("2023-07-01")
THRESHOLD_END = pd.Timestamp("2023-12-31 23:59:59")
SELECTION_START = pd.Timestamp("2024-01-01")
SELECTION_END = pd.Timestamp("2024-12-31 23:59:59")
LOCKED_START = pd.Timestamp("2025-01-01")
LOCKED_END = pd.Timestamp("2025-12-31 23:59:59")
REFERENCE_START = pd.Timestamp("2026-01-01")
REFERENCE_END = pd.Timestamp("2026-06-30 23:59:59")
DEVELOPMENT_START = TRAIN_START
DEVELOPMENT_END = LOCKED_END

CANDIDATES = (
    "Bear Breakdown Control",
    "SELL Logistic",
    "SELL Gradient Boosting",
    "SELL Probability Ensemble",
    "Dual-Horizon Consensus",
    "Adaptive Bear Confirmation",
)
HORIZONS = (4, 8, 12)


def run_v1_sell_specialist_lab(
    gold_m1: pd.DataFrame,
    frozen_payload: dict[str, object],
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
        "SELL Specialist v5",
        "SELL-only outcome engine",
        max_total_positions=1,
        max_same_direction=1,
    )
    spread_limit = float(
        data.loc[DEVELOPMENT_START:DEVELOPMENT_END, "SpreadPoints"].quantile(0.90)
    )
    frame = _sell_outcome_frame(data)
    candidate_frame = _bearish_candidate_frame(frame)
    runs, model_selection = _train_models(candidate_frame)
    raw_signals = _raw_sell_signals(candidate_frame, best)
    signals, funnel = _candidate_signals(
        data,
        raw_signals,
        candidate_frame,
        runs,
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
        development_results, signals, DEVELOPMENT_START, DEVELOPMENT_END
    )
    historical_reference = _result_table(
        reference_results, signals, REFERENCE_START, REFERENCE_END
    )
    periods = _period_validation(development_results, signals)
    folds = _fold_evaluation(development_results)
    classification = _classification_tables(candidate_frame, runs)
    monte_carlo = _monte_carlo_summary(development_results)
    concentration = _profit_concentration(development_results)
    decisions = _decision_table(
        development,
        periods,
        folds,
        monte_carlo,
        concentration,
    )
    ranking = _ranking(
        development,
        historical_reference,
        periods,
        classification["locked"],
        decisions,
    )
    winner = str(ranking.iloc[0]["Kandidat"])
    stress = _stress_summary(
        development_data,
        signals,
        best,
        config,
        [winner],
    )

    return {
        "methodology": {
            "Name": "v1 Directional Specialists Lab v5 - Bearish Outcome Engine",
            "Mandat": "Mesin hanya boleh SELL atau ABSTAIN; tidak pernah membuka BUY.",
            "Candidate universe": (
                "Candle H1 dengan minimal dua dari tiga bukti bearish: close di bawah "
                "EMA cepat, EMA cepat di bawah EMA lambat, dan momentum 6 jam negatif."
            ),
            "Outcome label": (
                "TP USD 25 harus tersentuh sebelum SL USD 10 pada horizon 4/8/12 jam; "
                "spread broker ikut membentuk barrier SELL."
            ),
            "Train": "2022",
            "Probability calibration": "2023H1",
            "Threshold calibration": "2023H2",
            "Model selection": "2024",
            "Locked confirmation": "2025",
            "Historical reference": "2026H1, tidak menentukan pemenang",
            "Execution": (
                "Equity USD 1.000 | lot 0.01 | maksimal 1 SELL | Fixed Delay 5m | "
                "spread, slippage, TP/SL broker-aware"
            ),
            "Baseline lock": (
                "Baseline v1 dan BUY Specialist v4 tidak diubah oleh eksperimen ini."
            ),
        },
        "data_audit": _extended_data_audit(data),
        "candidate_audit": _candidate_audit(candidate_frame),
        "model_selection": model_selection,
        "classification_threshold": classification["threshold"],
        "classification_selection": classification["selection"],
        "classification_locked": classification["locked"],
        "classification_reference": classification["reference"],
        "funnel": funnel,
        "development": development,
        "period_validation": periods,
        "historical_reference": historical_reference,
        "folds": folds,
        "monte_carlo_summary": monte_carlo,
        "profit_concentration": concentration,
        "stress_summary": stress,
        "decisions": decisions,
        "ranking": ranking,
        "winner": winner,
    }


def _sell_outcome_frame(data: pd.DataFrame) -> pd.DataFrame:
    base = _classifier_frame(data).drop(columns=["label"], errors="ignore")
    h1 = _ohlc_bars(data, "1h")
    h1_features = _timeframe_features(h1, "h1")
    frame = base.copy()
    frame["donchian_position_centered"] = (
        pd.to_numeric(frame["donchian_position"], errors="coerce") - 0.5
    ) * 2
    for feature in SIGNED_FEATURES:
        frame[feature] = -pd.to_numeric(frame[feature], errors="coerce")
    frame["breakout_support"] = pd.to_numeric(
        frame["breakout_down"], errors="coerce"
    )
    frame["raw_close"] = h1["Close"].reindex(frame.index)
    frame["raw_ema_fast"] = h1["Close"].ewm(span=10, adjust=False).mean().reindex(
        frame.index
    )
    frame["raw_ema_slow"] = h1["Close"].ewm(span=30, adjust=False).mean().reindex(
        frame.index
    )
    frame["raw_momentum_6"] = h1["Close"].pct_change(6).reindex(frame.index)
    frame["raw_atr"] = h1_features["atr"].reindex(frame.index)
    spread = (
        h1["SpreadPoints"].median()
        if "SpreadPoints" in h1
        else data["SpreadPoints"].median()
    )
    spread_price = float(spread) * 0.01
    for horizon in HORIZONS:
        frame[f"target_{horizon}h"] = _sell_barrier_target(
            h1, horizon, spread_price, take_profit=25.0, stop_loss=10.0
        ).reindex(frame.index)
    return frame.dropna(
        subset=[
            *SYMMETRIC_FEATURES,
            "raw_close",
            "raw_ema_fast",
            "raw_ema_slow",
            "raw_momentum_6",
            *[f"target_{horizon}h" for horizon in HORIZONS],
        ]
    )


def _sell_barrier_target(
    h1: pd.DataFrame,
    horizon: int,
    spread_price: float,
    *,
    take_profit: float,
    stop_loss: float,
) -> pd.Series:
    entry = h1["Close"]
    first_tp = pd.Series(np.inf, index=h1.index)
    first_sl = pd.Series(np.inf, index=h1.index)
    for step in range(1, horizon + 1):
        future_low = h1["Low"].shift(-step) + spread_price
        future_high = h1["High"].shift(-step) + spread_price
        first_tp = first_tp.mask(
            np.isinf(first_tp) & future_low.le(entry - take_profit),
            float(step),
        )
        first_sl = first_sl.mask(
            np.isinf(first_sl) & future_high.ge(entry + stop_loss),
            float(step),
        )
    target = first_tp.lt(first_sl).astype(float)
    unavailable = h1["Close"].shift(-horizon).isna()
    target.loc[unavailable] = np.nan
    return target


def _bearish_candidate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    evidence = pd.concat(
        [
            frame["raw_close"].lt(frame["raw_ema_fast"]),
            frame["raw_ema_fast"].lt(frame["raw_ema_slow"]),
            frame["raw_momentum_6"].lt(0),
        ],
        axis=1,
    ).sum(axis=1)
    candidates = frame.loc[evidence.ge(2)].copy()
    candidates["bearish_evidence"] = evidence.loc[candidates.index]
    candidates["strict_breakdown"] = (
        candidates["raw_close"].lt(candidates["raw_ema_fast"])
        & candidates["raw_ema_fast"].lt(candidates["raw_ema_slow"])
        & candidates["raw_momentum_6"].lt(0)
        & candidates["breakout_support"].gt(0)
    )
    candidates["trend_strength"] = (
        candidates["adx"].ge(22)
        & candidates["efficiency"].ge(0.25)
        & candidates["choppiness"].le(60)
    )
    return _cooldown_rows(candidates, hours=4)


def _cooldown_rows(frame: pd.DataFrame, hours: int) -> pd.DataFrame:
    keep = []
    last = None
    for timestamp in frame.index:
        if last is None or timestamp >= last + pd.Timedelta(hours=hours):
            keep.append(timestamp)
            last = timestamp
    return frame.loc[keep].copy()


def _train_models(frame: pd.DataFrame):
    train = frame.loc[TRAIN_START:TRAIN_END]
    calibration = frame.loc[CALIBRATION_START:CALIBRATION_END]
    threshold_period = frame.loc[THRESHOLD_START:THRESHOLD_END]
    runs = {}
    rows = []
    for horizon in HORIZONS:
        target = f"target_{horizon}h"
        logistic = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                C=0.4,
                random_state=50 + horizon,
            ),
        )
        boosting = HistGradientBoostingClassifier(
            learning_rate=0.04,
            max_iter=180,
            max_depth=3,
            min_samples_leaf=30,
            l2_regularization=1.5,
            random_state=60 + horizon,
        )
        logistic.fit(train[list(SYMMETRIC_FEATURES)], train[target].astype(int))
        boosting.fit(
            train[list(SYMMETRIC_FEATURES)],
            train[target].astype(int),
            sample_weight=_class_weights(train[target]),
        )
        raw_logistic = _predict(logistic, frame)
        raw_boosting = _predict(boosting, frame)
        raw_ensemble = (raw_logistic + raw_boosting) / 2
        for model_name, raw in (
            ("Logistic", raw_logistic),
            ("Gradient Boosting", raw_boosting),
            ("Ensemble", raw_ensemble),
        ):
            calibrator = _fit_platt(
                raw.reindex(calibration.index),
                calibration[target].astype(int),
            )
            probability = _apply_symmetric_calibration(raw, calibrator)
            threshold, audit = _select_threshold(
                threshold_period[target].astype(int),
                probability.reindex(threshold_period.index),
            )
            runs[(model_name, horizon)] = {
                "probability": probability,
                "threshold": threshold,
            }
            rows.append(
                {
                    "Model": model_name,
                    "Horizon": f"{horizon} jam",
                    "Threshold": threshold,
                    **audit,
                }
            )
    return runs, pd.DataFrame(rows)


def _predict(model, frame):
    return pd.Series(
        model.predict_proba(frame[list(SYMMETRIC_FEATURES)])[:, 1],
        index=frame.index,
    )


def _select_threshold(truth: pd.Series, probability: pd.Series):
    rows = []
    thresholds = sorted(
        {
            float(probability.quantile(quantile))
            for quantile in (0.50, 0.60, 0.70, 0.80, 0.90, 0.95)
        }
    )
    for threshold in thresholds:
        prediction = probability.ge(threshold)
        selected = truth.loc[prediction]
        precision = float(selected.mean()) if len(selected) else 0.0
        recall = float(
            ((prediction) & truth.eq(1)).sum() / max(int(truth.eq(1).sum()), 1)
        )
        expected_value = precision * 25 - (1 - precision) * 10
        eligible = len(selected) >= 12
        score = expected_value + recall * 4 + min(len(selected), 40) * 0.02
        rows.append((float(threshold), precision, recall, len(selected), score, eligible))
    eligible = [row for row in rows if row[-1]]
    selected = max(eligible or rows, key=lambda row: row[4])
    return selected[0], {
        "Precision threshold": selected[1],
        "Recall threshold": selected[2],
        "Sinyal threshold": selected[3],
        "Expected value proxy": selected[1] * 25 - (1 - selected[1]) * 10,
    }


def _raw_sell_signals(frame, best):
    threshold = max(float(best["Threshold entry (%)"]), 0.15)
    output = pd.DataFrame(index=frame.index)
    output["prediction"] = frame["raw_close"] * (1 - threshold / 100)
    output["reference_price"] = frame["raw_close"]
    output["expected_change_pct"] = -threshold
    output["lot"] = 0.01
    output["signal_date"] = frame.index
    output["strategy"] = "SELL Specialist v5 candidate"
    return output


def _candidate_signals(data, raw, frame, runs, best, spread_limit):
    probability = {
        "SELL Logistic": runs[("Logistic", 8)]["probability"],
        "SELL Gradient Boosting": runs[("Gradient Boosting", 8)]["probability"],
        "SELL Probability Ensemble": runs[("Ensemble", 8)]["probability"],
    }
    masks = {
        "Bear Breakdown Control": frame["strict_breakdown"] & frame["trend_strength"],
        "SELL Logistic": probability["SELL Logistic"].ge(
            runs[("Logistic", 8)]["threshold"]
        ),
        "SELL Gradient Boosting": probability["SELL Gradient Boosting"].ge(
            runs[("Gradient Boosting", 8)]["threshold"]
        ),
        "SELL Probability Ensemble": probability["SELL Probability Ensemble"].ge(
            runs[("Ensemble", 8)]["threshold"]
        ),
        "Dual-Horizon Consensus": (
            runs[("Ensemble", 4)]["probability"].ge(
                runs[("Ensemble", 4)]["threshold"]
            )
            & runs[("Ensemble", 12)]["probability"].ge(
                runs[("Ensemble", 12)]["threshold"]
            )
        ),
    }
    adaptive_probability = runs[("Ensemble", 8)]["probability"]
    adaptive_threshold = pd.Series(
        runs[("Ensemble", 8)]["threshold"] * 1.10,
        index=frame.index,
    )
    adaptive_threshold.loc[frame["trend_strength"]] *= 0.82
    entry_features = _entry_features(data)
    sell_direction = pd.Series("SELL", index=frame.index)
    m15 = _m15_alignment(entry_features.reindex(frame.index), sell_direction)
    masks["Adaptive Bear Confirmation"] = adaptive_probability.ge(
        adaptive_threshold.clip(lower=0.001)
    ) & (frame["trend_strength"] | m15)

    output = {}
    rows = []
    for candidate, mask in masks.items():
        before = raw.loc[mask.reindex(raw.index).fillna(False)].copy()
        delayed, events = _build_fixed_delay_signals(
            data, before, best, 5, spread_limit
        )
        output[candidate] = _unique_signals(delayed)
        rows.append(
            {
                "Kandidat": candidate,
                "Kandidat bearish H1": len(raw),
                "Lolos model": len(before),
                "Lolos Fixed Delay": len(output[candidate]),
                "Batal barrier": int(events["expired"].sum()) if not events.empty else 0,
                "Batal spread": int(
                    (~events["spread_ok"] & ~events["expired"]).sum()
                ) if not events.empty else 0,
            }
        )
    return output, pd.DataFrame(rows)


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
        "threshold": (THRESHOLD_START, THRESHOLD_END),
        "selection": (SELECTION_START, SELECTION_END),
        "locked": (LOCKED_START, LOCKED_END),
        "reference": (REFERENCE_START, REFERENCE_END),
    }
    output = {}
    for key, (start, end) in periods.items():
        selected = frame.loc[start:end]
        rows = []
        for candidate in CANDIDATES:
            probability, threshold, target = _candidate_probability(
                candidate, selected, runs
            )
            prediction = probability.ge(threshold)
            truth = selected[target].astype(int)
            rows.append(
                {
                    "Kandidat": candidate,
                    "Horizon label": target.replace("target_", "").replace("h", " jam"),
                    "Threshold": threshold,
                    "Precision": precision_score(truth, prediction, zero_division=0),
                    "Recall": recall_score(truth, prediction, zero_division=0),
                    "Coverage (%)": float(prediction.mean() * 100),
                    "Brier": float(brier_score_loss(truth, probability)),
                }
            )
        output[key] = pd.DataFrame(rows)
    return output


def _candidate_probability(candidate, frame, runs):
    if candidate == "Bear Breakdown Control":
        probability = (
            frame["strict_breakdown"] & frame["trend_strength"]
        ).astype(float)
        return probability, 0.5, "target_8h"
    if candidate == "SELL Logistic":
        run = runs[("Logistic", 8)]
        return run["probability"].reindex(frame.index), run["threshold"], "target_8h"
    if candidate == "SELL Gradient Boosting":
        run = runs[("Gradient Boosting", 8)]
        return run["probability"].reindex(frame.index), run["threshold"], "target_8h"
    if candidate == "SELL Probability Ensemble":
        run = runs[("Ensemble", 8)]
        return run["probability"].reindex(frame.index), run["threshold"], "target_8h"
    if candidate == "Dual-Horizon Consensus":
        short = runs[("Ensemble", 4)]
        long = runs[("Ensemble", 12)]
        probability = pd.concat(
            [
                short["probability"].reindex(frame.index),
                long["probability"].reindex(frame.index),
            ],
            axis=1,
        ).min(axis=1)
        return probability, max(short["threshold"], long["threshold"]), "target_12h"
    run = runs[("Ensemble", 8)]
    threshold = run["threshold"]
    return run["probability"].reindex(frame.index), threshold, "target_8h"


def _profit_concentration(results):
    rows = []
    for candidate, result in results.items():
        if result.trades.empty:
            concentration = 100.0
        else:
            positive = pd.to_numeric(
                result.trades["Net P/L"], errors="coerce"
            ).dropna()
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


def _ranking(development, reference, periods, classification, decisions):
    dev = development.set_index("Kandidat")
    ref = reference.set_index("Kandidat")
    period = periods.set_index(["Periode", "Kandidat"])
    cls = classification.set_index("Kandidat")
    decision = decisions.set_index("Kandidat")
    rows = []
    for candidate in CANDIDATES:
        rows.append(
            {
                "Kandidat": candidate,
                "Growth development (%)": float(dev.loc[candidate, "Growth (%)"]),
                "PF development": float(dev.loc[candidate, "Profit factor"]),
                "DD development (%)": float(dev.loc[candidate, "Max drawdown (%)"]),
                "Transaksi development": int(dev.loc[candidate, "Transaksi"]),
                "Growth selection 2024 (%)": float(
                    period.loc[("Model selection 2024", candidate), "Growth (%)"]
                ),
                "Growth locked 2025 (%)": float(
                    period.loc[("Locked confirmation 2025", candidate), "Growth (%)"]
                ),
                "Precision locked": float(cls.loc[candidate, "Precision"]),
                "Growth 2026H1 (%)": float(ref.loc[candidate, "Growth (%)"]),
                "Kriteria lolos": int(decision.loc[candidate, "Kriteria lolos"]),
                "Lulus": bool(decision.loc[candidate, "Lulus"]),
            }
        )
    ranking = pd.DataFrame(rows).sort_values(
        [
            "Lulus",
            "Kriteria lolos",
            "Growth selection 2024 (%)",
            "PF development",
            "DD development (%)",
        ],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)
    ranking.insert(0, "Peringkat", np.arange(1, len(ranking) + 1))
    return ranking


def _candidate_audit(frame):
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
        rows.append(
            {
                "Periode": label,
                "Kandidat bearish H1": len(selected),
                "TP-before-SL 4h (%)": float(selected["target_4h"].mean() * 100),
                "TP-before-SL 8h (%)": float(selected["target_8h"].mean() * 100),
                "TP-before-SL 12h (%)": float(selected["target_12h"].mean() * 100),
            }
        )
    return pd.DataFrame(rows)
