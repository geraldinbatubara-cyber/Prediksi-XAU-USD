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
from gold_forecast.v1_sell_specialist_v6 import _profit_concentration
from gold_forecast.v1_trend_strength_stability import _extended_data_audit
from gold_forecast.v1_unified_benchmark import _unified_best


CANDIDATES = (
    "Range Control",
    "Boundary Rejection",
    "Outcome Filter",
    "Midpoint Conservative",
    "Breakout Guard",
    "Adaptive Range Ensemble",
)
MODEL_FEATURES = (
    "adx",
    "efficiency",
    "choppiness",
    "trend_strength",
    "slope",
    "range_width_atr",
    "touch_lower",
    "touch_upper",
    "position_from_edge",
    "rsi_extreme",
    "rejection_body_atr",
    "distance_edge_atr",
    "reward_risk",
    "atr_percentile",
)


def run_v1_sideways_specialist_lab(
    gold_m1: pd.DataFrame,
    frozen_payload: dict[str, object],
    defense_payload: dict[str, object] | None = None,
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
        "Sideways Specialist v1",
        "Range detection and mean reversion",
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
    model_run, model_selection = _train_outcome_model(opportunities)
    signals, funnel = _candidate_signals(
        data, opportunities, model_run, best, spread_limit
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
    classification = _classification_tables(opportunities, model_run)
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
            "Name": "v1 Sideways Specialist Lab v1 - Range Detection & Mean Reversion",
            "Mandat": (
                "BUY lower boundary, SELL upper boundary, atau ABSTAIN. "
                "Tidak mengikuti breakout."
            ),
            "Range gate": (
                "Minimal 4/5 bukti sideways: ADX rendah, efficiency rendah, "
                "choppiness tinggi, MA gap kecil, dan slope MA lambat datar. "
                "Range harus memiliki minimal dua sentuhan pada kedua sisi."
            ),
            "Entry": (
                "Reversal M15 di boundary dengan RSI, reward/risk minimum, spread "
                "normal, dan range belum mengalami breakout acceptance."
            ),
            "Exit": (
                "TP menuju midpoint dengan cap USD 15, SL di luar boundary dengan "
                "buffer ATR dan cap USD 12, serta time stop 12 jam."
            ),
            "Label": (
                "TP-midpoint sebelum boundary invalidation dalam 12 jam; MFE, MAE, "
                "dan time-to-TP diaudit terpisah untuk BUY dan SELL."
            ),
            "Train": "2022",
            "Probability calibration": "2023H1",
            "Threshold calibration": "2023H2",
            "Model selection": "2024 saja",
            "Locked confirmation": "2025",
            "Historical reference": "2026H1, tidak menentukan pemenang",
            "Execution": (
                "Equity USD 1.000 | lot 0.01 | maksimal satu posisi total | "
                "spread/slippage MT5 | swap BUY berlaku | SELL swap nol."
            ),
            "Baseline lock": (
                "Baseline v1, BUY Specialist v4, eksperimen SELL, dan seluruh "
                "ledger paper live tidak dibaca atau ditulis."
            ),
        },
        "data_audit": _extended_data_audit(data),
        "range_audit": _range_audit(range_frame),
        "opportunity_audit": _opportunity_audit(opportunities),
        "path_audit": _path_audit(opportunities),
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
        "defense_reference": _defense_reference(defense_payload),
    }


def _range_quality_frame(features, h1):
    frame = features.copy()
    hourly = h1.reindex(frame.index, method="ffill")
    atr = pd.to_numeric(frame["atr"], errors="coerce").clip(lower=0.01)
    width = frame["range_high"] - frame["range_low"]
    frame["range_width_atr"] = width / atr
    lower_touch = hourly["Low"].le(frame["range_low"] + 0.18 * atr)
    upper_touch = hourly["High"].ge(frame["range_high"] - 0.18 * atr)
    frame["touch_lower"] = (
        lower_touch.resample("1h").max().rolling(30, min_periods=20).sum()
        .reindex(frame.index, method="ffill")
    )
    frame["touch_upper"] = (
        upper_touch.resample("1h").max().rolling(30, min_periods=20).sum()
        .reindex(frame.index, method="ffill")
    )
    checks = pd.concat(
        [
            frame["adx"].le(22),
            frame["efficiency"].le(0.35),
            frame["choppiness"].ge(55),
            frame["trend_strength"].le(0.45),
            frame["slope"].le(0.16),
        ],
        axis=1,
    )
    frame["sideways_votes"] = checks.fillna(False).sum(axis=1)
    frame["range_quality"] = (
        frame["sideways_votes"].ge(3)
        & frame["range_width_atr"].between(1.2, 6.0)
        & frame["touch_lower"].ge(2)
        & frame["touch_upper"].ge(2)
    )
    frame["range_confirmed"] = (
        frame["range_quality"].rolling(60, min_periods=60).sum().ge(60)
    )
    close = hourly["Close"]
    frame["breakout_up"] = close.gt(frame["range_high"] + 0.15 * atr)
    frame["breakout_down"] = close.lt(frame["range_low"] - 0.15 * atr)
    frame["breakout_guard"] = ~(
        frame["breakout_up"].rolling(180, min_periods=1).max().astype(bool)
        | frame["breakout_down"].rolling(180, min_periods=1).max().astype(bool)
    )
    h1_atr = atr.resample("1h").last()
    frame["atr_percentile"] = (
        h1_atr.rolling(240, min_periods=80)
        .rank(pct=True)
        .reindex(frame.index, method="ffill")
    )
    return frame.replace([np.inf, -np.inf], np.nan)


def _mean_reversion_opportunities(data, frame, m15, spread_limit):
    rows = []
    last_signal = {"BUY": pd.Timestamp.min, "SELL": pd.Timestamp.min}
    for timestamp, candle in m15.iterrows():
        if timestamp not in frame.index:
            continue
        state = frame.loc[timestamp]
        if not bool(state.get("range_confirmed", False)):
            continue
        range_high = float(state.get("range_high", np.nan))
        range_low = float(state.get("range_low", np.nan))
        range_mid = float(state.get("range_mid", np.nan))
        atr = float(state.get("atr", np.nan))
        close = float(candle["Close"])
        rsi = float(candle.get("rsi", np.nan))
        if not np.isfinite(
            [range_high, range_low, range_mid, atr, close, rsi]
        ).all():
            continue
        width = range_high - range_low
        if width <= 0:
            continue
        direction = None
        if (
            close <= range_low + 0.30 * width
            and rsi <= 50
            and bool(candle["bullish_reversal"])
        ):
            direction = "BUY"
            edge = range_low
            target_distance = range_mid - close
            stop_distance = close - (range_low - 0.25 * atr)
            position_from_edge = (close - range_low) / width
            rsi_extreme = (50 - rsi) / 50
            rejection_body = max(0.0, close - float(candle["Open"]))
        elif (
            close >= range_high - 0.30 * width
            and rsi >= 50
            and bool(candle["bearish_reversal"])
        ):
            direction = "SELL"
            edge = range_high
            target_distance = close - range_mid
            stop_distance = (range_high + 0.25 * atr) - close
            position_from_edge = (range_high - close) / width
            rsi_extreme = (rsi - 50) / 50
            rejection_body = max(0.0, float(candle["Open"]) - close)
        if direction is None:
            continue
        if timestamp < last_signal[direction] + pd.Timedelta(hours=2):
            continue
        tp_usd = min(max(target_distance, 5.0), 15.0)
        sl_usd = min(max(stop_distance, 5.0), 12.0)
        reward_risk = tp_usd / sl_usd
        if reward_risk < 0.70:
            continue
        entry_location = data.index.searchsorted(timestamp, side="right")
        if entry_location >= len(data):
            continue
        entry_time = pd.Timestamp(data.index[entry_location])
        spread = float(data.iloc[entry_location]["SpreadPoints"])
        if spread > spread_limit:
            continue
        row = {
            "timestamp": entry_time,
            "setup_time": timestamp,
            "direction": direction,
            "raw_close": float(data.iloc[entry_location]["Close"]),
            "tp_usd": tp_usd,
            "sl_usd": sl_usd,
            "range_mid": range_mid,
            "range_high": range_high,
            "range_low": range_low,
            "breakout_guard": bool(state.get("breakout_guard", False)),
            "strong_rejection": rejection_body / atr >= 0.08,
            "position_from_edge": position_from_edge,
            "rsi_extreme": rsi_extreme,
            "rejection_body_atr": rejection_body / atr,
            "distance_edge_atr": abs(close - edge) / atr,
            "reward_risk": reward_risk,
        }
        for feature in (
            "adx",
            "efficiency",
            "choppiness",
            "trend_strength",
            "slope",
            "range_width_atr",
            "touch_lower",
            "touch_upper",
            "atr_percentile",
        ):
            row[feature] = float(state.get(feature, np.nan))
        rows.append(row)
        last_signal[direction] = timestamp
    opportunities = pd.DataFrame(rows)
    if opportunities.empty:
        raise RuntimeError("Range detector tidak menghasilkan opportunity.")
    opportunities = opportunities.set_index("timestamp").sort_index()
    return _attach_path_labels(data, opportunities).dropna(
        subset=["target_12h", *MODEL_FEATURES]
    )


def _attach_path_labels(data, opportunities):
    output = opportunities.copy()
    labels = []
    for timestamp, row in output.iterrows():
        location = data.index.searchsorted(timestamp, side="left")
        future = data.iloc[location + 1 : location + 12 * 60 + 1]
        if len(future) < 12 * 60:
            labels.append((np.nan, np.nan, np.nan, np.nan))
            continue
        entry_bid = float(row["raw_close"])
        entry_spread = float(data.iloc[location]["SpreadPoints"]) * 0.01
        direction = str(row["direction"])
        tp = float(row["tp_usd"])
        sl = float(row["sl_usd"])
        spread = future["SpreadPoints"] * 0.01
        if direction == "BUY":
            entry = entry_bid + entry_spread
            favorable = future["High"] - entry
            adverse = entry - future["Low"]
        else:
            entry = entry_bid
            favorable = entry - (future["Low"] + spread)
            adverse = future["High"] + spread - entry
        tp_steps = np.flatnonzero(favorable.to_numpy() >= tp)
        sl_steps = np.flatnonzero(adverse.to_numpy() >= sl)
        first_tp = int(tp_steps[0] + 1) if len(tp_steps) else 10_000
        first_sl = int(sl_steps[0] + 1) if len(sl_steps) else 10_000
        labels.append(
            (
                float(first_tp < first_sl),
                float(favorable.max()),
                float(adverse.max()),
                float(first_tp / 60) if first_tp < 10_000 else np.nan,
            )
        )
    output[
        ["target_12h", "mfe_12h_usd", "mae_12h_usd", "time_to_tp_hours"]
    ] = labels
    return output


def _train_outcome_model(opportunities):
    train = opportunities.loc[TRAIN_START:TRAIN_END]
    calibration = opportunities.loc[CALIBRATION_START:CALIBRATION_END]
    threshold_period = opportunities.loc[THRESHOLD_START:THRESHOLD_END]
    if len(train) < 50 or train["target_12h"].nunique() < 2:
        raise RuntimeError("Data train range tidak cukup untuk klasifikasi.")
    logistic = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            C=0.30,
            random_state=90,
        ),
    )
    boosting = HistGradientBoostingClassifier(
        learning_rate=0.035,
        max_iter=180,
        max_depth=3,
        min_samples_leaf=20,
        l2_regularization=2.0,
        random_state=91,
    )
    logistic.fit(train[list(MODEL_FEATURES)], train["target_12h"].astype(int))
    boosting.fit(
        train[list(MODEL_FEATURES)],
        train["target_12h"].astype(int),
        sample_weight=_class_weights(train["target_12h"]),
    )
    raw = (
        pd.Series(
            logistic.predict_proba(opportunities[list(MODEL_FEATURES)])[:, 1],
            index=opportunities.index,
        )
        + pd.Series(
            boosting.predict_proba(opportunities[list(MODEL_FEATURES)])[:, 1],
            index=opportunities.index,
        )
    ) / 2
    calibrator = _fit_platt(
        raw.loc[CALIBRATION_START:CALIBRATION_END],
        calibration["target_12h"].astype(int),
    )
    probability = _apply_symmetric_calibration(raw, calibrator)
    threshold, audit = _select_threshold(
        threshold_period["target_12h"].astype(int),
        probability.loc[THRESHOLD_START:THRESHOLD_END],
    )
    return {
        "probability": probability,
        "threshold": threshold,
    }, pd.DataFrame(
        [
            {
                "Train opportunities": len(train),
                "Calibration opportunities": len(calibration),
                "Threshold opportunities": len(threshold_period),
                "Threshold": threshold,
                **audit,
            }
        ]
    )


def _select_threshold(truth, probability):
    rows = []
    for quantile in (0.40, 0.50, 0.60, 0.70, 0.80):
        threshold = float(probability.quantile(quantile))
        prediction = probability.ge(threshold)
        selected = truth.loc[prediction]
        precision = float(selected.mean()) if len(selected) else 0.0
        recall = float(
            (prediction & truth.eq(1)).sum() / max(int(truth.eq(1).sum()), 1)
        )
        eligible = len(selected) >= 12
        score = precision * 20 - (1 - precision) * 10 + recall * 3
        rows.append((threshold, precision, recall, len(selected), score, eligible))
    eligible_rows = [row for row in rows if row[-1]]
    selected = max(eligible_rows or rows, key=lambda row: row[4])
    return selected[0], {
        "Precision threshold": selected[1],
        "Recall threshold": selected[2],
        "Sinyal threshold": selected[3],
        "Expected value proxy": selected[1] * 20 - (1 - selected[1]) * 10,
    }


def _candidate_signals(data, opportunities, model, best, spread_limit):
    probability = model["probability"]
    selected = probability.ge(model["threshold"])
    masks = {
        "Range Control": pd.Series(True, index=opportunities.index),
        "Boundary Rejection": opportunities["strong_rejection"],
        "Outcome Filter": selected,
        "Midpoint Conservative": selected & opportunities["reward_risk"].ge(1.10),
        "Breakout Guard": (
            selected
            & opportunities["breakout_guard"]
            & opportunities["atr_percentile"].le(0.80)
        ),
    }
    output = {}
    funnel = []
    for candidate, mask in masks.items():
        frame = opportunities.loc[mask].copy()
        signals = _opportunities_to_signals(frame, best, candidate)
        output[candidate] = _unique_signals(signals)
        funnel.append(
            {
                "Kandidat": candidate,
                "Opportunity": len(opportunities),
                "Lolos filter": len(output[candidate]),
                "BUY": int(
                    (frame.loc[~frame.index.duplicated(keep="first"), "direction"] == "BUY").sum()
                ),
                "SELL": int(
                    (frame.loc[~frame.index.duplicated(keep="first"), "direction"] == "SELL").sum()
                ),
            }
        )
    adaptive_mask = (
        selected
        & opportunities["breakout_guard"]
        & (
            opportunities["strong_rejection"]
            | opportunities["reward_risk"].ge(1.20)
        )
    )
    adaptive_frame = opportunities.loc[adaptive_mask].copy()
    adaptive = _opportunities_to_signals(
        adaptive_frame, best, "Adaptive Range Ensemble"
    )
    if not adaptive.empty:
        source = adaptive_frame.loc[
            ~adaptive_frame.index.duplicated(keep="first")
        ].reindex(adaptive.index)
        high_quality = (
            source["reward_risk"].ge(1.30)
            & source["atr_percentile"].le(0.65)
        )
        adaptive.loc[high_quality, "tp_usd"] = np.minimum(
            adaptive.loc[high_quality, "tp_usd"] * 1.15, 17.0
        )
    output["Adaptive Range Ensemble"] = _unique_signals(adaptive)
    funnel.append(
        {
            "Kandidat": "Adaptive Range Ensemble",
            "Opportunity": len(opportunities),
            "Lolos filter": len(adaptive),
            "BUY": int(
                (adaptive_frame.loc[~adaptive_frame.index.duplicated(keep="first"), "direction"] == "BUY").sum()
            ),
            "SELL": int(
                (adaptive_frame.loc[~adaptive_frame.index.duplicated(keep="first"), "direction"] == "SELL").sum()
            ),
        }
    )
    return output, pd.DataFrame(funnel)


def _opportunities_to_signals(frame, best, strategy):
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "signal_date",
                "prediction",
                "expected_change_pct",
                "lot",
                "tp_usd",
                "sl_usd",
                "time_stop_hours",
                "strategy",
            ]
        )
    frame = frame.sort_values(
        ["reward_risk", "strong_rejection"], ascending=[False, False]
    )
    frame = frame.loc[~frame.index.duplicated(keep="first")].sort_index()
    threshold = max(float(best["Threshold entry (%)"]), 0.15)
    sign = np.where(frame["direction"].eq("BUY"), 1.0, -1.0)
    output = pd.DataFrame(index=frame.index)
    output["signal_date"] = pd.to_datetime(frame["setup_time"]).dt.normalize()
    output["expected_change_pct"] = sign * (threshold + 0.01)
    output["prediction"] = frame["raw_close"] * (
        1 + output["expected_change_pct"] / 100
    )
    output["lot"] = 0.01
    output["tp_usd"] = frame["tp_usd"]
    output["sl_usd"] = frame["sl_usd"]
    output["time_stop_hours"] = 12.0
    output["strategy"] = strategy
    return output


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
    for label, start, end in _periods():
        if label == "Reference 2026H1":
            continue
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


def _classification_tables(opportunities, model):
    output = {}
    for key, start, end in (
        ("selection", SELECTION_START, SELECTION_END),
        ("locked", LOCKED_START, LOCKED_END),
        ("reference", REFERENCE_START, REFERENCE_END),
    ):
        selected = opportunities.loc[start:end]
        probability = model["probability"].loc[start:end]
        prediction = probability.ge(model["threshold"])
        truth = selected["target_12h"].astype(int)
        rows = []
        for direction in ("ALL", "BUY", "SELL"):
            mask = (
                pd.Series(True, index=selected.index)
                if direction == "ALL"
                else selected["direction"].eq(direction)
            )
            rows.append(
                {
                    "Arah": direction,
                    "Threshold": model["threshold"],
                    "Opportunity": int(mask.sum()),
                    "Precision": precision_score(
                        truth.loc[mask], prediction.loc[mask], zero_division=0
                    ),
                    "Recall": recall_score(
                        truth.loc[mask], prediction.loc[mask], zero_division=0
                    ),
                    "Coverage (%)": float(prediction.loc[mask].mean() * 100),
                    "Brier": (
                        float(
                            brier_score_loss(
                                truth.loc[mask], probability.loc[mask]
                            )
                        )
                        if mask.sum()
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
    locked_precision = float(
        classification.loc[classification["Arah"].eq("ALL"), "Precision"].iloc[0]
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
                "Precision locked": locked_precision,
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


def _range_audit(frame):
    rows = []
    hourly = frame.resample("1h").last()
    for label, start, end in _periods():
        selected = hourly.loc[start:end]
        rows.append(
            {
                "Periode": label,
                "Observasi H1": len(selected),
                "Range quality": int(selected["range_quality"].sum()),
                "Range confirmed": int(selected["range_confirmed"].sum()),
                "Coverage confirmed (%)": float(
                    selected["range_confirmed"].mean() * 100
                ),
                "Breakout up": int(selected["breakout_up"].sum()),
                "Breakout down": int(selected["breakout_down"].sum()),
            }
        )
    return pd.DataFrame(rows)


def _opportunity_audit(opportunities):
    rows = []
    for label, start, end in _periods():
        selected = opportunities.loc[start:end]
        for direction in ("BUY", "SELL"):
            direction_frame = selected.loc[selected["direction"].eq(direction)]
            rows.append(
                {
                    "Periode": label,
                    "Arah": direction,
                    "Opportunity": len(direction_frame),
                    "TP-before-SL (%)": (
                        float(direction_frame["target_12h"].mean() * 100)
                        if len(direction_frame)
                        else np.nan
                    ),
                    "Median RR": (
                        float(direction_frame["reward_risk"].median())
                        if len(direction_frame)
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def _path_audit(opportunities):
    rows = []
    for label, start, end in _periods():
        selected = opportunities.loc[start:end]
        for direction in ("BUY", "SELL"):
            direction_frame = selected.loc[selected["direction"].eq(direction)]
            winners = direction_frame.loc[direction_frame["target_12h"].eq(1)]
            rows.append(
                {
                    "Periode": label,
                    "Arah": direction,
                    "Opportunity": len(direction_frame),
                    "Median MFE": (
                        float(direction_frame["mfe_12h_usd"].median())
                        if len(direction_frame)
                        else np.nan
                    ),
                    "Median MAE": (
                        float(direction_frame["mae_12h_usd"].median())
                        if len(direction_frame)
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


def _periods():
    return (
        ("Train 2022", TRAIN_START, TRAIN_END),
        ("Calibration 2023H1", CALIBRATION_START, CALIBRATION_END),
        ("Threshold 2023H2", THRESHOLD_START, THRESHOLD_END),
        ("Selection 2024", SELECTION_START, SELECTION_END),
        ("Locked 2025", LOCKED_START, LOCKED_END),
        ("Reference 2026H1", REFERENCE_START, REFERENCE_END),
    )


def _defense_reference(payload):
    if not payload:
        return pd.DataFrame()
    validation = payload.get("strategy_validation")
    if not isinstance(validation, pd.DataFrame) or validation.empty:
        return pd.DataFrame()
    return validation.loc[
        validation["Strategi"].eq("Sideways Mean Reversion")
    ].copy()
