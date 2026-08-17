from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from gold_forecast.v1_directional_specialization import (
    _apply_symmetric_calibration,
    _class_weights,
)
from gold_forecast.v1_regime_classifier_v3 import _fit_platt
from gold_forecast.v1_sell_specialist import (
    CALIBRATION_END,
    CALIBRATION_START,
    TRAIN_END,
    TRAIN_START,
)
from gold_forecast.v1_sideways_defense import _regime_features
from gold_forecast.v1_sideways_specialist import (
    _mean_reversion_opportunities,
    _range_quality_frame,
)
from gold_forecast.v1_sideways_specialist_v2 import (
    PERSISTENCE_FEATURES,
    _augment_opportunities,
)
from gold_forecast.v1_sideways_specialist_v9 import _expanded_range_frame


def build_moderate_regime_live_bundle(data, thresholds, spread_limit):
    features, h1, m15 = _regime_features(data)
    expanded = _expanded_range_frame(_range_quality_frame(features, h1))
    opportunities = _augment_opportunities(
        data,
        expanded,
        _mean_reversion_opportunities(data, expanded, m15, spread_limit),
    )
    train = opportunities.loc[TRAIN_START:TRAIN_END]
    calibration = opportunities.loc[CALIBRATION_START:CALIBRATION_END]
    target = "adverse_breakout_6h"
    if len(train) < 50 or train[target].nunique() < 2 or calibration.empty:
        raise RuntimeError("Data hazard Moderate Regime tidak cukup untuk artefak live.")

    logistic = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", C=0.30, random_state=171),
    )
    boosting = HistGradientBoostingClassifier(
        learning_rate=0.035,
        max_iter=180,
        max_depth=3,
        min_samples_leaf=20,
        l2_regularization=2.0,
        random_state=172,
    )
    columns = list(PERSISTENCE_FEATURES)
    logistic.fit(train[columns], train[target].astype(int))
    boosting.fit(
        train[columns],
        train[target].astype(int),
        sample_weight=_class_weights(train[target]),
    )
    raw = _raw_probability(opportunities, logistic, boosting)
    calibrator = _fit_platt(
        raw.loc[CALIBRATION_START:CALIBRATION_END],
        calibration[target].astype(int),
    )
    return {
        "strategy": "Moderate Regime - Sideways v9",
        "feature_columns": tuple(columns),
        "hazard_logistic": logistic,
        "hazard_boosting": boosting,
        "hazard_calibrator": calibrator,
        "thresholds": dict(thresholds),
        "spread_limit": float(spread_limit),
        "training_contract": (
            "Hazard train 2022 | calibration 2023H1 | threshold v9 dibekukan "
            "dari 2022-2023"
        ),
    }


def moderate_regime_live_signal(data, bundle, activation_utc, now_utc):
    state = {
        "Status": "MENUNGGU",
        "Detail": "Menunggu opportunity Moderate Regime terbaru.",
        "Opportunity": 0,
        "Boundary": False,
        "Rejection": False,
        "Hazard probability": np.nan,
        "Hazard extreme": False,
        "Checklist": [],
    }
    if data.empty or not bundle:
        state.update(
            {
                "Status": "MODEL / DATA BELUM TERSEDIA",
                "Detail": "Artefak inferensi atau candle M1 broker belum tersedia.",
            }
        )
        return None, state

    features, h1, m15 = _regime_features(data)
    expanded = _expanded_range_frame(_range_quality_frame(features, h1))
    opportunities = _safe_live_opportunities(
        data, expanded, m15, float(bundle["spread_limit"])
    )
    if opportunities.empty:
        state.update(
            {
                "Status": "MENUNGGU OPPORTUNITY",
                "Detail": "Belum ada setup mean-reversion pada cakupan candle M1 terbaru.",
            }
        )
        return None, state
    opportunities = _augment_live_opportunities(opportunities, expanded)
    columns = list(bundle["feature_columns"])
    opportunities = opportunities.dropna(subset=columns)
    recent = opportunities.loc[
        (opportunities.index >= activation_utc) & (opportunities.index <= now_utc)
    ].copy()
    state["Opportunity"] = int(len(recent))
    if recent.empty:
        state.update(
            {
                "Status": "MENUNGGU OPPORTUNITY",
                "Detail": (
                    "Belum ada setup mean-reversion dengan warm-up indikator lengkap "
                    "setelah aktivasi paper live."
                ),
            }
        )
        return None, state

    raw = _raw_probability(recent, bundle["hazard_logistic"], bundle["hazard_boosting"])
    hazard = _apply_symmetric_calibration(raw, bundle["hazard_calibrator"])
    thresholds = bundle["thresholds"]
    recent["hazard_probability"] = hazard
    recent["boundary_ok"] = (
        recent["position_from_edge"].le(float(thresholds["edge_position_relaxed"]))
        & recent["distance_edge_atr"].le(float(thresholds["edge_atr_relaxed"]))
        & recent["range_width_atr"].between(1.0, 7.0)
    )
    recent["rejection_ok"] = recent["rejection_body_atr"].ge(
        float(thresholds["rejection_relaxed"])
    )
    recent["hazard_extreme"] = recent["hazard_probability"].ge(
        float(thresholds["hazard_extreme"])
    )
    recent["eligible"] = recent["boundary_ok"] & recent["rejection_ok"] & ~recent["hazard_extreme"]
    latest = recent.iloc[-1]
    entry_time = pd.Timestamp(recent.index[-1])
    age_minutes = (now_utc - entry_time).total_seconds() / 60
    fresh_setup = -1 <= age_minutes <= 5
    eligible = bool(latest["eligible"]) and fresh_setup
    checklist = [
        {
            "Syarat": "Expanded sideways regime",
            "Status": "LOLOS",
            "Detail": "Minimal 2/5 bukti sideways dan range terkonfirmasi 12/15 M1.",
        },
        {
            "Syarat": "Lokasi dekat boundary",
            "Status": "LOLOS" if bool(latest["boundary_ok"]) else "BELUM",
            "Detail": (
                f"Edge position {float(latest['position_from_edge']):.3f} | "
                f"distance {float(latest['distance_edge_atr']):.3f} ATR."
            ),
        },
        {
            "Syarat": "Rejection candle",
            "Status": "LOLOS" if bool(latest["rejection_ok"]) else "BELUM",
            "Detail": f"Rejection body {float(latest['rejection_body_atr']):.3f} ATR.",
        },
        {
            "Syarat": "Extreme breakout hazard veto",
            "Status": "LOLOS" if not bool(latest["hazard_extreme"]) else "BLOKIR",
            "Detail": (
                f"P(hazard) {float(latest['hazard_probability']):.1%} vs batas "
                f"{float(thresholds['hazard_extreme']):.1%}."
            ),
        },
        {
            "Syarat": "Setup masih fresh",
            "Status": "LOLOS" if fresh_setup else "KEDALUWARSA",
            "Detail": f"Usia setup {age_minutes:.1f} menit; batas 5 menit.",
        },
    ]
    state.update(
        {
            "Status": "SIAP ENTRY" if eligible else "ABSTAIN / MENUNGGU",
            "Detail": (
                "Seluruh gerbang Moderate Regime terpenuhi."
                if eligible
                else "Opportunity terbaru belum memenuhi seluruh gerbang atau sudah kedaluwarsa."
            ),
            "Direction": str(latest["direction"]),
            "Setup time": pd.Timestamp(latest["setup_time"]),
            "Entry time": entry_time,
            "Boundary": bool(latest["boundary_ok"]),
            "Rejection": bool(latest["rejection_ok"]),
            "Hazard probability": float(latest["hazard_probability"]),
            "Hazard extreme": bool(latest["hazard_extreme"]),
            "Checklist": checklist,
        }
    )
    if not eligible:
        return None, state

    direction = str(latest["direction"])
    expected = 0.16 if direction == "BUY" else -0.16
    reference = float(latest["raw_close"])
    return {
        "signal_date": entry_time,
        "prediction": reference * (1 + expected / 100),
        "reference_price": reference,
        "expected_change_pct": expected,
        "arah": direction,
        "source": "Moderate Regime Sideways v9",
        "tp_usd": float(latest["tp_usd"]),
        "sl_usd": float(latest["sl_usd"]),
        "intraday_signal": True,
    }, state


def _raw_probability(frame, logistic, boosting):
    columns = list(PERSISTENCE_FEATURES)
    return (
        pd.Series(logistic.predict_proba(frame[columns])[:, 1], index=frame.index)
        + pd.Series(boosting.predict_proba(frame[columns])[:, 1], index=frame.index)
    ) / 2


def _augment_live_opportunities(opportunities, frame):
    if opportunities.empty:
        return opportunities.copy()
    output = opportunities.copy()
    confirmed = frame["range_confirmed"].fillna(False)
    group = (~confirmed).cumsum()
    atr = frame["atr"].clip(lower=0.01)
    width = frame["range_high"] - frame["range_low"]
    setup_times = pd.DatetimeIndex(pd.to_datetime(output["setup_time"]))
    mappings = {
        "range_age_hours": confirmed.groupby(group).cumcount() / 60,
        "midpoint_drift_atr": (frame["range_mid"] - frame["range_mid"].shift(180)) / atr,
        "range_width_change": width / width.shift(180) - 1,
        "adx_change_3h": frame["adx"] - frame["adx"].shift(180),
        "atr_acceleration": atr / atr.shift(360).rolling(180, min_periods=60).median() - 1,
        "touch_imbalance": (
            (frame["touch_upper"] - frame["touch_lower"])
            / (frame["touch_upper"] + frame["touch_lower"]).replace(0, np.nan)
        ),
    }
    for column, values in mappings.items():
        output[column] = values.reindex(setup_times).to_numpy()
    hour = setup_times.hour + setup_times.minute / 60
    output["session_sin"] = np.sin(2 * np.pi * hour / 24)
    output["session_cos"] = np.cos(2 * np.pi * hour / 24)
    output["direction_code"] = np.where(output["direction"].eq("BUY"), 1.0, -1.0)
    return output.replace([np.inf, -np.inf], np.nan)


def _safe_live_opportunities(data, expanded, m15, spread_limit):
    try:
        return _mean_reversion_opportunities(data, expanded, m15, spread_limit)
    except RuntimeError as exc:
        if str(exc) != "Range detector tidak menghasilkan opportunity.":
            raise
        return pd.DataFrame()
