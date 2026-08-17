import numpy as np
import pandas as pd

from gold_forecast.v1_sideways_specialist_v8 import (
    ADAPTIVE_BOUNDARY,
    BOUNDARY_HAZARD,
    BREAKOUT_HAZARD,
    CONTROL,
    FULL_V8,
    REJECTION_CONFIRMATION,
    _adaptive_thresholds,
    _candidate_signal_map,
)


def _opportunities():
    index = pd.to_datetime(
        ["2022-01-03 01:00", "2022-01-03 02:00", "2022-01-03 03:00"]
    )
    return pd.DataFrame(
        {
            "position_from_edge": [0.10, 0.20, 0.80],
            "distance_edge_atr": [0.05, 0.10, 0.90],
            "range_width_atr": [2.0, 2.5, 8.0],
            "range_width_change": [0.02, 0.03, 0.80],
            "midpoint_drift_atr": [0.02, 0.03, 0.80],
            "atr_percentile": [0.30, 0.40, 0.99],
            "rejection_body_atr": [0.20, 0.05, 0.01],
            "reward_risk": [1.40, 1.10, 0.80],
            "direction": ["BUY", "SELL", "BUY"],
            "setup_time": index,
            "raw_close": [2000.0, 2001.0, 2002.0],
            "tp_usd": [10.0, 10.0, 10.0],
            "sl_usd": [8.0, 8.0, 8.0],
            "range_low": [1990.0, 1990.0, 1990.0],
            "range_high": [2010.0, 2010.0, 2010.0],
            "range_mid": [2000.0, 2000.0, 2000.0],
            "strong_rejection": [True, False, False],
        },
        index=index,
    )


def _best():
    return {"Threshold entry (%)": 0.15}


def _control(frame):
    output = pd.DataFrame(index=frame.index[:1])
    output["signal_date"] = output.index.normalize()
    output["expected_change_pct"] = 0.16
    output["prediction"] = 2001.0
    output["lot"] = 0.01
    output["tp_usd"] = 10.0
    output["sl_usd"] = 8.0
    output["time_stop_hours"] = 12.0
    output["strategy"] = CONTROL
    for column in ("range_low", "range_high", "range_mid", "range_width_atr", "direction"):
        output[column] = frame.loc[output.index, column]
    return output


def test_thresholds_use_historical_distribution():
    thresholds = _adaptive_thresholds(_opportunities())
    assert thresholds["calibration"] == "2022-2023 saja"
    assert np.isfinite(thresholds["edge_atr_max"])


def test_six_candidates_are_isolated():
    opportunities = _opportunities()
    control = _control(opportunities)
    signals, _ = _candidate_signal_map(
        opportunities,
        control,
        control,
        _best(),
        {
            "edge_position_max": 0.30,
            "edge_atr_max": 0.30,
            "range_width_min": 1.0,
            "range_width_max": 5.0,
            "width_change_abs_max": 0.20,
            "midpoint_drift_abs_max": 0.20,
            "atr_percentile_max": 0.80,
            "rejection_min": 0.10,
            "rr_strong": 1.20,
        },
    )
    assert set(signals) == {
        CONTROL,
        ADAPTIVE_BOUNDARY,
        REJECTION_CONFIRMATION,
        BREAKOUT_HAZARD,
        BOUNDARY_HAZARD,
        FULL_V8,
    }
    assert len(signals[ADAPTIVE_BOUNDARY]) == 2
    assert len(signals[BREAKOUT_HAZARD]) == 1
    assert len(signals[FULL_V8]) == 1


def test_all_generated_entries_remain_micro_lot():
    opportunities = _opportunities()
    control = _control(opportunities)
    signals, _ = _candidate_signal_map(
        opportunities,
        control,
        control,
        _best(),
        {
            "edge_position_max": 1.0,
            "edge_atr_max": 1.0,
            "range_width_min": 1.0,
            "range_width_max": 10.0,
            "width_change_abs_max": 1.0,
            "midpoint_drift_abs_max": 1.0,
            "atr_percentile_max": 1.0,
            "rejection_min": 0.0,
            "rr_strong": 0.0,
        },
    )
    assert all(
        frame.empty or frame["lot"].eq(0.01).all()
        for frame in signals.values()
    )
