from __future__ import annotations

import base64
import pickle
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gold_forecast.broker_data import (
    BROKER_BARS_PATH,
    BROKER_QUOTE_PATH,
    apply_broker_clock_offset,
    load_broker_bars,
    load_broker_quote,
)
from gold_forecast.exact_broker_oos import _prepare_m1
from gold_forecast.v1_regime_classifier import (
    FEATURE_COLUMNS,
    _classifier_frame,
    _ohlc_bars,
)
from gold_forecast.v1_regime_classifier_v3 import (
    CALIBRATION_END,
    CALIBRATION_START,
    THRESHOLD_END,
    THRESHOLD_START,
    TRAIN_END,
    TRAIN_START,
    _apply_calibration,
    _calibrate_probabilities,
    _choose_thresholds,
    _fit_base_estimators,
    _label_frame,
    _raw_model_probabilities,
)
from scripts.build_v1_entry_timing import _load_cached_history


OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "precomputed" / "buy_specialist_v4_live.pkl.b64"
)
VERSION = "buy-specialist-v4-live-inference-2026-07-24-v1"
HORIZON_HOURS = 4
WARMUP_BARS = 250


def main() -> None:
    historical = _load_cached_history()
    recent = _recent_broker_m1()
    combined = pd.concat([historical, recent]).sort_index()
    combined = combined.loc[~combined.index.duplicated(keep="last")]
    data = _prepare_m1(combined)

    base = _classifier_frame(data).drop(columns=["label"], errors="ignore")
    frame = _label_frame(data, base, HORIZON_HOURS).dropna(
        subset=[*FEATURE_COLUMNS, "truth"]
    )
    train = frame.loc[TRAIN_START:TRAIN_END]
    calibration = frame.loc[CALIBRATION_START:CALIBRATION_END]
    threshold_data = frame.loc[THRESHOLD_START:THRESHOLD_END]
    if train.empty or calibration.empty or threshold_data.empty:
        raise RuntimeError("Periode train/calibration v4 tidak lengkap.")

    estimators = _fit_base_estimators(train)
    raw = _raw_model_probabilities(estimators, frame)[
        "Hierarchical Ensemble"
    ]
    calibrators = _calibrate_probabilities(raw, calibration)
    probabilities = _apply_calibration(raw, calibrators)
    thresholds = _choose_thresholds(
        threshold_data, probabilities.loc[threshold_data.index]
    )

    h1 = _ohlc_bars(data, "1h")
    h4 = _ohlc_bars(data, "4h")
    d1 = _ohlc_bars(data, "1D")
    spread = data["SpreadPoints"].resample(
        "1h", label="right", closed="left"
    ).agg(["median", lambda values: values.quantile(0.90)])
    spread.columns = ["spread_median", "spread_p90"]

    payload = {
        "strategy": "BUY Specialist v4 - Bullish Regime",
        "source_experiment": "Adaptive + Bear/Sideways Defense",
        "horizon_hours": HORIZON_HOURS,
        "feature_columns": FEATURE_COLUMNS,
        "estimators": estimators,
        "calibrators": calibrators,
        "thresholds": thresholds,
        "warmup_h1": h1.tail(WARMUP_BARS),
        "warmup_h4": h4.tail(WARMUP_BARS),
        "warmup_d1": d1.tail(WARMUP_BARS),
        "warmup_spread_h1": spread.tail(WARMUP_BARS),
        "generated_at_utc": pd.Timestamp.now(tz="UTC"),
        "training_contract": (
            "Train 2022 | calibration 2023H1 | threshold 2023H2 | "
            "horizon 4 jam | estimator dan threshold dibekukan"
        ),
    }
    artifact = pickle.dumps(
        {"version": VERSION, "payload": payload},
        protocol=pickle.HIGHEST_PROTOCOL,
    )
    OUTPUT_PATH.write_text(
        base64.b64encode(artifact).decode("ascii"), encoding="ascii"
    )
    print(
        f"BUY Specialist v4 live bundle selesai | train={len(train)} | "
        f"calibration={len(calibration)} | threshold={len(threshold_data)} | "
        f"latest={data.index.max()} | size={OUTPUT_PATH.stat().st_size:,} bytes"
    )


def _recent_broker_m1() -> pd.DataFrame:
    bars = load_broker_bars(BROKER_BARS_PATH)
    quotes = load_broker_quote(BROKER_QUOTE_PATH)
    bars, _ = apply_broker_clock_offset(bars, quotes)
    if bars.empty:
        raise RuntimeError(
            "Snapshot MT5 lokal tidak tersedia. Jalankan bridge sekali dengan --bars 50000."
        )
    frame = bars.rename(
        columns={
            "timestamp_utc": "timestamp_utc",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "spread_points": "SpreadPoints",
        }
    ).copy()
    frame["timestamp_utc"] = pd.to_datetime(
        frame["timestamp_utc"], utc=True, errors="coerce"
    )
    frame = frame.dropna(subset=["timestamp_utc"]).set_index("timestamp_utc")
    if frame.index.tz is not None:
        frame.index = frame.index.tz_convert("UTC").tz_localize(None)
    return frame[["Open", "High", "Low", "Close", "SpreadPoints"]]


if __name__ == "__main__":
    main()
