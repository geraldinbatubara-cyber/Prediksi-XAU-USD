import numpy as np
import pandas as pd

from gold_forecast.h1_forecast import MINIMUM_H1_BARS, build_h1_forecast


def _h1_bars(periods: int) -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=periods, freq="1h", tz="UTC")
    trend = np.linspace(4000.0, 4200.0, periods)
    cycle = np.sin(np.arange(periods) / 7) * 4
    close = trend + cycle
    return pd.DataFrame(
        {
            "timestamp_utc": index,
            "open": close - 0.4,
            "high": close + 1.2,
            "low": close - 1.1,
            "close": close,
            "tick_volume": 100,
            "spread_points": 15.0,
        }
    )


def test_h1_forecast_refuses_insufficient_data():
    result = build_h1_forecast(
        _h1_bars(MINIMUM_H1_BARS - 1),
        now=pd.Timestamp("2026-02-01", tz="UTC"),
    )
    assert not result["valid"]
    assert result["signal"] == "TIDAK VALID"


def test_h1_forecast_uses_only_completed_candles_and_two_horizons():
    bars = _h1_bars(500)
    current_start = pd.Timestamp(bars.iloc[-1]["timestamp_utc"])
    now = current_start + pd.Timedelta(minutes=30)
    result = build_h1_forecast(bars, now=now)

    assert result["valid"]
    assert result["source_time"] == current_start - pd.Timedelta(hours=1)
    assert set(result["forecasts"]) == {1, 6}
    assert result["forecasts"][1]["training_rows"] > 180
    assert result["signal"] in {"BULLISH", "BEARISH", "NETRAL"}
