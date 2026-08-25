import numpy as np
import pandas as pd

from gold_forecast.technical_analysis import (
    analyze_timeframe,
    completed_m5_bars,
    detect_price_pattern,
)


def _bars(periods: int = 100, freq: str = "1h") -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=periods, freq=freq, tz="UTC")
    close = np.linspace(4000.0, 4100.0, periods)
    return pd.DataFrame(
        {
            "timestamp_utc": index,
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "tick_volume": 100,
            "spread_points": 20,
        }
    )


def test_completed_m5_excludes_unfinished_bucket():
    frame = _bars(periods=11, freq="1min")
    now = pd.Timestamp("2026-01-01 00:10:30", tz="UTC")
    result = completed_m5_bars(frame, now=now)
    assert len(result) == 2
    assert result.index.max() == pd.Timestamp("2026-01-01 00:05:00", tz="UTC")


def test_head_and_shoulders_requires_neckline_confirmation():
    periods = 45
    index = pd.date_range("2026-01-01", periods=periods, freq="1h", tz="UTC")
    close = np.full(periods, 104.0)
    high = np.full(periods, 105.0)
    low = np.full(periods, 103.0)
    for position, peak in [(20, 110.0), (25, 115.0), (30, 110.5)]:
        high[position] = peak
        close[position] = peak - 1.0
        low[position] = peak - 2.0
    low[22] = 100.0
    low[28] = 99.5
    close[-1] = 98.0
    high[-1] = 99.0
    low[-1] = 97.5
    frame = pd.DataFrame(
        {"open": close + 0.2, "high": high, "low": low, "close": close},
        index=index,
    )
    result = detect_price_pattern(frame, atr_value=2.0)
    assert result["name"] == "Head and Shoulders"
    assert result["status"] == "TERKONFIRMASI"
    assert result["bias"] == -1


def test_analysis_returns_invalid_instead_of_fabricating_pattern():
    result = analyze_timeframe(_bars(periods=20), "H1", "test", minimum_bars=80)
    assert not result["valid"]
    assert result["signal"] == "TIDAK VALID"
    assert result["pattern"] == "Belum cukup data"


def test_bullish_trend_produces_auditable_analysis():
    result = analyze_timeframe(_bars(periods=120), "H1", "test", minimum_bars=80)
    assert result["valid"]
    assert result["trend"] == "BULLISH"
    assert result["signal"] in {"BUY", "WAIT"}
    assert result["bars"] == 120
