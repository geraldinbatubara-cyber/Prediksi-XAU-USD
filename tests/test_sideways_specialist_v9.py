import pandas as pd

from gold_forecast.v1_sideways_specialist_v9 import _expanded_range_frame


def test_expanded_regime_is_looser_than_strict_control():
    index = pd.date_range("2022-01-01", periods=30, freq="min")
    strict = pd.DataFrame(
        {
            "range_confirmed": False,
            "sideways_votes": 2,
            "range_width_atr": 2.0,
            "touch_lower": 1.0,
            "touch_upper": 1.0,
        },
        index=index,
    )
    expanded = _expanded_range_frame(strict)
    assert not expanded["strict_range_confirmed"].any()
    assert expanded["range_confirmed"].iloc[-1]


def test_expanded_regime_keeps_risk_boundaries():
    index = pd.date_range("2022-01-01", periods=30, freq="min")
    strict = pd.DataFrame(
        {
            "range_confirmed": False,
            "sideways_votes": 2,
            "range_width_atr": 9.0,
            "touch_lower": 1.0,
            "touch_upper": 1.0,
        },
        index=index,
    )
    expanded = _expanded_range_frame(strict)
    assert not expanded["range_confirmed"].any()


def test_expansion_requires_both_range_sides_touched():
    index = pd.date_range("2022-01-01", periods=30, freq="min")
    strict = pd.DataFrame(
        {
            "range_confirmed": False,
            "sideways_votes": 3,
            "range_width_atr": 2.0,
            "touch_lower": 1.0,
            "touch_upper": 0.0,
        },
        index=index,
    )
    expanded = _expanded_range_frame(strict)
    assert not expanded["range_confirmed"].any()
