import pandas as pd

from gold_forecast.data import _merge_gold_revisions


def _frame(rows):
    return pd.DataFrame(rows).set_index(pd.to_datetime([row.pop("Date") for row in rows]))


def test_partial_flat_revision_does_not_replace_completed_candle():
    cached = _frame(
        [{"Date": "2026-09-04", "Open": 4522.0, "High": 4537.8, "Low": 4412.0, "Close": 4476.6, "Volume": 186451}]
    )
    latest = _frame(
        [{"Date": "2026-09-04", "Open": 4429.8, "High": 4429.8, "Low": 4429.8, "Close": 4429.8, "Volume": 16}]
    )

    merged = _merge_gold_revisions(cached, latest)

    assert merged.iloc[0]["Close"] == 4476.6
    assert merged.iloc[0]["Volume"] == 186451


def test_healthy_revision_can_replace_cached_candle():
    cached = _frame(
        [{"Date": "2026-09-04", "Open": 4522.0, "High": 4537.8, "Low": 4412.0, "Close": 4476.6, "Volume": 186451}]
    )
    latest = _frame(
        [{"Date": "2026-09-04", "Open": 4522.0, "High": 4538.0, "Low": 4412.0, "Close": 4477.2, "Volume": 187000}]
    )

    merged = _merge_gold_revisions(cached, latest)

    assert merged.iloc[0]["Close"] == 4477.2
    assert merged.iloc[0]["Volume"] == 187000
