import pandas as pd

from gold_forecast.post_entry_audit import build_post_entry_audit


def _frame(rows):
    timestamps = pd.DatetimeIndex([row[0] for row in rows], tz="Asia/Jayapura")
    return pd.DataFrame(
        {
            "timestamp_utc": timestamps.tz_convert("UTC"),
            "Open": [row[1] for row in rows],
            "High": [row[2] for row in rows],
            "Low": [row[3] for row in rows],
            "Close": [row[4] for row in rows],
            "SpreadPoints": [10 for _ in rows],
        }
    )


def test_buy_tp_and_sell_sl_are_evaluated_after_exact_entry():
    bars = _frame(
        [
            ("2025-01-02 08:00", 100.0, 101.0, 99.5, 100.5),
            ("2025-01-02 09:00", 100.5, 126.0, 100.0, 125.0),
            ("2025-01-03 04:00", 120.0, 120.0, 120.0, 120.0),
        ]
    )
    result = build_post_entry_audit(bars, "2025-01-02", "2025-01-02")
    outcomes = result.paths.set_index("Arah")["Outcome"].to_dict()
    assert outcomes == {"BUY": "TP", "SELL": "SL"}


def test_same_candle_uses_conservative_sl_first_rule():
    bars = _frame(
        [
            ("2025-01-02 08:00", 100.0, 130.0, 70.0, 100.0),
            ("2025-01-03 04:00", 100.0, 100.0, 100.0, 100.0),
        ]
    )
    result = build_post_entry_audit(bars, "2025-01-02", "2025-01-02")
    assert result.paths["Outcome"].tolist() == ["SL", "SL"]
    assert result.paths["TP dan SL satu candle"].all()
