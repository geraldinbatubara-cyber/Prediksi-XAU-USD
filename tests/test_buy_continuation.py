import base64
import pickle
from pathlib import Path

import pandas as pd

from gold_forecast.v1_buy_continuation import (
    CONTROL,
    _cooldown_filter,
    _signal_frame,
)


def test_continuation_cooldown_keeps_one_signal_per_twelve_hours() -> None:
    index = pd.to_datetime(
        ["2026-01-02 00:00", "2026-01-02 04:00", "2026-01-02 12:00"]
    )
    signals = pd.DataFrame({"expected_change_pct": [0.2, 0.2, 0.2]}, index=index)

    selected = _cooldown_filter(signals)

    assert selected.index.tolist() == [index[0], index[2]]


def test_continuation_signal_is_buy_and_uses_frozen_lot() -> None:
    index = pd.date_range("2026-01-02 00:00", periods=3, freq="1min")
    data = pd.DataFrame({"Close": [4000.0, 4001.0, 4002.0]}, index=index)

    signal = _signal_frame(
        data,
        pd.DatetimeIndex([index[1]]),
        0.16,
        {"Lot": 0.01},
        "continuation-test",
    ).iloc[0]

    assert signal["expected_change_pct"] > 0
    assert signal["lot"] == 0.01
    assert signal["prediction"] > 4001.0


def test_precomputed_v41_keeps_v4_as_winner() -> None:
    path = Path("data/precomputed/v1_buy_continuation.pkl.b64")
    artifact = pickle.loads(base64.b64decode(path.read_text(encoding="ascii")))
    payload = artifact["payload"]

    assert payload["winner"] == CONTROL
    assert payload["methodology"]["Control"] == "BUY Specialist v4 tidak diubah"
    assert payload["methodology"]["Isolation"].startswith("Tidak membaca")
    assert payload["ranking"].iloc[0]["Lulus"]
