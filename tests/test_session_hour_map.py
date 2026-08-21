import pandas as pd

from gold_forecast.session_hour_map import build_session_hour_map


def test_session_mapping_uses_exact_wit_observation_and_first_extreme():
    wit_index = pd.DatetimeIndex(
        [
            "2025-01-02 07:00",
            "2025-01-02 08:00",
            "2025-01-02 09:15",
            "2025-01-02 10:30",
            "2025-01-03 04:00",
        ],
        tz="Asia/Jayapura",
    )
    frame = pd.DataFrame(
        {
            "timestamp_utc": wit_index.tz_convert("UTC"),
            "Open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "High": [101.0, 102.0, 110.0, 110.0, 105.0],
            "Low": [99.0, 98.0, 100.0, 101.0, 98.0],
            "Close": [100.5, 101.5, 102.5, 103.5, 104.5],
        }
    )

    result = build_session_hour_map(frame, "2025-01-02", "2025-01-02")
    row = result.daily.iloc[0]

    assert row["Open +1 Jam (08 WIT)"] == 101.0
    assert row["Close -1 Jam (04 WIT)"] == 104.0
    assert row["Jam High WIT"] == 9
    assert row["Jumlah kemunculan High"] == 2
    assert row["Jam Low WIT"] == 8
    assert row["Jumlah kemunculan Low"] == 2


def test_missing_exact_observation_is_not_interpolated():
    wit_index = pd.DatetimeIndex(
        ["2025-01-02 07:00", "2025-01-02 08:01", "2025-01-03 03:59"],
        tz="Asia/Jayapura",
    )
    frame = pd.DataFrame(
        {
            "timestamp_utc": wit_index.tz_convert("UTC"),
            "Open": [100.0, 101.0, 102.0],
            "High": [101.0, 102.0, 103.0],
            "Low": [99.0, 100.0, 101.0],
            "Close": [100.5, 101.5, 102.5],
        }
    )

    result = build_session_hour_map(frame, "2025-01-02", "2025-01-02")
    row = result.daily.iloc[0]

    assert pd.isna(row["Open +1 Jam (08 WIT)"])
    assert pd.isna(row["Close -1 Jam (04 WIT)"])
    assert row["Status Open +1"] == "Tidak tersedia"
    assert row["Status Close -1"] == "Tidak tersedia"
