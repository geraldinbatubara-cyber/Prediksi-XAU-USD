import pandas as pd

from gold_forecast.session_hour_map import (
    build_session_hour_map,
    summarize_hour_frequency,
    summarize_monthly_frequency,
)


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


def test_hour_frequency_summary_uses_counts_and_session_windows():
    hourly = pd.DataFrame(
        {
            "Jam WIT": range(24),
            "Frekuensi High": [10 if hour == 4 else 5 if hour == 7 else 0 for hour in range(24)],
            "Frekuensi Low": [12 if hour == 7 else 4 if hour == 23 else 0 for hour in range(24)],
        }
    )

    summary = summarize_hour_frequency(hourly, sessions=20)

    assert summary["high_rank"][0] == {"hour": 4, "count": 10, "percentage": 50.0}
    assert summary["low_rank"][0] == {"hour": 7, "count": 12, "percentage": 60.0}
    assert summary["late_high_pct"] == 50.0
    assert summary["late_low_pct"] == 20.0
    assert summary["opening_high_pct"] == 25.0
    assert summary["opening_low_pct"] == 60.0


def test_monthly_summary_counts_tied_peaks_and_window_dominance():
    monthly = pd.DataFrame(
        [
            {"Bulan": "2025-01", "Jam WIT": 4, "Frekuensi High": 3, "Frekuensi Low": 1},
            {"Bulan": "2025-01", "Jam WIT": 7, "Frekuensi High": 1, "Frekuensi Low": 3},
            {"Bulan": "2025-02", "Jam WIT": 4, "Frekuensi High": 2, "Frekuensi Low": 0},
            {"Bulan": "2025-02", "Jam WIT": 7, "Frekuensi High": 2, "Frekuensi Low": 4},
        ]
    )

    summary = summarize_monthly_frequency(monthly)

    assert summary["month_count"] == 2
    assert summary["recurring_high_hour"] == 4
    assert summary["recurring_high_months"] == 2
    assert summary["recurring_low_hour"] == 7
    assert summary["recurring_low_months"] == 2
    assert summary["late_high_dominant_months"] == 1
    assert summary["opening_low_dominant_months"] == 2
    assert summary["strongest_high"]["percentage"] == 75.0
    assert summary["strongest_low"]["percentage"] == 100.0
