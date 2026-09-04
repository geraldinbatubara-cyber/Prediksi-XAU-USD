from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


WIT = "Asia/Jayapura"
SESSION_OPEN_HOUR = 7
SESSION_CLOSE_HOUR = 5
OPEN_PLUS_ONE_HOUR = 8
CLOSE_MINUS_ONE_HOUR = 4
EXPECTED_SESSION_MINUTES = 22 * 60


@dataclass(frozen=True)
class SessionHourMap:
    daily: pd.DataFrame
    hourly_frequency: pd.DataFrame
    monthly_frequency: pd.DataFrame
    metadata: dict[str, object]


def summarize_hour_frequency(
    hourly_frequency: pd.DataFrame,
    sessions: int,
) -> dict[str, object]:
    """Summarize when session extremes occur without implying a trading edge."""
    required = {"Jam WIT", "Frekuensi High", "Frekuensi Low"}
    missing = required.difference(hourly_frequency.columns)
    if missing:
        raise ValueError(f"Kolom frekuensi tidak lengkap: {', '.join(sorted(missing))}")
    if sessions <= 0:
        raise ValueError("Jumlah sesi harus lebih besar dari nol.")

    frame = hourly_frequency.loc[:, sorted(required)].copy()
    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(int)

    high_rank = frame.sort_values(
        ["Frekuensi High", "Jam WIT"], ascending=[False, True]
    ).head(3)
    low_rank = frame.sort_values(
        ["Frekuensi Low", "Jam WIT"], ascending=[False, True]
    ).head(3)
    opening = frame["Jam WIT"].between(7, 10)
    late_session = frame["Jam WIT"].isin([22, 23, 0, 1, 2, 3, 4])

    def _rank_rows(rank: pd.DataFrame, count_column: str) -> list[dict[str, object]]:
        return [
            {
                "hour": int(row["Jam WIT"]),
                "count": int(row[count_column]),
                "percentage": float(row[count_column]) / sessions * 100.0,
            }
            for _, row in rank.iterrows()
        ]

    return {
        "high_rank": _rank_rows(high_rank, "Frekuensi High"),
        "low_rank": _rank_rows(low_rank, "Frekuensi Low"),
        "top_three_high_pct": float(high_rank["Frekuensi High"].sum()) / sessions * 100.0,
        "top_three_low_pct": float(low_rank["Frekuensi Low"].sum()) / sessions * 100.0,
        "opening_high_pct": float(frame.loc[opening, "Frekuensi High"].sum()) / sessions * 100.0,
        "opening_low_pct": float(frame.loc[opening, "Frekuensi Low"].sum()) / sessions * 100.0,
        "late_high_pct": float(frame.loc[late_session, "Frekuensi High"].sum()) / sessions * 100.0,
        "late_low_pct": float(frame.loc[late_session, "Frekuensi Low"].sum()) / sessions * 100.0,
    }


def _prepare_bars(bars: pd.DataFrame) -> pd.DataFrame:
    frame = bars.copy()
    if "timestamp_utc" in frame.columns:
        frame["timestamp_utc"] = pd.to_datetime(
            frame["timestamp_utc"], errors="coerce", utc=True
        )
        frame = frame.set_index("timestamp_utc")
    else:
        index = pd.to_datetime(frame.index, errors="coerce", utc=True)
        frame.index = index

    frame = frame.loc[~frame.index.isna()].sort_index()
    frame = frame.loc[~frame.index.duplicated(keep="last")]
    required = ["Open", "High", "Low", "Close"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Kolom OHLC tidak lengkap: {', '.join(missing)}")
    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    valid = (
        frame[required].notna().all(axis=1)
        & frame["High"].ge(frame[["Open", "Close"]].max(axis=1))
        & frame["Low"].le(frame[["Open", "Close"]].min(axis=1))
        & frame["High"].ge(frame["Low"])
    )
    return frame.loc[valid, required]


def _exact_open(frame: pd.DataFrame, timestamp: pd.Timestamp) -> float | None:
    if timestamp not in frame.index:
        return None
    value = frame.loc[timestamp, "Open"]
    if isinstance(value, pd.Series):
        value = value.iloc[0]
    return float(value)


def build_session_hour_map(
    bars: pd.DataFrame,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
) -> SessionHourMap:
    frame = _prepare_bars(bars)
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()

    local_index = frame.index.tz_convert(WIT)
    local_hour = local_index.hour
    in_session = (local_hour >= SESSION_OPEN_HOUR) | (local_hour < SESSION_CLOSE_HOUR)
    frame = frame.loc[in_session].copy()
    local_index = frame.index.tz_convert(WIT)
    session_dates = local_index.tz_localize(None).normalize()
    session_dates = session_dates.where(
        local_index.hour >= SESSION_OPEN_HOUR,
        session_dates - pd.Timedelta(days=1),
    )
    frame["_timestamp_wit"] = local_index
    frame["_session_date"] = session_dates
    frame = frame.loc[frame["_session_date"].between(start, end)]

    rows: list[dict[str, object]] = []
    for session_date, session in frame.groupby("_session_date", sort=True):
        session = session.sort_index()
        session_date = pd.Timestamp(session_date)
        open_plus_one_at = (session_date + pd.Timedelta(hours=OPEN_PLUS_ONE_HOUR)).tz_localize(WIT)
        close_minus_one_at = (
            session_date + pd.Timedelta(days=1, hours=CLOSE_MINUS_ONE_HOUR)
        ).tz_localize(WIT)

        high_value = float(session["High"].max())
        low_value = float(session["Low"].min())
        high_rows = session.loc[session["High"].eq(high_value)]
        low_rows = session.loc[session["Low"].eq(low_value)]
        first_high = pd.Timestamp(high_rows.iloc[0]["_timestamp_wit"])
        first_low = pd.Timestamp(low_rows.iloc[0]["_timestamp_wit"])

        open_plus_one = _exact_open(session, open_plus_one_at.tz_convert("UTC"))
        close_minus_one = _exact_open(session, close_minus_one_at.tz_convert("UTC"))
        bar_count = int(len(session))
        rows.append(
            {
                "Tanggal sesi": session_date,
                "Open +1 Jam (08 WIT)": open_plus_one,
                "Status Open +1": "Tersedia" if open_plus_one is not None else "Tidak tersedia",
                "Close -1 Jam (04 WIT)": close_minus_one,
                "Status Close -1": "Tersedia" if close_minus_one is not None else "Tidak tersedia",
                "High": high_value,
                "Jam High WIT": int(first_high.hour),
                "Waktu High pertama WIT": first_high,
                "Jumlah kemunculan High": int(len(high_rows)),
                "Low": low_value,
                "Jam Low WIT": int(first_low.hour),
                "Waktu Low pertama WIT": first_low,
                "Jumlah kemunculan Low": int(len(low_rows)),
                "Jumlah candle M1": bar_count,
                "Cakupan sesi (%)": bar_count / EXPECTED_SESSION_MINUTES * 100.0,
            }
        )

    daily = pd.DataFrame(rows)
    if daily.empty:
        raise ValueError("Tidak ada sesi trading dalam periode yang diminta.")

    hourly = pd.DataFrame({"Jam WIT": range(24)})
    high_counts = daily["Jam High WIT"].value_counts()
    low_counts = daily["Jam Low WIT"].value_counts()
    hourly["Frekuensi High"] = hourly["Jam WIT"].map(high_counts).fillna(0).astype(int)
    hourly["Frekuensi Low"] = hourly["Jam WIT"].map(low_counts).fillna(0).astype(int)
    hourly["Persentase High (%)"] = hourly["Frekuensi High"] / len(daily) * 100.0
    hourly["Persentase Low (%)"] = hourly["Frekuensi Low"] / len(daily) * 100.0

    monthly_source = daily.copy()
    monthly_source["Bulan"] = monthly_source["Tanggal sesi"].dt.to_period("M").astype(str)
    monthly = (
        monthly_source.groupby(["Bulan", "Jam High WIT"]).size().rename("Frekuensi High").reset_index()
        .merge(
            monthly_source.groupby(["Bulan", "Jam Low WIT"]).size().rename("Frekuensi Low").reset_index(),
            left_on=["Bulan", "Jam High WIT"],
            right_on=["Bulan", "Jam Low WIT"],
            how="outer",
        )
    )
    monthly["Jam WIT"] = monthly["Jam High WIT"].fillna(monthly["Jam Low WIT"]).astype(int)
    monthly = monthly.drop(columns=["Jam High WIT", "Jam Low WIT"]).fillna(0)
    monthly[["Frekuensi High", "Frekuensi Low"]] = monthly[
        ["Frekuensi High", "Frekuensi Low"]
    ].astype(int)
    monthly = monthly.sort_values(["Bulan", "Jam WIT"]).reset_index(drop=True)

    metadata = {
        "period_start": start.strftime("%Y-%m-%d"),
        "period_end": end.strftime("%Y-%m-%d"),
        "session_definition": "07:00 WIT sampai sebelum 05:00 WIT hari berikutnya",
        "open_plus_one_definition": "Open candle M1 tepat 08:00 WIT",
        "close_minus_one_definition": "Open candle M1 tepat 04:00 WIT hari berikutnya",
        "sessions": int(len(daily)),
        "open_plus_one_available": int(daily["Open +1 Jam (08 WIT)"].notna().sum()),
        "close_minus_one_available": int(daily["Close -1 Jam (04 WIT)"].notna().sum()),
        "median_session_coverage_pct": float(daily["Cakupan sesi (%)"].median()),
    }
    return SessionHourMap(daily, hourly, monthly, metadata)
