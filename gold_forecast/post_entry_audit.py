from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


WIT = "Asia/Jayapura"


@dataclass(frozen=True)
class PostEntryAudit:
    paths: pd.DataFrame
    summary: pd.DataFrame
    metadata: dict[str, object]


def _prepare_bars(bars: pd.DataFrame) -> pd.DataFrame:
    frame = bars.copy()
    frame["timestamp_utc"] = pd.to_datetime(
        frame["timestamp_utc"], errors="coerce", utc=True
    )
    frame = frame.dropna(subset=["timestamp_utc"]).set_index("timestamp_utc")
    frame = frame.sort_index().loc[lambda value: ~value.index.duplicated(keep="last")]
    required = ["Open", "High", "Low", "Close", "SpreadPoints"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Kolom audit jalur tidak lengkap: {', '.join(missing)}")
    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    valid = (
        frame[required].notna().all(axis=1)
        & frame["SpreadPoints"].ge(0)
        & frame["High"].ge(frame[["Open", "Close"]].max(axis=1))
        & frame["Low"].le(frame[["Open", "Close"]].min(axis=1))
        & frame["High"].ge(frame["Low"])
    )
    return frame.loc[valid, required]


def _first_index(mask: pd.Series) -> pd.Timestamp | None:
    matches = mask.index[mask]
    return None if len(matches) == 0 else pd.Timestamp(matches[0])


def _evaluate_direction(
    session_date: pd.Timestamp,
    path: pd.DataFrame,
    entry_row: pd.Series,
    close_row: pd.Series,
    direction: str,
    tp_usd: float,
    sl_usd: float,
    point_size: float,
) -> dict[str, object]:
    entry_bid = float(entry_row["Open"])
    entry_spread = float(entry_row["SpreadPoints"]) * point_size
    entry_price = entry_bid + entry_spread if direction == "BUY" else entry_bid
    spread = path["SpreadPoints"] * point_size

    if direction == "BUY":
        tp_level = entry_price + tp_usd
        sl_level = entry_price - sl_usd
        tp_mask = path["High"].ge(tp_level)
        sl_mask = path["Low"].le(sl_level)
        favorable = path["High"] - entry_price
        adverse = entry_price - path["Low"]
        close_price = float(close_row["Open"])
        close_pnl = close_price - entry_price
    else:
        tp_level = entry_price - tp_usd
        sl_level = entry_price + sl_usd
        ask_low = path["Low"] + spread
        ask_high = path["High"] + spread
        tp_mask = ask_low.le(tp_level)
        sl_mask = ask_high.ge(sl_level)
        favorable = entry_price - ask_low
        adverse = ask_high - entry_price
        close_price = float(close_row["Open"]) + float(close_row["SpreadPoints"]) * point_size
        close_pnl = entry_price - close_price

    tp_at = _first_index(tp_mask)
    sl_at = _first_index(sl_mask)
    ambiguous = tp_at is not None and sl_at is not None and tp_at == sl_at
    if sl_at is not None and (tp_at is None or sl_at <= tp_at):
        outcome = "SL"
        exit_at = sl_at
        exit_price = sl_level
        pnl = -sl_usd
    elif tp_at is not None:
        outcome = "TP"
        exit_at = tp_at
        exit_price = tp_level
        pnl = tp_usd
    else:
        outcome = "CLOSE_04_PROFIT" if close_pnl > 0 else "CLOSE_04_LOSS"
        if close_pnl == 0:
            outcome = "CLOSE_04_FLAT"
        exit_at = pd.Timestamp(close_row.name)
        exit_price = close_price
        pnl = float(close_pnl)

    mfe_at = pd.Timestamp(favorable.idxmax())
    mae_at = pd.Timestamp(adverse.idxmax())
    return {
        "Tanggal sesi": session_date,
        "Arah": direction,
        "Status data": "Valid",
        "Entry WIT": pd.Timestamp(entry_row.name).tz_convert(WIT),
        "Entry price": entry_price,
        "Spread entry (USD/oz)": entry_spread,
        "TP level": tp_level,
        "SL level": sl_level,
        "Outcome": outcome,
        "Exit WIT": exit_at.tz_convert(WIT),
        "Exit price": exit_price,
        "P/L 0.01 lot (USD)": pnl,
        "MFE (USD/oz)": float(favorable.max()),
        "MFE WIT": mfe_at.tz_convert(WIT),
        "MAE (USD/oz)": float(adverse.max()),
        "MAE WIT": mae_at.tz_convert(WIT),
        "Candle jalur": int(len(path)),
        "TP dan SL satu candle": bool(ambiguous),
    }


def _unavailable_rows(session_date: pd.Timestamp, reason: str) -> list[dict[str, object]]:
    return [
        {
            "Tanggal sesi": session_date,
            "Arah": direction,
            "Status data": reason,
            "Outcome": "DATA_UNAVAILABLE",
        }
        for direction in ("BUY", "SELL")
    ]


def _summarize(paths: pd.DataFrame) -> pd.DataFrame:
    valid = paths.loc[paths["Status data"].eq("Valid")].copy()
    valid["Periode"] = np.where(
        valid["Tanggal sesi"].dt.year.eq(2025), "2025", "2026H1"
    )
    valid = pd.concat([valid.assign(Periode="Seluruh periode"), valid], ignore_index=True)
    rows = []
    for (period, direction), group in valid.groupby(["Periode", "Arah"], sort=False):
        profits = group["P/L 0.01 lot (USD)"].clip(lower=0).sum()
        losses = -group["P/L 0.01 lot (USD)"].clip(upper=0).sum()
        rows.append(
            {
                "Periode": period,
                "Arah": direction,
                "Sesi valid": int(len(group)),
                "TP": int(group["Outcome"].eq("TP").sum()),
                "SL": int(group["Outcome"].eq("SL").sum()),
                "Close 04 profit": int(group["Outcome"].eq("CLOSE_04_PROFIT").sum()),
                "Close 04 loss": int(group["Outcome"].eq("CLOSE_04_LOSS").sum()),
                "Close 04 flat": int(group["Outcome"].eq("CLOSE_04_FLAT").sum()),
                "Win rate (%)": float(group["P/L 0.01 lot (USD)"].gt(0).mean() * 100),
                "Total P/L (USD)": float(group["P/L 0.01 lot (USD)"].sum()),
                "Rata-rata P/L (USD)": float(group["P/L 0.01 lot (USD)"].mean()),
                "Profit factor": float(profits / losses) if losses > 0 else np.nan,
                "Rata-rata MFE": float(group["MFE (USD/oz)"].mean()),
                "Rata-rata MAE": float(group["MAE (USD/oz)"].mean()),
                "Barrier ambigu": int(group["TP dan SL satu candle"].sum()),
            }
        )
    return pd.DataFrame(rows)


def build_post_entry_audit(
    bars: pd.DataFrame,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    tp_usd: float = 25.0,
    sl_usd: float = 10.0,
    point_size: float = 0.01,
) -> PostEntryAudit:
    frame = _prepare_bars(bars)
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    rows: list[dict[str, object]] = []

    for session_date in pd.date_range(start, end, freq="D"):
        entry_at = (session_date + pd.Timedelta(hours=8)).tz_localize(WIT).tz_convert("UTC")
        close_at = (
            session_date + pd.Timedelta(days=1, hours=4)
        ).tz_localize(WIT).tz_convert("UTC")
        session = frame.loc[(frame.index >= entry_at) & (frame.index <= close_at)]
        if session.empty:
            continue
        if entry_at not in session.index:
            rows.extend(_unavailable_rows(session_date, "Candle 08:00 tidak tersedia"))
            continue
        if close_at not in session.index:
            rows.extend(_unavailable_rows(session_date, "Candle 04:00 tidak tersedia"))
            continue
        path = session.loc[session.index < close_at]
        if path.empty:
            rows.extend(_unavailable_rows(session_date, "Jalur setelah entry kosong"))
            continue
        entry_row = session.loc[entry_at]
        close_row = session.loc[close_at]
        for direction in ("BUY", "SELL"):
            rows.append(
                _evaluate_direction(
                    session_date,
                    path,
                    entry_row,
                    close_row,
                    direction,
                    tp_usd,
                    sl_usd,
                    point_size,
                )
            )

    paths = pd.DataFrame(rows)
    if paths.empty:
        raise ValueError("Tidak ada jalur harga setelah 08:00 dalam periode audit.")
    paths["Tanggal sesi"] = pd.to_datetime(paths["Tanggal sesi"])
    summary = _summarize(paths)
    metadata = {
        "period_start": start.strftime("%Y-%m-%d"),
        "period_end": end.strftime("%Y-%m-%d"),
        "entry_definition": "Open candle M1 tepat 08:00 WIT",
        "monitoring_definition": "Candle M1 mulai 08:00 WIT sampai sebelum 04:00 WIT",
        "fallback_exit_definition": "Open candle M1 tepat 04:00 WIT",
        "tp_usd": tp_usd,
        "sl_usd": sl_usd,
        "lot": 0.01,
        "spread": "SpreadPoints historis dikonversi dengan point size 0.01",
        "slippage": "Tidak diterapkan karena data slippage historis tidak tersedia",
        "same_candle_rule": "SL dianggap tersentuh lebih dahulu",
        "valid_sessions": int(paths.loc[paths["Status data"].eq("Valid"), "Tanggal sesi"].nunique()),
        "unavailable_sessions": int(paths.loc[~paths["Status data"].eq("Valid"), "Tanggal sesi"].nunique()),
    }
    return PostEntryAudit(paths, summary, metadata)
