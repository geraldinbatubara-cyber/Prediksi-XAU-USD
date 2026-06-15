from __future__ import annotations

import pandas as pd
import yfinance as yf


def load_gold_data(period: str = "10y") -> pd.DataFrame:
    """Download and normalize daily COMEX gold futures prices."""
    data = yf.download(
        "GC=F",
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False,
    )
    if data.empty:
        raise RuntimeError("Data harga emas tidak tersedia dari Yahoo Finance.")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise RuntimeError(f"Kolom data tidak lengkap: {', '.join(missing)}")

    frame = data[required].dropna(subset=["Close"]).copy()
    frame.index = pd.to_datetime(frame.index).tz_localize(None)
    return frame
