from __future__ import annotations

import time
from pathlib import Path
from tempfile import gettempdir

import pandas as pd
import yfinance as yf


YFINANCE_CACHE_DIR = Path(gettempdir()) / "prediksi-xau-usd-yfinance"
YFINANCE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
if hasattr(yf, "set_tz_cache_location"):
    yf.set_tz_cache_location(str(YFINANCE_CACHE_DIR))

MARKET_SYMBOLS = {
    "gold": "GC=F",
    "dollar": "DX-Y.NYB",
    "treasury_10y": "^TNX",
    "oil": "CL=F",
    "vix": "^VIX",
    "silver": "SI=F",
}
CACHE_DIR = Path("data/cache")
GOLD_CACHE_PATH = CACHE_DIR / "gold_ohlc.csv"
MARKET_CACHE_PATH = CACHE_DIR / "market_data.csv"
REQUIRED_CACHE_START = pd.Timestamp("2025-01-01")


def _read_cached_frame(path: Path, required_columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        return pd.DataFrame()
    frame.index = pd.to_datetime(frame.index).tz_localize(None)
    frame = frame.loc[frame.index >= REQUIRED_CACHE_START]
    return frame[required_columns].sort_index().dropna(how="all")


def _write_cached_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = frame.copy()
    output.index = pd.to_datetime(output.index).tz_localize(None)
    output = output[~output.index.duplicated(keep="last")].sort_index()
    output.to_csv(path, index_label="Date")


def _download_gold_data(period: str = "10y") -> pd.DataFrame:
    """Download and normalize daily COMEX gold futures prices from Yahoo."""
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


def _download_market_data(period: str = "10y") -> pd.DataFrame:
    """Download aligned close prices for gold and related global markets from Yahoo."""
    symbols = list(MARKET_SYMBOLS.values())
    data = pd.DataFrame()
    for attempt in range(3):
        data = yf.download(
            symbols,
            period=period,
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
            group_by="column",
        )
        if not data.empty:
            break
        time.sleep(1 + attempt)

    series: dict[str, pd.Series] = {}
    if not data.empty and isinstance(data.columns, pd.MultiIndex):
        close_data = data["Close"]
        for name, symbol in MARKET_SYMBOLS.items():
            if symbol in close_data.columns and close_data[symbol].notna().any():
                series[name] = close_data[symbol].rename(name)

    if "gold" not in series:
        gold = _download_gold_data(period)
        series["gold"] = gold["Close"].rename("gold")

    if "gold" not in series:
        raise RuntimeError("Data harga emas tidak tersedia dari Yahoo Finance.")

    frame = pd.concat(series.values(), axis=1).sort_index()
    for name in MARKET_SYMBOLS:
        if name not in frame.columns:
            frame[name] = pd.NA
    frame = frame[list(MARKET_SYMBOLS.keys())]
    frame = frame.ffill(limit=3).dropna(subset=["gold"])
    return frame


def load_gold_data(period: str = "10y") -> pd.DataFrame:
    required = ["Open", "High", "Low", "Close", "Volume"]
    cached = _read_cached_frame(GOLD_CACHE_PATH, required)
    if not cached.empty:
        return cached.dropna(subset=["Close"])
    return _download_gold_data(period)


def load_market_data(period: str = "10y") -> pd.DataFrame:
    required = list(MARKET_SYMBOLS.keys())
    cached = _read_cached_frame(MARKET_CACHE_PATH, required)
    if not cached.empty:
        return cached.ffill(limit=3).dropna(subset=["gold"])
    return _download_market_data(period)


def refresh_market_cache(incremental_period: str = "14d", bootstrap_period: str = "10y") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Update local CSV caches. Use a short Yahoo window when cache already exists."""
    gold_cached = _read_cached_frame(GOLD_CACHE_PATH, ["Open", "High", "Low", "Close", "Volume"])
    market_cached = _read_cached_frame(MARKET_CACHE_PATH, list(MARKET_SYMBOLS.keys()))
    has_required_history = (
        not gold_cached.empty
        and not market_cached.empty
        and gold_cached.index.min() <= REQUIRED_CACHE_START
        and market_cached.index.min() <= REQUIRED_CACHE_START
    )
    period = incremental_period if has_required_history else bootstrap_period

    gold_latest = _download_gold_data(period)
    market_latest = _download_market_data(period)
    gold = pd.concat([gold_cached, gold_latest]).sort_index()
    market = pd.concat([market_cached, market_latest]).sort_index()

    gold = gold[~gold.index.duplicated(keep="last")].dropna(subset=["Close"])
    market = market[~market.index.duplicated(keep="last")].ffill(limit=3).dropna(subset=["gold"])
    gold = gold.loc[gold.index >= REQUIRED_CACHE_START]
    market = market.loc[market.index >= REQUIRED_CACHE_START]
    _write_cached_frame(gold, GOLD_CACHE_PATH)
    _write_cached_frame(market, MARKET_CACHE_PATH)
    return gold, market
