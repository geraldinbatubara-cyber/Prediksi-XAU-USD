from __future__ import annotations

import numpy as np
import pandas as pd


OHLC_COLUMNS = ["open", "high", "low", "close"]


def _normalize_bars(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=OHLC_COLUMNS)
    data = frame.copy()
    data.columns = [str(column).strip().lower() for column in data.columns]
    if "timestamp_utc" in data.columns:
        timestamps = pd.to_datetime(data["timestamp_utc"], errors="coerce", utc=True)
        data = data.assign(timestamp_utc=timestamps).dropna(subset=["timestamp_utc"])
        data = data.set_index("timestamp_utc")
    elif not isinstance(data.index, pd.DatetimeIndex):
        return pd.DataFrame(columns=OHLC_COLUMNS)
    else:
        index = pd.to_datetime(data.index, errors="coerce", utc=True)
        data.index = index
        data = data.loc[data.index.notna()]
    if not set(OHLC_COLUMNS).issubset(data.columns):
        return pd.DataFrame(columns=OHLC_COLUMNS)
    optional = [column for column in ["tick_volume", "spread_points"] if column in data.columns]
    data[OHLC_COLUMNS + optional] = data[OHLC_COLUMNS + optional].apply(
        pd.to_numeric, errors="coerce"
    )
    return (
        data[OHLC_COLUMNS + optional]
        .dropna(subset=OHLC_COLUMNS)
        .sort_index()
        .loc[lambda value: ~value.index.duplicated(keep="last")]
    )


def completed_m5_bars(
    m1_bars: pd.DataFrame | None,
    now: pd.Timestamp | None = None,
) -> pd.DataFrame:
    data = _normalize_bars(m1_bars)
    if data.empty:
        return data
    aggregation: dict[str, str] = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }
    if "tick_volume" in data.columns:
        aggregation["tick_volume"] = "sum"
    if "spread_points" in data.columns:
        aggregation["spread_points"] = "mean"
    resampled = data.resample("5min", label="left", closed="left").agg(aggregation)
    resampled = resampled.dropna(subset=OHLC_COLUMNS)
    now_utc = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if now_utc.tzinfo is None:
        now_utc = now_utc.tz_localize("UTC")
    else:
        now_utc = now_utc.tz_convert("UTC")
    return resampled.loc[resampled.index + pd.Timedelta(minutes=5) <= now_utc]


def _atr(data: pd.DataFrame, length: int = 14) -> pd.Series:
    previous_close = data["close"].shift(1)
    true_range = pd.concat(
        [
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()


def _rsi(close: pd.Series, length: int = 14) -> pd.Series:
    change = close.diff()
    gain = change.clip(lower=0).ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    loss = (-change.clip(upper=0)).ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    relative_strength = gain / loss.replace(0, np.nan)
    result = 100 - (100 / (1 + relative_strength))
    return result.where(loss.ne(0), 100.0).fillna(50.0)


def _pivot_positions(values: pd.Series, mode: str, window: int = 2) -> list[int]:
    array = values.to_numpy(dtype=float)
    positions: list[int] = []
    for position in range(window, len(array) - window):
        neighborhood = array[position - window : position + window + 1]
        target = array[position]
        if mode == "high" and target == np.max(neighborhood) and np.sum(neighborhood == target) == 1:
            positions.append(position)
        if mode == "low" and target == np.min(neighborhood) and np.sum(neighborhood == target) == 1:
            positions.append(position)
    return positions


def _pattern_result(
    name: str,
    status: str,
    bias: int,
    confidence: int,
    detail: str,
) -> dict[str, object]:
    return {
        "name": name,
        "status": status,
        "bias": bias,
        "confidence": confidence,
        "detail": detail,
    }


def detect_price_pattern(data: pd.DataFrame, atr_value: float) -> dict[str, object]:
    recent = data.tail(120).copy()
    if len(recent) < 25 or not np.isfinite(atr_value) or atr_value <= 0:
        return _pattern_result("Belum cukup data", "TIDAK VALID", 0, 0, "Pola memerlukan candle selesai dan ATR valid.")

    close = float(recent.iloc[-1]["close"])
    highs = _pivot_positions(recent["high"], "high")
    lows = _pivot_positions(recent["low"], "low")
    shoulder_tolerance = max(1.5 * atr_value, close * 0.012)
    prominence = max(0.5 * atr_value, close * 0.003)

    if len(highs) >= 3:
        left, head, right = highs[-3:]
        left_value, head_value, right_value = recent["high"].iloc[[left, head, right]]
        between_lows = recent["low"].iloc[left : right + 1]
        neckline = float(between_lows.min())
        geometry = (
            head_value > left_value + prominence
            and head_value > right_value + prominence
            and abs(left_value - right_value) <= shoulder_tolerance
        )
        if geometry:
            confirmed = close < neckline
            return _pattern_result(
                "Head and Shoulders",
                "TERKONFIRMASI" if confirmed else "KANDIDAT",
                -1 if confirmed else 0,
                82 if confirmed else 62,
                (
                    f"Neckline {neckline:,.2f} sudah ditembus; bias bearish."
                    if confirmed
                    else f"Geometri tiga puncak terbentuk, tetapi close belum di bawah neckline {neckline:,.2f}."
                ),
            )

    if len(lows) >= 3:
        left, head, right = lows[-3:]
        left_value, head_value, right_value = recent["low"].iloc[[left, head, right]]
        between_highs = recent["high"].iloc[left : right + 1]
        neckline = float(between_highs.max())
        geometry = (
            head_value < left_value - prominence
            and head_value < right_value - prominence
            and abs(left_value - right_value) <= shoulder_tolerance
        )
        if geometry:
            confirmed = close > neckline
            return _pattern_result(
                "Inverse Head and Shoulders",
                "TERKONFIRMASI" if confirmed else "KANDIDAT",
                1 if confirmed else 0,
                82 if confirmed else 62,
                (
                    f"Neckline {neckline:,.2f} sudah ditembus; bias bullish."
                    if confirmed
                    else f"Geometri tiga lembah terbentuk, tetapi close belum di atas neckline {neckline:,.2f}."
                ),
            )

    if len(highs) >= 2:
        first, second = highs[-2:]
        first_value, second_value = recent["high"].iloc[[first, second]]
        valley = float(recent["low"].iloc[first : second + 1].min())
        if abs(first_value - second_value) <= max(atr_value, close * 0.008) and min(first_value, second_value) - valley >= atr_value:
            confirmed = close < valley
            return _pattern_result(
                "Double Top",
                "TERKONFIRMASI" if confirmed else "KANDIDAT",
                -1 if confirmed else 0,
                76 if confirmed else 58,
                f"Support antar puncak {valley:,.2f} {'sudah' if confirmed else 'belum'} ditembus.",
            )

    if len(lows) >= 2:
        first, second = lows[-2:]
        first_value, second_value = recent["low"].iloc[[first, second]]
        peak = float(recent["high"].iloc[first : second + 1].max())
        if abs(first_value - second_value) <= max(atr_value, close * 0.008) and peak - max(first_value, second_value) >= atr_value:
            confirmed = close > peak
            return _pattern_result(
                "Double Bottom",
                "TERKONFIRMASI" if confirmed else "KANDIDAT",
                1 if confirmed else 0,
                76 if confirmed else 58,
                f"Resistance antar lembah {peak:,.2f} {'sudah' if confirmed else 'belum'} ditembus.",
            )

    previous = recent.iloc[-2]
    latest = recent.iloc[-1]
    latest_body = abs(float(latest["close"] - latest["open"]))
    previous_body = abs(float(previous["close"] - previous["open"]))
    candle_range = max(float(latest["high"] - latest["low"]), 1e-9)
    if latest["close"] > latest["open"] and previous["close"] < previous["open"] and latest["open"] <= previous["close"] and latest["close"] >= previous["open"]:
        return _pattern_result("Bullish Engulfing", "TERKONFIRMASI", 1, 68, "Body candle bullish menelan body bearish sebelumnya.")
    if latest["close"] < latest["open"] and previous["close"] > previous["open"] and latest["open"] >= previous["close"] and latest["close"] <= previous["open"]:
        return _pattern_result("Bearish Engulfing", "TERKONFIRMASI", -1, 68, "Body candle bearish menelan body bullish sebelumnya.")
    if latest_body / candle_range <= 0.12:
        return _pattern_result("Doji", "TERKONFIRMASI", 0, 55, "Body sangat kecil; pasar menunjukkan keraguan, bukan arah entry mandiri.")
    lower_wick = min(float(latest["open"]), float(latest["close"])) - float(latest["low"])
    upper_wick = float(latest["high"]) - max(float(latest["open"]), float(latest["close"]))
    if lower_wick >= 2 * max(latest_body, 1e-9) and upper_wick <= latest_body:
        return _pattern_result("Hammer / Bullish Pin Bar", "TERKONFIRMASI", 1, 62, "Penolakan harga bawah terdeteksi; perlu konfirmasi tren dan support.")
    if upper_wick >= 2 * max(latest_body, 1e-9) and lower_wick <= latest_body:
        return _pattern_result("Shooting Star / Bearish Pin Bar", "TERKONFIRMASI", -1, 62, "Penolakan harga atas terdeteksi; perlu konfirmasi tren dan resistance.")

    prior_high = float(recent["high"].iloc[-21:-1].max())
    prior_low = float(recent["low"].iloc[-21:-1].min())
    if close > prior_high:
        return _pattern_result("Breakout Resistance", "TERKONFIRMASI", 1, 72, f"Close menembus resistance 20 candle {prior_high:,.2f}.")
    if close < prior_low:
        return _pattern_result("Breakdown Support", "TERKONFIRMASI", -1, 72, f"Close menembus support 20 candle {prior_low:,.2f}.")

    recent_range = float(recent["high"].tail(20).max() - recent["low"].tail(20).min())
    previous_range = float(recent["high"].iloc[-40:-20].max() - recent["low"].iloc[-40:-20].min())
    if previous_range > 0 and recent_range / previous_range < 0.72:
        return _pattern_result("Range Compression", "TERKONFIRMASI", 0, 60, "Range 20 candle menyempit; tunggu breakout terkonfirmasi.")

    return _pattern_result("Tidak ada pola valid", "NETRAL", 0, 40, "Tidak ada geometri atau candle pattern yang memenuhi ambang objektif.")


def analyze_timeframe(
    frame: pd.DataFrame | None,
    timeframe: str,
    source: str,
    minimum_bars: int,
) -> dict[str, object]:
    data = _normalize_bars(frame)
    if len(data) < minimum_bars:
        return {
            "timeframe": timeframe,
            "valid": False,
            "signal": "TIDAK VALID",
            "confidence": 0,
            "pattern": "Belum cukup data",
            "pattern_status": "TIDAK VALID",
            "trend": "-",
            "rsi": np.nan,
            "atr": np.nan,
            "support": np.nan,
            "resistance": np.nan,
            "last_close": np.nan if data.empty else float(data.iloc[-1]["close"]),
            "last_timestamp": pd.NaT if data.empty else data.index.max(),
            "source": source,
            "interpretation": f"Memerlukan minimal {minimum_bars} candle selesai; tersedia {len(data)}.",
            "bars": len(data),
        }

    close = data["close"]
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    atr_series = _atr(data)
    rsi_series = _rsi(close)
    latest_close = float(close.iloc[-1])
    atr_value = float(atr_series.iloc[-1])
    rsi_value = float(rsi_series.iloc[-1])
    slope = float(ema20.iloc[-1] - ema20.iloc[-6])
    if latest_close > ema20.iloc[-1] > ema50.iloc[-1] and slope > 0:
        trend = "BULLISH"
        trend_bias = 1
    elif latest_close < ema20.iloc[-1] < ema50.iloc[-1] and slope < 0:
        trend = "BEARISH"
        trend_bias = -1
    else:
        trend = "SIDEWAYS / TRANSITION"
        trend_bias = 0

    support = float(data["low"].iloc[-21:-1].min())
    resistance = float(data["high"].iloc[-21:-1].max())
    pattern = detect_price_pattern(data, atr_value)
    score = trend_bias + int(pattern["bias"]) * 2
    if rsi_value >= 58:
        score += 1
    elif rsi_value <= 42:
        score -= 1
    if latest_close > resistance:
        score += 1
    elif latest_close < support:
        score -= 1
    signal = "BUY" if score >= 2 else "SELL" if score <= -2 else "WAIT"
    confidence = min(90, 50 + abs(score) * 10)
    interpretation = (
        f"Bias {signal}: tren {trend.lower()}, RSI {rsi_value:.1f}, dan pola "
        f"{pattern['name']} ({str(pattern['status']).lower()}). {pattern['detail']}"
    )
    return {
        "timeframe": timeframe,
        "valid": True,
        "signal": signal,
        "confidence": confidence,
        "pattern": pattern["name"],
        "pattern_status": pattern["status"],
        "pattern_confidence": pattern["confidence"],
        "trend": trend,
        "rsi": rsi_value,
        "atr": atr_value,
        "support": support,
        "resistance": resistance,
        "last_close": latest_close,
        "last_timestamp": data.index.max(),
        "source": source,
        "interpretation": interpretation,
        "bars": len(data),
    }


def build_multitimeframe_analysis(
    m1_bars: pd.DataFrame | None,
    h1_bars: pd.DataFrame | None,
    d1_bars: pd.DataFrame | None,
    now: pd.Timestamp | None = None,
) -> dict[str, dict[str, object]]:
    m5 = completed_m5_bars(m1_bars, now=now)
    return {
        "M5": analyze_timeframe(m5, "M5", "MT5 M1 diresample, candle selesai", 60),
        "H1": analyze_timeframe(h1_bars, "H1", "MT5 H1 langsung, candle selesai", 80),
        "D1": analyze_timeframe(d1_bars, "D1", "MT5 D1 langsung, candle selesai", 60),
    }
