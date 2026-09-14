from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from gold_forecast.technical_analysis import completed_timeframe_bars


MINIMUM_H1_BARS = 240
FEATURE_COLUMNS = (
    "return_1",
    "return_3",
    "return_6",
    "return_12",
    "return_24",
    "atr_relative",
    "rsi_14",
    "ema_8_21",
    "ema_21_55",
    "volatility_6",
    "volatility_24",
    "range_position_24",
    "body_atr",
    "upper_wick_atr",
    "lower_wick_atr",
    "hour_sin",
    "hour_cos",
)


def build_h1_forecast(
    h1_bars: pd.DataFrame | None,
    now: object | None = None,
) -> dict[str, object]:
    completed = completed_timeframe_bars(
        h1_bars,
        pd.Timedelta(hours=1),
        now=pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now),
    )
    if len(completed) < MINIMUM_H1_BARS:
        return _invalid_result(
            completed,
            f"Memerlukan minimal {MINIMUM_H1_BARS} candle H1 selesai; tersedia {len(completed)}.",
        )

    features = _feature_frame(completed)
    latest_features = features.dropna(subset=list(FEATURE_COLUMNS))
    if latest_features.empty:
        return _invalid_result(completed, "Warm-up fitur H1 belum lengkap.")

    forecasts = {}
    for horizon in (1, 6):
        forecast = _fit_horizon(completed, features, horizon)
        if forecast is None:
            return _invalid_result(
                completed,
                f"Sampel temporal H+{horizon} belum cukup untuk training dan validasi.",
            )
        forecasts[horizon] = forecast

    source_time = pd.Timestamp(completed.index.max())
    source_close = float(completed.iloc[-1]["close"])
    now_utc = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    now_utc = (
        now_utc.tz_localize("UTC")
        if now_utc.tzinfo is None
        else now_utc.tz_convert("UTC")
    )
    source_utc = (
        source_time.tz_localize("UTC")
        if source_time.tzinfo is None
        else source_time.tz_convert("UTC")
    )
    age_hours = max(
        (now_utc - (source_utc + pd.Timedelta(hours=1))).total_seconds() / 3600,
        0.0,
    )
    primary = forecasts[1]
    threshold = max(0.0005, float(primary["mae_return"]) * 0.35)
    predicted_return = float(primary["predicted_return"])
    signal = (
        "BULLISH"
        if predicted_return >= threshold
        else "BEARISH"
        if predicted_return <= -threshold
        else "NETRAL"
    )
    return {
        "valid": True,
        "fresh": age_hours <= 3.0,
        "status": "AKTIF" if age_hours <= 3.0 else "SNAPSHOT H1 TERTINGGAL",
        "signal": signal,
        "confidence": float(primary["directional_accuracy"]),
        "source_time": source_time,
        "source_close": source_close,
        "age_hours": age_hours,
        "bars": len(completed),
        "feature_rows": len(latest_features),
        "forecasts": forecasts,
        "model": "Ridge H1 dengan holdout temporal",
    }


def _feature_frame(data: pd.DataFrame) -> pd.DataFrame:
    close = data["close"].astype(float)
    high = data["high"].astype(float)
    low = data["low"].astype(float)
    open_price = data["open"].astype(float)
    returns = close.pct_change()
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = true_range.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    change = close.diff()
    gain = change.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    loss = (-change.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    relative_strength = gain / loss.replace(0, np.nan)
    rsi = (100 - 100 / (1 + relative_strength)).where(loss.ne(0), 100.0)
    ema8 = close.ewm(span=8, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    ema55 = close.ewm(span=55, adjust=False).mean()
    rolling_high = high.rolling(24).max()
    rolling_low = low.rolling(24).min()
    rolling_range = (rolling_high - rolling_low).replace(0, np.nan)
    candle_body = (close - open_price).abs()
    upper_wick = high - pd.concat([open_price, close], axis=1).max(axis=1)
    lower_wick = pd.concat([open_price, close], axis=1).min(axis=1) - low
    hour = pd.Series(data.index.hour, index=data.index, dtype=float)

    frame = pd.DataFrame(index=data.index)
    for window in (1, 3, 6, 12, 24):
        frame[f"return_{window}"] = close.pct_change(window)
    frame["atr_relative"] = atr / close
    frame["rsi_14"] = rsi / 100
    frame["ema_8_21"] = (ema8 - ema21) / close
    frame["ema_21_55"] = (ema21 - ema55) / close
    frame["volatility_6"] = returns.rolling(6).std()
    frame["volatility_24"] = returns.rolling(24).std()
    frame["range_position_24"] = (close - rolling_low) / rolling_range
    frame["body_atr"] = candle_body / atr
    frame["upper_wick_atr"] = upper_wick / atr
    frame["lower_wick_atr"] = lower_wick / atr
    frame["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    frame["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    return frame.replace([np.inf, -np.inf], np.nan)


def _fit_horizon(
    data: pd.DataFrame,
    features: pd.DataFrame,
    horizon: int,
) -> dict[str, float] | None:
    target = data["close"].shift(-horizon) / data["close"] - 1
    dataset = features.loc[:, FEATURE_COLUMNS].copy()
    dataset["target"] = target
    dataset["reference_close"] = data["close"]
    dataset = dataset.dropna()
    if len(dataset) < 180:
        return None

    validation_rows = max(40, min(80, len(dataset) // 5))
    train = dataset.iloc[:-validation_rows]
    validation = dataset.iloc[-validation_rows:]
    if len(train) < 120:
        return None

    validation_model = make_pipeline(StandardScaler(), Ridge(alpha=4.0))
    validation_model.fit(train.loc[:, FEATURE_COLUMNS], train["target"])
    validation_prediction = validation_model.predict(
        validation.loc[:, FEATURE_COLUMNS]
    )
    residual = validation["target"].to_numpy() - validation_prediction
    mae_return = float(np.mean(np.abs(residual)))
    mae_price = float(
        np.mean(np.abs(residual) * validation["reference_close"].to_numpy())
    )
    directional_accuracy = float(
        np.mean(
            np.sign(validation_prediction)
            == np.sign(validation["target"].to_numpy())
        )
        * 100
    )

    final_model = make_pipeline(StandardScaler(), Ridge(alpha=4.0))
    final_model.fit(dataset.loc[:, FEATURE_COLUMNS], dataset["target"])
    latest = features.loc[:, FEATURE_COLUMNS].dropna().tail(1)
    predicted_return = float(final_model.predict(latest)[0])
    source_close = float(data.loc[latest.index[-1], "close"])
    lower_return = predicted_return + float(np.quantile(residual, 0.10))
    upper_return = predicted_return + float(np.quantile(residual, 0.90))
    return {
        "prediction": source_close * (1 + predicted_return),
        "predicted_return": predicted_return,
        "lower": source_close * (1 + lower_return),
        "upper": source_close * (1 + upper_return),
        "mae_price": mae_price,
        "mae_return": mae_return,
        "directional_accuracy": directional_accuracy,
        "training_rows": float(len(dataset)),
        "validation_rows": float(validation_rows),
    }


def _invalid_result(data: pd.DataFrame, reason: str) -> dict[str, object]:
    return {
        "valid": False,
        "fresh": False,
        "status": "TIDAK VALID",
        "signal": "TIDAK VALID",
        "confidence": 0.0,
        "source_time": pd.NaT if data.empty else data.index.max(),
        "source_close": np.nan if data.empty else float(data.iloc[-1]["close"]),
        "age_hours": np.nan,
        "bars": len(data),
        "feature_rows": 0,
        "forecasts": {},
        "model": "Ridge H1 dengan holdout temporal",
        "reason": reason,
    }
