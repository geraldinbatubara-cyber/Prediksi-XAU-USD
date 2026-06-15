from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


LAGS = (1, 2, 3, 5, 10, 20)


@dataclass
class ForecastResult:
    forecast: pd.DataFrame
    metrics: dict[str, float]


def _features(close: pd.Series) -> pd.DataFrame:
    frame = pd.DataFrame(index=close.index)
    frame["current_close"] = close
    for lag in LAGS:
        frame[f"lag_{lag}"] = close.shift(lag)
    frame["return_1d"] = close.pct_change()
    frame["return_5d"] = close.pct_change(5)
    frame["ma_5_ratio"] = close / close.rolling(5).mean() - 1
    frame["ma_20_ratio"] = close / close.rolling(20).mean() - 1
    frame["volatility_10d"] = close.pct_change().rolling(10).std()
    return frame


class RidgeRegressor:
    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha

    def fit(self, features: pd.DataFrame, target: pd.Series) -> None:
        values = features.to_numpy(dtype=float)
        self.mean = values.mean(axis=0)
        self.scale = values.std(axis=0)
        self.scale[self.scale == 0] = 1.0
        normalized = (values - self.mean) / self.scale
        design = np.column_stack([np.ones(len(normalized)), normalized])
        penalty = np.eye(design.shape[1]) * self.alpha
        penalty[0, 0] = 0
        self.coefficients = np.linalg.solve(
            design.T @ design + penalty,
            design.T @ target.to_numpy(dtype=float),
        )

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        values = (features.to_numpy(dtype=float) - self.mean) / self.scale
        design = np.column_stack([np.ones(len(values)), values])
        return design @ self.coefficients


def train_and_forecast(close: pd.Series, horizon: int = 7) -> ForecastResult:
    features = _features(close)
    dataset = features.copy()
    dataset["target"] = close.shift(-1)
    dataset = dataset.dropna()
    if len(dataset) < 250:
        raise ValueError("Minimal 250 observasi bersih diperlukan untuk pelatihan.")

    split = int(len(dataset) * 0.8)
    train, test = dataset.iloc[:split], dataset.iloc[split:]
    feature_names = list(features.columns)
    estimator = RidgeRegressor(alpha=10.0)
    estimator.fit(train[feature_names], train["target"])
    predicted = pd.Series(estimator.predict(test[feature_names]), index=test.index)
    actual = test["target"]
    residuals = actual - predicted
    current_test_price = close.reindex(test.index)

    metrics = {
        "MAE": float(np.mean(np.abs(actual - predicted))),
        "RMSE": float(np.sqrt(np.mean((actual - predicted) ** 2))),
        "MAPE": float(np.mean(np.abs((actual - predicted) / actual)) * 100),
        "Akurasi arah": float(((actual > current_test_price) == (predicted > current_test_price)).mean() * 100),
    }

    estimator.fit(dataset[feature_names], dataset["target"])
    history = close.copy()
    rows: list[dict[str, float | pd.Timestamp]] = []
    residual_std = float(residuals.std())
    for step in range(1, horizon + 1):
        feature_row = _features(history).iloc[[-1]][feature_names]
        point = float(estimator.predict(feature_row)[0])
        next_date = history.index[-1] + pd.offsets.BDay(1)
        uncertainty = 1.96 * residual_std * np.sqrt(step)
        rows.append(
            {
                "Tanggal": next_date,
                "Estimasi": point,
                "Batas bawah": point - uncertainty,
                "Batas atas": point + uncertainty,
            }
        )
        history.loc[next_date] = point

    return ForecastResult(pd.DataFrame(rows).set_index("Tanggal"), metrics)
