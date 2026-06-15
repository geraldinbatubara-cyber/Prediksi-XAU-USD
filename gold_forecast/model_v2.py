from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


HORIZONS = range(1, 8)


@dataclass
class ModelV2Result:
    forecast: pd.DataFrame
    metrics: dict[str, float]
    horizon_metrics: pd.DataFrame
    feature_count: int


def _market_features(market: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=market.index)
    gold = market["gold"]
    for lag in (1, 2, 3, 5, 10, 20, 60):
        features[f"gold_return_{lag}d"] = gold.pct_change(lag)
    for window in (5, 10, 20, 60):
        features[f"gold_ma_{window}"] = gold / gold.rolling(window).mean() - 1
        features[f"gold_vol_{window}"] = gold.pct_change().rolling(window).std()
    features["gold_range_20"] = gold.rolling(20).max() / gold.rolling(20).min() - 1

    for column in market.columns.drop("gold"):
        values = market[column]
        features[f"{column}_return_1d"] = values.pct_change()
        features[f"{column}_return_5d"] = values.pct_change(5)
        features[f"{column}_ma_20"] = values / values.rolling(20).mean() - 1
        features[f"{column}_vol_10"] = values.pct_change().rolling(10).std()

    features["gold_dollar_corr_20"] = gold.pct_change().rolling(20).corr(
        market.get("dollar", gold).pct_change()
    )
    features["weekday"] = market.index.dayofweek
    return features.replace([np.inf, -np.inf], np.nan)


def _estimator() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        learning_rate=0.045,
        max_iter=220,
        max_leaf_nodes=15,
        min_samples_leaf=18,
        l2_regularization=2.0,
        random_state=42,
    )


def _metrics(actual: pd.Series, predicted: pd.Series, current: pd.Series) -> dict[str, float]:
    return {
        "MAE": float(np.mean(np.abs(actual - predicted))),
        "RMSE": float(np.sqrt(np.mean((actual - predicted) ** 2))),
        "MAPE": float(np.mean(np.abs((actual - predicted) / actual)) * 100),
        "Akurasi arah": float(((actual > current) == (predicted > current)).mean() * 100),
    }


def train_model_v2(market: pd.DataFrame) -> ModelV2Result:
    features = _market_features(market)
    gold = market["gold"]
    clean_features = features.dropna()
    if len(clean_features) < 500:
        raise ValueError("Model 2 memerlukan minimal 500 observasi lintas pasar.")

    latest_features = clean_features.iloc[[-1]]
    forecasts: list[dict[str, float | pd.Timestamp]] = []
    horizon_rows: list[dict[str, float]] = []
    all_actual: list[pd.Series] = []
    all_predicted: list[pd.Series] = []
    all_current: list[pd.Series] = []

    for horizon in HORIZONS:
        dataset = clean_features.copy()
        dataset["target_return"] = gold.shift(-horizon) / gold - 1
        dataset = dataset.dropna()
        split = int(len(dataset) * 0.8)
        train, test = dataset.iloc[:split], dataset.iloc[split:]
        feature_names = list(clean_features.columns)

        estimator = _estimator()
        estimator.fit(train[feature_names], train["target_return"])
        predicted_return = pd.Series(
            estimator.predict(test[feature_names]), index=test.index
        )
        current = gold.reindex(test.index)
        predicted = current * (1 + predicted_return)
        actual = current * (1 + test["target_return"])
        horizon_metric = _metrics(actual, predicted, current)
        horizon_rows.append({"Horizon": horizon, **horizon_metric})
        all_actual.append(actual)
        all_predicted.append(predicted)
        all_current.append(current)

        estimator.fit(dataset[feature_names], dataset["target_return"])
        point_return = float(estimator.predict(latest_features)[0])
        point = float(gold.iloc[-1] * (1 + point_return))
        residual_std = float((actual - predicted).std())
        next_date = gold.index[-1] + pd.offsets.BDay(horizon)
        forecasts.append(
            {
                "Tanggal": next_date,
                "Estimasi": point,
                "Batas bawah": point - 1.96 * residual_std,
                "Batas atas": point + 1.96 * residual_std,
            }
        )

    actual_all = pd.concat(all_actual, ignore_index=True)
    predicted_all = pd.concat(all_predicted, ignore_index=True)
    current_all = pd.concat(all_current, ignore_index=True)
    return ModelV2Result(
        forecast=pd.DataFrame(forecasts).set_index("Tanggal"),
        metrics=_metrics(actual_all, predicted_all, current_all),
        horizon_metrics=pd.DataFrame(horizon_rows).set_index("Horizon"),
        feature_count=len(clean_features.columns),
    )
