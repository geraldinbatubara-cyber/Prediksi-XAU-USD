from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score

from gold_forecast.model_v2 import HORIZONS, _market_features


THRESHOLDS = (0.50, 0.55, 0.60, 0.65, 0.70)


@dataclass
class DirectionModelResult:
    latest_probabilities: pd.DataFrame
    threshold_metrics: pd.DataFrame


def _classifier() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.035,
        max_iter=180,
        max_leaf_nodes=12,
        min_samples_leaf=20,
        l2_regularization=1.5,
        random_state=42,
    )


def _direction(probability_up: float, threshold: float) -> str:
    if probability_up >= threshold:
        return "Bullish"
    if probability_up <= 1 - threshold:
        return "Bearish"
    return "Netral"


def train_direction_model(market: pd.DataFrame) -> DirectionModelResult:
    features = _market_features(market).dropna()
    gold = market["gold"]
    if len(features) < 500:
        raise ValueError("Model arah memerlukan minimal 500 observasi lintas pasar.")

    feature_names = list(features.columns)
    latest_features = features.iloc[[-1]]
    probability_rows: list[dict[str, float | str]] = []
    metric_rows: list[dict[str, float | int]] = []

    for horizon in HORIZONS:
        dataset = features.copy()
        dataset["target_up"] = (gold.shift(-horizon) > gold).astype(float)
        dataset = dataset.dropna()
        split = int(len(dataset) * 0.8)
        train = dataset.iloc[:split]
        test = dataset.iloc[split:]
        y_train = train["target_up"].astype(int)
        y_test = test["target_up"].astype(int)

        classifier = _classifier()
        classifier.fit(train[feature_names], y_train)
        probability_up = pd.Series(
            classifier.predict_proba(test[feature_names])[:, 1],
            index=test.index,
        )
        prediction = (probability_up >= 0.5).astype(int)
        all_accuracy = accuracy_score(y_test, prediction) * 100

        for threshold in THRESHOLDS:
            actionable = (probability_up >= threshold) | (probability_up <= 1 - threshold)
            actionable_count = int(actionable.sum())
            actionable_accuracy = (
                accuracy_score(y_test[actionable], prediction[actionable]) * 100
                if actionable_count
                else np.nan
            )
            metric_rows.append(
                {
                    "Horizon": horizon,
                    "Threshold": threshold,
                    "Akurasi semua hari": all_accuracy,
                    "Akurasi actionable": actionable_accuracy,
                    "Coverage": actionable.mean() * 100,
                    "Jumlah sinyal": actionable_count,
                }
            )

        classifier.fit(dataset[feature_names], dataset["target_up"].astype(int))
        latest_probability = float(classifier.predict_proba(latest_features)[0, 1])
        probability_rows.append(
            {
                "Horizon": horizon,
                "Probabilitas naik": latest_probability * 100,
                "Probabilitas turun": (1 - latest_probability) * 100,
                "Sinyal 60%": _direction(latest_probability, 0.60),
                "Sinyal 65%": _direction(latest_probability, 0.65),
            }
        )

    return DirectionModelResult(
        latest_probabilities=pd.DataFrame(probability_rows).set_index("Horizon"),
        threshold_metrics=pd.DataFrame(metric_rows).set_index(["Horizon", "Threshold"]),
    )
