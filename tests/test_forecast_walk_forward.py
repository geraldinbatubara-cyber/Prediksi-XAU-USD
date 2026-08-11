import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor

from gold_forecast.model import train_and_forecast
import gold_forecast.model_v2 as model_v2


def _market(rows: int = 560) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    index = pd.bdate_range("2023-01-02", periods=rows)
    return pd.DataFrame(
        {
            "gold": 1900 * np.exp(np.cumsum(rng.normal(0.0003, 0.008, rows))),
            "dollar": 100 * np.exp(np.cumsum(rng.normal(0.0, 0.002, rows))),
            "silver": 24 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, rows))),
        },
        index=index,
    )


def test_model_1_walk_forward_uses_four_expanding_folds():
    result = train_and_forecast(
        _market(360)["gold"], evaluate_walk_forward=True
    )
    assert len(result.forecast) == 7
    assert result.walk_forward_metrics is not None
    assert result.walk_forward_metrics["Fold"] == 4.0
    assert result.walk_forward_metrics["MAE"] >= 0


def test_model_2_walk_forward_is_optional_and_precomputed(monkeypatch):
    monkeypatch.setattr(model_v2, "_estimator", lambda: DummyRegressor())
    result = model_v2.train_model_v2(
        _market(), evaluate_walk_forward=True
    )
    assert len(result.forecast) == 7
    assert result.walk_forward_metrics is not None
    assert result.walk_forward_metrics["Fold"] == 4.0
    assert 0 <= result.walk_forward_metrics["Akurasi arah"] <= 100


def test_model_2_ignores_unavailable_cross_market_factor(monkeypatch):
    monkeypatch.setattr(model_v2, "_estimator", lambda: DummyRegressor())
    market = _market(400)
    market["unavailable_factor"] = np.nan
    result = model_v2.train_model_v2(market)
    assert len(result.forecast) == 7
    assert result.feature_count > 0
