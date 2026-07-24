from pathlib import Path

import pandas as pd

from gold_forecast.model_comparison import EXPERIMENTS, build_model_comparison


PRECOMPUTED = Path("data/precomputed")


def test_comparison_contains_all_26_experiments() -> None:
    payload = build_model_comparison(PRECOMPUTED)

    assert len(EXPERIMENTS) == 26
    assert payload["experiment_count"] == 26
    assert payload["group_count"] == 5
    assert set(payload["master"]["Eksperimen"]) == {item.name for item in EXPERIMENTS}


def test_scores_are_bounded_and_ranked() -> None:
    master = build_model_comparison(PRECOMPUTED)["master"]

    assert master["Skor Total"].between(0, 100).all()
    assert master["Peringkat"].tolist() == list(range(1, 27))
    assert master["Skor Total"].is_monotonic_decreasing
    assert pd.to_numeric(master["Growth OOS (%)"], errors="coerce").notna().all()


def test_every_group_has_one_rank_one() -> None:
    master = build_model_comparison(PRECOMPUTED)["master"]

    leaders = master.loc[master["Peringkat kelompok"] == 1]
    assert len(leaders) == 5
    assert set(leaders["Kelompok"]) == set(master["Kelompok"])
