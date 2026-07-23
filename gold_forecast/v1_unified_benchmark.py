from __future__ import annotations

import numpy as np
import pandas as pd

from gold_forecast.exact_broker_oos import _prepare_m1
from gold_forecast.v1_entry_outcome import (
    _balanced_signals,
    _safe_monthly_summary,
    _safe_monte_carlo,
)
from gold_forecast.v1_entry_quality import _stress_test
from gold_forecast.v1_entry_quality_path import (
    CONFIRMATION_END,
    CONFIRMATION_START,
    DEVELOPMENT_END,
    DEVELOPMENT_START,
    FOLDS,
    _data_audit,
    _unique_signals,
)
from gold_forecast.v1_fixed_delay import _build_fixed_delay_signals
from gold_forecast.v1_risk_control import (
    RiskControlConfig,
    _entry_signals_for_period,
    _metric_values,
    _simulate_risk_control,
)
from gold_forecast.v1_sideways_defense import (
    RegimeConfig,
    SidewaysConfig,
    _gate_trend_signals,
    _merge_signals,
    _regime_features,
    _regime_states,
    _sideways_signals,
)
from gold_forecast.v1_signal_quality import SignalQualityConfig, _entry_features


STRATEGIES = (
    "Baseline v1",
    "Balanced Entry",
    "Fixed Delay 5m",
    "Sideways Defense",
)
UNIFIED_TP_USD = 25.0
UNIFIED_SL_USD = 10.0
UNIFIED_LOT = 0.01
MAX_DRAWDOWN_PCT = 10.0
PROFIT_FACTOR_TARGET = 1.50
MAX_MONTE_CARLO_LOSS_PCT = 10.0


def run_v1_unified_benchmark(
    gold_m1: pd.DataFrame,
    signal_daily: pd.DataFrame,
    frozen_payload: dict[str, object],
    sideways_payload: dict[str, object],
) -> dict[str, object]:
    data = _prepare_m1(gold_m1)
    _, leaderboard, _ = frozen_payload["v1"]
    source_best = leaderboard.iloc[0].to_dict()
    best = _unified_best(source_best)
    entry_features = _entry_features(data)
    regime_features, _, m15 = _regime_features(data)
    spread_limit = float(
        data.loc[DEVELOPMENT_START:DEVELOPMENT_END, "SpreadPoints"].quantile(0.90)
    )
    balanced_config = SignalQualityConfig(
        "Balanced Entry Frozen",
        "Trend engine",
        conviction_multiplier=1.05,
        require_h1_trend=True,
        wait_hours=2,
    )
    regime_config = RegimeConfig(**sideways_payload["selected_regime_config"])
    sideways_config = SidewaysConfig(**sideways_payload["selected_sideways_config"])
    states = _regime_states(regime_features, regime_config)
    config = RiskControlConfig(
        "Unified Strategy Benchmark",
        "Kontrak eksekusi identik",
        max_total_positions=1,
        max_same_direction=1,
    )

    development_signals = _strategy_signals(
        data,
        signal_daily,
        best,
        entry_features,
        regime_features,
        m15,
        states,
        balanced_config,
        sideways_config,
        spread_limit,
        DEVELOPMENT_START,
        DEVELOPMENT_END,
    )
    confirmation_signals = _strategy_signals(
        data,
        signal_daily,
        best,
        entry_features,
        regime_features,
        m15,
        states,
        balanced_config,
        sideways_config,
        spread_limit,
        CONFIRMATION_START,
        CONFIRMATION_END,
    )

    development_data = data.loc[DEVELOPMENT_START:DEVELOPMENT_END]
    confirmation_data = data.loc[CONFIRMATION_START:CONFIRMATION_END]
    development_results = _simulate_all(
        development_data, development_signals, best, config
    )
    confirmation_results = _simulate_all(
        confirmation_data, confirmation_signals, best, config
    )
    development = _result_table(development_results, development_signals)
    confirmation = _result_table(confirmation_results, confirmation_signals)
    folds = _fold_evaluation(
        data, development_signals, best, config
    )
    stress = _stress_all(
        confirmation_data, confirmation_signals, best, config
    )
    monte_carlo, monte_carlo_summary = _monte_carlo_all(confirmation_results)
    monthly = _monthly_all(confirmation_results)
    decisions = _decision_table(
        development,
        confirmation,
        folds,
        stress,
        monte_carlo_summary,
    )
    ranking = _ranking_table(development, confirmation, decisions)
    direction = _direction_audit(
        confirmation_data, confirmation_signals, best, config
    )
    signal_overlap = _signal_overlap(confirmation_signals)

    return {
        "methodology": {
            "Development": "01 Jan 2022 - 31 Des 2025",
            "Quarterly robustness": "12 folds, 01 Jan 2023 - 31 Des 2025",
            "Historical test": "01 Jan 2026 - 30 Jun 2026; bukan true unseen OOS",
            "Data": "Candle M1 MT5 yang sama untuk seluruh strategi",
            "Execution contract": (
                "Modal USD 1.000 | lot 0.01 | maksimal 1 posisi | TP USD 25 | "
                "SL USD 10 | tanpa close-all target equity"
            ),
            "Costs": "Spread M1 aktual | slippage 2 points/sisi | swap BUY berlaku, SELL 0",
            "Baseline entry": "Sinyal Optimizer v1 langsung",
            "Balanced entry": "Baseline + H1 trend + conviction 1.05 + wait 2 jam",
            "Fixed delay entry": (
                "Balanced Entry + delay 5 menit; batal jika barrier tersentuh/spread P90 terlampaui"
            ),
            "Sideways entry": (
                f"Trend gate + {sideways_config.name}; classifier {regime_config.name}"
            ),
            "No retuning": "Tidak ada parameter yang dipilih menggunakan data 2026H1",
            "Baseline lock": "Live Trading, ledger, dan seluruh sub-tab lama tidak diubah",
        },
        "data_audit": _extended_data_audit(data),
        "development": development,
        "confirmation": confirmation,
        "folds": folds,
        "stress": stress,
        "monte_carlo_summary": monte_carlo_summary,
        "monthly": monthly,
        "decisions": decisions,
        "ranking": ranking,
        "direction_audit": direction,
        "signal_overlap": signal_overlap,
        "signal_counts": pd.DataFrame(
            [
                {
                    "Strategi": strategy,
                    "Development": len(development_signals[strategy]),
                    "2026H1": len(confirmation_signals[strategy]),
                }
                for strategy in STRATEGIES
            ]
        ),
    }


def _unified_best(source: dict[str, object]) -> dict[str, object]:
    best = dict(source)
    best.update(
        {
            "TP (USD)": UNIFIED_TP_USD,
            "SL (USD)": UNIFIED_SL_USD,
            "Lot": UNIFIED_LOT,
            "Max BUY": 1,
            "Max SELL": 1,
            "Close-all target equity": False,
            "Profit protection aktif (USD)": None,
            "Profit protection floor (USD)": None,
            "Profit protection trail (USD)": None,
            "Floating profit close (USD)": None,
        }
    )
    return best


def _strategy_signals(
    data: pd.DataFrame,
    daily: pd.DataFrame,
    best: dict[str, object],
    entry_features: pd.DataFrame,
    regime_features: pd.DataFrame,
    m15: pd.DataFrame,
    states: pd.Series,
    balanced_config: SignalQualityConfig,
    sideways_config: SidewaysConfig,
    spread_limit: float,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, pd.DataFrame]:
    baseline = _unique_signals(
        _entry_signals_for_period(data, daily, best, start, end)
    )
    balanced = _unique_signals(
        _balanced_signals(
            data,
            daily,
            best,
            entry_features,
            balanced_config,
            spread_limit,
            start,
            end,
        )
    )
    delayed, _ = _build_fixed_delay_signals(
        data, balanced, best, 5, spread_limit
    )
    trend = _gate_trend_signals(
        _normalize_signals(balanced, "Balanced Trend"), states
    )
    sideways = _sideways_signals(
        data,
        regime_features,
        m15,
        states,
        best,
        sideways_config,
        start,
        end,
    )
    hybrid = _merge_signals(trend, sideways)
    return {
        "Baseline v1": _normalize_signals(baseline, "Baseline v1"),
        "Balanced Entry": _normalize_signals(balanced, "Balanced Entry"),
        "Fixed Delay 5m": _normalize_signals(delayed, "Fixed Delay 5m"),
        "Sideways Defense": _normalize_signals(hybrid, "Sideways Defense"),
    }


def _normalize_signals(signals: pd.DataFrame, strategy: str) -> pd.DataFrame:
    normalized = signals.copy()
    if normalized.empty:
        return normalized
    normalized["lot"] = UNIFIED_LOT
    normalized["tp_usd"] = UNIFIED_TP_USD
    normalized["sl_usd"] = UNIFIED_SL_USD
    normalized["time_stop_hours"] = np.nan
    normalized["strategy"] = strategy
    return normalized.sort_index()


def _simulate_all(
    data: pd.DataFrame,
    signals: dict[str, pd.DataFrame],
    best: dict[str, object],
    config: RiskControlConfig,
) -> dict[str, object]:
    return {
        strategy: _simulate_risk_control(data, signals[strategy], best, config)
        for strategy in STRATEGIES
    }


def _result_table(
    results: dict[str, object],
    signals: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Strategi": strategy,
                "Sinyal tersedia": len(signals[strategy]),
                **_metric_values(results[strategy]),
            }
            for strategy in STRATEGIES
        ]
    )


def _fold_evaluation(
    data: pd.DataFrame,
    signals: dict[str, pd.DataFrame],
    best: dict[str, object],
    config: RiskControlConfig,
) -> pd.DataFrame:
    rows = []
    for fold in FOLDS:
        period_data = data.loc[fold.test_start:fold.test_end]
        for strategy in STRATEGIES:
            period_signals = signals[strategy].loc[fold.test_start:fold.test_end]
            result = _simulate_risk_control(
                period_data, period_signals, best, config
            )
            metrics = _metric_values(result)
            rows.append(
                {
                    "Fold": fold.name,
                    "Strategi": strategy,
                    "Test mulai": fold.test_start,
                    "Test akhir": fold.test_end,
                    **metrics,
                    "Profitable": bool(metrics["Growth (%)"] > 0),
                }
            )
    return pd.DataFrame(rows)


def _stress_all(
    data: pd.DataFrame,
    signals: dict[str, pd.DataFrame],
    best: dict[str, object],
    config: RiskControlConfig,
) -> pd.DataFrame:
    rows = []
    for strategy in STRATEGIES:
        table = _stress_test(data, signals[strategy], best, config).copy()
        table.insert(0, "Strategi", strategy)
        rows.append(table)
    return pd.concat(rows, ignore_index=True)


def _monte_carlo_all(
    results: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    simulations = []
    summaries = []
    for strategy in STRATEGIES:
        monte_carlo, summary = _safe_monte_carlo(results[strategy].trades)
        monte_carlo.insert(0, "Strategi", strategy)
        simulations.append(monte_carlo)
        summaries.append({"Strategi": strategy, **summary})
    return pd.concat(simulations, ignore_index=True), pd.DataFrame(summaries)


def _monthly_all(results: dict[str, object]) -> pd.DataFrame:
    rows = []
    for strategy in STRATEGIES:
        monthly = _safe_monthly_summary(results[strategy]).copy()
        monthly.insert(0, "Strategi", strategy)
        rows.append(monthly)
    return pd.concat(rows, ignore_index=True)


def _decision_table(
    development: pd.DataFrame,
    confirmation: pd.DataFrame,
    folds: pd.DataFrame,
    stress: pd.DataFrame,
    monte_carlo: pd.DataFrame,
) -> pd.DataFrame:
    dev = development.set_index("Strategi")
    test = confirmation.set_index("Strategi")
    mc = monte_carlo.set_index("Strategi")
    rows = []
    for strategy in STRATEGIES:
        strategy_folds = folds[folds["Strategi"].eq(strategy)]
        strategy_stress = stress[stress["Strategi"].eq(strategy)]
        criteria = {
            "Dev growth positif": float(dev.loc[strategy, "Growth (%)"]) > 0,
            "Dev DD <= 10%": float(dev.loc[strategy, "Max drawdown (%)"]) <= MAX_DRAWDOWN_PCT,
            "Dev PF >= 1.50": float(dev.loc[strategy, "Profit factor"]) >= PROFIT_FACTOR_TARGET,
            "Fold profitable >= 9/12": int(strategy_folds["Profitable"].sum()) >= 9,
            "Test growth positif": float(test.loc[strategy, "Growth (%)"]) > 0,
            "Test DD <= 10%": float(test.loc[strategy, "Max drawdown (%)"]) <= MAX_DRAWDOWN_PCT,
            "Test PF >= 1.50": float(test.loc[strategy, "Profit factor"]) >= PROFIT_FACTOR_TARGET,
            "Test transaksi >= 30": float(test.loc[strategy, "Transaksi"]) >= 30,
            "Stress profitable 9/9": len(strategy_stress) == 9 and bool(
                strategy_stress["Growth (%)"].gt(0).all()
            ),
            "Monte Carlo rugi <= 10%": float(
                mc.loc[strategy, "Probabilitas equity akhir < modal awal (%)"]
            ) <= MAX_MONTE_CARLO_LOSS_PCT,
        }
        rows.append(
            {
                "Strategi": strategy,
                **{key: bool(value) for key, value in criteria.items()},
                "Fold profitable": int(strategy_folds["Profitable"].sum()),
                "Kriteria lolos": int(sum(criteria.values())),
                "Total kriteria": len(criteria),
                "Lulus": bool(all(criteria.values())),
            }
        )
    return pd.DataFrame(rows)


def _ranking_table(
    development: pd.DataFrame,
    confirmation: pd.DataFrame,
    decisions: pd.DataFrame,
) -> pd.DataFrame:
    dev = development.set_index("Strategi")
    test = confirmation.set_index("Strategi")
    decision = decisions.set_index("Strategi")
    rows = []
    for strategy in STRATEGIES:
        rows.append(
            {
                "Strategi": strategy,
                "Growth development (%)": float(dev.loc[strategy, "Growth (%)"]),
                "PF development": float(dev.loc[strategy, "Profit factor"]),
                "DD development (%)": float(dev.loc[strategy, "Max drawdown (%)"]),
                "Growth test (%)": float(test.loc[strategy, "Growth (%)"]),
                "PF test": float(test.loc[strategy, "Profit factor"]),
                "DD test (%)": float(test.loc[strategy, "Max drawdown (%)"]),
                "Transaksi test": int(test.loc[strategy, "Transaksi"]),
                "Fold profitable": int(decision.loc[strategy, "Fold profitable"]),
                "Kriteria lolos": int(decision.loc[strategy, "Kriteria lolos"]),
                "Lulus": bool(decision.loc[strategy, "Lulus"]),
            }
        )
    ranking = pd.DataFrame(rows)
    ranking["Skor robustness"] = (
        ranking["Kriteria lolos"] * 100
        + ranking["Fold profitable"] * 5
        + ranking["PF test"].clip(upper=5) * 2
        + ranking["Growth test (%)"].clip(lower=-100, upper=100) / 10
        - ranking["DD test (%)"].clip(upper=100) / 10
    )
    ranking = ranking.sort_values(
        ["Lulus", "Skor robustness", "PF test", "Growth test (%)"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    ranking.insert(0, "Peringkat", np.arange(1, len(ranking) + 1))
    return ranking


def _direction_audit(
    data: pd.DataFrame,
    signals: dict[str, pd.DataFrame],
    best: dict[str, object],
    config: RiskControlConfig,
) -> pd.DataFrame:
    rows = []
    for strategy in STRATEGIES:
        expected = pd.to_numeric(
            signals[strategy]["expected_change_pct"], errors="coerce"
        )
        for direction, mask in (
            ("BUY", expected.gt(0)),
            ("SELL", expected.lt(0)),
        ):
            selected = signals[strategy].loc[mask.fillna(False)]
            result = _simulate_risk_control(data, selected, best, config)
            rows.append(
                {
                    "Strategi": strategy,
                    "Arah": direction,
                    "Sinyal": len(selected),
                    **_metric_values(result),
                }
            )
    return pd.DataFrame(rows)


def _signal_overlap(signals: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for left in STRATEGIES:
        left_dates = set(signals[left].index)
        for right in STRATEGIES:
            right_dates = set(signals[right].index)
            union = left_dates | right_dates
            rows.append(
                {
                    "Strategi A": left,
                    "Strategi B": right,
                    "Sinyal sama": len(left_dates & right_dates),
                    "Jaccard (%)": (
                        len(left_dates & right_dates) / len(union) * 100
                        if union
                        else 0.0
                    ),
                }
            )
    return pd.DataFrame(rows)


def _extended_data_audit(data: pd.DataFrame) -> pd.DataFrame:
    audit = _data_audit(data)
    expected = set(pd.period_range("2022-01", "2026-06", freq="M"))
    actual = set(
        data.loc[DEVELOPMENT_START:CONFIRMATION_END].index.to_period("M").unique()
    )
    coverage = audit["Pemeriksaan"].str.startswith("Cakupan bulan")
    audit.loc[coverage, "Pemeriksaan"] = "Cakupan bulan 2022-01 sampai 2026-06"
    audit.loc[coverage, "Status"] = "LOLOS" if expected.issubset(actual) else "BELUM"
    audit.loc[coverage, "Detail"] = f"{len(expected & actual)}/{len(expected)} bulan tersedia"
    return audit

