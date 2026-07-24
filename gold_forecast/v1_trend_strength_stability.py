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
from gold_forecast.v1_fixed_delay_quality import (
    CALIBRATION_END,
    DEVELOPMENT_CONFIRMATION_END,
    DEVELOPMENT_CONFIRMATION_START,
    MAX_DRAWDOWN_PCT,
    MIN_RETENTION_PCT,
    PROFIT_FACTOR_TARGET,
    VALIDATION_END,
    VALIDATION_START,
    _guard_features,
)
from gold_forecast.v1_risk_control import (
    MAX_MONTE_CARLO_LOSS_PCT,
    RiskControlConfig,
    _metric_values,
    _simulate_risk_control,
)
from gold_forecast.v1_signal_quality import SignalQualityConfig, _entry_features
from gold_forecast.v1_unified_benchmark import _unified_best


QUANTILES = (0.10, 0.20, 0.25, 0.30, 0.40)
CANDIDATES = (
    "Fixed Delay 5m Control",
    "Trend Strength Q10",
    "Trend Strength Q20",
    "Trend Strength Q25",
    "Trend Strength Q30",
    "Trend Strength Q40",
)


def run_v1_trend_strength_stability_lab(
    gold_m1: pd.DataFrame,
    signal_daily: pd.DataFrame,
    frozen_payload: dict[str, object],
) -> dict[str, object]:
    data = _prepare_m1(gold_m1)
    _, leaderboard, _ = frozen_payload["v1"]
    best = _unified_best(leaderboard.iloc[0].to_dict())
    entry_features = _entry_features(data)
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
    config = RiskControlConfig(
        "Trend Strength Stability",
        "Entry filter only",
        max_total_positions=1,
        max_same_direction=1,
    )

    development_control, development_events = _fixed_delay_control(
        data,
        signal_daily,
        best,
        entry_features,
        balanced_config,
        spread_limit,
        DEVELOPMENT_START,
        DEVELOPMENT_END,
    )
    confirmation_control, confirmation_events = _fixed_delay_control(
        data,
        signal_daily,
        best,
        entry_features,
        balanced_config,
        spread_limit,
        CONFIRMATION_START,
        CONFIRMATION_END,
    )
    development_features = _guard_features(
        development_control, development_events, entry_features
    )
    confirmation_features = _guard_features(
        confirmation_control, confirmation_events, entry_features
    )
    calibration_strength = development_features.loc[
        DEVELOPMENT_START:CALIBRATION_END, "trend_strength_atr"
    ].dropna()
    thresholds = {
        quantile: float(calibration_strength.quantile(quantile))
        for quantile in QUANTILES
    }
    decile_edges = _decile_edges(calibration_strength)
    development_signals = _candidate_signals(
        development_control, development_features, thresholds
    )
    confirmation_signals = _candidate_signals(
        confirmation_control, confirmation_features, thresholds
    )

    development_data = data.loc[DEVELOPMENT_START:DEVELOPMENT_END]
    confirmation_data = data.loc[CONFIRMATION_START:CONFIRMATION_END]
    development_results = _simulate_candidates(
        development_data, development_signals, best, config
    )
    confirmation_results = _simulate_candidates(
        confirmation_data, confirmation_signals, best, config
    )
    development = _result_table(development_results, development_signals)
    confirmation = _result_table(confirmation_results, confirmation_signals)
    retention = _retention_table(development_signals, confirmation_signals)
    period_validation = _period_validation(
        data, development_signals, best, config
    )
    folds = _fold_evaluation(data, development_signals, best, config)
    monte_carlo, monte_carlo_summary = _monte_carlo_all(development_results)
    decisions = _decision_table(
        development,
        period_validation,
        folds,
        retention,
        monte_carlo_summary,
    )
    ranking = _ranking_table(
        development, confirmation, decisions, retention
    )
    winner = str(ranking.iloc[0]["Kandidat"])
    stability = _stability_summary(
        development, confirmation, period_validation, decisions
    )

    return {
        "methodology": {
            "Question": (
                "Apakah keunggulan Trend Strength stabil di beberapa threshold, "
                "atau hanya cocok pada Q25 dan kondisi 2026H1?"
            ),
            "Calibration": (
                "01 Jan 2022 - 31 Des 2023; threshold Q10-Q40 dihitung dari distribusi "
                "trend_strength_atr tanpa label profit"
            ),
            "Validation": "01 Jan 2024 - 31 Des 2024",
            "Development confirmation": "01 Jan 2025 - 31 Des 2025",
            "Historical reference": (
                "01 Jan 2026 - 30 Jun 2026; tidak dipakai memilih kandidat"
            ),
            "Trend strength": "|EMA H1 cepat - EMA H1 lambat| / ATR H1",
            "Execution contract": (
                "Equity USD 1.000 | lot 0.01 | maksimal 1 posisi | TP USD 25 | "
                "SL USD 10 | Fixed Delay 5m | spread M1 aktual | slippage 2 points/sisi"
            ),
            "Baseline lock": (
                "Baseline v1, Fixed Delay paper live, ledger, dan eksperimen lama tidak diubah"
            ),
        },
        "thresholds": _threshold_table(thresholds),
        "data_audit": _extended_data_audit(data),
        "development": development,
        "period_validation": period_validation,
        "confirmation": confirmation,
        "retention": retention,
        "folds": folds,
        "monte_carlo_summary": monte_carlo_summary,
        "decisions": decisions,
        "ranking": ranking,
        "winner": winner,
        "stability": stability,
        "winner_stress": _stress_test(
            development_data, development_signals[winner], best, config
        ),
        "winner_monthly": _safe_monthly_summary(
            development_results[winner]
        ),
        "winner_monte_carlo": monte_carlo[
            monte_carlo["Kandidat"].eq(winner)
        ].copy(),
        "direction_audit": _trade_direction_audit(
            development_results, confirmation_results
        ),
        "accept_reject_audit": _accept_reject_audit(
            development_results["Fixed Delay 5m Control"],
            confirmation_results["Fixed Delay 5m Control"],
            development_signals,
            confirmation_signals,
        ),
        "trend_decile_audit": _trend_decile_audit(
            development_results["Fixed Delay 5m Control"],
            confirmation_results["Fixed Delay 5m Control"],
            development_features,
            confirmation_features,
            decile_edges,
        ),
        "signal_counts": pd.DataFrame(
            [
                {
                    "Kandidat": candidate,
                    "Development": len(development_signals[candidate]),
                    "2026H1": len(confirmation_signals[candidate]),
                }
                for candidate in CANDIDATES
            ]
        ),
    }


def _fixed_delay_control(
    data: pd.DataFrame,
    daily: pd.DataFrame,
    best: dict[str, object],
    entry_features: pd.DataFrame,
    balanced_config: SignalQualityConfig,
    spread_limit: float,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    delayed, events = _build_fixed_delay_signals(
        data, balanced, best, 5, spread_limit
    )
    return _unique_signals(delayed), events


def _candidate_signals(
    control: pd.DataFrame,
    features: pd.DataFrame,
    thresholds: dict[float, float],
) -> dict[str, pd.DataFrame]:
    trend = pd.to_numeric(features["trend_strength_atr"], errors="coerce")
    output = {"Fixed Delay 5m Control": control.copy()}
    for quantile in QUANTILES:
        name = f"Trend Strength Q{int(quantile * 100)}"
        output[name] = control.loc[
            trend.ge(thresholds[quantile]).fillna(False)
        ].copy()
    return output


def _simulate_candidates(
    data: pd.DataFrame,
    signals: dict[str, pd.DataFrame],
    best: dict[str, object],
    config: RiskControlConfig,
) -> dict[str, object]:
    return {
        candidate: _simulate_risk_control(
            data, signals[candidate], best, config
        )
        for candidate in CANDIDATES
    }


def _result_table(
    results: dict[str, object],
    signals: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Kandidat": candidate,
                "Sinyal tersedia": len(signals[candidate]),
                **_metric_values(results[candidate]),
            }
            for candidate in CANDIDATES
        ]
    )


def _retention_table(
    development: dict[str, pd.DataFrame],
    confirmation: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    development_control = max(
        len(development["Fixed Delay 5m Control"]), 1
    )
    confirmation_control = max(
        len(confirmation["Fixed Delay 5m Control"]), 1
    )
    return pd.DataFrame(
        [
            {
                "Kandidat": candidate,
                "Sinyal development": len(development[candidate]),
                "Retensi development (%)": (
                    len(development[candidate]) / development_control * 100
                ),
                "Sinyal 2026H1": len(confirmation[candidate]),
                "Retensi 2026H1 (%)": (
                    len(confirmation[candidate]) / confirmation_control * 100
                ),
            }
            for candidate in CANDIDATES
        ]
    )


def _period_validation(
    data: pd.DataFrame,
    signals: dict[str, pd.DataFrame],
    best: dict[str, object],
    config: RiskControlConfig,
) -> pd.DataFrame:
    periods = (
        ("Calibration 2022-2023", DEVELOPMENT_START, CALIBRATION_END),
        ("Validation 2024", VALIDATION_START, VALIDATION_END),
        (
            "Development confirmation 2025",
            DEVELOPMENT_CONFIRMATION_START,
            DEVELOPMENT_CONFIRMATION_END,
        ),
    )
    rows = []
    for label, start, end in periods:
        period_data = data.loc[start:end]
        for candidate in CANDIDATES:
            selected = signals[candidate].loc[start:end]
            result = _simulate_risk_control(
                period_data, selected, best, config
            )
            rows.append(
                {
                    "Periode": label,
                    "Kandidat": candidate,
                    "Sinyal tersedia": len(selected),
                    **_metric_values(result),
                }
            )
    return pd.DataFrame(rows)


def _fold_evaluation(
    data: pd.DataFrame,
    signals: dict[str, pd.DataFrame],
    best: dict[str, object],
    config: RiskControlConfig,
) -> pd.DataFrame:
    rows = []
    for fold in FOLDS:
        period_data = data.loc[fold.test_start:fold.test_end]
        for candidate in CANDIDATES:
            selected = signals[candidate].loc[
                fold.test_start:fold.test_end
            ]
            result = _simulate_risk_control(
                period_data, selected, best, config
            )
            metrics = _metric_values(result)
            rows.append(
                {
                    "Fold": fold.name,
                    "Kandidat": candidate,
                    "Test mulai": fold.test_start,
                    "Test akhir": fold.test_end,
                    **metrics,
                    "Profitable": bool(metrics["Growth (%)"] > 0),
                }
            )
    return pd.DataFrame(rows)


def _monte_carlo_all(
    results: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    summaries = []
    for candidate in CANDIDATES:
        simulations, summary = _safe_monte_carlo(
            results[candidate].trades
        )
        simulations.insert(0, "Kandidat", candidate)
        frames.append(simulations)
        summaries.append({"Kandidat": candidate, **summary})
    return pd.concat(frames, ignore_index=True), pd.DataFrame(summaries)


def _decision_table(
    development: pd.DataFrame,
    period_validation: pd.DataFrame,
    folds: pd.DataFrame,
    retention: pd.DataFrame,
    monte_carlo: pd.DataFrame,
) -> pd.DataFrame:
    dev = development.set_index("Kandidat")
    period = period_validation.set_index(["Periode", "Kandidat"])
    retained = retention.set_index("Kandidat")
    mc = monte_carlo.set_index("Kandidat")
    rows = []
    for candidate in CANDIDATES:
        candidate_folds = folds[folds["Kandidat"].eq(candidate)]
        criteria = {
            "Growth development positif": float(
                dev.loc[candidate, "Growth (%)"]
            ) > 0,
            "PF development >= 1.50": float(
                dev.loc[candidate, "Profit factor"]
            ) >= PROFIT_FACTOR_TARGET,
            "DD development <= 10%": float(
                dev.loc[candidate, "Max drawdown (%)"]
            ) <= MAX_DRAWDOWN_PCT,
            "Retensi transaksi >= 60%": float(
                retained.loc[candidate, "Retensi development (%)"]
            ) >= MIN_RETENTION_PCT,
            "Validation 2024 positif": float(
                period.loc[("Validation 2024", candidate), "Growth (%)"]
            ) > 0,
            "Confirmation 2025 positif": float(
                period.loc[
                    ("Development confirmation 2025", candidate),
                    "Growth (%)",
                ]
            ) > 0,
            "Fold profitable >= 9/12": int(
                candidate_folds["Profitable"].sum()
            ) >= 9,
            "Monte Carlo rugi <= 10%": float(
                mc.loc[
                    candidate,
                    "Probabilitas equity akhir < modal awal (%)",
                ]
            ) <= MAX_MONTE_CARLO_LOSS_PCT,
        }
        rows.append(
            {
                "Kandidat": candidate,
                **{name: bool(value) for name, value in criteria.items()},
                "Fold profitable": int(
                    candidate_folds["Profitable"].sum()
                ),
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
    retention: pd.DataFrame,
) -> pd.DataFrame:
    dev = development.set_index("Kandidat")
    test = confirmation.set_index("Kandidat")
    decision = decisions.set_index("Kandidat")
    retained = retention.set_index("Kandidat")
    rows = []
    for candidate in CANDIDATES:
        rows.append(
            {
                "Kandidat": candidate,
                "Growth development (%)": float(
                    dev.loc[candidate, "Growth (%)"]
                ),
                "PF development": float(
                    dev.loc[candidate, "Profit factor"]
                ),
                "DD development (%)": float(
                    dev.loc[candidate, "Max drawdown (%)"]
                ),
                "Transaksi development": int(
                    dev.loc[candidate, "Transaksi"]
                ),
                "Retensi development (%)": float(
                    retained.loc[candidate, "Retensi development (%)"]
                ),
                "Growth 2026H1 (%)": float(
                    test.loc[candidate, "Growth (%)"]
                ),
                "PF 2026H1": float(test.loc[candidate, "Profit factor"]),
                "DD 2026H1 (%)": float(
                    test.loc[candidate, "Max drawdown (%)"]
                ),
                "Transaksi 2026H1": int(
                    test.loc[candidate, "Transaksi"]
                ),
                "Fold profitable": int(
                    decision.loc[candidate, "Fold profitable"]
                ),
                "Kriteria lolos": int(
                    decision.loc[candidate, "Kriteria lolos"]
                ),
                "Lulus": bool(decision.loc[candidate, "Lulus"]),
            }
        )
    ranking = pd.DataFrame(rows)
    ranking = ranking.sort_values(
        [
            "Lulus",
            "Kriteria lolos",
            "PF development",
            "DD development (%)",
            "Growth development (%)",
        ],
        ascending=[False, False, False, True, False],
    ).reset_index(drop=True)
    ranking.insert(0, "Peringkat", np.arange(1, len(ranking) + 1))
    return ranking


def _threshold_table(thresholds: dict[float, float]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Kandidat": f"Trend Strength Q{int(quantile * 100)}",
                "Quantile": quantile,
                "Minimum trend strength (ATR)": value,
                "Retensi teoritis (%)": (1 - quantile) * 100,
            }
            for quantile, value in thresholds.items()
        ]
    )


def _stability_summary(
    development: pd.DataFrame,
    confirmation: pd.DataFrame,
    period_validation: pd.DataFrame,
    decisions: pd.DataFrame,
) -> dict[str, object]:
    trend_candidates = [
        candidate for candidate in CANDIDATES if candidate != CANDIDATES[0]
    ]
    dev = development.set_index("Kandidat").loc[trend_candidates]
    test = confirmation.set_index("Kandidat").loc[trend_candidates]
    decision = decisions.set_index("Kandidat").loc[trend_candidates]
    period = period_validation[
        period_validation["Kandidat"].isin(trend_candidates)
    ]
    every_stage_positive = bool(period["Growth (%)"].gt(0).all())
    neighborhood = dev.loc[
        ["Trend Strength Q20", "Trend Strength Q25", "Trend Strength Q30"]
    ]
    return {
        "Semua Q10-Q40 development growth positif": bool(
            dev["Growth (%)"].gt(0).all()
        ),
        "Semua Q10-Q40 2026H1 growth positif": bool(
            test["Growth (%)"].gt(0).all()
        ),
        "Semua Q10-Q40 setiap tahap development positif": every_stage_positive,
        "PF neighborhood Q20-Q30 range": float(
            neighborhood["Profit factor"].max()
            - neighborhood["Profit factor"].min()
        ),
        "DD neighborhood Q20-Q30 range (pp)": float(
            neighborhood["Max drawdown (%)"].max()
            - neighborhood["Max drawdown (%)"].min()
        ),
        "Kandidat trend yang lulus": int(decision["Lulus"].sum()),
        "Interpretasi": (
            "STABIL"
            if bool(
                dev["Growth (%)"].gt(0).all()
                and test["Growth (%)"].gt(0).all()
                and every_stage_positive
                and (
                    neighborhood["Profit factor"].max()
                    - neighborhood["Profit factor"].min()
                )
                <= 0.25
            )
            else "BELUM STABIL"
        ),
    }


def _trade_direction_audit(
    development_results: dict[str, object],
    confirmation_results: dict[str, object],
) -> pd.DataFrame:
    rows = []
    for period, results in (
        ("Development 2022-2025", development_results),
        ("Historical reference 2026H1", confirmation_results),
    ):
        for candidate in CANDIDATES:
            trades = results[candidate].trades
            for direction in ("BUY", "SELL"):
                selected = trades[
                    trades.get(
                        "Arah", pd.Series(index=trades.index, dtype=object)
                    ).eq(direction)
                ]
                rows.append(
                    {
                        "Periode": period,
                        "Kandidat": candidate,
                        "Arah": direction,
                        **_trade_metrics(selected),
                    }
                )
    return pd.DataFrame(rows)


def _accept_reject_audit(
    development_control_result,
    confirmation_control_result,
    development_signals: dict[str, pd.DataFrame],
    confirmation_signals: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []
    for period, control_result, signals in (
        (
            "Development 2022-2025",
            development_control_result,
            development_signals,
        ),
        (
            "Historical reference 2026H1",
            confirmation_control_result,
            confirmation_signals,
        ),
    ):
        trades = control_result.trades.copy()
        entry_times = pd.to_datetime(
            trades.get("Tanggal entry"), errors="coerce"
        )
        for candidate in CANDIDATES[1:]:
            accepted_times = set(signals[candidate].index)
            accepted_mask = entry_times.isin(accepted_times)
            for status, mask in (
                ("DITERIMA", accepted_mask),
                ("DITOLAK", ~accepted_mask),
            ):
                rows.append(
                    {
                        "Periode": period,
                        "Kandidat": candidate,
                        "Status oleh filter": status,
                        **_trade_metrics(trades.loc[mask]),
                    }
                )
    return pd.DataFrame(rows)


def _decile_edges(strength: pd.Series) -> np.ndarray:
    values = strength.dropna().to_numpy(dtype=float)
    if values.size == 0:
        raise RuntimeError("Trend strength calibration kosong.")
    edges = np.quantile(values, np.linspace(0, 1, 11))
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def _trend_decile_audit(
    development_control_result,
    confirmation_control_result,
    development_features: pd.DataFrame,
    confirmation_features: pd.DataFrame,
    edges: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for period, result, features in (
        (
            "Development 2022-2025",
            development_control_result,
            development_features,
        ),
        (
            "Historical reference 2026H1",
            confirmation_control_result,
            confirmation_features,
        ),
    ):
        trades = result.trades.copy()
        entry_times = pd.to_datetime(
            trades.get("Tanggal entry"), errors="coerce"
        )
        strength = pd.to_numeric(
            features["trend_strength_atr"], errors="coerce"
        ).reindex(entry_times)
        strength.index = trades.index
        bins = pd.cut(
            strength,
            bins=edges,
            labels=[f"D{number}" for number in range(1, 11)],
            include_lowest=True,
        )
        for label in [f"D{number}" for number in range(1, 11)]:
            selected = trades.loc[bins.eq(label)]
            rows.append(
                {
                    "Periode": period,
                    "Desil trend strength": label,
                    "Rata-rata trend strength": float(
                        strength.loc[bins.eq(label)].mean()
                    ),
                    **_trade_metrics(selected),
                }
            )
    return pd.DataFrame(rows)


def _trade_metrics(trades: pd.DataFrame) -> dict[str, float]:
    if trades.empty:
        return {
            "Transaksi": 0,
            "Net P/L": 0.0,
            "Profit factor": np.nan,
            "Win rate (%)": np.nan,
        }
    net = pd.to_numeric(trades["Net P/L"], errors="coerce").fillna(0.0)
    profit = float(net[net > 0].sum())
    loss = float(-net[net < 0].sum())
    return {
        "Transaksi": int(len(trades)),
        "Net P/L": float(net.sum()),
        "Profit factor": profit / loss if loss > 0 else np.inf,
        "Win rate (%)": float(net.gt(0).mean() * 100),
    }


def _extended_data_audit(data: pd.DataFrame) -> pd.DataFrame:
    audit = _data_audit(data)
    expected = set(pd.period_range("2022-01", "2026-06", freq="M"))
    actual = set(
        data.loc[DEVELOPMENT_START:CONFIRMATION_END]
        .index.to_period("M")
        .unique()
    )
    coverage = audit["Pemeriksaan"].str.startswith("Cakupan bulan")
    audit.loc[coverage, "Pemeriksaan"] = (
        "Cakupan bulan 2022-01 sampai 2026-06"
    )
    audit.loc[coverage, "Status"] = (
        "LOLOS" if expected.issubset(actual) else "BELUM"
    )
    audit.loc[coverage, "Detail"] = (
        f"{len(expected & actual)}/{len(expected)} bulan tersedia"
    )
    return audit
