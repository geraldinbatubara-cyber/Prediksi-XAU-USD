from __future__ import annotations

import numpy as np
import pandas as pd

from gold_forecast.exact_broker_oos import _compact_curve, _prepare_m1
from gold_forecast.v1_entry_outcome import (
    OUTCOME_HORIZON_DAYS,
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
from gold_forecast.v1_entry_timing import _micro_event_frame
from gold_forecast.v1_risk_control import (
    MAX_DRAWDOWN_PCT,
    MAX_MONTE_CARLO_LOSS_PCT,
    RiskControlConfig,
    _metric_values,
    _simulate_risk_control,
)
from gold_forecast.v1_sideways_defense import (
    RegimeConfig,
    _regime_features,
    _regime_states,
)
from gold_forecast.v1_signal_quality import SignalQualityConfig, _entry_features


FIXED_DELAY_MINUTES = 5
SENSITIVITY_DELAYS = (4, 5, 6)
PROFIT_FACTOR_TARGET = 1.50


def run_v1_fixed_delay_lab(
    gold_m1: pd.DataFrame,
    signal_daily: pd.DataFrame,
    frozen_payload: dict[str, object],
) -> dict[str, object]:
    data = _prepare_m1(gold_m1)
    _, leaderboard, _ = frozen_payload["v1"]
    best = leaderboard.iloc[0].to_dict()
    balanced_config = SignalQualityConfig(
        "Balanced Entry Frozen",
        "Trend engine",
        conviction_multiplier=1.05,
        require_h1_trend=True,
        wait_hours=2,
    )
    entry_features = _entry_features(data)
    regime_features, _, _ = _regime_features(data)
    spread_limit = float(
        data.loc[DEVELOPMENT_START:DEVELOPMENT_END, "SpreadPoints"].quantile(0.90)
    )
    development_signals = _unique_signals(
        _balanced_signals(
            data,
            signal_daily,
            best,
            entry_features,
            balanced_config,
            spread_limit,
            DEVELOPMENT_START,
            DEVELOPMENT_END,
        )
    )
    confirmation_signals = _unique_signals(
        _balanced_signals(
            data,
            signal_daily,
            best,
            entry_features,
            balanced_config,
            spread_limit,
            CONFIRMATION_START,
            CONFIRMATION_END,
        )
    )
    config = RiskControlConfig(
        "Fixed Delay 5m Robustness",
        "Delay tepat lima menit tanpa micro gate",
        max_total_positions=1,
        max_same_direction=1,
    )

    development_delayed, development_events = _build_fixed_delay_signals(
        data, development_signals, best, FIXED_DELAY_MINUTES, spread_limit
    )
    confirmation_delayed, confirmation_events = _build_fixed_delay_signals(
        data, confirmation_signals, best, FIXED_DELAY_MINUTES, spread_limit
    )
    folds = _fold_evaluation(
        data, development_signals, development_delayed, best, config
    )
    development_economic = _economic_comparison(
        data.loc[DEVELOPMENT_START:DEVELOPMENT_END],
        development_signals,
        development_delayed,
        best,
        config,
    )
    confirmation_data = data.loc[CONFIRMATION_START:CONFIRMATION_END]
    confirmation_economic = _economic_comparison(
        confirmation_data,
        confirmation_signals,
        confirmation_delayed,
        best,
        config,
    )
    delayed_result = _simulate_risk_control(
        confirmation_data, confirmation_delayed, best, config
    )
    stress = _stress_test(
        confirmation_data, confirmation_delayed, best, config
    )
    sensitivity, sensitivity_events = _delay_sensitivity(
        data, confirmation_signals, best, config, spread_limit
    )
    monte_carlo, monte_carlo_summary = _safe_monte_carlo(delayed_result.trades)
    regime_audit = _regime_audit(
        confirmation_data,
        confirmation_delayed,
        regime_features,
        best,
        config,
    )
    direction_audit = _direction_audit(
        confirmation_data, confirmation_delayed, best, config
    )
    cancellation_audit = _cancellation_audit(
        confirmation_signals, sensitivity_events
    )
    concentration = _profit_concentration(delayed_result.trades)
    monthly = _safe_monthly_summary(delayed_result)
    selected_metrics = _metric_values(delayed_result)
    development_metrics = (
        development_economic.set_index("Strategi")
        .loc["Fixed Delay 5m Validated"]
        .to_dict()
    )
    decision = _decision(
        selected_metrics,
        development_metrics,
        folds,
        stress,
        sensitivity,
        monte_carlo_summary,
        concentration,
    )

    return {
        "methodology": {
            "Baseline lock": (
                "v1 Exact Baseline, Balanced Entry, ledger, dan Live Trading tidak diubah"
            ),
            "Rule": "Entry tepat 5 menit setelah sinyal Balanced Entry",
            "Parameter search": "Tidak ada; delay 5 menit dikunci sebelum pengujian",
            "Development robustness": "12 quarterly folds, 01 Jan 2023 - 31 Des 2025",
            "Historical confirmation": "01 Jan 2026 - 30 Jun 2026; bukan true OOS baru",
            "Barrier rule": (
                "Sinyal dibatalkan bila TP atau SL awal telah tersentuh sebelum entry"
            ),
            "Spread rule": (
                f"Entry dibatalkan bila spread di atas P90 development ({spread_limit:.2f} points)"
            ),
            "Sensitivity": "Delay 4, 5, dan 6 menit; 5 menit tetap rule utama",
            "Caveat": (
                "2026H1 sudah pernah diamati. Hasil ini historical confirmation dan "
                "tidak mengubah baseline maupun paper live trading."
            ),
        },
        "data_audit": _extended_data_audit(data),
        "folds": folds,
        "development_economic": development_economic,
        "confirmation_economic": confirmation_economic,
        "delay_sensitivity": sensitivity,
        "cancellation_audit": cancellation_audit,
        "regime_audit": regime_audit,
        "direction_audit": direction_audit,
        "monthly": monthly,
        "stress": stress,
        "profit_concentration": concentration,
        "decision": decision,
        "selected_result": _compact_curve(delayed_result),
        "selected_monte_carlo": monte_carlo,
        "selected_monte_carlo_summary": monte_carlo_summary,
        "confirmation_events": _compact_events(confirmation_events),
    }


def _build_fixed_delay_signals(
    data: pd.DataFrame,
    signals: pd.DataFrame,
    best: dict[str, object],
    delay_minutes: int,
    spread_limit: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    events = _micro_event_frame(
        data, signals, best, delay_minutes, spread_limit
    )
    if events.empty:
        return signals.iloc[0:0].copy(), events
    accepted = events.loc[~events["expired"] & events["spread_ok"]]
    rows = []
    for signal_time, event in accepted.iterrows():
        if signal_time not in signals.index:
            continue
        row = signals.loc[signal_time].copy()
        row.name = pd.Timestamp(event["confirmation_time"])
        row["original_signal_time"] = signal_time
        row["strategy"] = f"Fixed Delay {delay_minutes}m"
        rows.append(row)
    if not rows:
        return signals.iloc[0:0].copy(), events
    output = pd.DataFrame(rows).sort_index()
    output = output.loc[~output.index.duplicated(keep="first")]
    return output, events


def _economic_comparison(
    data: pd.DataFrame,
    immediate: pd.DataFrame,
    delayed: pd.DataFrame,
    best: dict[str, object],
    config: RiskControlConfig,
) -> pd.DataFrame:
    rows = []
    for strategy, signals in (
        ("Balanced Entry Langsung", immediate),
        ("Fixed Delay 5m Validated", delayed),
    ):
        result = _simulate_risk_control(data, signals, best, config)
        rows.append({"Strategi": strategy, **_metric_values(result)})
    return pd.DataFrame(rows)


def _fold_evaluation(
    data: pd.DataFrame,
    immediate: pd.DataFrame,
    delayed: pd.DataFrame,
    best: dict[str, object],
    config: RiskControlConfig,
) -> pd.DataFrame:
    rows = []
    for fold in FOLDS:
        period_data = data.loc[fold.test_start:fold.test_end]
        for strategy, signals in (
            ("Balanced Entry Langsung", immediate),
            ("Fixed Delay 5m Validated", delayed),
        ):
            period_signals = signals.loc[fold.test_start:fold.test_end]
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


def _delay_sensitivity(
    data: pd.DataFrame,
    signals: pd.DataFrame,
    best: dict[str, object],
    config: RiskControlConfig,
    spread_limit: float,
) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    rows = []
    events_by_delay = {}
    period_data = data.loc[CONFIRMATION_START:CONFIRMATION_END]
    for delay in SENSITIVITY_DELAYS:
        delayed, events = _build_fixed_delay_signals(
            data, signals, best, delay, spread_limit
        )
        events_by_delay[delay] = events
        result = _simulate_risk_control(period_data, delayed, best, config)
        rows.append(
            {
                "Delay (menit)": delay,
                "Entry diterima": len(delayed),
                **_metric_values(result),
            }
        )
    return pd.DataFrame(rows), events_by_delay


def _cancellation_audit(
    signals: pd.DataFrame,
    events_by_delay: dict[int, pd.DataFrame],
) -> pd.DataFrame:
    rows = []
    for delay, events in events_by_delay.items():
        expired = int(events["expired"].sum()) if not events.empty else 0
        spread_rejected = (
            int((~events["spread_ok"] & ~events["expired"]).sum())
            if not events.empty
            else 0
        )
        accepted = (
            int((~events["expired"] & events["spread_ok"]).sum())
            if not events.empty
            else 0
        )
        rows.append(
            {
                "Delay (menit)": delay,
                "Sinyal awal": len(signals),
                "Event tersedia": len(events),
                "Entry diterima": accepted,
                "Batal barrier tersentuh": expired,
                "Batal spread": spread_rejected,
                "Data M1 tidak tersedia": max(len(signals) - len(events), 0),
            }
        )
    return pd.DataFrame(rows)


def _regime_audit(
    data: pd.DataFrame,
    signals: pd.DataFrame,
    features: pd.DataFrame,
    best: dict[str, object],
    config: RiskControlConfig,
) -> pd.DataFrame:
    regime_config = RegimeConfig(
        "RG-B Balanced", 20.0, 0.30, 58.0, 0.35, 0.12, 3
    )
    states = _regime_states(features, regime_config)
    rows = []
    labels = pd.Series(
        [states.get(pd.Timestamp(timestamp), "UNCERTAIN") for timestamp in signals.index],
        index=signals.index,
    )
    for regime in ("TRENDING", "SIDEWAYS", "UNCERTAIN"):
        selected = signals.loc[labels.eq(regime)]
        result = _simulate_risk_control(data, selected, best, config)
        rows.append(
            {
                "Regime": regime,
                "Entry": len(selected),
                **_metric_values(result),
            }
        )
    return pd.DataFrame(rows)


def _direction_audit(
    data: pd.DataFrame,
    signals: pd.DataFrame,
    best: dict[str, object],
    config: RiskControlConfig,
) -> pd.DataFrame:
    rows = []
    expected = pd.to_numeric(
        signals.get("expected_change_pct", pd.Series(index=signals.index)),
        errors="coerce",
    )
    for direction, mask in (
        ("BUY", expected.gt(0)),
        ("SELL", expected.lt(0)),
    ):
        selected = signals.loc[mask.fillna(False)]
        result = _simulate_risk_control(data, selected, best, config)
        rows.append(
            {
                "Arah": direction,
                "Entry": len(selected),
                **_metric_values(result),
            }
        )
    return pd.DataFrame(rows)


def _profit_concentration(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty or "Net P/L" not in trades.columns:
        return pd.DataFrame(
            [{"Gross profit": 0.0, "Profit top 2": 0.0, "Konsentrasi top 2 (%)": 100.0}]
        )
    net = pd.to_numeric(trades["Net P/L"], errors="coerce").fillna(0.0)
    profits = net[net > 0].sort_values(ascending=False)
    gross_profit = float(profits.sum())
    top_two = float(profits.head(2).sum())
    concentration = top_two / gross_profit * 100 if gross_profit > 0 else 100.0
    return pd.DataFrame(
        [
            {
                "Gross profit": gross_profit,
                "Profit top 2": top_two,
                "Konsentrasi top 2 (%)": concentration,
            }
        ]
    )


def _decision(
    economic: dict[str, float],
    development: dict[str, float],
    folds: pd.DataFrame,
    stress: pd.DataFrame,
    sensitivity: pd.DataFrame,
    monte_carlo: dict[str, float],
    concentration: pd.DataFrame,
) -> dict[str, object]:
    selected_folds = folds[folds["Strategi"].eq("Fixed Delay 5m Validated")]
    criteria = {
        "Minimal 9 dari 12 fold profitable": int(selected_folds["Profitable"].sum()) >= 9,
        "Minimal 10 dari 12 fold drawdown <= 10%": int(
            selected_folds["Max drawdown (%)"].le(MAX_DRAWDOWN_PCT).sum()
        ) >= 10,
        "Growth development agregat positif": development["Growth (%)"] > 0,
        "Max drawdown development agregat <= 10%": (
            development["Max drawdown (%)"] <= MAX_DRAWDOWN_PCT
        ),
        "Profit factor development agregat >= 1.50": (
            development["Profit factor"] >= PROFIT_FACTOR_TARGET
        ),
        "Growth historical confirmation positif": economic["Growth (%)"] > 0,
        "Max drawdown <= 10%": economic["Max drawdown (%)"] <= MAX_DRAWDOWN_PCT,
        "Profit factor >= 1.50": economic["Profit factor"] >= PROFIT_FACTOR_TARGET,
        "Minimal 30 transaksi confirmation": economic["Transaksi"] >= 30,
        "Stress profitable 9/9": len(stress) == 9 and bool((stress["Growth (%)"] > 0).all()),
        "Delay 4-6 menit seluruhnya profitable": bool((sensitivity["Growth (%)"] > 0).all()),
        "Monte Carlo rugi <= 10%": (
            monte_carlo["Probabilitas equity akhir < modal awal (%)"]
            <= MAX_MONTE_CARLO_LOSS_PCT
        ),
        "Konsentrasi profit top 2 <= 40%": (
            float(concentration.iloc[0]["Konsentrasi top 2 (%)"]) <= 40.0
        ),
        "Rule fixed tanpa fallback": True,
    }
    return {
        **{key: bool(value) for key, value in criteria.items()},
        "Jumlah fold profitable": int(selected_folds["Profitable"].sum()),
        "Jumlah kriteria lolos": int(sum(bool(value) for value in criteria.values())),
        "Jumlah kriteria": len(criteria),
        "Lulus seluruh kriteria": bool(all(criteria.values())),
    }


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


def _compact_events(events: pd.DataFrame) -> pd.DataFrame:
    output = events.copy()
    output["status"] = np.select(
        [output["expired"], ~output["spread_ok"]],
        ["BATAL_BARRIER", "BATAL_SPREAD"],
        default="ENTRY",
    )
    columns = [
        "confirmation_time",
        "direction",
        "status",
        "expired",
        "observed_adverse_usd",
        "observed_favorable_usd",
        "spread_points",
        "delayed_outcome",
        "delayed_mfe_usd",
        "delayed_mae_usd",
        "delayed_hours_to_outcome",
    ]
    return output[[column for column in columns if column in output.columns]]
