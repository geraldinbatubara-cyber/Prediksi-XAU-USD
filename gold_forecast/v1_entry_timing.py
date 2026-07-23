from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd

from gold_forecast.exact_broker_oos import _compact_curve, _prepare_m1
from gold_forecast.v1_entry_outcome import (
    OUTCOME_HORIZON_DAYS,
    _balanced_signals,
    _delay_signals,
    _outcome_features,
    _safe_monte_carlo,
)
from gold_forecast.v1_entry_quality import _event_economic_metrics, _stress_test
from gold_forecast.v1_entry_quality_path import (
    CONFIRMATION_END,
    CONFIRMATION_START,
    DEVELOPMENT_END,
    DEVELOPMENT_START,
    FOLDS,
    _data_audit,
    _first_barrier_path,
    _path_aware_events,
    _unique_signals,
)
from gold_forecast.v1_risk_control import (
    MAX_DRAWDOWN_PCT,
    MAX_MONTE_CARLO_LOSS_PCT,
    PROFIT_FACTOR_TARGET,
    RiskControlConfig,
    _metric_values,
    _simulate_risk_control,
)
from gold_forecast.v1_sideways_defense import _regime_features
from gold_forecast.v1_signal_quality import SignalQualityConfig, _entry_features


DELAYS = (1, 3, 5, 10, 15)


@dataclass(frozen=True)
class TimingRule:
    delay_minutes: int
    min_return_atr: float
    min_momentum_atr: float
    max_adverse_usd: float
    require_ema_alignment: bool

    @property
    def name(self) -> str:
        ema = "EMA" if self.require_ema_alignment else "NoEMA"
        return (
            f"D{self.delay_minutes}|Ret{self.min_return_atr:+.2f}|"
            f"Mom{self.min_momentum_atr:+.2f}|Adv{self.max_adverse_usd:.1f}|{ema}"
        )


def run_v1_entry_timing_lab(
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
    regime_features, _, m15 = _regime_features(data)
    spread_limit = float(
        data.loc[DEVELOPMENT_START:DEVELOPMENT_END, "SpreadPoints"].quantile(0.90)
    )
    development_signals = _unique_signals(
        _balanced_signals(
            data, signal_daily, best, entry_features, balanced_config, spread_limit,
            DEVELOPMENT_START, DEVELOPMENT_END,
        )
    )
    confirmation_signals = _unique_signals(
        _balanced_signals(
            data, signal_daily, best, entry_features, balanced_config, spread_limit,
            CONFIRMATION_START, CONFIRMATION_END,
        )
    )
    feature_frame = _outcome_features(data, regime_features, m15, best)
    development_immediate = _path_aware_events(
        data, development_signals, feature_frame, best
    )
    confirmation_immediate = _path_aware_events(
        data, confirmation_signals, feature_frame, best
    )
    micro_development = {
        delay: _micro_event_frame(data, development_signals, best, delay, spread_limit)
        for delay in DELAYS
    }
    micro_confirmation = {
        delay: _micro_event_frame(data, confirmation_signals, best, delay, spread_limit)
        for delay in DELAYS
    }

    rules = [
        TimingRule(*values)
        for values in product(
            DELAYS,
            (-0.10, 0.00, 0.10),
            (-0.10, 0.00),
            (2.5, 5.0, 7.5),
            (False, True),
        )
    ]
    candidates, fold_results = _evaluate_candidates(
        rules, micro_development, development_immediate, best
    )
    selected_row, threshold_fallback = _select_candidate(candidates)
    selected_rule = TimingRule(
        int(selected_row["Delay (menit)"]),
        float(selected_row["Min return/ATR"]),
        float(selected_row["Min momentum/ATR"]),
        float(selected_row["Max adverse (USD)"]),
        bool(selected_row["Require EMA"]),
    )

    selected_development_folds = fold_results[
        fold_results["Rule"].eq(selected_rule.name)
    ].copy()
    confirmation_micro = micro_confirmation[selected_rule.delay_minutes]
    selected_signals = _timed_signals(
        confirmation_signals, confirmation_micro, selected_rule,
        "v1 Micro Confirmation",
    )
    fixed_delay_signals = _delay_signals(
        confirmation_signals, data.index, selected_rule.delay_minutes
    )
    config = RiskControlConfig(
        "Entry Timing Lab",
        "Micro confirmation",
        max_total_positions=1,
        max_same_direction=1,
    )
    confirmation_data = data.loc[CONFIRMATION_START:CONFIRMATION_END]
    economic_signals = {
        "Balanced Entry Langsung": confirmation_signals,
        f"Fixed Delay {selected_rule.delay_minutes} menit": fixed_delay_signals,
        "v1 Micro Confirmation": selected_signals,
    }
    economic_results = {
        name: _simulate_risk_control(confirmation_data, signals, best, config)
        for name, signals in economic_signals.items()
    }
    economic = pd.DataFrame([
        {"Strategi": name, **_metric_values(result)}
        for name, result in economic_results.items()
    ])
    selected_result = economic_results["v1 Micro Confirmation"]
    selected_metrics = _metric_values(selected_result)
    baseline_net = float(
        economic_results["Balanced Entry Langsung"].summary["Total net P/L"]
    )
    selected_net = float(selected_result.summary["Total net P/L"])
    profit_retention = selected_net / baseline_net * 100 if baseline_net > 0 else 0.0
    monte_carlo, monte_carlo_summary = _safe_monte_carlo(selected_result.trades)
    stress = _stress_test(confirmation_data, selected_signals, best, config)
    delay_sensitivity = _confirmation_delay_sensitivity(
        data, confirmation_signals, micro_confirmation, selected_rule, best, config
    )
    selection_audit = _selection_audit(
        confirmation_immediate, confirmation_micro, selected_rule
    )
    path_audit = _accepted_path_audit(confirmation_micro, selected_rule)
    decision = _decision(
        selected_metrics,
        selected_development_folds,
        stress,
        delay_sensitivity,
        monte_carlo_summary,
        len(selected_signals),
        len(confirmation_signals),
        profit_retention,
        selection_audit,
        threshold_fallback,
    )

    return {
        "methodology": {
            "Baseline lock": "v1 Exact Baseline, Balanced Entry, ledger, dan Live Trading tidak diubah",
            "Development": "01 Jan 2022 - 31 Des 2025; rule dipilih dari OOF 2023-2025",
            "Walk-forward": "12 expanding quarterly folds dengan purge 14 hari",
            "Historical confirmation": "01 Jan 2026 - 30 Jun 2026; bukan true OOS baru",
            "Observation windows": "1, 3, 5, 10, dan 15 menit setelah sinyal",
            "Selected rule": selected_rule.name,
            "Selected delay": selected_rule.delay_minutes,
            "Min return/ATR": selected_rule.min_return_atr,
            "Min momentum/ATR": selected_rule.min_momentum_atr,
            "Max adverse (USD)": selected_rule.max_adverse_usd,
            "Require EMA alignment": selected_rule.require_ema_alignment,
            "Spread ceiling points": spread_limit,
            "Rule fallback": threshold_fallback,
            "Caveat": (
                "2026H1 sudah pernah diamati. Hasil ini hanya historical confirmation dan "
                "tidak mengubah baseline atau paper live trading."
            ),
        },
        "data_audit": _extended_data_audit(data),
        "candidate_summary": candidates.head(60).copy(),
        "selected_folds": selected_development_folds,
        "economic": economic,
        "selection_audit": selection_audit,
        "accepted_path_audit": path_audit,
        "delay_sensitivity": delay_sensitivity,
        "stress": stress,
        "decision": decision,
        "selected_result": _compact_curve(selected_result),
        "selected_monte_carlo": monte_carlo,
        "selected_monte_carlo_summary": monte_carlo_summary,
        "confirmation_events": _compact_confirmation_events(
            confirmation_micro, selected_rule
        ),
        "signal_counts": {
            name: int(len(signals)) for name, signals in economic_signals.items()
        },
        "profit_retention": profit_retention,
    }


def _micro_event_frame(
    data: pd.DataFrame,
    signals: pd.DataFrame,
    best: dict[str, object],
    delay_minutes: int,
    spread_limit: float,
) -> pd.DataFrame:
    close = data["Close"]
    previous_close = close.shift(1)
    true_range = pd.concat([
        data["High"] - data["Low"],
        (data["High"] - previous_close).abs(),
        (data["Low"] - previous_close).abs(),
    ], axis=1).max(axis=1)
    atr = true_range.rolling(14, min_periods=5).mean().replace(0, np.nan)
    ema_fast = close.ewm(span=3, adjust=False).mean()
    ema_slow = close.ewm(span=10, adjust=False).mean()
    rows = []
    for signal_time, signal in signals.iterrows():
        target_time = pd.Timestamp(signal_time) + pd.Timedelta(minutes=delay_minutes)
        location = data.index.searchsorted(target_time, side="left")
        if location >= len(data.index):
            continue
        confirmation_time = data.index[location]
        if confirmation_time > signal_time + pd.Timedelta(minutes=delay_minutes + 5):
            continue
        window = data.loc[(data.index > signal_time) & (data.index <= confirmation_time)]
        if window.empty:
            continue
        expected = float(signal["expected_change_pct"])
        direction = "BUY" if expected > 0 else "SELL"
        sign = 1.0 if direction == "BUY" else -1.0
        lot = float(signal.get("lot", best.get("Lot", 0.01)) or 0.01)
        units = lot * 100.0
        reference = float(data.loc[signal_time, "Close"])
        current = float(data.loc[confirmation_time, "Close"])
        atr_value = float(atr.loc[confirmation_time])
        if not np.isfinite(atr_value) or atr_value <= 0:
            continue
        momentum_location = max(0, location - 3)
        momentum_reference = float(close.iloc[momentum_location])
        signed_return = sign * (current - reference) / atr_value
        signed_momentum = sign * (current - momentum_reference) / atr_value
        signed_ema_gap = sign * (
            float(ema_fast.loc[confirmation_time]) - float(ema_slow.loc[confirmation_time])
        ) / atr_value
        if direction == "BUY":
            adverse = max((reference - float(window["Low"].min())) * units, 0.0)
            favorable = max((float(window["High"].max()) - reference) * units, 0.0)
        else:
            adverse = max((float(window["High"].max()) - reference) * units, 0.0)
            favorable = max((reference - float(window["Low"].min())) * units, 0.0)
        expired = bool(
            adverse >= float(best["SL (USD)"])
            or favorable >= float(best["TP (USD)"])
        )
        outcome = _first_barrier_path(
            data, confirmation_time, direction, lot, best
        )
        rows.append({
            "signal_time": signal_time,
            "confirmation_time": confirmation_time,
            "direction": direction,
            "signed_return_atr": signed_return,
            "signed_momentum_atr": signed_momentum,
            "signed_ema_gap_atr": signed_ema_gap,
            "observed_adverse_usd": adverse,
            "observed_favorable_usd": favorable,
            "spread_points": float(data.loc[confirmation_time, "SpreadPoints"]),
            "spread_ok": float(data.loc[confirmation_time, "SpreadPoints"]) <= spread_limit,
            "expired": expired,
            "delayed_outcome": outcome["raw_outcome"],
            "delayed_target": float(outcome["raw_outcome"] == "TP_FIRST"),
            "delayed_mfe_usd": outcome["mfe_usd"],
            "delayed_mae_usd": outcome["mae_usd"],
            "delayed_hours_to_outcome": outcome["hours_to_outcome"],
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("signal_time").sort_index()


def _rule_mask(events: pd.DataFrame, rule: TimingRule) -> pd.Series:
    return (
        ~events["expired"]
        & events["spread_ok"]
        & events["signed_return_atr"].ge(rule.min_return_atr)
        & events["signed_momentum_atr"].ge(rule.min_momentum_atr)
        & events["observed_adverse_usd"].le(rule.max_adverse_usd)
        & (
            events["signed_ema_gap_atr"].ge(0)
            if rule.require_ema_alignment
            else pd.Series(True, index=events.index)
        )
    )


def _evaluate_candidates(
    rules: list[TimingRule],
    micro: dict[int, pd.DataFrame],
    immediate: pd.DataFrame,
    best: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    oof_index = immediate.loc[pd.Timestamp("2023-01-01"):DEVELOPMENT_END].index
    baseline_values = pd.Series(
        np.where(immediate.loc[oof_index, "target"].eq(1), float(best["TP (USD)"]), -float(best["SL (USD)"])),
        index=oof_index,
    )
    baseline_net = float(baseline_values.sum())
    rows = []
    fold_rows = []
    for rule in rules:
        events = micro[rule.delay_minutes].reindex(oof_index).dropna(
            subset=["delayed_target"]
        )
        accepted = events.loc[_rule_mask(events, rule)]
        values = pd.Series(
            np.where(accepted["delayed_target"].eq(1), float(best["TP (USD)"]), -float(best["SL (USD)"])),
            index=accepted.index,
        )
        metrics = _event_economic_metrics(values)
        net = float(values.sum())
        profitable_folds = 0
        fold_trade_counts = []
        for fold in FOLDS:
            fold_values = values.loc[fold.test_start:fold.test_end]
            fold_metrics = _event_economic_metrics(fold_values)
            profitable = bool(fold_metrics["Growth (%)"] > 0)
            profitable_folds += int(profitable)
            fold_trade_counts.append(len(fold_values))
            fold_rows.append({
                "Rule": rule.name,
                "Fold": fold.name,
                "Test mulai": fold.test_start,
                "Test akhir": fold.test_end,
                **fold_metrics,
                "Profitable": profitable,
            })
        profit_retention = net / baseline_net * 100 if baseline_net > 0 else 0.0
        eligible = bool(
            len(values) >= 100
            and metrics["Growth (%)"] > 0
            and metrics["Max drawdown (%)"] <= MAX_DRAWDOWN_PCT
            and metrics["Profit factor"] >= PROFIT_FACTOR_TARGET
            and profit_retention >= 70
            and profitable_folds >= 8
            and min(fold_trade_counts, default=0) >= 3
        )
        rows.append({
            "Rule": rule.name,
            "Delay (menit)": rule.delay_minutes,
            "Min return/ATR": rule.min_return_atr,
            "Min momentum/ATR": rule.min_momentum_atr,
            "Max adverse (USD)": rule.max_adverse_usd,
            "Require EMA": rule.require_ema_alignment,
            **metrics,
            "Entry tersedia": len(events),
            "Entry diterima": len(values),
            "Retensi entry (%)": len(values) / len(events) * 100 if len(events) else 0.0,
            "Retensi net profit (%)": profit_retention,
            "Fold profitable": profitable_folds,
            "Eligible": eligible,
        })
    candidates = pd.DataFrame(rows).sort_values(
        ["Eligible", "Fold profitable", "Profit factor", "Growth (%)"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    return candidates, pd.DataFrame(fold_rows)


def _select_candidate(candidates: pd.DataFrame) -> tuple[pd.Series, bool]:
    eligible = candidates[candidates["Eligible"]]
    fallback = eligible.empty
    pool = eligible if not eligible.empty else candidates[
        (candidates["Entry diterima"] >= 80)
        & (candidates["Growth (%)"] > 0)
    ]
    if pool.empty:
        pool = candidates[candidates["Entry diterima"] > 0]
    selected = pool.sort_values(
        ["Fold profitable", "Profit factor", "Growth (%)", "Retensi net profit (%)"],
        ascending=[False, False, False, False],
    ).iloc[0]
    return selected, fallback


def _timed_signals(
    signals: pd.DataFrame,
    events: pd.DataFrame,
    rule: TimingRule,
    strategy: str,
) -> pd.DataFrame:
    accepted = events.loc[_rule_mask(events, rule)]
    rows = []
    for signal_time, event in accepted.iterrows():
        if signal_time not in signals.index:
            continue
        row = signals.loc[signal_time].copy()
        row.name = pd.Timestamp(event["confirmation_time"])
        row["original_signal_time"] = signal_time
        row["strategy"] = strategy
        rows.append(row)
    if not rows:
        return signals.iloc[0:0].copy()
    output = pd.DataFrame(rows).sort_index()
    return output.loc[~output.index.duplicated(keep="last")]


def _confirmation_delay_sensitivity(
    data: pd.DataFrame,
    signals: pd.DataFrame,
    micro: dict[int, pd.DataFrame],
    selected_rule: TimingRule,
    best: dict[str, object],
    config: RiskControlConfig,
) -> pd.DataFrame:
    rows = []
    period_data = data.loc[CONFIRMATION_START:CONFIRMATION_END]
    for delay in DELAYS:
        rule = TimingRule(
            delay,
            selected_rule.min_return_atr,
            selected_rule.min_momentum_atr,
            selected_rule.max_adverse_usd,
            selected_rule.require_ema_alignment,
        )
        selected = _timed_signals(signals, micro[delay], rule, "Delay sensitivity")
        result = _simulate_risk_control(period_data, selected, best, config)
        rows.append({
            "Delay (menit)": delay,
            "Entry diterima": len(selected),
            **_metric_values(result),
        })
    return pd.DataFrame(rows)


def _selection_audit(
    immediate: pd.DataFrame,
    micro: pd.DataFrame,
    rule: TimingRule,
) -> pd.DataFrame:
    aligned = immediate.reindex(micro.index)
    accepted = _rule_mask(micro, rule)
    winner = aligned["target"].eq(1)
    groups = [
        ("Winner diterima", accepted & winner),
        ("Winner terlewat", ~accepted & winner),
        ("Loser dihindari", ~accepted & ~winner),
        ("Loser tetap masuk", accepted & ~winner),
    ]
    rows = []
    for label, mask in groups:
        rows.append({
            "Kelompok": label,
            "Events": int(mask.sum()),
            "Proporsi (%)": float(mask.mean() * 100),
            "Median return/ATR": float(micro.loc[mask, "signed_return_atr"].median()) if mask.any() else np.nan,
            "Median adverse": float(micro.loc[mask, "observed_adverse_usd"].median()) if mask.any() else np.nan,
        })
    return pd.DataFrame(rows)


def _accepted_path_audit(
    micro: pd.DataFrame,
    rule: TimingRule,
) -> pd.DataFrame:
    accepted = micro.loc[_rule_mask(micro, rule)]
    rows = []
    for outcome, frame in accepted.groupby("delayed_outcome"):
        rows.append({
            "Outcome setelah entry": outcome,
            "Events": len(frame),
            "Median MFE": float(frame["delayed_mfe_usd"].median()),
            "Median MAE": float(frame["delayed_mae_usd"].median()),
            "Median jam outcome": float(frame["delayed_hours_to_outcome"].median()),
            "Median observed adverse": float(frame["observed_adverse_usd"].median()),
        })
    return pd.DataFrame(rows)


def _extended_data_audit(data: pd.DataFrame) -> pd.DataFrame:
    audit = _data_audit(data)
    expected = set(pd.period_range("2022-01", "2026-06", freq="M"))
    actual = set(data.loc[DEVELOPMENT_START:CONFIRMATION_END].index.to_period("M").unique())
    coverage = audit["Pemeriksaan"].str.startswith("Cakupan bulan")
    audit.loc[coverage, "Pemeriksaan"] = "Cakupan bulan 2022-01 sampai 2026-06"
    audit.loc[coverage, "Status"] = "LOLOS" if expected.issubset(actual) else "BELUM"
    audit.loc[coverage, "Detail"] = f"{len(expected & actual)}/{len(expected)} bulan tersedia"
    return audit


def _decision(
    economic: dict[str, float],
    folds: pd.DataFrame,
    stress: pd.DataFrame,
    delay_sensitivity: pd.DataFrame,
    monte_carlo: dict[str, float],
    selected_count: int,
    available_count: int,
    profit_retention: float,
    selection_audit: pd.DataFrame,
    fallback: bool,
) -> dict[str, object]:
    audit = selection_audit.set_index("Kelompok")["Events"]
    winner_total = int(audit.get("Winner diterima", 0) + audit.get("Winner terlewat", 0))
    winner_retention = (
        float(audit.get("Winner diterima", 0) / winner_total * 100)
        if winner_total else 0.0
    )
    criteria = {
        "Minimal 8 dari 12 fold profitable": int(folds["Profitable"].sum()) >= 8,
        "Growth historical confirmation positif": economic["Growth (%)"] > 0,
        "Max drawdown <= 10%": economic["Max drawdown (%)"] <= MAX_DRAWDOWN_PCT,
        "Profit factor >= 1.30": economic["Profit factor"] >= PROFIT_FACTOR_TARGET,
        "Minimal 30 transaksi confirmation": economic["Transaksi"] >= 30,
        "Retensi entry >= 25%": selected_count >= max(1, int(available_count * 0.25)),
        "Retensi winner >= 70%": winner_retention >= 70,
        "Minimal satu loser dihindari": int(audit.get("Loser dihindari", 0)) > 0,
        "Retensi net profit >= 70%": profit_retention >= 70,
        "Stress profitable 9/9": len(stress) == 9 and bool((stress["Growth (%)"] > 0).all()),
        "Minimal 3 dari 5 delay profitable": int((delay_sensitivity["Growth (%)"] > 0).sum()) >= 3,
        "Monte Carlo rugi <= 10%": monte_carlo["Probabilitas equity akhir < modal awal (%)"] <= MAX_MONTE_CARLO_LOSS_PCT,
        "Rule tanpa fallback": not fallback,
    }
    return {
        **{key: bool(value) for key, value in criteria.items()},
        "Jumlah kriteria lolos": int(sum(bool(value) for value in criteria.values())),
        "Jumlah kriteria": len(criteria),
        "Lulus seluruh kriteria": bool(all(criteria.values())),
        "Retensi winner (%)": winner_retention,
        "Retensi net profit (%)": float(profit_retention),
    }


def _compact_confirmation_events(
    events: pd.DataFrame,
    rule: TimingRule,
) -> pd.DataFrame:
    output = events.copy()
    output["gate_status"] = np.where(
        _rule_mask(events, rule), "DITERIMA", "DITOLAK"
    )
    columns = [
        "confirmation_time", "direction", "gate_status", "expired",
        "signed_return_atr", "signed_momentum_atr", "signed_ema_gap_atr",
        "observed_adverse_usd", "observed_favorable_usd", "spread_points",
        "delayed_outcome", "delayed_mfe_usd", "delayed_mae_usd",
        "delayed_hours_to_outcome",
    ]
    return output[[column for column in columns if column in output.columns]]
