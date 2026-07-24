from __future__ import annotations

import base64
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


MODEL_COMPARISON_VERSION = "optimizer-v1-model-comparison-26-experiments-v1"


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    group: str
    filename: str
    source: str
    candidate: str = ""
    result_key: str = ""
    comparable: bool = True
    note: str = ""


EXPERIMENTS = (
    ExperimentSpec(
        "Optimizer v1 OOS",
        "Baseline & Validasi",
        "optimizer_oos.pkl",
        "tuple",
        candidate="Optimizer v1 Frozen 2025",
        result_key="2",
        comparable=False,
        note="OOS harian; tidak memodelkan fill M1 broker.",
    ),
    ExperimentSpec(
        "Optimizer v1 Exact Broker-Aware OOS",
        "Baseline & Validasi",
        "exact_broker_oos.pkl",
        "tuple",
        candidate="Optimizer v1 Exact",
        result_key="0",
        note="OOS 2026H1 dengan eksekusi M1, spread, dan slippage.",
    ),
    ExperimentSpec(
        "Optimizer v1 Robustness Test",
        "Baseline & Validasi",
        "v1_robustness.pkl",
        "result",
        result_key="baseline_result",
        note="Baseline exact diuji dengan stress dan Monte Carlo.",
    ),
    ExperimentSpec(
        "v1 Risk-Control Lab",
        "General / Balanced",
        "v1_risk_control.pkl",
        "result",
        candidate="RC-D Profit Protection",
        result_key="winner_result",
    ),
    ExperimentSpec(
        "v1 Balanced Entry",
        "General / Balanced",
        "v1_signal_quality.pkl.b64",
        "result",
        candidate="BE-E H1 Trend Wait-2h",
        result_key="winner_result",
    ),
    ExperimentSpec(
        "v1 Balanced Robustness",
        "General / Balanced",
        "v1_balanced_robustness.pkl.b64",
        "result",
        candidate="Balanced Entry",
        result_key="center_result",
    ),
    ExperimentSpec(
        "v1 Regime Classifier v2",
        "General / Balanced",
        "v1_regime_classifier.pkl.b64",
        "result",
        candidate="Classifier v2 Trend Gate",
        result_key="selected_result",
    ),
    ExperimentSpec(
        "v1 Entry Outcome Lab",
        "General / Balanced",
        "v1_entry_outcome.pkl.b64",
        "result",
        candidate="v1 Probability-Gated Entry",
        result_key="selected_result",
    ),
    ExperimentSpec(
        "v1 Entry Quality Lab v2",
        "General / Balanced",
        "v1_entry_quality.pkl.b64",
        "result",
        candidate="v1 Entry Quality v2",
        result_key="selected_result",
    ),
    ExperimentSpec(
        "v1 Entry Quality Path-Aware v3",
        "General / Balanced",
        "v1_entry_quality_path.pkl.b64",
        "result",
        candidate="v1 Entry Quality Path-Aware",
        result_key="selected_result",
    ),
    ExperimentSpec(
        "v1 Entry Timing Lab v1",
        "General / Balanced",
        "v1_entry_timing.pkl.b64",
        "result",
        candidate="v1 Micro Confirmation",
        result_key="selected_result",
    ),
    ExperimentSpec(
        "v1 Fixed Delay 5m Robustness",
        "General / Balanced",
        "v1_fixed_delay.pkl.b64",
        "result",
        candidate="Fixed Delay 5 menit",
        result_key="selected_result",
    ),
    ExperimentSpec(
        "v1 Unified Strategy Benchmark",
        "General / Balanced",
        "v1_unified_benchmark.pkl.b64",
        "ranking",
        candidate="Fixed Delay 5m",
    ),
    ExperimentSpec(
        "v1 Fixed Delay Quality Guard",
        "General / Balanced",
        "v1_fixed_delay_quality.pkl.b64",
        "ranking",
        candidate="Volatility Corridor",
    ),
    ExperimentSpec(
        "v1 Trend Strength Stability",
        "General / Balanced",
        "v1_trend_strength_stability.pkl.b64",
        "ranking",
        candidate="Fixed Delay 5m Control",
    ),
    ExperimentSpec(
        "v1 Trend-Regime Fusion",
        "General / Balanced",
        "v1_trend_regime_fusion.pkl.b64",
        "ranking",
        candidate="Q40 Only",
    ),
    ExperimentSpec(
        "v1 Regime Classifier v3",
        "General / Balanced",
        "v1_regime_classifier_v3.pkl.b64",
        "ranking",
        candidate="Ensemble Adaptive Confirmation",
        note="Ekonomi kuat, tetapi classifier tidak lulus gerbang klasifikasi.",
    ),
    ExperimentSpec(
        "BUY Specialist v4",
        "BUY Specialist",
        "v1_directional_specialization.pkl.b64",
        "tables",
        candidate="Adaptive + Bear/Sideways Defense",
        note="Mesin spesialis BUY; tidak dinilai sebagai mesin universal dua arah.",
    ),
    ExperimentSpec(
        "SELL Specialist v5",
        "SELL Specialist",
        "v1_sell_specialist.pkl.b64",
        "ranking",
        candidate="SELL Probability Ensemble",
    ),
    ExperimentSpec(
        "SELL Specialist v6",
        "SELL Specialist",
        "v1_sell_specialist_v6.pkl.b64",
        "ranking",
        candidate="Breakdown Retest 15m",
    ),
    ExperimentSpec(
        "SELL Specialist v7",
        "SELL Specialist",
        "v1_sell_specialist_v7.pkl.b64",
        "ranking",
        candidate="Adaptive Event Ensemble",
    ),
    ExperimentSpec(
        "v1 Sideways Defense",
        "Sideways Specialist",
        "v1_sideways_defense.pkl.b64",
        "result",
        candidate="Regime Strategy",
        result_key="hybrid_result",
    ),
    ExperimentSpec(
        "Sideways Specialist v1",
        "Sideways Specialist",
        "v1_sideways_specialist.pkl.b64",
        "ranking",
        candidate="Adaptive Range Ensemble",
    ),
    ExperimentSpec(
        "Sideways Specialist v2",
        "Sideways Specialist",
        "v1_sideways_specialist_v2.pkl.b64",
        "ranking",
        candidate="Breakout Hazard Gate",
    ),
    ExperimentSpec(
        "Sideways Specialist v3",
        "Sideways Specialist",
        "v1_sideways_specialist_v3.pkl.b64",
        "ranking",
        candidate="Hazard Acceleration Protection",
    ),
    ExperimentSpec(
        "Sideways Specialist v4",
        "Sideways Specialist",
        "v1_sideways_specialist_v4.pkl.b64",
        "ranking",
        candidate="Breakout Hazard v2 Control",
    ),
)


def _load_payload(path: Path) -> dict[str, Any]:
    if path.suffix == ".b64":
        saved = pickle.loads(base64.b64decode(path.read_text(encoding="ascii")))
    else:
        with path.open("rb") as file:
            saved = pickle.load(file)
    return saved["payload"]


def _number(value: Any, default: float = np.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def _lookup(mapping: dict[str, Any], names: tuple[str, ...], default: float = np.nan) -> float:
    for name in names:
        if name in mapping:
            return _number(mapping[name], default)
    return default


def _candidate_column(frame: pd.DataFrame) -> str | None:
    for column in ("Kandidat", "Strategi", "Model"):
        if column in frame.columns:
            return column
    return None


def _matching_row(frame: Any, candidate: str) -> dict[str, Any]:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return {}
    column = _candidate_column(frame)
    if column is None:
        return frame.iloc[0].to_dict()
    exact = frame.loc[frame[column].astype(str).str.casefold() == candidate.casefold()]
    if not exact.empty:
        return exact.iloc[0].to_dict()
    partial = frame.loc[
        frame[column].astype(str).str.casefold().str.contains(candidate.casefold(), regex=False)
    ]
    return partial.iloc[0].to_dict() if not partial.empty else {}


def _summary_metrics(summary: dict[str, Any]) -> dict[str, float]:
    capital = _lookup(summary, ("Modal awal",), 1000.0)
    drawdown = _lookup(summary, ("Max drawdown (%)",))
    if np.isnan(drawdown):
        absolute = _lookup(summary, ("Max drawdown",))
        drawdown = absolute / capital * 100.0 if capital > 0 and not np.isnan(absolute) else np.nan
    return {
        "oos_growth": _lookup(summary, ("Growth total", "Growth (%)")),
        "oos_pf": _lookup(summary, ("Profit factor",)),
        "oos_dd": drawdown,
        "oos_trades": _lookup(summary, ("Jumlah transaksi", "Transaksi")),
        "win_rate": _lookup(summary, ("Win rate", "Win rate (%)")),
    }


def _ranking_metrics(row: dict[str, Any]) -> dict[str, float]:
    return {
        "oos_growth": _lookup(
            row,
            (
                "Growth 2026H1 (%)",
                "Growth test (%)",
                "Growth confirmation (%)",
                "Growth locked 2025 (%)",
                "Growth (%)",
            ),
        ),
        "oos_pf": _lookup(
            row,
            ("PF 2026H1", "PF test", "Profit factor", "PF locked 2025"),
        ),
        "oos_dd": _lookup(
            row,
            ("DD 2026H1 (%)", "DD test (%)", "Max drawdown (%)", "DD locked 2025 (%)"),
        ),
        "oos_trades": _lookup(
            row,
            ("Transaksi 2026H1", "Transaksi test", "Transaksi", "Transaksi locked 2025"),
        ),
        "dev_growth": _lookup(row, ("Growth development (%)",)),
        "dev_pf": _lookup(row, ("PF development",)),
        "dev_dd": _lookup(row, ("DD development (%)",)),
        "dev_trades": _lookup(row, ("Transaksi development",)),
        "locked_growth": _lookup(row, ("Growth locked 2025 (%)",)),
    }


def _table_metrics(payload: dict[str, Any], candidate: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    development = _matching_row(payload.get("development"), candidate)
    reference = _matching_row(payload.get("historical_reference"), candidate)
    if development:
        metrics.update(
            {
                "dev_growth": _lookup(development, ("Growth (%)",)),
                "dev_pf": _lookup(development, ("Profit factor",)),
                "dev_dd": _lookup(development, ("Max drawdown (%)",)),
                "dev_trades": _lookup(development, ("Transaksi",)),
            }
        )
    if reference:
        metrics.update(
            {
                "oos_growth": _lookup(reference, ("Growth (%)",)),
                "oos_pf": _lookup(reference, ("Profit factor",)),
                "oos_dd": _lookup(reference, ("Max drawdown (%)",)),
                "oos_trades": _lookup(reference, ("Transaksi",)),
                "win_rate": _lookup(reference, ("Win rate (%)",)),
            }
        )
    return metrics


def _augment_period_metrics(
    metrics: dict[str, float], payload: dict[str, Any], candidate: str
) -> None:
    development = _matching_row(payload.get("development"), candidate)
    if development:
        metrics.setdefault("dev_growth", _lookup(development, ("Growth (%)",)))
        metrics.setdefault("dev_pf", _lookup(development, ("Profit factor",)))
        metrics.setdefault("dev_dd", _lookup(development, ("Max drawdown (%)",)))
        metrics.setdefault("dev_trades", _lookup(development, ("Transaksi",)))

    historical = _matching_row(payload.get("historical_reference"), candidate)
    if historical:
        for key, value in {
            "oos_growth": _lookup(historical, ("Growth (%)",)),
            "oos_pf": _lookup(historical, ("Profit factor",)),
            "oos_dd": _lookup(historical, ("Max drawdown (%)",)),
            "oos_trades": _lookup(historical, ("Transaksi",)),
            "win_rate": _lookup(historical, ("Win rate (%)",)),
        }.items():
            if np.isnan(metrics.get(key, np.nan)) and not np.isnan(value):
                metrics[key] = value

    period = payload.get("period_validation")
    if isinstance(period, pd.DataFrame) and not period.empty:
        matched = period
        column = _candidate_column(period)
        if column is not None:
            matched = matched.loc[
                matched[column].astype(str).str.casefold() == candidate.casefold()
            ]
        if not matched.empty:
            locked = matched.loc[
                matched.get("Periode", pd.Series(index=matched.index, dtype=str))
                .astype(str)
                .str.contains("2025", regex=False)
            ]
            if not locked.empty:
                metrics["locked_growth"] = _lookup(locked.iloc[-1].to_dict(), ("Growth (%)",))

    for key in ("monte_carlo_summary", "winner_monte_carlo_summary", "center_monte_carlo_summary"):
        row = _matching_row(payload.get(key), candidate)
        if row:
            metrics["mc_loss"] = _lookup(
                row, ("Probabilitas equity akhir < modal awal (%)",)
            )
            break

    decisions = _matching_row(payload.get("decisions"), candidate)
    if not decisions:
        decisions = _matching_row(payload.get("decision"), candidate)
    if decisions:
        passed = _lookup(decisions, ("Kriteria lolos", "Total kriteria lolos"))
        total = _lookup(decisions, ("Total kriteria",))
        if not np.isnan(passed) and total > 0:
            metrics["criteria_ratio"] = passed / total
        if "Lulus" in decisions:
            metrics["lab_passed"] = bool(decisions["Lulus"])

    folds = payload.get("folds")
    if isinstance(folds, pd.DataFrame) and not folds.empty:
        matched = folds
        column = _candidate_column(folds)
        if column is not None:
            matched = matched.loc[
                matched[column].astype(str).str.casefold() == candidate.casefold()
            ]
        profitable_column = next(
            (column for column in matched.columns if "profitable" in column.casefold()),
            None,
        )
        if profitable_column and not matched.empty:
            values = matched[profitable_column]
            metrics["fold_ratio"] = float(pd.to_numeric(values, errors="coerce").fillna(0).mean())
        else:
            growth_column = next(
                (
                    column
                    for column in matched.columns
                    if "growth" in column.casefold() and "train" not in column.casefold()
                ),
                None,
            )
            if growth_column and not matched.empty:
                values = pd.to_numeric(matched[growth_column], errors="coerce").dropna()
                if not values.empty:
                    metrics["fold_ratio"] = float((values > 0).mean())


def _raw_metrics(spec: ExperimentSpec, payload: dict[str, Any]) -> dict[str, float]:
    if spec.source == "tuple":
        result = payload["v1"][int(spec.result_key)]
        metrics = _summary_metrics(result.summary)
    elif spec.source == "result":
        result = payload[spec.result_key]
        metrics = _summary_metrics(result.summary)
    elif spec.source == "ranking":
        row = _matching_row(payload.get("ranking"), spec.candidate)
        metrics = _ranking_metrics(row)
    elif spec.source == "tables":
        metrics = _table_metrics(payload, spec.candidate)
    else:
        raise ValueError(f"Unknown comparison source: {spec.source}")

    _augment_period_metrics(metrics, payload, spec.candidate)
    if spec.name == "Optimizer v1 Robustness Test":
        summary = payload["summary"]
        metrics["mc_loss"] = _lookup(
            summary, ("Probabilitas equity akhir < modal awal (%)",)
        )
        metrics["criteria_ratio"] = _lookup(summary, ("Skenario profitable",), 0.0) / max(
            _lookup(summary, ("Total skenario",), 1.0), 1.0
        )
    return metrics


def _clamp(value: float, low: float, high: float) -> float:
    return float(np.clip(value, low, high))


def _score(metrics: dict[str, float]) -> dict[str, float]:
    growth = metrics.get("oos_growth", np.nan)
    pf = metrics.get("oos_pf", np.nan)
    dd = metrics.get("oos_dd", np.nan)
    trades = metrics.get("oos_trades", np.nan)
    mc_loss = metrics.get("mc_loss", np.nan)

    profitability = 12.5 if np.isnan(growth) else _clamp((growth + 5.0) / 35.0 * 25.0, 0.0, 25.0)
    quality = 10.0 if np.isnan(pf) else _clamp((pf - 0.75) / 1.25 * 20.0, 0.0, 20.0)
    dd_score = 7.5 if np.isnan(dd) else _clamp((20.0 - dd) / 15.0 * 15.0, 0.0, 15.0)
    mc_score = 5.0 if np.isnan(mc_loss) else _clamp((25.0 - mc_loss) / 20.0 * 10.0, 0.0, 10.0)
    risk = dd_score + mc_score

    dev_growth = metrics.get("dev_growth", np.nan)
    locked_growth = metrics.get("locked_growth", np.nan)
    fold_ratio = metrics.get("fold_ratio", np.nan)
    criteria_ratio = metrics.get("criteria_ratio", np.nan)
    robustness = (
        (2.5 if np.isnan(dev_growth) else (5.0 if dev_growth > 0 else 0.0))
        + (2.5 if np.isnan(locked_growth) else (5.0 if locked_growth > 0 else 0.0))
        + (2.5 if np.isnan(fold_ratio) else _clamp(fold_ratio, 0.0, 1.0) * 5.0)
        + (2.5 if np.isnan(criteria_ratio) else _clamp(criteria_ratio, 0.0, 1.0) * 5.0)
    )
    sample = 3.5 if np.isnan(trades) else _clamp(trades / 50.0, 0.0, 1.0) * 7.0
    available = sum(
        not np.isnan(metrics.get(key, np.nan))
        for key in ("oos_growth", "oos_pf", "oos_dd", "oos_trades", "dev_growth", "mc_loss")
    )
    sample += available / 6.0 * 3.0

    penalty = 0.0
    if not np.isnan(growth) and growth < 0:
        penalty += 8.0
    if not np.isnan(pf) and pf < 1.0:
        penalty += 5.0
    if not np.isnan(dd) and dd > 15.0:
        penalty += 8.0
    if not np.isnan(mc_loss) and mc_loss > 25.0:
        penalty += 5.0
    if not np.isnan(trades) and trades < 10:
        penalty += 4.0

    total = _clamp(profitability + quality + risk + robustness + sample - penalty, 0.0, 100.0)
    return {
        "Skor Profitabilitas": profitability,
        "Skor Kualitas Profit": quality,
        "Skor Risiko": risk,
        "Skor Robustness": robustness,
        "Skor Sampel": sample,
        "Penalti": penalty,
        "Skor Total": total,
    }


def _grade(row: dict[str, Any]) -> str:
    score = row["Skor Total"]
    growth = row["Growth OOS (%)"]
    pf = row["PF OOS"]
    dd = row["DD OOS (%)"]
    mc_loss = row["Prob. Rugi MC (%)"]
    gates = (
        growth > 0
        and pf >= 1.30
        and dd <= 10
        and (np.isnan(mc_loss) or mc_loss <= 10)
    )
    if score >= 80 and gates and row["Lulus Lab"]:
        return "A - Kandidat kuat"
    if score >= 65:
        return "B - Menjanjikan"
    if score >= 50:
        return "C - Eksperimental"
    if score >= 35:
        return "D - Lemah"
    return "E - Tidak layak"


def build_model_comparison(precomputed_dir: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for spec in EXPERIMENTS:
        payload = _load_payload(precomputed_dir / spec.filename)
        metrics = _raw_metrics(spec, payload)
        scores = _score(metrics)
        row = {
            "Kelompok": spec.group,
            "Eksperimen": spec.name,
            "Kandidat wakil": spec.candidate or spec.name,
            "Growth OOS (%)": metrics.get("oos_growth", np.nan),
            "PF OOS": metrics.get("oos_pf", np.nan),
            "DD OOS (%)": metrics.get("oos_dd", np.nan),
            "Transaksi OOS": metrics.get("oos_trades", np.nan),
            "Growth development (%)": metrics.get("dev_growth", np.nan),
            "Growth locked (%)": metrics.get("locked_growth", np.nan),
            "Prob. Rugi MC (%)": metrics.get("mc_loss", np.nan),
            "Fold profitable (%)": (
                metrics.get("fold_ratio", np.nan) * 100.0
                if not np.isnan(metrics.get("fold_ratio", np.nan))
                else np.nan
            ),
            "Lulus Lab": bool(metrics.get("lab_passed", False)),
            "Komparabilitas": "Setara M1/OOS" if spec.comparable else "Terbatas",
            "Catatan": spec.note,
            **scores,
        }
        row["Grade"] = _grade(row)
        rows.append(row)

    master = pd.DataFrame(rows).sort_values(
        ["Skor Total", "Growth OOS (%)"], ascending=[False, False]
    )
    master.insert(0, "Peringkat", range(1, len(master) + 1))
    master["Peringkat kelompok"] = (
        master.groupby("Kelompok")["Skor Total"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    group_rankings = {
        group: frame.sort_values(
            ["Peringkat kelompok", "Skor Total"], ascending=[True, False]
        ).reset_index(drop=True)
        for group, frame in master.groupby("Kelompok", sort=False)
    }
    winners = (
        master.sort_values(["Kelompok", "Peringkat kelompok"])
        .groupby("Kelompok", as_index=False)
        .first()[
            [
                "Kelompok",
                "Eksperimen",
                "Kandidat wakil",
                "Skor Total",
                "Growth OOS (%)",
                "PF OOS",
                "DD OOS (%)",
                "Grade",
            ]
        ]
    )
    methodology = pd.DataFrame(
        [
            {"Dimensi": "Profitabilitas OOS", "Bobot": 25, "Dasar": "Growth periode test/OOS"},
            {"Dimensi": "Kualitas profit", "Bobot": 20, "Dasar": "Profit factor OOS"},
            {"Dimensi": "Risiko", "Bobot": 25, "Dasar": "Drawdown OOS dan probabilitas rugi Monte Carlo"},
            {"Dimensi": "Robustness", "Bobot": 20, "Dasar": "Development, locked period, fold, dan kriteria lab"},
            {"Dimensi": "Kecukupan sampel", "Bobot": 10, "Dasar": "Jumlah transaksi dan kelengkapan bukti"},
        ]
    )
    return {
        "master": master.reset_index(drop=True),
        "group_rankings": group_rankings,
        "winners": winners,
        "methodology": methodology,
        "experiment_count": len(master),
        "group_count": master["Kelompok"].nunique(),
    }
