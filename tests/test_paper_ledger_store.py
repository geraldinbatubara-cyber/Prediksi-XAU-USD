from pathlib import Path

import pandas as pd

from gold_forecast.paper_ledger_store import (
    load_recovery_manual_exits,
    load_recovery_positions,
    merge_ledger_frames,
    strategy_id_for_path,
)
from gold_forecast.live_trading import _close_hit_positions_quote


def test_strategy_paths_are_isolated():
    assert strategy_id_for_path(Path("data/live_trading_optimizer.csv")) == "baseline_v1"
    assert (
        strategy_id_for_path(Path("data/live_trading_fixed_delay_5m.csv"))
        == "fixed_delay_5m"
    )
    assert (
        strategy_id_for_path(Path("data/live_trading_buy_specialist_v4.csv"))
        == "buy_specialist_v4"
    )


def test_merge_keeps_newest_state_without_duplicate_position():
    persistent = pd.DataFrame(
        [
            {
                "position_id": 5,
                "status": "OPEN",
                "last_update_wit": "2026-07-24 16:00:00 WIT",
            }
        ]
    )
    local = pd.DataFrame(
        [
            {
                "position_id": 5,
                "status": "CLOSED",
                "last_update_wit": "2026-07-24 17:00:00 WIT",
            }
        ]
    )
    recovery = pd.DataFrame(
        [
            {
                "position_id": 5,
                "status": "OPEN",
                "last_update_wit": "2026-07-24 15:59:00 WIT",
            },
            {
                "position_id": 6,
                "status": "OPEN",
                "last_update_wit": "2026-07-24 15:59:30 WIT",
            },
        ]
    )
    merged = merge_ledger_frames(
        persistent,
        local,
        recovery,
        "position_id",
    )
    assert merged["position_id"].tolist() == [5, 6]
    assert merged.loc[merged["position_id"].eq(5), "status"].iloc[0] == "CLOSED"


def test_recovery_preserves_optimizer_and_manual_observations_separately():
    positions = load_recovery_positions("baseline_v1")
    manual = load_recovery_manual_exits("baseline_v1")

    position_5 = positions.loc[positions["position_id"].eq(5)].iloc[0]
    position_6 = positions.loc[positions["position_id"].eq(6)].iloc[0]
    manual_5 = manual.loc[manual["position_id"].eq(5)].iloc[0]

    assert position_5["status"] == "OPEN"
    assert position_5["arah"] == "SELL"
    assert position_5["entry_price"] == 4029.0
    assert position_6["status"] == "OPEN"
    assert position_6["entry_price"] == 4032.0
    assert manual_5["manual_exit_price"] == 4020.0
    assert manual_5["manual_net_pl"] == 9.0


def test_recovered_position_uses_contract_barrier_not_late_quote():
    positions = load_recovery_positions("baseline_v1")
    position_5 = positions.loc[positions["position_id"].eq(5)].copy()
    position_5["exit_time_wit"] = ""
    position_5["exit_reason"] = ""
    closed = _close_hit_positions_quote(
        position_5,
        bid=4119.0,
        ask=4120.0,
        now=pd.Timestamp("2026-07-24 17:00:00", tz="Asia/Jayapura"),
    )
    assert closed.iloc[0]["status"] == "CLOSED"
    assert closed.iloc[0]["exit_price"] == 4039.0
    assert closed.iloc[0]["gross_pl"] == -10.0
    assert "pemulihan ledger" in closed.iloc[0]["exit_reason"]
