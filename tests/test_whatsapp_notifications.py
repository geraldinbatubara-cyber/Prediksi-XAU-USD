from gold_forecast.whatsapp_notifications import format_notification


def test_formats_entry_notification_with_targets():
    message = format_notification(
        {
            "notification_type": "ENTRY",
            "strategy_id": "baseline_v1",
            "position_id": 8,
            "payload": {
                "arah": "SELL",
                "lot": 0.01,
                "entry_price": 4315.5,
                "tp_usd": 25,
                "cl_usd": 10,
                "entry_time_wit": "2026-08-18 09:00:00 WIT",
                "catatan": "Seluruh syarat terpenuhi.",
            },
        }
    )
    assert "NEW ENTRY" in message
    assert "#8 SELL" in message
    assert "TP: USD 4,290.50" in message
    assert "CL/SL: USD 4,325.50" in message


def test_formats_exit_notification_with_net_result():
    message = format_notification(
        {
            "notification_type": "EXIT",
            "strategy_id": "fixed_delay_5m",
            "position_id": 3,
            "payload": {
                "arah": "BUY",
                "entry_price": 4300,
                "exit_price": 4325,
                "exit_reason": "TP tersentuh",
                "gross_pl": 25,
                "swap": -0.02,
                "net_pl": 24.98,
                "exit_time_wit": "2026-08-18 11:00:00 WIT",
            },
        }
    )
    assert "POSITION CLOSED" in message
    assert "TP tersentuh" in message
    assert "Net P/L: USD +24.98" in message
