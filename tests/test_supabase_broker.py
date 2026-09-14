import gold_forecast.supabase_broker as supabase_broker
import pandas as pd


def test_broker_feed_paginates_past_postgrest_row_limit(monkeypatch):
    calls = []

    def request(*args, **kwargs):
        table = args[2]
        query = kwargs["query"]
        if table == "broker_latest_quote":
            return []
        calls.append(query)
        offset = int(query["offset"])
        limit = int(query["limit"])
        return [
            {
                "timestamp_utc": (
                    pd.Timestamp("2026-09-14T12:00:00Z")
                    - pd.Timedelta(minutes=offset + row)
                ).isoformat(),
                "open": 4400.0,
                "high": 4401.0,
                "low": 4399.0,
                "close": 4400.5,
                "tick_volume": 10,
                "spread_points": 10.0,
                "symbol": "XAUUSD",
                "source": "test",
            }
            for row in range(limit)
        ]

    monkeypatch.setattr(supabase_broker, "_request_json", request)
    bars, _ = supabase_broker.load_supabase_broker_feed(
        "https://example.supabase.co", "key", bars_limit=2500
    )

    assert len(calls) == 3
    assert [call["offset"] for call in calls] == ["0", "1000", "2000"]
    assert [call["limit"] for call in calls] == ["1000", "1000", "500"]
    assert len(bars) == 2500
