from scripts.mt5_data_bridge import _initialize_mt5


class FakeMT5:
    def __init__(self, outcomes: list[bool]) -> None:
        self.outcomes = iter(outcomes)
        self.initialize_calls = []
        self.shutdown_calls = 0

    def initialize(self, **kwargs) -> bool:
        self.initialize_calls.append(kwargs)
        return next(self.outcomes)

    def last_error(self):
        return (-10005, "IPC timeout")

    def shutdown(self) -> None:
        self.shutdown_calls += 1


def test_initialize_mt5_retries_with_explicit_terminal_path() -> None:
    mt5 = FakeMT5([False, False, True])
    delays = []

    _initialize_mt5(
        mt5,
        r"C:\Program Files\MetaTrader 5\terminal64.exe",
        attempts=4,
        timeout_ms=30000,
        retry_delay=10,
        sleep_func=delays.append,
    )

    assert len(mt5.initialize_calls) == 3
    assert mt5.initialize_calls[-1] == {
        "path": r"C:\Program Files\MetaTrader 5\terminal64.exe",
        "timeout": 30000,
    }
    assert mt5.shutdown_calls == 2
    assert delays == [10, 10]
