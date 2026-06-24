from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gold_forecast.monitoring import capture_estimate, update_actuals


def main() -> None:
    parser = argparse.ArgumentParser(description="Run XAU/USD monitoring jobs.")
    parser.add_argument("mode", choices=["capture-estimate", "capture-actual"])
    args = parser.parse_args()

    if args.mode == "capture-estimate":
        frame = capture_estimate()
        print(f"Saved estimate rows: {len(frame)}")
    else:
        frame = update_actuals()
        print(f"Updated monitoring rows: {len(frame)}")


if __name__ == "__main__":
    main()
