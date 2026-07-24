from __future__ import annotations

import base64
import pickle
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gold_forecast.model_comparison import (  # noqa: E402
    MODEL_COMPARISON_VERSION,
    build_model_comparison,
)


def main() -> None:
    output = ROOT / "data" / "precomputed" / "model_comparison.pkl.b64"
    payload = build_model_comparison(ROOT / "data" / "precomputed")
    saved = {"version": MODEL_COMPARISON_VERSION, "payload": payload}
    output.write_text(
        base64.b64encode(
            pickle.dumps(saved, protocol=pickle.HIGHEST_PROTOCOL)
        ).decode("ascii"),
        encoding="ascii",
    )
    master = payload["master"]
    print(f"Saved {output}")
    print(
        master[
            [
                "Peringkat",
                "Kelompok",
                "Eksperimen",
                "Skor Total",
                "Growth OOS (%)",
                "PF OOS",
                "DD OOS (%)",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
