#!/usr/bin/env python3
"""Thin v3 batch runner — same pipeline as v2, v3-friendly defaults.

Defaults (superset of v2 group outputs; viz ON unless ``--skip-viz``):
  --data-dir data/test_data_v3
  --out-dir  data/outputs/v3
  --mode     full          # offline: full ball track + render viz/dashboard
  --expected-persons 4     # lineup A–D

Examples
--------
  # Full batch (group0 enroll + groups 1–5), with viz videos + dashboard
  PYTHONPATH=. python scripts/run_v3_testset.py --groups all

  # Re-render viz/dashboard from existing sessions (keeps motion/report JSON)
  PYTHONPATH=. python scripts/run_v3_testset.py --groups 1,2,3,4,5 --rerender-viz

  # Fast classroom path without videos (explicit opt-out)
  PYTHONPATH=. python scripts/run_v3_testset.py --groups 1 --mode realtime --skip-viz
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_v2_testset import main as _v2_main  # noqa: E402


def main() -> None:
    # Inject v3 defaults before argparse sees argv (only if user did not pass them).
    argv = sys.argv[1:]
    flags = set()
    i = 0
    while i < len(argv):
        a = argv[i]
        if a.startswith("--"):
            flags.add(a.split("=", 1)[0])
        i += 1

    injected: list[str] = []
    if "--data-dir" not in flags:
        injected += ["--data-dir", str(ROOT / "data" / "test_data_v3")]
    if "--out-dir" not in flags:
        injected += ["--out-dir", str(ROOT / "data" / "outputs" / "v3")]
    if "--mode" not in flags:
        # full → skip_viz=False unless user passes --skip-viz
        injected += ["--mode", "full"]
    if "--expected-persons" not in flags:
        injected += ["--expected-persons", "4"]
    if "--groups" not in flags:
        injected += ["--groups", "all"]

    sys.argv = [sys.argv[0], *injected, *argv]
    _v2_main()


if __name__ == "__main__":
    main()
