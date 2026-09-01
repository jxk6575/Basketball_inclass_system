#!/usr/bin/env python3
"""Run offline analysis pipeline for a session."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.privacy.db import init_db
from src.orchestrator.session_pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="Basketball classroom session pipeline")
    parser.add_argument("--session-id", required=True)
    parser.add_argument(
        "--from-stage",
        choices=["perception", "sync", "pose3d", "action", "shot", "scoring"],
        default="perception",
        help="Resume from this stage through scoring (skip earlier stages; upstream outputs must exist)",
    )
    parser.add_argument("--init-db", action="store_true", help="Initialize SQLite before run")
    args = parser.parse_args()

    if args.init_db:
        init_db()

    result = run_pipeline(args.session_id, from_stage=args.from_stage)
    print(result)


if __name__ == "__main__":
    main()
