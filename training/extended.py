"""
Phase 2 (optional): retrain the BEST configuration of each algorithm with a longer, equal budget,
to check whether performance was still improving at 100k frames.

    uv run python -m training.extended --algo ppo --frames 300000
    uv run python -m training.extended --algo all --frames 300000

Runs are saved as "<best_run_id>_long" and appear in the same logs/tables, so the comparison
between the 100k sweep and the longer runs is fully documented.
"""

import argparse
import csv

from training.common import LOG_DIR, MODEL_DIR
from training.configs import get_config


def best_run_id(algo: str) -> str:
    with open(LOG_DIR / f"{algo}_summary.csv", newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if not r["run_id"].endswith("_long")]
    return max(rows, key=lambda r: float(r["final_return_mean"]))["run_id"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", default="all", choices=["all", "dqn", "reinforce", "a2c", "ppo"])
    p.add_argument("--frames", type=int, default=300_000)
    args = p.parse_args()

    from training.dqn_training import train_dqn
    from training.pg_training import train_pg

    for algo in (["dqn", "reinforce", "a2c", "ppo"] if args.algo == "all" else [args.algo]):
        rid = best_run_id(algo)
        long_id = f"{rid}_long"
        sub = "dqn" if algo == "dqn" else "pg"
        if (MODEL_DIR / sub / f"{long_id}.pt").exists():
            print(f"skip {long_id} (already trained)")
            continue
        cfg = get_config(algo, rid, args.frames)
        cfg["run_id"] = long_id
        cfg["why"] = f"best 100k config ({rid}) retrained for {args.frames:,} frames"
        print(f"\n######## {algo.upper()} extended run: {long_id} ({args.frames:,} frames) ########")
        train_dqn(cfg) if algo == "dqn" else train_pg(algo, cfg)


if __name__ == "__main__":
    main()
