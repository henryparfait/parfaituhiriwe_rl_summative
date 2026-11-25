"""
Run the hyperparameter sweeps (>= 10 runs per algorithm), then build figures and tables.

    uv run python -m training.sweep --algo all          # everything, one after another
    uv run python -m training.sweep --algo dqn          # one algorithm (open 4 terminals to run 4 in parallel)
    uv run python -m training.sweep --algo ppo --frames 50000

Already-trained runs are skipped, so you can stop (Ctrl+C) and resume at any time.
"""

import argparse
import time

from training.common import MODEL_DIR
from training.configs import SWEEPS, TOTAL_FRAMES, get_config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", default="all", choices=["all", "dqn", "reinforce", "a2c", "ppo"])
    p.add_argument("--frames", type=int, default=TOTAL_FRAMES)
    p.add_argument("--no-analyze", action="store_true")
    args = p.parse_args()

    from training.dqn_training import train_dqn
    from training.pg_training import train_pg

    algos = list(SWEEPS) if args.algo == "all" else [args.algo]
    t0 = time.time()
    for algo in algos:
        runs = SWEEPS[algo][1]
        for i, run in enumerate(runs, 1):
            rid = run["run_id"]
            sub = "dqn" if algo == "dqn" else "pg"
            if (MODEL_DIR / sub / f"{rid}.pt").exists():
                print(f"skip {rid} (already trained)")
                continue
            print(f"\n######## {algo.upper()} run {i}/{len(runs)}: {rid} - {run['why']} "
                  f"(elapsed {(time.time() - t0) / 60:.1f} min) ########")
            cfg = get_config(algo, rid, args.frames)
            train_dqn(cfg) if algo == "dqn" else train_pg(algo, cfg)
    print(f"\nSweep finished in {(time.time() - t0) / 60:.1f} min.")
    if not args.no_analyze:
        from training.analyze import main as analyze
        analyze()


if __name__ == "__main__":
    main()
