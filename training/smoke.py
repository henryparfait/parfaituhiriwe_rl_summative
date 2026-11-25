"""
Quick check that all four algorithms run end-to-end on this machine (about 1-3 minutes).
Writes only to logs/_smoke and models/_smoke, so the real results are never touched.

    uv run python -m training.smoke
"""

import os
import traceback

os.environ["KAGUGU_SMOKE"] = "1"   # must be set before training modules are imported

from training.configs import get_config  # noqa: E402

FRAMES = 3_000


def main():
    import torch
    import torchrl
    print(f"torch {torch.__version__} | torchrl {torchrl.__version__}\n")
    from training.dqn_training import train_dqn
    from training.pg_training import train_pg

    results = {}
    jobs = [("dqn", "dqn_01_baseline"), ("reinforce", "reinforce_01_baseline"),
            ("a2c", "a2c_01_baseline"), ("ppo", "ppo_01_baseline")]
    for algo, rid in jobs:
        print(f"===== {algo.upper()} smoke test ({FRAMES} frames) =====")
        cfg = get_config(algo, rid, FRAMES)
        cfg.update(eval_every=1_000, learning_starts=500)
        try:
            s = train_dqn(cfg) if algo == "dqn" else train_pg(algo, cfg)
            results[algo] = f"OK   (held-out return {s['final_return_mean']:.1f}, {s['train_minutes']} min)"
        except Exception:
            traceback.print_exc()
            results[algo] = "FAILED - copy the error above and send it"
        print()
    print("===== SMOKE TEST SUMMARY =====")
    for k, v in results.items():
        print(f"  {k:10s} {v}")


if __name__ == "__main__":
    main()
