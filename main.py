"""
Kagugu Follow-Up Coordinator - run the best trained agent with the live 3D viewer.

    uv run main.py                       # best agent overall, 3 episodes, opens the browser
    uv run main.py --agent ppo           # a specific algorithm's best model
    uv run main.py --agent heuristic     # rule-based baseline (no model needed)
    uv run main.py --stress              # unseen stress scenario (generalisation)
    uv run main.py --delay 0.2 --episodes 1

Training:  uv run python -m training.dqn_training --help   /   uv run python -m training.pg_training --help
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from environment.custom_env import (
    ACTIONS_PER_DAY, CHW_WEEKLY_BUDGET, HORIZON_DAYS, N_ACTIONS, STRESS_PARAMS,
    KaguguFollowUpEnv, describe_action, heuristic_policy,
)
from environment.rendering import LiveViewer

ROOT = Path(__file__).resolve().parent
BEST_FILE = ROOT / "models" / "best_model.json"
MODEL_PATHS = {
    "dqn": ROOT / "models" / "dqn" / "dqn_best.pt",
    "reinforce": ROOT / "models" / "pg" / "reinforce_best.pt",
    "a2c": ROOT / "models" / "pg" / "a2c_best.pt",
    "ppo": ROOT / "models" / "pg" / "ppo_best.pt",
}

os.system("")  # enables ANSI colours in the Windows terminal
C = {"b": "\033[1m", "g": "\033[92m", "r": "\033[91m", "y": "\033[93m", "c": "\033[96m",
     "d": "\033[2m", "x": "\033[0m"}


def banner(agent_name: str, scenario: str):
    print(f"""{C['b']}{C['c']}
==========================================================================
  KAGUGU FOLLOW-UP COORDINATOR  -  Reinforcement Learning agent demo
=========================================================================={C['x']}
  Problem   : HPV-positive women at Kagugu Health Centre (Rwanda) are missed
              at triage or lost before treatment of cervical precancer.
  Agent     : the follow-up coordinator. {ACTIONS_PER_DAY} decisions/day for {HORIZON_DAYS} days,
              caseload of 8 women, {CHW_WEEKLY_BUDGET} CHW home visits per week.
  Actions   : wait | SMS reminder | CHW home visit | invite to AI triage |
              invite for thermal ablation   ({N_ACTIONS} discrete actions)
  Rewards   : +10 precancer treated, +1 triage done, -8 precancer lost,
              -6 missed lesion, -3 other woman lost, -0.5 invalid action,
              small costs per SMS/visit/invitation.
  Objective : get as many true precancers treated as possible, as early as
              possible, without losing women to follow-up.
  Agent     : {C['b']}{agent_name}{C['x']}     Scenario: {scenario}
--------------------------------------------------------------------------""")


def load_agent(name: str):
    """Returns (act_fn, scores_fn or None, display_name)."""
    if name == "heuristic":
        return heuristic_policy, None, "Rule-based heuristic baseline"
    if name == "random":
        rng = np.random.default_rng(0)
        return (lambda o: int(rng.integers(N_ACTIONS))), None, "Random policy"

    if name == "best":
        if not BEST_FILE.exists():
            print(f"{C['y']}No trained model found yet (models/best_model.json missing).\n"
                  f"Run the training first - see README. Falling back to the heuristic baseline.{C['x']}")
            return heuristic_policy, None, "Rule-based heuristic baseline (no trained model yet)"
        best = json.loads(BEST_FILE.read_text())
        path, algo = ROOT / best["path"], best["algo"]
    else:
        path, algo = MODEL_PATHS[name], name
    if not path.exists():
        sys.exit(f"Model file not found: {path}. Train it first (see README).")

    from training.common import load_policy  # imported here so the baseline works without torch
    act, scores, ckpt = load_policy(path)
    res = ckpt.get("results", {})
    label = (f"{algo.upper()} ({ckpt['config'].get('run_id', path.stem)}; held-out eval return "
             f"{res.get('final_return_mean', float('nan')):.1f})")
    return act, scores, label


def fmt_scores(scores_fn, obs, chosen: int, algo_is_q: bool) -> str:
    if scores_fn is None:
        return ""
    s = scores_fn(obs)
    top = np.argsort(s)[::-1][:3]
    short = ["SMS", "CHW", "Triage", "Treat"]

    def name(a: int) -> str:
        return "Wait" if a == 0 else f"{short[(a - 1) % 4]}-W{(a - 1) // 4 + 1}"

    kind = "Q" if algo_is_q else "p"
    alts = ", ".join(f"{name(int(a))}={s[a]:.2f}" for a in top)
    return f"{C['d']}   [top {kind}: {alts}]{C['x']}"


def run(args):
    act, scores, label = load_agent(args.agent)
    is_q = "DQN" in label
    scenario = "STRESS (unseen: fewer phones, faster disengagement)" if args.stress else "standard"
    banner(label, scenario)

    viewer = None
    if not args.no_viewer:
        viewer = LiveViewer(port=args.port, open_browser=not args.no_browser)
        print(f"  Live 3D viewer : {C['b']}{viewer.url}{C['x']}   (API: {viewer.url}api/state)")
        print("  Waiting a few seconds for the browser to load...\n")
        time.sleep(args.wait_start)

    env = KaguguFollowUpEnv(render_mode="human" if viewer else None, viewer=viewer,
                            params=STRESS_PARAMS if args.stress else None)
    if viewer:
        viewer.set_recommender(lambda: {"action": act(env._obs()), "text": describe_action(act(env._obs()))})

    all_returns, all_info = [], []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        print(f"{C['b']}{C['c']}>>> Episode {ep + 1}/{args.episodes}  (seed {args.seed + ep}){C['x']}")
        done, ret = False, 0.0
        while not done:
            a = act(obs)
            day, dec = env.day, ACTIONS_PER_DAY - env.decisions_left + 1
            extra = fmt_scores(scores, obs, a, is_q) if args.verbose else ""
            obs, r, term, trunc, info = env.step(a)
            ret += r
            done = term or trunc
            col = C["g"] if r >= 1 else (C["r"] if r <= -0.5 else "")
            print(f"[Day {day + 1:02d} {dec}/{ACTIONS_PER_DAY} | CHW left {info['chw_budget']}] "
                  f"{describe_action(a):48s} -> {col}{env.last_event}{C['x']}  "
                  f"r={r:+.2f} R={ret:7.2f}{extra}")
            time.sleep(args.delay)
        all_returns.append(ret)
        all_info.append(info)
        print(f"{C['b']}--- Episode {ep + 1} summary: return {ret:.1f} | precancers treated "
              f"{info['treated_precancer']} | precancers lost {info['lost_precancer']} | missed lesions "
              f"{info['missed_lesions']} | other lost {info['lost_other']} | invalid actions "
              f"{info['invalid_actions']} | precancer treatment rate "
              f"{100 * info['precancer_treatment_rate']:.0f}%{C['x']}\n")
        time.sleep(min(2.0, args.delay * 4))

    print(f"{C['b']}{C['c']}=== {args.episodes} episode(s): mean return {np.mean(all_returns):.1f} "
          f"± {np.std(all_returns):.1f}, mean precancer treatment rate "
          f"{100 * np.mean([i['precancer_treatment_rate'] for i in all_info]):.0f}% ==={C['x']}")
    if viewer:
        print(f"{C['d']}Viewer stays open for {args.hold:.0f} s...{C['x']}")
        time.sleep(args.hold)
        viewer.close()


def main():
    p = argparse.ArgumentParser(description="Run the Kagugu follow-up agent with the 3D viewer.")
    p.add_argument("--agent", default="best", choices=["best", "dqn", "reinforce", "a2c", "ppo", "heuristic", "random"])
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--delay", type=float, default=0.35, help="seconds between agent decisions")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--stress", action="store_true", help="unseen stress scenario (generalisation)")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-browser", action="store_true", help="do not open the browser automatically")
    p.add_argument("--no-viewer", action="store_true", help="terminal output only")
    p.add_argument("--wait-start", type=float, default=4.0)
    p.add_argument("--hold", type=float, default=10.0, help="seconds to keep the viewer open at the end")
    p.add_argument("--quiet", dest="verbose", action="store_false", help="hide the agent's top-3 action scores")
    run(p.parse_args())


if __name__ == "__main__":
    main()
