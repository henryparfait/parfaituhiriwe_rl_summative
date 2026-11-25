"""
Shared utilities for all four algorithms so the comparison is fair:
same environment wrapper, same network builder, same evaluation protocol,
same logging format.
"""

from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from environment.custom_env import (
    N_ACTIONS, OBS_DIM, STRESS_PARAMS, KaguguFollowUpEnv,
)

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"
MODEL_DIR = ROOT / "models"
os.environ.setdefault("RL_WARNINGS", "False")   # silence "total_frames not divisible" notices
if os.environ.get("KAGUGU_SMOKE"):          # smoke tests never touch the real results
    LOG_DIR = LOG_DIR / "_smoke"
    MODEL_DIR = MODEL_DIR / "_smoke"

EVAL_SEEDS = list(range(10_000, 10_010))          # periodic evaluation during training
FINAL_SEEDS = list(range(20_000, 20_030))         # final held-out evaluation (never trained on)
STRESS_SEEDS = list(range(30_000, 30_030))        # unseen stress scenario (generalisation)


# --------------------------------------------------------------------------- networks
def build_mlp(in_dim: int, out_dim: int, hidden: int = 128, depth: int = 2,
              activation: str = "relu") -> nn.Sequential:
    act = {"relu": nn.ReLU, "tanh": nn.Tanh}[activation]
    layers, d = [], in_dim
    for _ in range(depth):
        layers += [nn.Linear(d, hidden), act()]
        d = hidden
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


# --------------------------------------------------------------------------- TorchRL env
def make_torchrl_env(seed: int = 0, params: dict | None = None):
    """KaguguFollowUpEnv wrapped for TorchRL, with step counting and episode-return tracking."""
    from torchrl.envs import Compose, GymWrapper, InitTracker, RewardSum, StepCounter, TransformedEnv
    from torchrl.envs.libs.gym import set_gym_backend

    with set_gym_backend("gymnasium"):
        base = GymWrapper(KaguguFollowUpEnv(params=params), device="cpu")
    env = TransformedEnv(base, Compose(InitTracker(), StepCounter(), RewardSum()))
    env.set_seed(seed)
    return env


def get_collector_cls():
    """TorchRL 0.15 renamed SyncDataCollector -> Collector; support both."""
    try:
        from torchrl.collectors import Collector
        return Collector
    except ImportError:  # older TorchRL
        from torchrl.collectors import SyncDataCollector
        return SyncDataCollector


def episode_returns_from_batch(data) -> list[float]:
    """Completed-episode returns inside a collected batch (uses the RewardSum transform)."""
    done = data.get(("next", "done")).reshape(-1)
    if not done.any():
        return []
    ep_r = data.get(("next", "episode_reward")).reshape(-1)
    return ep_r[done].tolist()


# --------------------------------------------------------------------------- evaluation
def greedy_act_fn(net: nn.Module):
    """Deterministic policy from a Q-network or a policy-logits network."""
    @torch.no_grad()
    def act(obs: np.ndarray) -> int:
        x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        return int(net(x).argmax(-1).item())
    return act


def evaluate(act_fn, seeds, params: dict | None = None) -> dict:
    """Run full episodes on the raw Gymnasium env and return mean metrics."""
    env = KaguguFollowUpEnv(params=params)
    rets, infos = [], []
    for s in seeds:
        obs, _ = env.reset(seed=s)
        done, ret = False, 0.0
        while not done:
            obs, r, term, trunc, info = env.step(act_fn(obs))
            ret += r
            done = term or trunc
        rets.append(ret)
        infos.append(info)
    out = {"return_mean": float(np.mean(rets)), "return_std": float(np.std(rets))}
    for k in ("precancer_treatment_rate", "treated_precancer", "lost_precancer", "lost_other",
              "missed_lesions", "invalid_actions", "chw_visits", "sms_sent"):
        out[k] = float(np.mean([i[k] for i in infos]))
    return out


def final_evaluation(net: nn.Module) -> dict:
    act = greedy_act_fn(net)
    std = evaluate(act, FINAL_SEEDS)
    stress = evaluate(act, STRESS_SEEDS, params=STRESS_PARAMS)
    res = {f"final_{k}": v for k, v in std.items()}
    res.update({f"stress_{k}": v for k, v in stress.items()})
    return res


# --------------------------------------------------------------------------- logging
class RunLogger:
    """Writes logs/<algo>/<run_id>/progress.csv + config.json, and a summary row."""

    FIELDS = ["frames", "episodes", "wall_time", "train_return", "eval_return", "eval_treat_rate",
              "loss", "loss_actor", "loss_critic", "loss_entropy", "entropy", "epsilon", "q_mean",
              "grad_norm"]

    def __init__(self, algo: str, run_id: str, config: dict):
        self.algo, self.run_id, self.config = algo, run_id, config
        self.dir = LOG_DIR / algo / run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "config.json").write_text(json.dumps(config, indent=2))
        self._f = open(self.dir / "progress.csv", "w", newline="")
        self._w = csv.DictWriter(self._f, fieldnames=self.FIELDS)
        self._w.writeheader()
        self.t0 = time.time()
        self.best_eval = -np.inf
        self.history_eval: list[tuple[int, float]] = []

    def log(self, **row):
        row["wall_time"] = round(time.time() - self.t0, 2)
        clean = {k: (round(float(v), 5) if isinstance(v, (int, float, np.floating)) and v is not None else v)
                 for k, v in row.items() if k in self.FIELDS}
        self._w.writerow(clean)
        self._f.flush()
        if row.get("eval_return") is not None:
            self.history_eval.append((int(row["frames"]), float(row["eval_return"])))

    def frames_to_converge(self, frac: float = 0.9) -> int | None:
        """First frame count at which periodic eval reached `frac` of the final eval level
        (measured relative to the first evaluation, so negative returns are handled)."""
        if len(self.history_eval) < 2:
            return None
        start, end = self.history_eval[0][1], self.history_eval[-1][1]
        target = start + frac * (end - start)
        for f, v in self.history_eval:
            if (end >= start and v >= target) or (end < start and v <= target):
                return f
        return None

    def finish(self, results: dict):
        self._f.close()
        summary = {"algo": self.algo, "run_id": self.run_id, **self.config, **results,
                   "frames_to_90pct": self.frames_to_converge(),
                   "train_minutes": round((time.time() - self.t0) / 60, 2)}
        (self.dir / "summary.json").write_text(json.dumps(summary, indent=2))
        path = LOG_DIR / f"{self.algo}_summary.csv"
        rows = []
        if path.exists():
            with open(path, newline="") as f:
                rows = [r for r in csv.DictReader(f) if r["run_id"] != self.run_id]
        rows.append({k: v for k, v in summary.items()})
        fields = list(dict.fromkeys(k for r in rows for k in r.keys()))
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        return summary


# --------------------------------------------------------------------------- serialisation
def save_policy(net: nn.Module, path: Path, algo: str, arch: dict, config: dict, results: dict):
    """Plain-PyTorch checkpoint: loading it needs only torch (no TorchRL) -> easy to deploy."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"algo": algo, "arch": arch, "state_dict": net.state_dict(),
                "config": config, "results": results}, path)


def load_policy(path: Path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    a = ckpt["arch"]
    net = build_mlp(a["in_dim"], a["out_dim"], a["hidden"], a["depth"], a["activation"])
    net.load_state_dict(ckpt["state_dict"])
    net.eval()

    @torch.no_grad()
    def scores(obs: np.ndarray) -> np.ndarray:
        out = net(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))[0]
        if ckpt["algo"] != "dqn":          # policy logits -> probabilities
            out = torch.softmax(out, -1)
        return out.numpy()

    def act(obs: np.ndarray) -> int:
        return int(np.argmax(scores(obs)))

    return act, scores, ckpt


def arch_dict(hidden: int, depth: int, activation: str) -> dict:
    return {"in_dim": OBS_DIM, "out_dim": N_ACTIONS, "hidden": hidden, "depth": depth,
            "activation": activation}


# --------------------------------------------------------------------------- model selection
def update_best_models() -> dict | None:
    """Copy each algorithm's best run (highest held-out return) to <algo>_best.pt and
    record the overall winner in models/best_model.json (read by main.py)."""
    import shutil

    best_overall = None
    for algo in ("dqn", "reinforce", "a2c", "ppo"):
        path = LOG_DIR / f"{algo}_summary.csv"
        if not path.exists():
            continue
        with open(path, newline="") as f:
            rows = [r for r in csv.DictReader(f) if r.get("final_return_mean")]
        if not rows:
            continue
        best = max(rows, key=lambda r: float(r["final_return_mean"]))
        sub = "dqn" if algo == "dqn" else "pg"
        src = MODEL_DIR / sub / f"{best['run_id']}.pt"
        dst = MODEL_DIR / sub / f"{algo}_best.pt"
        if src.exists():
            shutil.copyfile(src, dst)
        cand = {"algo": algo, "run_id": best["run_id"], "path": str(dst.relative_to(ROOT)).replace("\\", "/"),
                "final_return_mean": float(best["final_return_mean"])}
        if best_overall is None or cand["final_return_mean"] > best_overall["final_return_mean"]:
            best_overall = cand
    if best_overall:
        (MODEL_DIR / "best_model.json").write_text(json.dumps(best_overall, indent=2))
    return best_overall
