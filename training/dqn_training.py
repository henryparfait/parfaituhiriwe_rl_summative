"""
Deep Q-Network (value-based) training with TorchRL.

    uv run python -m training.dqn_training --run dqn_01_baseline
    uv run python -m training.dqn_training --run dqn_01_baseline --frames 5000     # quick smoke test
    uv run python -m training.dqn_training --all                                    # the full DQN sweep

Components (all TorchRL): GymWrapper env -> QValueActor + EGreedyModule policy -> Collector ->
TensorDictReplayBuffer -> DQNLoss (target network, optional Double DQN) -> HardUpdate target sync.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from tensordict.nn import TensorDictSequential
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from torchrl.modules import EGreedyModule, QValueActor
from torchrl.objectives import DQNLoss, HardUpdate

from environment.custom_env import N_ACTIONS, OBS_DIM
from training.common import (
    EVAL_SEEDS, MODEL_DIR, RunLogger, arch_dict, build_mlp, episode_returns_from_batch, evaluate,
    final_evaluation, get_collector_cls, greedy_act_fn, make_torchrl_env, save_policy,
    update_best_models,
)
from training.configs import DQN_RUNS, TOTAL_FRAMES, get_config

# only what the DQN loss needs -> consistent replay storage, smaller memory footprint
RB_KEYS = ("observation", "action", ("next", "observation"), ("next", "reward"),
           ("next", "done"), ("next", "terminated"))


def train_dqn(cfg: dict, verbose: bool = True) -> dict:
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.set_num_threads(1)          # lets several sweeps run in parallel on one laptop

    env = make_torchrl_env(seed=cfg["seed"])
    net = build_mlp(OBS_DIM, N_ACTIONS, cfg["hidden"], cfg["depth"], "relu")
    qactor = QValueActor(net, in_keys=["observation"], spec=env.action_spec)
    egreedy = EGreedyModule(env.action_spec, eps_init=cfg["eps_start"], eps_end=cfg["eps_end"],
                            annealing_num_steps=int(cfg["eps_decay_frac"] * cfg["total_frames"]))
    policy = TensorDictSequential(qactor, egreedy)

    Collector = get_collector_cls()
    # epsilon starts at 1.0, so the first `learning_starts` frames are already uniformly random
    collector = Collector(env, policy, frames_per_batch=cfg["frames_per_batch"],
                          total_frames=cfg["total_frames"])
    rb = TensorDictReplayBuffer(storage=LazyTensorStorage(cfg["buffer_size"]), batch_size=cfg["batch_size"])

    loss_fn = DQNLoss(qactor, action_space=env.action_spec, delay_value=True,
                      double_dqn=cfg["double_dqn"], loss_function="l2")
    loss_fn.make_value_estimator(gamma=cfg["gamma"])
    target_updater = HardUpdate(loss_fn, value_network_update_interval=cfg["target_update"])
    optim = torch.optim.Adam(loss_fn.parameters(), lr=cfg["lr"])

    logger = RunLogger("dqn", cfg["run_id"], cfg)
    frames, episodes, next_eval = 0, 0, 0
    recent: list[float] = []
    n_updates = max(1, int(cfg["utd_ratio"] * cfg["frames_per_batch"]))

    for data in collector:
        n = data.numel()
        frames += n
        rb.extend(data.reshape(-1).select(*RB_KEYS))
        if frames > cfg["learning_starts"]:
            egreedy.step(n)
        rets = episode_returns_from_batch(data)
        episodes += len(rets)
        recent = (recent + rets)[-20:]

        losses, qs, gnorms = [], [], []
        if len(rb) >= cfg["learning_starts"]:
            for _ in range(n_updates):
                sample = rb.sample()
                loss = loss_fn(sample)["loss"]
                optim.zero_grad()
                loss.backward()
                gnorms.append(float(torch.nn.utils.clip_grad_norm_(loss_fn.parameters(), cfg["grad_clip"])))
                optim.step()
                target_updater.step()
                losses.append(loss.item())
            with torch.no_grad():
                qs.append(net(sample["observation"]).max(-1).values.mean().item())

        row = dict(frames=frames, episodes=episodes,
                   train_return=np.mean(recent) if recent else None,
                   loss=np.mean(losses) if losses else None, q_mean=np.mean(qs) if qs else None,
                   grad_norm=np.mean(gnorms) if gnorms else None, epsilon=float(egreedy.eps))
        if frames >= next_eval:
            ev = evaluate(greedy_act_fn(net), EVAL_SEEDS)
            row.update(eval_return=ev["return_mean"], eval_treat_rate=ev["precancer_treatment_rate"])
            next_eval += cfg["eval_every"]
            if verbose:
                print(f"[{cfg['run_id']}] frames {frames:>7} | eps {float(egreedy.eps):.3f} | "
                      f"train R {row['train_return'] if row['train_return'] is not None else float('nan'):7.2f} | "
                      f"eval R {ev['return_mean']:7.2f} | loss {row['loss'] or 0:.4f} | "
                      f"treat-rate {ev['precancer_treatment_rate']:.2f}", flush=True)
        logger.log(**row)

    collector.shutdown()
    results = final_evaluation(net)
    save_policy(net, MODEL_DIR / "dqn" / f"{cfg['run_id']}.pt", "dqn",
                arch_dict(cfg["hidden"], cfg["depth"], "relu"), cfg, results)
    summary = logger.finish(results)
    update_best_models()
    if verbose:
        print(f"[{cfg['run_id']}] DONE  held-out return {results['final_return_mean']:.2f} "
              f"± {results['final_return_std']:.2f} | stress {results['stress_return_mean']:.2f} | "
              f"treat-rate {results['final_precancer_treatment_rate']:.2f} | {summary['train_minutes']} min")
    return summary


def main():
    p = argparse.ArgumentParser(description="Train DQN on the Kagugu follow-up environment.")
    p.add_argument("--run", default="dqn_01_baseline", help="run id from training/configs.py")
    p.add_argument("--all", action="store_true", help="run the whole DQN sweep")
    p.add_argument("--frames", type=int, default=TOTAL_FRAMES, help="frame budget per run")
    p.add_argument("--skip-existing", action="store_true")
    args = p.parse_args()
    run_ids = [r["run_id"] for r in DQN_RUNS] if args.all else [args.run]
    for rid in run_ids:
        if args.skip_existing and (MODEL_DIR / "dqn" / f"{rid}.pt").exists():
            print(f"skip {rid} (already trained)")
            continue
        train_dqn(get_config("dqn", rid, args.frames))


if __name__ == "__main__":
    main()
