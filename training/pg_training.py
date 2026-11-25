"""
Policy-gradient training with TorchRL: REINFORCE, A2C (actor-critic) and PPO.

    uv run python -m training.pg_training --algo ppo --run ppo_01_baseline
    uv run python -m training.pg_training --algo reinforce --run reinforce_01_baseline --frames 5000   # smoke test
    uv run python -m training.pg_training --algo a2c --all                                             # full A2C sweep

Shared pieces (all TorchRL): ProbabilisticActor (OneHotCategorical policy) + ValueOperator critic.
- REINFORCE: complete episodes via env.rollout(), Monte-Carlo returns-to-go, ReinforceLoss
             (critic = learned baseline, can be switched off).
- A2C:       short on-policy rollouts from Collector, GAE advantages, A2CLoss, one update per rollout.
- PPO:       longer rollouts, GAE, ClipPPOLoss, several epochs of minibatch updates per rollout.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from tensordict.nn import TensorDictModule
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import OneHotCategorical, ProbabilisticActor, ValueOperator
from torchrl.objectives import A2CLoss, ClipPPOLoss, ReinforceLoss
from torchrl.objectives.value import GAE

from environment.custom_env import N_ACTIONS, OBS_DIM
from training.common import (
    EVAL_SEEDS, MODEL_DIR, RunLogger, arch_dict, build_mlp, episode_returns_from_batch, evaluate,
    final_evaluation, get_collector_cls, greedy_act_fn, make_torchrl_env, save_policy,
    update_best_models,
)
from training.configs import SWEEPS, TOTAL_FRAMES, get_config


# --------------------------------------------------------------------------- model
def build_actor_critic(env, cfg):
    actor_net = build_mlp(OBS_DIM, N_ACTIONS, cfg["hidden"], cfg["depth"], "tanh")
    actor = ProbabilisticActor(
        TensorDictModule(actor_net, in_keys=["observation"], out_keys=["logits"]),
        in_keys=["logits"], spec=env.action_spec, distribution_class=OneHotCategorical,
        return_log_prob=True,
    )
    critic_net = build_mlp(OBS_DIM, 1, cfg["hidden"], cfg["depth"], "tanh")
    critic = ValueOperator(critic_net, in_keys=["observation"])
    # initialise lazy shapes with one forward pass
    td = env.reset()
    with torch.no_grad():
        actor(td)
        critic(td)
    return actor_net, actor, critic, td


def log_prob_key_of(td) -> str | None:
    """Name of the log-prob entry written by the actor (differs across TorchRL versions)."""
    for k in td.keys():
        if isinstance(k, str) and k.endswith("log_prob"):
            return k
    return None


def align_log_prob_key(loss_fn, td):
    key = log_prob_key_of(td)
    if key is None:
        return
    for name in ("sample_log_prob", "log_prob"):
        try:
            loss_fn.set_keys(**{name: key})
            return
        except (KeyError, ValueError, TypeError, AttributeError):
            continue


def policy_entropy(actor, td) -> torch.Tensor:
    dist = actor.get_dist(td.select("observation"))
    return dist.entropy().mean()


def discounted_returns(rewards: torch.Tensor, gamma: float) -> torch.Tensor:
    out = torch.zeros_like(rewards)
    g = 0.0
    for t in range(rewards.shape[0] - 1, -1, -1):
        g = rewards[t] + gamma * g
        out[t] = g
    return out


# --------------------------------------------------------------------------- training loops
def train_pg(algo: str, cfg: dict, verbose: bool = True) -> dict:
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.set_num_threads(1)

    env = make_torchrl_env(seed=cfg["seed"])
    actor_net, actor, critic, probe = build_actor_critic(env, cfg)
    logger = RunLogger(algo, cfg["run_id"], cfg)
    state = {"frames": 0, "episodes": 0, "next_eval": 0, "recent": []}

    def log_step(extra: dict, rets: list[float], n_frames: int):
        state["frames"] += n_frames
        state["episodes"] += len(rets)
        state["recent"] = (state["recent"] + rets)[-20:]
        row = dict(frames=state["frames"], episodes=state["episodes"],
                   train_return=np.mean(state["recent"]) if state["recent"] else None, **extra)
        if state["frames"] >= state["next_eval"]:
            ev = evaluate(greedy_act_fn(actor_net), EVAL_SEEDS)
            row.update(eval_return=ev["return_mean"], eval_treat_rate=ev["precancer_treatment_rate"])
            state["next_eval"] += cfg["eval_every"]
            if verbose:
                tr = row["train_return"] if row["train_return"] is not None else float("nan")
                print(f"[{cfg['run_id']}] frames {state['frames']:>7} | train R {tr:7.2f} | "
                      f"eval R {ev['return_mean']:7.2f} | entropy {extra.get('entropy', 0):.3f} | "
                      f"treat-rate {ev['precancer_treatment_rate']:.2f}", flush=True)
        logger.log(**row)

    if algo == "reinforce":
        _train_reinforce(env, actor, critic, cfg, log_step, state)
    else:
        _train_actor_critic(algo, env, actor, critic, cfg, log_step, probe)

    results = final_evaluation(actor_net)
    save_policy(actor_net, MODEL_DIR / "pg" / f"{cfg['run_id']}.pt", algo,
                arch_dict(cfg["hidden"], cfg["depth"], "tanh"), cfg, results)
    summary = logger.finish(results)
    update_best_models()
    if verbose:
        print(f"[{cfg['run_id']}] DONE  held-out return {results['final_return_mean']:.2f} "
              f"± {results['final_return_std']:.2f} | stress {results['stress_return_mean']:.2f} | "
              f"treat-rate {results['final_precancer_treatment_rate']:.2f} | {summary['train_minutes']} min")
    return summary


def _train_reinforce(env, actor, critic, cfg, log_step, state):
    loss_fn = ReinforceLoss(actor_network=actor, critic_network=critic, loss_critic_type="smooth_l1")
    optim = torch.optim.Adam(loss_fn.parameters(), lr=cfg["lr"])
    while state["frames"] < cfg["total_frames"]:
        episodes, rets = [], []
        with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
            for _ in range(cfg["episodes_per_batch"]):
                ep = env.rollout(max_steps=1_000, policy=actor, break_when_any_done=True)
                r = ep.get(("next", "reward")).reshape(-1)
                ep.set("value_target", discounted_returns(r, cfg["gamma"]).unsqueeze(-1))
                episodes.append(ep)
                rets.append(float(r.sum()))
        data = torch.cat(episodes, 0)
        returns = data.get("value_target")
        with torch.no_grad():
            adv = returns - critic(data.select("observation").clone()).get("state_value") \
                if cfg["baseline"] else returns.clone()
            if cfg["normalize_adv"]:
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        data.set("advantage", adv)

        out = loss_fn(data)
        ent = policy_entropy(actor, data)
        loss = out["loss_actor"] - cfg["entropy_coeff"] * ent
        if cfg["baseline"]:
            loss = loss + cfg["value_coeff"] * out["loss_value"]
        optim.zero_grad()
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(loss_fn.parameters(), cfg["grad_clip"]))
        optim.step()
        log_step(dict(loss=loss.item(), loss_actor=out["loss_actor"].item(),
                      loss_critic=out["loss_value"].item() if cfg["baseline"] else None,
                      entropy=ent.item(), grad_norm=gn), rets, data.numel())


def _train_actor_critic(algo, env, actor, critic, cfg, log_step, probe):
    gae = GAE(gamma=cfg["gamma"], lmbda=cfg["gae_lambda"], value_network=critic, average_gae=False)
    common = dict(actor_network=actor, critic_network=critic, entropy_bonus=cfg["entropy_coeff"] > 0,
                  entropy_coeff=cfg["entropy_coeff"], critic_coeff=cfg["critic_coeff"],
                  loss_critic_type="smooth_l1")
    if algo == "ppo":
        loss_fn = ClipPPOLoss(clip_epsilon=cfg["clip"], normalize_advantage=cfg["normalize_adv"], **common)
    else:
        loss_fn = A2CLoss(**common)
    align_log_prob_key(loss_fn, probe)
    optim = torch.optim.Adam(loss_fn.parameters(), lr=cfg["lr"])

    Collector = get_collector_cls()
    collector = Collector(env, actor, frames_per_batch=cfg["frames_per_batch"],
                          total_frames=cfg["total_frames"])
    for data in collector:
        rets = episode_returns_from_batch(data)
        epochs = cfg.get("epochs", 1)
        losses, lobj, lcrit, lent, ents, gns = [], [], [], [], [], []
        for _ in range(epochs):
            with torch.no_grad():
                gae(data)
            flat = data.reshape(-1)
            mb = cfg.get("minibatch", flat.numel())
            for idx in torch.randperm(flat.numel()).split(mb):
                sub = flat[idx]
                out = loss_fn(sub)
                loss = out["loss_objective"] + out.get("loss_critic", 0.0) + out.get("loss_entropy", 0.0)
                optim.zero_grad()
                loss.backward()
                gns.append(float(torch.nn.utils.clip_grad_norm_(loss_fn.parameters(), cfg["grad_clip"])))
                optim.step()
                losses.append(loss.item())
                lobj.append(out["loss_objective"].item())
                if "loss_critic" in out.keys():
                    lcrit.append(out["loss_critic"].item())
                if "loss_entropy" in out.keys():
                    lent.append(out["loss_entropy"].item())
        with torch.no_grad():
            ents.append(policy_entropy(actor, data.reshape(-1)).item())
        log_step(dict(loss=np.mean(losses), loss_actor=np.mean(lobj),
                      loss_critic=np.mean(lcrit) if lcrit else None,
                      loss_entropy=np.mean(lent) if lent else None,
                      entropy=np.mean(ents), grad_norm=np.mean(gns)), rets, data.numel())
    collector.shutdown()


def main():
    p = argparse.ArgumentParser(description="Train REINFORCE / A2C / PPO on the Kagugu environment.")
    p.add_argument("--algo", required=True, choices=["reinforce", "a2c", "ppo"])
    p.add_argument("--run", default=None, help="run id from training/configs.py (default: baseline)")
    p.add_argument("--all", action="store_true", help="run the whole sweep for this algorithm")
    p.add_argument("--frames", type=int, default=TOTAL_FRAMES, help="frame budget per run")
    p.add_argument("--skip-existing", action="store_true")
    args = p.parse_args()
    runs = [r["run_id"] for r in SWEEPS[args.algo][1]]
    run_ids = runs if args.all else [args.run or runs[0]]
    for rid in run_ids:
        if args.skip_existing and (MODEL_DIR / "pg" / f"{rid}.pt").exists():
            print(f"skip {rid} (already trained)")
            continue
        train_pg(args.algo, get_config(args.algo, rid, args.frames))


if __name__ == "__main__":
    main()
