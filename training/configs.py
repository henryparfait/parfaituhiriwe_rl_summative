"""
Hyperparameter sweep definitions: >= 10 meaningfully different runs per algorithm.
Every run uses the SAME environment and the SAME frame budget (TOTAL_FRAMES) so results
are directly comparable. Each run changes one (sometimes two) controls relative to the
algorithm's baseline, and `why` states the hypothesis being tested.
"""

TOTAL_FRAMES = 100_000
EVAL_EVERY = 5_000

DQN_BASE = dict(lr=5e-4, gamma=0.99, batch_size=64, buffer_size=50_000, eps_start=1.0,
                eps_end=0.05, eps_decay_frac=0.4, target_update=500, hidden=128, depth=2,
                double_dqn=False, frames_per_batch=200, utd_ratio=0.25, learning_starts=2_000,
                grad_clip=10.0, seed=0)

DQN_RUNS = [
    dict(run_id="dqn_01_baseline", why="reference configuration"),
    dict(run_id="dqn_02_lr_low", lr=1e-4, why="smaller steps: slower but steadier learning?"),
    dict(run_id="dqn_03_lr_high", lr=2e-3, why="larger steps: faster but less stable?"),
    dict(run_id="dqn_04_gamma_090", gamma=0.90, why="myopic agent: ignores delayed treatment reward"),
    dict(run_id="dqn_05_small_buffer", buffer_size=5_000, why="small replay: more correlated samples"),
    dict(run_id="dqn_06_batch_256", batch_size=256, why="larger batch: lower gradient variance"),
    dict(run_id="dqn_07_fast_eps", eps_decay_frac=0.1, eps_end=0.01, why="exploit early: risk of premature convergence"),
    dict(run_id="dqn_08_slow_eps", eps_decay_frac=0.8, eps_end=0.10, why="explore long: better coverage, slower return growth"),
    dict(run_id="dqn_09_target_fast", target_update=50, why="frequent target sync: moving-target instability"),
    dict(run_id="dqn_10_target_slow", target_update=2_000, why="slow target sync: stable but stale targets"),
    dict(run_id="dqn_11_double_wide", double_dqn=True, hidden=256, why="Double DQN + wider net: less Q over-estimation"),
    dict(run_id="dqn_12_high_utd", utd_ratio=1.0, why="more gradient updates per collected frame"),
]

REINFORCE_BASE = dict(lr=1e-3, gamma=0.99, episodes_per_batch=8, baseline=True,
                      normalize_adv=True, entropy_coeff=0.0, value_coeff=0.5, hidden=128, depth=2,
                      grad_clip=1.0, seed=0)

REINFORCE_RUNS = [
    dict(run_id="reinforce_01_baseline", why="REINFORCE with learned value baseline"),
    dict(run_id="reinforce_02_no_baseline", baseline=False, why="vanilla REINFORCE (normalised returns)"),
    dict(run_id="reinforce_03_pure", baseline=False, normalize_adv=False, why="textbook REINFORCE: raw returns, high variance"),
    dict(run_id="reinforce_04_lr_low", lr=3e-4, why="smaller policy steps"),
    dict(run_id="reinforce_05_lr_high", lr=3e-3, why="larger policy steps: collapse risk"),
    dict(run_id="reinforce_06_gamma_095", gamma=0.95, why="shorter credit-assignment horizon"),
    dict(run_id="reinforce_07_batch_2", episodes_per_batch=2, why="fewer episodes per update: noisier gradient"),
    dict(run_id="reinforce_08_batch_16", episodes_per_batch=16, why="more episodes per update: fewer but cleaner updates"),
    dict(run_id="reinforce_09_entropy", entropy_coeff=0.01, why="entropy bonus keeps exploring"),
    dict(run_id="reinforce_10_small_net", hidden=64, depth=1, why="lower capacity policy"),
    dict(run_id="reinforce_11_gamma_0999", gamma=0.999, why="near-undiscounted returns"),
]

A2C_BASE = dict(lr=7e-4, gamma=0.99, frames_per_batch=120, gae_lambda=0.95, entropy_coeff=0.01,
                critic_coeff=0.5, hidden=128, depth=2, grad_clip=0.5, seed=0)

A2C_RUNS = [
    dict(run_id="a2c_01_baseline", why="reference configuration"),
    dict(run_id="a2c_02_lr_low", lr=2.5e-4, why="smaller steps"),
    dict(run_id="a2c_03_lr_high", lr=2e-3, why="larger steps: instability?"),
    dict(run_id="a2c_04_rollout_32", frames_per_batch=32, why="short rollouts: frequent, noisy updates"),
    dict(run_id="a2c_05_rollout_480", frames_per_batch=480, why="long rollouts: fewer, smoother updates"),
    dict(run_id="a2c_06_lambda_1", gae_lambda=1.0, why="Monte-Carlo-like advantage: low bias, high variance"),
    dict(run_id="a2c_07_lambda_0", gae_lambda=0.0, why="TD(0) advantage: low variance, high bias"),
    dict(run_id="a2c_08_no_entropy", entropy_coeff=0.0, why="no exploration bonus: early collapse?"),
    dict(run_id="a2c_09_entropy_high", entropy_coeff=0.05, why="strong exploration bonus"),
    dict(run_id="a2c_10_critic_gamma", critic_coeff=1.0, gamma=0.95, why="stronger critic, shorter horizon"),
    dict(run_id="a2c_11_wide", hidden=256, why="wider networks"),
]

PPO_BASE = dict(lr=3e-4, gamma=0.99, frames_per_batch=1024, epochs=4, minibatch=128, clip=0.2,
                gae_lambda=0.95, entropy_coeff=0.01, critic_coeff=0.5, normalize_adv=True,
                hidden=128, depth=2, grad_clip=0.5, seed=0)

PPO_RUNS = [
    dict(run_id="ppo_01_baseline", why="reference configuration"),
    dict(run_id="ppo_02_lr_low", lr=1e-4, why="smaller steps"),
    dict(run_id="ppo_03_lr_high", lr=1e-3, why="larger steps"),
    dict(run_id="ppo_04_clip_01", clip=0.1, why="tighter trust region: conservative updates"),
    dict(run_id="ppo_05_clip_03", clip=0.3, why="looser trust region"),
    dict(run_id="ppo_06_epochs_10", epochs=10, why="more reuse of each batch: overfitting/instability?"),
    dict(run_id="ppo_07_small_batch", frames_per_batch=256, minibatch=64, why="short rollouts"),
    dict(run_id="ppo_08_large_batch", frames_per_batch=4096, minibatch=256, why="long rollouts: fewer updates"),
    dict(run_id="ppo_09_no_entropy", entropy_coeff=0.0, why="no exploration bonus"),
    dict(run_id="ppo_10_entropy_high", entropy_coeff=0.05, why="strong exploration bonus"),
    dict(run_id="ppo_11_lambda_gamma", gae_lambda=0.8, gamma=0.95, why="shorter horizon, more bias"),
]

SWEEPS = {
    "dqn": (DQN_BASE, DQN_RUNS),
    "reinforce": (REINFORCE_BASE, REINFORCE_RUNS),
    "a2c": (A2C_BASE, A2C_RUNS),
    "ppo": (PPO_BASE, PPO_RUNS),
}


def get_config(algo: str, run_id: str, total_frames: int = TOTAL_FRAMES) -> dict:
    base, runs = SWEEPS[algo]
    run = next(r for r in runs if r["run_id"] == run_id)
    return {**base, **run, "algo": algo, "total_frames": total_frames, "eval_every": EVAL_EVERY}
