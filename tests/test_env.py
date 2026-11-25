"""Sanity tests for the Kagugu follow-up environment. Run with:  uv run pytest -q"""
import numpy as np

from environment.custom_env import (
    ACTIONS_PER_DAY, AWAIT_TREAT, AWAIT_TRIAGE, CHW_VISIT, HORIZON_DAYS, N_ACTIONS, OBS_DIM,
    R_INVALID, STRESS_PARAMS, TREAT, TRIAGE, KaguguFollowUpEnv, decode_action, encode_action,
    heuristic_policy,
)


def run_episode(env, policy, seed):
    obs, _ = env.reset(seed=seed)
    total, steps, done = 0.0, 0, False
    while not done:
        obs, r, term, trunc, info = env.step(policy(obs))
        assert obs.shape == (OBS_DIM,) and obs.dtype == np.float32
        assert obs.min() >= 0.0 and obs.max() <= 1.0
        total += r
        steps += 1
        done = term or trunc
    return total, steps, info


def test_action_encoding_roundtrip():
    for a in range(1, N_ACTIONS):
        s, iv = decode_action(a)
        assert encode_action(s, iv) == a
    assert decode_action(0) == (-1, -1)


def test_start_state():
    env = KaguguFollowUpEnv()
    obs, info = env.reset(seed=0)
    assert sum(w.active for w in env.women) == 5
    assert env.day == 0 and env.decisions_left == ACTIONS_PER_DAY
    assert info["referrals"] == 5


def test_seed_determinism():
    env = KaguguFollowUpEnv()
    rng = np.random.default_rng(1)
    actions = rng.integers(N_ACTIONS, size=400)
    out = []
    for _ in range(2):
        env.reset(seed=42)
        rs = []
        for a in actions:
            _, r, term, _, _ = env.step(int(a))
            rs.append(r)
            if term:
                break
        out.append(rs)
    assert out[0] == out[1]


def test_episode_terminates_within_horizon():
    env = KaguguFollowUpEnv()
    _, steps, info = run_episode(env, lambda o: 0, seed=3)
    assert steps <= HORIZON_DAYS * ACTIONS_PER_DAY


def test_invalid_actions_are_penalised():
    env = KaguguFollowUpEnv()
    env.reset(seed=0)
    empty = next(i for i, w in enumerate(env.women) if not w.active)
    _, r, *_ = env.step(encode_action(empty, TRIAGE))
    assert r <= R_INVALID
    waiting = next(i for i, w in enumerate(env.women) if w.stage == AWAIT_TRIAGE)
    _, r, *_ = env.step(encode_action(waiting, TREAT))   # cannot treat before triage
    assert r <= R_INVALID


def test_chw_budget_is_enforced():
    env = KaguguFollowUpEnv()
    env.reset(seed=0)
    i = next(i for i, w in enumerate(env.women) if w.active)
    budget = env.chw_budget
    for _ in range(budget):
        env.women[i].stage = AWAIT_TRIAGE      # keep her active
        env.step(encode_action(i, CHW_VISIT))
    assert env.chw_budget == 0
    _, r, *_ = env.step(encode_action(i, CHW_VISIT))
    assert r <= R_INVALID


def test_treatment_requires_positive_triage_and_pays_off():
    env = KaguguFollowUpEnv()
    env.reset(seed=0)
    w = env.women[0]
    w.stage, w.precancer, w.engagement = AWAIT_TREAT, True, 1.0
    _, r, *_ = env.step(encode_action(0, TREAT))
    assert r > 5   # treated true precancer (attendance prob ~0.98 at engagement 1.0)
    assert not env.women[0].active


def test_heuristic_beats_random_and_waiting():
    env = KaguguFollowUpEnv()
    rng = np.random.default_rng(0)
    rand = np.mean([run_episode(env, lambda o: int(rng.integers(N_ACTIONS)), s)[0] for s in range(40)])
    wait = np.mean([run_episode(env, lambda o: 0, s)[0] for s in range(40)])
    heur = np.mean([run_episode(env, heuristic_policy, s)[0] for s in range(40)])
    assert heur > rand > wait


def test_stress_scenario_runs():
    env = KaguguFollowUpEnv(params=STRESS_PARAMS)
    total, steps, info = run_episode(env, heuristic_policy, seed=7)
    assert steps > 0 and "precancer_treatment_rate" in info
