"""Checks that TorchRL's ReinforceLoss computes the textbook REINFORCE gradient on our actor:
loss_actor = -mean( log pi(a_t|s_t) * A_t ) for the actions that were actually taken.
Skipped automatically if torchrl is not installed."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchrl")


def test_reinforce_loss_matches_manual_policy_gradient():
    from torchrl.envs.utils import ExplorationType, set_exploration_type
    from torchrl.objectives import ReinforceLoss

    from training.common import make_torchrl_env
    from training.pg_training import build_actor_critic

    torch.manual_seed(0)
    env = make_torchrl_env(seed=0)
    _, actor, critic, _ = build_actor_critic(env, {"hidden": 32, "depth": 1})
    with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
        td = env.rollout(60, policy=actor)
    n = td.shape[0]
    adv = torch.randn(n, 1)
    td.set("advantage", adv)
    td.set("value_target", torch.randn(n, 1))

    out = ReinforceLoss(actor_network=actor, critic_network=critic)(td.clone())
    logp = actor.get_dist(td.select("observation")).log_prob(td.get("action"))
    manual = -(logp.reshape(-1) * adv.reshape(-1)).mean()
    assert torch.allclose(out["loss_actor"], manual, atol=1e-5), (out["loss_actor"].item(), manual.item())
