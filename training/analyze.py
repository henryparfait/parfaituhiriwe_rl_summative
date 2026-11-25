"""
Turn the training logs into the report's figures and tables.

    uv run python -m training.analyze

Outputs (assets/):
  fig_compare_rewards.png   held-out evaluation return vs frames, best run of each algorithm + baselines
  fig_cumulative_rewards.png  episode return vs training episodes, one subplot per method (template a)
  fig_episodes_to_converge.png  held-out return vs episodes with 90%-of-final marker (template b)
  fig_dqn_objective.png     DQN TD-loss and mean max-Q across all DQN runs
  fig_pg_entropy.png        policy entropy for REINFORCE / A2C / PPO (all runs, best highlighted)
  fig_convergence.png       training return of every run, one panel per algorithm (stability)
  fig_generalization.png    held-out vs unseen stress scenario, best run per algorithm + baselines
  fig_hparam_effects.png    final return of every sweep run (spread = sensitivity to hyperparameters)
  tables/<algo>_sweep.md|csv  hyperparameter tables for the report
  results_summary.md        headline numbers
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from environment.custom_env import N_ACTIONS, STRESS_PARAMS, KaguguFollowUpEnv, heuristic_policy  # noqa: E402
from training.configs import SWEEPS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LOGS, ASSETS = ROOT / "logs", ROOT / "assets"
ALGOS = ["dqn", "reinforce", "a2c", "ppo"]
NAMES = {"dqn": "DQN", "reinforce": "REINFORCE", "a2c": "A2C", "ppo": "PPO"}
COLORS = {"dqn": "#257de0", "reinforce": "#f47230", "a2c": "#00b894", "ppo": "#f2a600"}  # validated categorical order
GREY, INK, MUTED = "#9aa0a8", "#1d2433", "#5b6577"
FINAL_SEEDS = list(range(20_000, 20_030))
STRESS_SEEDS = list(range(30_000, 30_030))

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "font.size": 10, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.edgecolor": "#c9ced6", "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": "#e6e9ee",
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "lines.linewidth": 2.0,
})


from matplotlib.ticker import FuncFormatter  # noqa: E402

KFMT = FuncFormatter(lambda x, _: f"{x / 1000:.0f}k" if x else "0")


def smooth(y: pd.Series, k: int = 10) -> pd.Series:
    return y.rolling(k, min_periods=1).mean()


def baseline_scores() -> dict:
    """Random and rule-based policies on the same held-out seeds (no torch needed)."""
    out = {}
    rng = np.random.default_rng(0)
    for name, pol in (("Random", lambda o: int(rng.integers(N_ACTIONS))), ("Heuristic", heuristic_policy)):
        for scen, seeds, params in (("final", FINAL_SEEDS, None), ("stress", STRESS_SEEDS, STRESS_PARAMS)):
            env = KaguguFollowUpEnv(params=params)
            rets, rates = [], []
            for s in seeds:
                o, _ = env.reset(seed=s)
                done, ret = False, 0.0
                while not done:
                    o, r, t, tr, info = env.step(pol(o))
                    ret += r
                    done = t or tr
                rets.append(ret)
                rates.append(info["precancer_treatment_rate"])
            out[(name, scen)] = (float(np.mean(rets)), float(np.std(rets)), float(np.mean(rates)))
    return out


def load_algo(algo: str):
    path = LOGS / f"{algo}_summary.csv"
    if not path.exists():
        return None, {}
    summ = pd.read_csv(path)
    prog = {}
    for rid in summ["run_id"]:
        f = LOGS / algo / rid / "progress.csv"
        if f.exists():
            prog[rid] = pd.read_csv(f)
    return summ, prog


def best_run(summ: pd.DataFrame) -> str:
    """Best run of the equal-budget (100k) sweep; extended '_long' runs are reported separately."""
    sw = summ[~summ["run_id"].astype(str).str.endswith("_long")]
    return sw.loc[sw["final_return_mean"].idxmax(), "run_id"]


def to_markdown(df: pd.DataFrame) -> str:
    head = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "---|" * len(df.columns)
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *body]) + "\n"


def save(fig, name: str):
    for ax in fig.axes:   # compact "25k" frame labels
        if "Frames" in ax.get_xlabel() or "frames" in ax.get_xlabel():
            ax.xaxis.set_major_formatter(KFMT)
    ASSETS.mkdir(exist_ok=True)
    fig.savefig(ASSETS / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", ASSETS / name)


def _mr(r) -> str:
    return f"{r['final_return_mean']:.1f} ± {r['final_return_std']:.1f}"


def template_table(algo: str, summ: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in summ.iterrows():
        net = f"{int(r['hidden'])}x{int(r['depth'])}"
        if algo == "dqn":
            rows.append({"Run": r["run_id"], "Learning Rate": r["lr"], "Gamma": r["gamma"],
                         "Replay Buffer Size": int(r["buffer_size"]), "Batch Size": int(r["batch_size"]),
                         "Exploration Strategy": f"ε-greedy 1.0→{r['eps_end']} over {r['eps_decay_frac']:.0%} of frames",
                         "Target Update (steps)": int(r["target_update"]),
                         "Other": ("Double DQN, " if str(r["double_dqn"]) == "True" else "") + f"net {net}, UTD {r['utd_ratio']}",
                         "Mean Reward": _mr(r)})
        elif algo == "reinforce":
            rows.append({"Run": r["run_id"], "Learning Rate": r["lr"], "Gamma": r["gamma"],
                         "Episodes / Update": int(r["episodes_per_batch"]),
                         "Baseline": "learned V(s)" if str(r["baseline"]) == "True" else "none",
                         "Return Normalisation": "yes" if str(r["normalize_adv"]) == "True" else "no",
                         "Entropy Coef.": r["entropy_coeff"], "Network": net, "Mean Reward": _mr(r)})
        elif algo == "a2c":
            rows.append({"Run": r["run_id"], "Learning Rate": r["lr"], "Gamma": r["gamma"],
                         "Rollout Length": int(r["frames_per_batch"]), "GAE λ": r["gae_lambda"],
                         "Entropy Coef.": r["entropy_coeff"], "Critic Coef.": r["critic_coeff"],
                         "Network": net, "Mean Reward": _mr(r)})
        else:
            rows.append({"Run": r["run_id"], "Learning Rate": r["lr"], "Gamma": r["gamma"],
                         "Rollout / Minibatch": f"{int(r['frames_per_batch'])} / {int(r['minibatch'])}",
                         "Epochs": int(r["epochs"]), "Clip ε": r["clip"], "GAE λ": r["gae_lambda"],
                         "Entropy Coef.": r["entropy_coeff"], "Mean Reward": _mr(r)})
    return pd.DataFrame(rows)


def main():
    data = {a: load_algo(a) for a in ALGOS}
    data = {a: v for a, v in data.items() if v[0] is not None and len(v[0])}
    if not data:
        raise SystemExit("No logs found. Train first (see README).")
    base = baseline_scores()
    best = {a: best_run(s) for a, (s, _) in data.items()}

    # 1) comparative rewards -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for a, (summ, prog) in data.items():
        df = prog[best[a]].dropna(subset=["eval_return"])
        ax.plot(df["frames"], df["eval_return"], color=COLORS[a], label=f"{NAMES[a]} ({best[a]})")
    ax.axhline(base[("Heuristic", "final")][0], color=GREY, ls="--", lw=1.5, label="Rule-based heuristic")
    ax.axhline(base[("Random", "final")][0], color=GREY, ls=":", lw=1.5, label="Random policy")
    ax.set(title="Evaluation return during training (best run per algorithm)",
           xlabel="Environment frames", ylabel="Mean return, 10 held-out episodes")
    ax.legend(loc="lower right", fontsize=8.5)
    save(fig, "fig_compare_rewards.png")

    # 1b) template figure: episode returns over EPISODES, one subplot per method (best runs)
    conv_eps = {}
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4), sharey=True)
    for ax, a in zip(axes.flat, ALGOS):
        if a not in data:
            ax.set_visible(False)
            continue
        df = data[a][1][best[a]].dropna(subset=["train_return"])
        ax.plot(df["episodes"], df["train_return"], color=COLORS[a])
        ax.axhline(base[("Heuristic", "final")][0], color=GREY, ls="--", lw=1)
        ax.set(title=f"{NAMES[a]} ({best[a]})", xlabel="Training episodes")
    axes[0][0].set_ylabel("Episode return (moving avg. of 20)")
    axes[1][0].set_ylabel("Episode return (moving avg. of 20)")
    fig.suptitle("Cumulative reward per episode, best model of each method (dashed = rule-based heuristic)",
                 fontsize=10, color=MUTED)
    fig.tight_layout()
    save(fig, "fig_cumulative_rewards.png")

    # 1c) template figure: episodes to converge (held-out eval return vs episodes, 90% marker)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4), sharey=True)
    for ax, a in zip(axes.flat, ALGOS):
        if a not in data:
            ax.set_visible(False)
            continue
        df = data[a][1][best[a]]
        ev = df.dropna(subset=["eval_return"])
        ax.plot(ev["episodes"], ev["eval_return"], color=COLORS[a], marker="o", markersize=3.5)
        f90 = data[a][0].set_index("run_id").loc[best[a]].get("frames_to_90pct")
        if pd.notna(f90):
            ep = int(df.loc[df["frames"] >= float(f90), "episodes"].iloc[0])
            conv_eps[a] = ep
            ax.axvline(ep, color=MUTED, ls=":", lw=1.2)
            ax.annotate(f"90% of final level\n@ {ep} episodes", xy=(ep, ev["eval_return"].min()),
                        xytext=(6, 4), textcoords="offset points", fontsize=8, color=MUTED)
        ax.set(title=f"{NAMES[a]}: held-out evaluation", xlabel="Training episodes")
    axes[0][0].set_ylabel("Mean return, 10 held-out episodes")
    axes[1][0].set_ylabel("Mean return, 10 held-out episodes")
    fig.tight_layout()
    save(fig, "fig_episodes_to_converge.png")

    # 2) DQN objective ---------------------------------------------------------------------
    if "dqn" in data:
        summ, prog = data["dqn"]
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
        for rid, df in prog.items():
            if rid.endswith("_long"):
                continue
            is_best = rid == best["dqn"]
            kw = dict(color=COLORS["dqn"] if is_best else GREY, lw=2 if is_best else 0.9,
                      alpha=1 if is_best else 0.55, zorder=3 if is_best else 1,
                      label=f"best: {rid}" if is_best else None)
            d = df.dropna(subset=["loss"])
            axes[0].plot(d["frames"], smooth(d["loss"]), **kw)
            d = df.dropna(subset=["q_mean"])
            axes[1].plot(d["frames"], smooth(d["q_mean"]), **kw)
        axes[0].set(title="DQN objective: TD loss (smoothed)", xlabel="Frames", ylabel="MSE TD loss", yscale="log")
        axes[1].set(title="Mean max-Q on replay samples", xlabel="Frames", ylabel="Q-value")
        axes[0].plot([], [], color=GREY, lw=0.9, label="other sweep runs")
        axes[0].legend(fontsize=8.5)
        save(fig, "fig_dqn_objective.png")

    # 3) policy-gradient entropy ----------------------------------------------------------
    pg = [a for a in ("reinforce", "a2c", "ppo") if a in data]
    if pg:
        fig, axes = plt.subplots(1, len(pg), figsize=(4 * len(pg), 3.6), sharey=True, squeeze=False)
        for ax, a in zip(axes[0], pg):
            for rid, df in data[a][1].items():
                if rid.endswith("_long"):
                    continue
                is_best = rid == best[a]
                d = df.dropna(subset=["entropy"])
                ax.plot(d["frames"], smooth(d["entropy"], 5), color=COLORS[a] if is_best else GREY,
                        lw=2 if is_best else 0.9, alpha=1 if is_best else 0.55, zorder=3 if is_best else 1)
            ax.axhline(np.log(N_ACTIONS), color=MUTED, ls=":", lw=1)
            ax.set(title=f"{NAMES[a]} policy entropy", xlabel="Frames")
        axes[0][0].set_ylabel("Entropy (nats); dotted = uniform")
        save(fig, "fig_pg_entropy.png")

    # 4) convergence / stability ---------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4), sharey=True)
    for ax, a in zip(axes.flat, ALGOS):
        if a not in data:
            ax.set_visible(False)
            continue
        for rid, df in data[a][1].items():
            if rid.endswith("_long"):
                continue
            is_best = rid == best[a]
            d = df.dropna(subset=["train_return"])
            ax.plot(d["frames"], smooth(d["train_return"], 8), color=COLORS[a] if is_best else GREY,
                    lw=2 if is_best else 0.9, alpha=1 if is_best else 0.5, zorder=3 if is_best else 1)
        ax.axhline(base[("Heuristic", "final")][0], color=MUTED, ls="--", lw=1)
        ax.set(title=f"{NAMES[a]}: training return, all {sum(not r.endswith('_long') for r in data[a][1])} runs", xlabel="Frames")
    axes[0][0].set_ylabel("Mean return (last 20 episodes)")
    axes[1][0].set_ylabel("Mean return (last 20 episodes)")
    fig.suptitle("Convergence and stability across the hyperparameter sweep (coloured = best run; dashed = heuristic)",
                 fontsize=10, color=MUTED)
    fig.tight_layout()
    save(fig, "fig_convergence.png")

    # 5) generalisation -----------------------------------------------------------------
    labels, std_v, std_e, str_v, str_e, cols = [], [], [], [], [], []
    for a, (summ, _) in data.items():
        r = summ.set_index("run_id").loc[best[a]]
        labels.append(NAMES[a]); cols.append(COLORS[a])
        std_v.append(r["final_return_mean"]); std_e.append(r["final_return_std"])
        str_v.append(r["stress_return_mean"]); str_e.append(r["stress_return_std"])
    for name in ("Heuristic", "Random"):
        labels.append(name); cols.append(GREY)
        std_v.append(base[(name, "final")][0]); std_e.append(base[(name, "final")][1])
        str_v.append(base[(name, "stress")][0]); str_e.append(base[(name, "stress")][1])
    x = np.arange(len(labels)); w = 0.38
    fig, ax = plt.subplots(figsize=(9, 4))
    ek = dict(ecolor=MUTED, elinewidth=1, capsize=3)
    ax.bar(x - w / 2 - 0.01, std_v, w, yerr=std_e, color=cols, error_kw=ek)
    ax.bar(x + w / 2 + 0.01, str_v, w, yerr=str_e, color=cols, alpha=0.45, hatch="//", edgecolor="white",
           error_kw=ek)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=GREY, label="Held-out seeds (standard scenario)"),
                       Patch(facecolor=GREY, alpha=0.45, hatch="//", edgecolor="white",
                             label="Unseen stress scenario")], fontsize=8.5)
    ax.axhline(0, color="#c9ced6", lw=1)
    ax.set_xticks(x, labels)
    ax.set(title="Generalisation: 30 held-out episodes each", ylabel="Mean return ± 1 s.d.")
    save(fig, "fig_generalization.png")

    # 6) hyperparameter sensitivity -----------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for i, (a, (summ, _)) in enumerate(data.items()):
        v = summ.loc[~summ["run_id"].astype(str).str.endswith("_long"), "final_return_mean"].values
        jitter = np.random.default_rng(i).uniform(-0.12, 0.12, len(v))
        ax.scatter(np.full(len(v), i) + jitter, v, s=46, color=COLORS[a], edgecolor="white", linewidth=1.5, zorder=3)
    ax.axhline(base[("Heuristic", "final")][0], color=GREY, ls="--", lw=1.2, label="Heuristic")
    ax.set_xticks(range(len(data)), [NAMES[a] for a in data])
    ax.set(title="Held-out return of every sweep run (spread = hyperparameter sensitivity)", ylabel="Mean return")
    ax.legend(fontsize=8.5)
    save(fig, "fig_hparam_effects.png")

    # 6b) extended runs (phase 2), if any
    longs = {a: [r for r in data[a][1] if r.endswith("_long")] for a in data}
    if any(longs.values()):
        fig, ax = plt.subplots(figsize=(8, 4.2))
        for a, rids in longs.items():
            for rid in rids:
                df = data[a][1][rid].dropna(subset=["eval_return"])
                ax.plot(df["frames"], smooth(df["eval_return"], 3), color=COLORS[a], label=f"{NAMES[a]} ({rid})")
        ax.axvline(100_000, color=MUTED, ls=":", lw=1)
        ax.axhline(base[("Heuristic", "final")][0], color=GREY, ls="--", lw=1.2, label="Heuristic")
        ax.set(title="Extended training of each method's best configuration", xlabel="Frames",
               ylabel="Held-out eval return (smoothed)")
        ax.legend(fontsize=8)
        save(fig, "fig_extended_training.png")
        ext = ["", "| Extended run | Held-out return | Stress return | Treat-rate |", "|---|---|---|---|"]
        for a, rids in longs.items():
            for rid in rids:
                r = data[a][0].set_index("run_id").loc[rid]
                ext.append(f"| {rid} | {r['final_return_mean']:.1f} ± {r['final_return_std']:.1f} | "
                           f"{r['stress_return_mean']:.1f} | {r['final_precancer_treatment_rate']:.0%} |")
    else:
        ext = []

    # 7) tables + summary -----------------------------------------------------------------
    tdir = ASSETS / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    lines = ["# Results summary", "",
             f"Baselines (30 held-out episodes): heuristic {base[('Heuristic', 'final')][0]:.1f} "
             f"(treat-rate {base[('Heuristic', 'final')][2]:.0%}), random {base[('Random', 'final')][0]:.1f}", "",
             "| Algorithm | Best run | Held-out return | Stress return | Precancer treat-rate | Precancers lost | Invalid actions | Frames to 90% |",
             "|---|---|---|---|---|---|---|---|"]
    for a, (summ, _) in data.items():
        base_cfg, _runs = SWEEPS[a]
        rows = []
        for _, r in summ.iterrows():
            changed = []
            for k, v in base_cfg.items():
                if k in r and str(r[k]) != str(v):
                    try:
                        if float(r[k]) == float(v):
                            continue
                    except (TypeError, ValueError):
                        pass
                    changed.append(f"{k}={r[k]}")
            rows.append({
                "Run": r["run_id"], "Change vs baseline": ", ".join(changed) or "(baseline)",
                "Hypothesis": r.get("why", ""),
                "Held-out return": f"{r['final_return_mean']:.1f} ± {r['final_return_std']:.1f}",
                "Stress return": f"{r['stress_return_mean']:.1f}",
                "Treat-rate": f"{r['final_precancer_treatment_rate']:.0%}",
                "Invalid/ep": f"{r['final_invalid_actions']:.1f}",
                "Frames to 90%": "" if pd.isna(r.get("frames_to_90pct")) else int(r["frames_to_90pct"]),
                "Minutes": r.get("train_minutes", ""),
            })
        tab = pd.DataFrame(rows)
        tab.to_csv(tdir / f"{a}_sweep.csv", index=False, encoding="utf-8")
        (tdir / f"{a}_sweep.md").write_text(to_markdown(tab), encoding="utf-8")
        b = summ.set_index("run_id").loc[best[a]]
        f90 = b.get("frames_to_90pct")
        lines.append(f"| {NAMES[a]} | {best[a]} | {b['final_return_mean']:.1f} ± {b['final_return_std']:.1f} | "
                     f"{b['stress_return_mean']:.1f} | {b['final_precancer_treatment_rate']:.0%} | "
                     f"{b['final_lost_precancer']:.2f} | {b['final_invalid_actions']:.1f} | "
                     f"{'' if pd.isna(f90) else int(f90)} |")
    # template-shaped tables (columns requested by the report template)
    for a, (summ, _) in data.items():
        tab = template_table(a, summ)
        tab.to_csv(tdir / f"{a}_template_table.csv", index=False)
        (tdir / f"{a}_template_table.md").write_text(to_markdown(tab), encoding="utf-8")
    lines += ["", "Episodes to reach 90% of final evaluation level (best run): " +
              ", ".join(f"{NAMES[a]} {e}" for a, e in conv_eps.items())]
    lines += ext
    (ASSETS / "results_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (ASSETS / "baselines.json").write_text(json.dumps({f"{k[0]}_{k[1]}": v for k, v in base.items()}, indent=2), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
