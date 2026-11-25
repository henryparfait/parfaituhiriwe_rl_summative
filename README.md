# Kagugu Follow-Up Coordinator: Reinforcement Learning Summative

**Author:** Henry Christian Parfait · **Repository:** [henryparfait/parfaituhiriwe_rl_summative](https://github.com/henryparfait/parfaituhiriwe_rl_summative)

**Mission:** AI-Assisted Triage and Community Follow-Up App for HPV-Positive Women at Kagugu Health Centre, Rwanda.
Late detection of cervical precancer happens when HPV-positive women are missed at triage or lost before treatment.
This project trains an RL agent to act as the health centre's **follow-up coordinator**: every day it decides who
gets a Kinyarwanda SMS reminder, who gets a community health worker (CHW) home visit, who is invited to
AI-assisted triage, and who is invited for thermal ablation. The goal is to treat as many true precancers as possible without
losing women to follow-up.

Four algorithms are implemented with **TorchRL** and compared on the *same* environment and frame budget:
**DQN** (value-based) and **REINFORCE, A2C, PPO** (policy-gradient).

---

## Quick start (uv only)

```bash
git clone https://github.com/henryparfait/parfaituhiriwe_rl_summative.git
cd parfaituhiriwe_rl_summative
uv sync                 # installs Python deps (CPU PyTorch) into .venv
uv run main.py          # runs the best trained agent with the live 3D viewer + verbose terminal log
```

`main.py` opens the 3D three.js viewer in your browser at `http://127.0.0.1:8765/`. You need an internet connection
the first time, because three.js loads from a CDN (a 2D fallback is used if it can't load). Useful options:

```bash
uv run main.py --agent ppo            # dqn | reinforce | a2c | ppo | heuristic | random
uv run main.py --stress               # unseen stress scenario (generalisation test)
uv run main.py --episodes 1 --delay 0.2
uv run play.py                        # same as main.py
uv run pytest -q                      # environment unit tests
```

## Training and experiments

```bash
uv run python -m training.smoke                       # 1-3 min check that all 4 algorithms run
uv run python -m training.sweep --algo all            # full sweep: >= 10 runs per algorithm, then plots
uv run python -m training.sweep --algo dqn            # or one algorithm per terminal, in parallel
uv run python -m training.dqn_training --run dqn_03_lr_high
uv run python -m training.pg_training --algo ppo --run ppo_04_clip_01
uv run python -m training.analyze                     # rebuild figures/tables from logs/
```

Every run trains for the same **100,000 environment frames**, evaluates greedily every 5,000 frames on 10 fixed
held-out seeds, and finishes with 30 held-out episodes plus 30 episodes of an **unseen stress scenario**.
Logs go to `logs/<algo>/<run_id>/progress.csv`, summaries to `logs/<algo>_summary.csv`, models to `models/dqn/` and `models/pg/`.
The best run of each algorithm is copied to `<algo>_best.pt`, and the overall winner is recorded in `models/best_model.json`.
The sweep definitions (what each run changes and why) are in `training/configs.py`.

---

## The environment (`environment/custom_env.py`)

| Element | Design |
|---|---|
| **Agent** | Follow-up coordinator at Kagugu Health Centre (Gasabo District, Kigali) |
| **Episode** | One programme period: 40 working days × 3 decisions/day = up to 120 steps |
| **Caseload** | 8 slots of HPV-positive women; an empty slot receives a new referral with p = 0.6/day |
| **Pathway** | HPV+ referral → AI-assisted VIA triage → (if positive) thermal ablation |
| **Scarce resources** | 3 decisions/day; 6 CHW home visits per 5-day week |

### Action space: `Discrete(33)`

`0` = wait. Actions `1-32` = (woman slot 1-8) × (intervention 1-4). Each one maps to a real action in the app:

| Intervention | Real-world meaning | Effect in simulation |
|---|---|---|
| SMS reminder | Kinyarwanda reminder from the CHW app | +0.12 engagement (diminishing if repeated); fails without a phone |
| CHW home visit | CHW visits the woman's home | +0.35 engagement, reveals exact engagement; uses weekly budget |
| Invite to triage | Nurse examination with AI cervical-image analysis | Attends with p = f(engagement, distance); AI risk score → positive/negative |
| Invite for ablation | Same-facility thermal ablation | Attends with p = f(engagement, distance); precancer removed |

Impossible actions (empty slot, treating before triage, no CHW budget left) are allowed but penalised. The agent has
to learn the pathway's rules.

### Observation space: `Box(0, 1, (84,))`, 10 features × 8 slots + 4 global

| Feature | Meaning | Encoding / range | Source in the real system | Update cadence |
|---|---|---|---|---|
| active | slot holds a woman | 0/1 | referral register | on referral / exit |
| stage_await_triage | waiting for triage | 0/1 | pathway status in app | on event |
| stage_await_treatment | triage-positive, waiting for ablation | 0/1 | nurse app result | on event |
| days_since_referral | time in pathway | days/40, clipped | HPV result date | daily |
| days_since_contact | time since last SMS reply / visit | days/14, clipped | CHW app log | daily |
| engagement_estimate | believed likelihood she stays engaged | 0-1 (last measured value minus expected decay) | SMS replies, CHW visit notes | on contact, decays daily |
| distance_km | distance from health centre | km/8 | village / cell at registration | static |
| has_phone | reachable by SMS | 0/1 | registration | static |
| ai_risk_score | AI cervical-image risk score | 0-1 (0 before triage) | nurse-side image model | at triage |
| missed_appointments | missed invitations | count/3, clipped | attendance log | on event |
| programme_day | progress through the period | day/40 | system clock | daily |
| decisions_left_today | coordinator capacity left today | 0-1 | scheduler | each step |
| chw_budget_left | CHW visits left this week | visits/6 | CHW roster | each step, resets weekly |
| weekday | position in the week | 0-1 | system clock | daily |

The **true** precancer status and true engagement are hidden (partial observability), just as in real life.

### Rewards

| Event | Reward | Why |
|---|---|---|
| True precancer (CIN2+) treated | **+10** | the mission's core outcome |
| Screen-positive woman without CIN2+ treated | +0.5 | guideline-consistent but low value (over-treatment) |
| Triage visit completed | +1 | progress along the pathway |
| Triage negative and truly lesion-free | +1 | correct discharge to routine rescreening |
| Triage negative but had a precancer | −6 | missed lesion (delayed, hidden consequence) |
| Precancer lost to follow-up | **−8** | the failure the mission targets |
| Other woman lost to follow-up | −3 | lost to the programme |
| Each day a precancer stays untreated | −0.05 | earlier treatment is better |
| Missed appointment | −0.3 | wasted clinic slot |
| SMS / CHW visit / invitation | −0.02 / −0.2 / −0.1 | resource cost |
| Invalid action | −0.5 | impossible in the current state |
| Precancer still untreated at period close | −3 each | handover risk |
| 3 precancers lost (programme audit) | −10, episode ends | catastrophic failure |

### Start and terminal conditions
- **Start:** day 0, 5 newly referred women, 3 empty slots, full CHW budget. Each woman's precancer status (p = 0.18),
  engagement (U[0.35, 0.85]), distance (U[0.5, 8] km) and phone access (p = 0.75) are randomised.
- **Terminal:** (a) 40 working days completed, or (b) 3 precancers lost to follow-up (programme audit).

### Simulation rules and parameter sources
- Daily engagement decay is 0.015 + 0.003 × distance_km. Below 0.25, a woman is lost with p = 0.15/day.
- Attendance probability = clip(engagement − 0.03 × distance + 0.15, 0.05, 0.98).
- AI triage score ~ N(0.68, 0.15) if precancer, N(0.32, 0.15) otherwise; positive if ≥ 0.5 (≈ 88% sensitivity/specificity, a modelling assumption).
- P(CIN2+ | HPV+) = 0.18, within the 12.9-19.1% CIN2+ reported among HPV-positive women in East African screen-and-treat cohorts (Sci. Rep. 2023, doi:10.1038/s41598-023-43467-2).
- Loss to follow-up between visits is a documented problem in Rwanda: in a careHPV + VIA pilot, 13% of high-grade lesions were lost between visits (BMC Women's Health 2018, doi:10.1186/s12905-018-0549-5).
- Rwanda's programme uses HPV testing → facility triage → thermal ablation, and CHWs bring screening to communities (CHAI, Nov 2026).
- All patient data is **simulated**. No real patient records are used.

### Why reinforcement learning?
Each decision changes the future caseload. A CHW visit spent today isn't available on Thursday. A woman who
isn't contacted keeps losing engagement until she's lost. Treatment is only possible after triage. The penalty for
neglect arrives days later. Ranking women once and acting greedily ignores these delayed, resource-coupled effects. A
sequential policy that weighs the value of each action now against its consequences later is exactly what RL optimises.
A rule-based coordinator (`heuristic_policy`) is included as a baseline the agents must beat.

## Results (30 held-out episodes with unseen initial states)

| Policy | Held-out return | Unseen stress scenario | Precancers treated | Precancers lost / period |
|---|---|---|---|---|
| **PPO, 300k frames (best, used by `main.py`)** | **102.3 ± 30.3** | **86.4** | **81%** | **0.6** |
| DQN, 300k frames | 90.3 ± 34.4 | 62.6 | 75% | 1.3 |
| Rule-based coordinator (baseline) | 73.6 ± 37.0 | 53.4 | 71% | 1.6 |
| DQN, best of 100k sweep (γ = 0.90) | 66.7 ± 43.9 | 50.2 | 68% | 1.7 |
| PPO, best of 100k sweep (10 epochs) | 58.2 ± 33.9 | 45.7 | 69% | 1.5 |
| A2C, 300k frames | 47.6 ± 39.6 | 26.1 | 62% | 1.9 |
| REINFORCE, 300k frames | 15.1 ± 37.9 | 11.3 | 61% | 2.2 |
| Random policy | −11.2 ± 17.4 | −30.5 | – | – |

With equal 100k-frame budgets (45 sweep runs), DQN learned fastest. With 300k frames, PPO learned the best policy:
it uses about 32 CHW home visits per period, against 4 for the rule-based coordinator, to keep at-risk women engaged, and loses 63% fewer precancers.
All figures and per-run tables are in `assets/`.

---

## Visualisation and integration
- `environment/rendering.py` runs a small local HTTP server; `environment/viewer/index.html` renders a **three.js
  WebGL 3D scene**. It shows the health centre, 8 homes placed by distance, stage beacons, engagement bars, phone
  markers, and animated SMS, CHW visits and patient trips, plus a live HUD and outcome counters.
- **API:** `GET /api/state` (the current snapshot as JSON), `GET /api/recommend` (the agent's recommended next action),
  `GET /api/health`. `/api/recommend` is the contract the CHW and nurse mobile app would call.
- **Serialisation:** trained policies are saved as plain PyTorch checkpoints (`state_dict` + architecture), so they
  load with `torch` alone, without TorchRL. That makes them easy to export to a mobile backend.

## Repository structure
```
pyproject.toml  uv.lock  README.md  main.py  play.py
environment/  custom_env.py  rendering.py  viewer/index.html
training/     dqn_training.py  pg_training.py  common.py  configs.py  sweep.py  analyze.py  smoke.py
models/dqn/  models/pg/   trained checkpoints (+ best_model.json)
logs/        per-run CSV logs and summaries
assets/      figures and tables for the report
tests/       environment unit tests (pytest)
```

## Limitations and ethics
This is a decision-support simulation, not a clinical tool. The parameters are literature-anchored assumptions and
would need calibration against Kagugu Health Centre's real referral and attendance data before any deployment. Clinical
decisions stay with nurses and doctors; the agent only prioritises follow-up actions.
