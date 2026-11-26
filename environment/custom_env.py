"""
Kagugu Follow-Up Coordinator environment.

Mission: AI-Assisted Triage and Community Follow-Up for HPV-positive women at
Kagugu Health Centre (Gasabo District, Kigali, Rwanda).

The agent plays the follow-up coordinator. Every working day it makes
ACTIONS_PER_DAY decisions about a caseload of N_SLOTS HPV-positive women who
are somewhere on the pathway

    HPV+ referral -> AI-assisted VIA triage -> thermal ablation (treatment)

Resources are scarce (weekly CHW home-visit budget, limited daily decisions),
women's engagement fades every day without contact, and the cost of a missed
precancer only appears later (when she is lost to follow-up or discharged with
an undetected lesion). Because today's choices change tomorrow's caseload,
this is a sequential decision problem rather than a one-shot ranking problem.

All patient-level quantities are SIMULATED. Rates are anchored to published
Rwandan / East African screen-and-treat data where available and are otherwise
stated as modelling assumptions (see README / report, "Observation table").
"""

from __future__ import annotations

from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
N_SLOTS = 8                 # women in the active caseload at one time
ACTIONS_PER_DAY = 3         # coordinator decisions per working day
HORIZON_DAYS = 40           # one programme period = 8 working weeks
CHW_WEEKLY_BUDGET = 6       # CHW home visits available per 5-day week
DAYS_PER_WEEK = 5

# Interventions available for each woman (each maps to a button in the app)
SMS, CHW_VISIT, TRIAGE, TREAT = 0, 1, 2, 3
INTERVENTIONS = ["SMS reminder (Kinyarwanda)", "CHW home visit",
                 "Invite to AI-assisted triage", "Invite for thermal ablation"]
N_INTERVENTIONS = len(INTERVENTIONS)
N_ACTIONS = 1 + N_SLOTS * N_INTERVENTIONS   # 0 = wait / no action

# Pathway stages
EMPTY, AWAIT_TRIAGE, AWAIT_TREAT = 0, 1, 2

# Clinical / behavioural parameters
P_PRECANCER = 0.18          # P(CIN2+ | HPV+); 12.9-19.1% reported in East African SAT cohorts
P_HAS_PHONE = 0.75          # assumption: share of women reachable by SMS
P_NEW_REFERRAL = 0.60       # daily probability an empty slot receives a new HPV+ referral
ENGAGEMENT_DECAY = 0.015     # daily engagement loss ...
DISTANCE_DECAY = 0.003      # ... plus this much per km from the health centre
DROPOUT_THRESHOLD = 0.25    # below this engagement a woman can be lost to follow-up
P_DROPOUT = 0.15            # daily probability of loss once below the threshold
SMS_GAIN = 0.12             # engagement gain from an SMS (diminishing if repeated)
VISIT_GAIN = 0.35           # engagement gain from a CHW home visit
MISSED_APPT_LOSS = 0.08     # engagement loss after a missed appointment
AI_MEAN_POS, AI_MEAN_NEG, AI_SD = 0.68, 0.32, 0.15   # AI risk-score model (sens/spec ~0.88)
AI_THRESHOLD = 0.5

# Rewards
R_TREAT_PRECANCER = 10.0    # true precancer treated: the mission's core outcome
R_TREAT_SCREEN_POS = 0.5    # screen-positive without CIN2+ treated per protocol
R_TRIAGE_DONE = 1.0         # triage visit completed
R_CORRECT_DISCHARGE = 1.0   # triage-negative and truly lesion-free
R_MISSED_LESION = -6.0      # triage-negative but had a precancer (false negative)
R_LOST_PRECANCER = -8.0     # precancer lost to follow-up
R_LOST_OTHER = -3.0         # other woman lost to follow-up
R_UNTREATED_DAILY = -0.05   # each day a precancer remains untreated
R_MISSED_APPT = -0.3        # woman invited but did not attend
R_INVALID = -0.5            # action impossible in current state (wrong stage, empty slot, no budget)
COST = {SMS: -0.02, CHW_VISIT: -0.2, TRIAGE: -0.1, TREAT: -0.1}
R_PROGRAMME_FAILURE = -10.0 # >= 3 precancers lost -> programme audit, episode ends
R_HANDOVER_UNTREATED = -3.0 # precancer still untreated when the period closes
MAX_LOST_PRECANCER = 3

DEFAULT_PARAMS = {
    "p_precancer": P_PRECANCER, "p_has_phone": P_HAS_PHONE, "p_new_referral": P_NEW_REFERRAL,
    "engagement_decay": ENGAGEMENT_DECAY, "distance_decay": DISTANCE_DECAY, "p_dropout": P_DROPOUT,
}
# Unseen scenario used only for generalisation testing (never seen during training):
# fewer women reachable by phone, faster disengagement, more referrals (rainy season / campaign surge)
STRESS_PARAMS = {**DEFAULT_PARAMS, "p_has_phone": 0.55, "engagement_decay": 0.02,
                 "distance_decay": 0.004, "p_new_referral": 0.8}

OBS_PER_SLOT = 10
OBS_GLOBAL = 4
OBS_DIM = N_SLOTS * OBS_PER_SLOT + OBS_GLOBAL

SLOT_FEATURES = [
    "active", "stage_await_triage", "stage_await_treatment",
    "days_since_referral", "days_since_contact", "engagement_estimate",
    "distance_km", "has_phone", "ai_risk_score", "missed_appointments",
]
GLOBAL_FEATURES = ["programme_day", "decisions_left_today", "chw_budget_left", "weekday"]


def decode_action(action: int) -> tuple[int, int]:
    """Map a flat action id to (slot, intervention). Returns (-1, -1) for wait."""
    action = int(action)
    if action == 0:
        return -1, -1
    return (action - 1) // N_INTERVENTIONS, (action - 1) % N_INTERVENTIONS


def encode_action(slot: int, intervention: int) -> int:
    return 1 + slot * N_INTERVENTIONS + intervention


def describe_action(action: int) -> str:
    slot, iv = decode_action(action)
    if slot < 0:
        return "Wait / no action"
    return f"{INTERVENTIONS[iv]} -> woman in slot {slot + 1}"


@dataclass
class Woman:
    uid: int = 0
    stage: int = EMPTY
    precancer: bool = False          # hidden ground truth
    engagement: float = 0.0          # hidden true engagement
    engagement_est: float = 0.0      # what the app knows (updated on contact)
    distance_km: float = 0.0
    has_phone: bool = False
    days_since_referral: int = 0
    days_since_contact: int = 0
    recent_sms: int = 0
    ai_risk: float = 0.0
    missed: int = 0

    @property
    def active(self) -> bool:
        return self.stage != EMPTY


@dataclass
class EpisodeStats:
    treated_precancer: int = 0
    treated_other: int = 0
    triaged: int = 0
    missed_lesions: int = 0
    correct_discharges: int = 0
    lost_precancer: int = 0
    lost_other: int = 0
    invalid_actions: int = 0
    missed_appointments: int = 0
    sms_sent: int = 0
    chw_visits: int = 0
    referrals: int = 0
    precancers_seen: int = 0
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "extra"}
        # share of precancers that reached treatment among those whose outcome is known
        resolved = self.treated_precancer + self.lost_precancer + self.missed_lesions
        d["precancer_treatment_rate"] = self.treated_precancer / resolved if resolved else 0.0
        return d


class KaguguFollowUpEnv(gym.Env):
    """Gymnasium environment for the Kagugu HPV follow-up coordination task."""

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(self, render_mode: str | None = None, viewer=None, params: dict | None = None):
        super().__init__()
        self.p = {**DEFAULT_PARAMS, **(params or {})}
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(OBS_DIM,), dtype=np.float32)
        self._viewer = viewer
        self.women: list[Woman] = []
        self.stats = EpisodeStats()
        self.day = 0
        self.decisions_left = ACTIONS_PER_DAY
        self.chw_budget = CHW_WEEKLY_BUDGET
        self._uid = 0
        self.last_event = ""
        self.last_action = 0
        self.last_reward = 0.0
        self.episode_return = 0.0
        self.t = 0

    #  helpers
    def _new_woman(self) -> Woman:
        rng = self.np_random
        self._uid += 1
        w = Woman(
            uid=self._uid,
            stage=AWAIT_TRIAGE,
            precancer=bool(rng.random() < self.p["p_precancer"]),
            engagement=float(rng.uniform(0.35, 0.85)),
            distance_km=float(rng.uniform(0.5, 8.0)),
            has_phone=bool(rng.random() < self.p["p_has_phone"]),
        )
        w.engagement_est = float(np.clip(w.engagement + rng.normal(0, 0.1), 0, 1))
        self.stats.referrals += 1
        self.stats.precancers_seen += int(w.precancer)
        return w

    def _attend_prob(self, w: Woman) -> float:
        return float(np.clip(w.engagement - 0.03 * w.distance_km + 0.15, 0.05, 0.98))

    def _contact(self, w: Woman, exact: bool) -> None:
        w.days_since_contact = 0
        noise = 0.0 if exact else self.np_random.normal(0, 0.08)
        w.engagement_est = float(np.clip(w.engagement + noise, 0, 1))

    def _exit(self, slot: int) -> None:
        self.women[slot] = Woman()

    #  gym API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._uid = 0
        self.stats = EpisodeStats()
        self.day = 0
        self.decisions_left = ACTIONS_PER_DAY
        self.chw_budget = CHW_WEEKLY_BUDGET
        # start state: 5 newly referred women, 3 empty slots
        self.women = [self._new_woman() if i < 5 else Woman() for i in range(N_SLOTS)]
        self.last_event = "Programme period starts: 5 new HPV-positive referrals"
        self.last_action = 0
        self.last_reward = 0.0
        self.episode_return = 0.0
        self.t = 0
        self._render_frame()
        return self._obs(), self._info()

    def step(self, action):
        reward = 0.0
        slot, iv = decode_action(action)
        rng = self.np_random
        event = "Coordinator waits"

        if slot >= 0:
            w = self.women[slot]
            if not w.active:
                reward += R_INVALID
                self.stats.invalid_actions += 1
                event = f"Invalid: slot {slot + 1} is empty"
            elif iv == SMS:
                reward += COST[SMS]
                self.stats.sms_sent += 1
                if w.has_phone:
                    gain = SMS_GAIN / (1 + w.recent_sms)
                    w.engagement = min(1.0, w.engagement + gain)
                    w.recent_sms += 1
                    self._contact(w, exact=False)
                    event = f"SMS sent to slot {slot + 1} (engagement +{gain:.2f})"
                else:
                    event = f"SMS failed: slot {slot + 1} has no phone"
            elif iv == CHW_VISIT:
                if self.chw_budget <= 0:
                    reward += R_INVALID
                    self.stats.invalid_actions += 1
                    event = "Invalid: CHW visit budget used up this week"
                else:
                    reward += COST[CHW_VISIT]
                    self.chw_budget -= 1
                    self.stats.chw_visits += 1
                    w.engagement = min(1.0, w.engagement + VISIT_GAIN)
                    w.recent_sms = 0
                    self._contact(w, exact=True)
                    event = f"CHW home visit to slot {slot + 1}"
            elif iv == TRIAGE:
                if w.stage != AWAIT_TRIAGE:
                    reward += R_INVALID
                    self.stats.invalid_actions += 1
                    event = f"Invalid: slot {slot + 1} is not awaiting triage"
                else:
                    reward += COST[TRIAGE]
                    if rng.random() < self._attend_prob(w):
                        mean = AI_MEAN_POS if w.precancer else AI_MEAN_NEG
                        score = float(np.clip(rng.normal(mean, AI_SD), 0, 1))
                        self.stats.triaged += 1
                        reward += R_TRIAGE_DONE
                        if score >= AI_THRESHOLD:
                            w.stage = AWAIT_TREAT
                            w.ai_risk = score
                            w.engagement = min(1.0, w.engagement + 0.1)
                            self._contact(w, exact=True)
                            event = f"Slot {slot + 1} triaged: AI risk {score:.2f} -> needs treatment"
                        else:
                            if w.precancer:
                                reward += R_MISSED_LESION
                                self.stats.missed_lesions += 1
                                event = f"Slot {slot + 1} triaged negative (lesion missed!)"
                            else:
                                reward += R_CORRECT_DISCHARGE
                                self.stats.correct_discharges += 1
                                event = f"Slot {slot + 1} triaged negative -> routine rescreening"
                            self._exit(slot)
                    else:
                        reward += R_MISSED_APPT
                        self.stats.missed_appointments += 1
                        w.missed += 1
                        w.engagement = max(0.0, w.engagement - MISSED_APPT_LOSS)
                        event = f"Slot {slot + 1} did not attend triage"
            elif iv == TREAT:
                if w.stage != AWAIT_TREAT:
                    reward += R_INVALID
                    self.stats.invalid_actions += 1
                    event = f"Invalid: slot {slot + 1} has not been triaged positive"
                else:
                    reward += COST[TREAT]
                    if rng.random() < self._attend_prob(w):
                        if w.precancer:
                            reward += R_TREAT_PRECANCER
                            self.stats.treated_precancer += 1
                            event = f"Slot {slot + 1} treated: precancer removed (+10)"
                        else:
                            reward += R_TREAT_SCREEN_POS
                            self.stats.treated_other += 1
                            event = f"Slot {slot + 1} treated (screen-positive, no CIN2+)"
                        self._exit(slot)
                    else:
                        reward += R_MISSED_APPT
                        self.stats.missed_appointments += 1
                        w.missed += 1
                        w.engagement = max(0.0, w.engagement - MISSED_APPT_LOSS)
                        event = f"Slot {slot + 1} did not attend treatment"

        #  advance the clock
        self.decisions_left -= 1
        terminated = False
        if self.decisions_left == 0:
            reward += self._end_of_day()
            self.decisions_left = ACTIONS_PER_DAY
            if self.stats.lost_precancer >= MAX_LOST_PRECANCER:
                reward += R_PROGRAMME_FAILURE
                terminated = True
                event += " | PROGRAMME AUDIT: 3 precancers lost"
            elif self.day >= HORIZON_DAYS:
                untreated = sum(1 for w in self.women if w.active and w.precancer)
                reward += R_HANDOVER_UNTREATED * untreated
                terminated = True
                event += " | Programme period closed"

        self.last_event = event
        self.last_action = int(action)
        self.last_reward = float(reward)
        self.episode_return += reward
        self.t += 1
        self._render_frame()
        return self._obs(), float(reward), terminated, False, self._info()

    def _end_of_day(self) -> float:
        rng = self.np_random
        reward = 0.0
        for i, w in enumerate(self.women):
            if not w.active:
                if rng.random() < self.p["p_new_referral"]:
                    self.women[i] = self._new_woman()
                continue
            w.engagement = max(0.0, w.engagement - self.p["engagement_decay"] - self.p["distance_decay"] * w.distance_km)
            w.days_since_referral += 1
            w.days_since_contact += 1
            if w.days_since_contact >= 3:
                w.recent_sms = 0
            if w.precancer:
                reward += R_UNTREATED_DAILY
            if w.engagement < DROPOUT_THRESHOLD and rng.random() < self.p["p_dropout"]:
                if w.precancer:
                    reward += R_LOST_PRECANCER
                    self.stats.lost_precancer += 1
                else:
                    reward += R_LOST_OTHER
                    self.stats.lost_other += 1
                self._exit(i)
        self.day += 1
        if self.day % DAYS_PER_WEEK == 0:
            self.chw_budget = CHW_WEEKLY_BUDGET
        return reward

    #  observation
    def _obs(self) -> np.ndarray:
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        for i, w in enumerate(self.women):
            if not w.active:
                continue
            est = max(0.0, w.engagement_est - (self.p["engagement_decay"] + self.p["distance_decay"] * w.distance_km)
                      * w.days_since_contact)
            obs[i * OBS_PER_SLOT:(i + 1) * OBS_PER_SLOT] = [
                1.0,
                float(w.stage == AWAIT_TRIAGE),
                float(w.stage == AWAIT_TREAT),
                min(w.days_since_referral / HORIZON_DAYS, 1.0),
                min(w.days_since_contact / 14.0, 1.0),
                est,
                w.distance_km / 8.0,
                float(w.has_phone),
                w.ai_risk,
                min(w.missed / 3.0, 1.0),
            ]
        g = N_SLOTS * OBS_PER_SLOT
        obs[g:] = [
            min(self.day / HORIZON_DAYS, 1.0),
            self.decisions_left / ACTIONS_PER_DAY,
            self.chw_budget / CHW_WEEKLY_BUDGET,
            (self.day % DAYS_PER_WEEK) / (DAYS_PER_WEEK - 1),
        ]
        return obs

    def _info(self) -> dict:
        return {"day": self.day, "chw_budget": self.chw_budget, **self.stats.as_dict()}

    #  rendering
    def render_state(self) -> dict:
        """Serializable snapshot consumed by the three.js viewer (and the API)."""
        women = []
        for i, w in enumerate(self.women):
            women.append({
                "slot": i + 1, "active": w.active, "stage": int(w.stage),
                "engagement": round(float(w.engagement_est), 3),
                "distance_km": round(w.distance_km, 2), "has_phone": w.has_phone,
                "ai_risk": round(w.ai_risk, 3), "missed": w.missed,
                "days_since_contact": w.days_since_contact, "uid": w.uid,
            })
        slot, iv = decode_action(self.last_action)
        return {
            "t": self.t, "day": self.day, "horizon": HORIZON_DAYS,
            "decisions_left": self.decisions_left, "chw_budget": self.chw_budget,
            "chw_weekly_budget": CHW_WEEKLY_BUDGET,
            "last_action": {"id": self.last_action, "slot": slot + 1 if slot >= 0 else 0,
                            "intervention": iv, "text": describe_action(self.last_action)},
            "last_event": self.last_event, "last_reward": round(self.last_reward, 3),
            "episode_return": round(self.episode_return, 3),
            "women": women, "stats": self.stats.as_dict(),
        }

    def render(self):
        if self.render_mode == "ansi":
            s = self.render_state()
            lines = [f"Day {s['day']}/{HORIZON_DAYS} | CHW visits left {s['chw_budget']} | "
                     f"return {s['episode_return']:.2f}"]
            for w in s["women"]:
                if w["active"]:
                    stage = "triage" if w["stage"] == AWAIT_TRIAGE else "TREAT"
                    lines.append(f"  slot {w['slot']}: {stage:6s} eng={w['engagement']:.2f} "
                                 f"dist={w['distance_km']:.1f}km phone={int(w['has_phone'])}")
            return "\n".join(lines)
        self._render_frame()
        return None

    def _render_frame(self):
        if self.render_mode == "human" and self._viewer is not None:
            self._viewer.update(self.render_state())

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None


# ----------------------------------------------------------------------------
# Rule-based baseline (what a careful coordinator might do without RL)
# ----------------------------------------------------------------------------
def heuristic_policy(obs: np.ndarray) -> int:
    """Treat triage-positives first, then triage, then shore up the least engaged."""
    slots = obs[: N_SLOTS * OBS_PER_SLOT].reshape(N_SLOTS, OBS_PER_SLOT)
    budget = obs[N_SLOTS * OBS_PER_SLOT + 2]
    active = slots[:, 0] > 0.5
    eng = slots[:, 5]
    # 1) treatment for triage-positive women who are engaged enough to come
    cand = np.where(active & (slots[:, 2] > 0.5))[0]
    if len(cand):
        i = cand[np.argmax(eng[cand])]
        if eng[i] >= 0.45:
            return encode_action(int(i), TREAT)
        return encode_action(int(i), CHW_VISIT if budget > 0 else SMS)
    # 2) triage for engaged women awaiting triage
    cand = np.where(active & (slots[:, 1] > 0.5))[0]
    if len(cand):
        i = cand[np.argmax(eng[cand])]
        if eng[i] >= 0.45:
            return encode_action(int(i), TRIAGE)
        j = cand[np.argmin(eng[cand])]
        if slots[j, 7] > 0.5:
            return encode_action(int(j), SMS)
        if budget > 0:
            return encode_action(int(j), CHW_VISIT)
    return 0
