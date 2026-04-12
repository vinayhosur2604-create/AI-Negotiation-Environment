import random
import time
from dataclasses import dataclass, asdict
from typing import Optional


# ─── Typed Models ────────────────────────────────────────────────────────────

@dataclass
class Observation:
    buyer_budget: int
    seller_min_price: int
    current_offer: int
    round_number: int
    max_rounds: int
    last_action: Optional[str]
    negotiation_zone: str
    progress_ratio: float
    task_id: str

    def model_dump(self):
        return asdict(self)


@dataclass
class Action:
    move: str
    counter_value: Optional[int] = None


@dataclass
class Reward:
    value: float
    reason: str
    partial_credit: float

    def model_dump(self):
        return asdict(self)


# ─── Tasks ───────────────────────────────────────────────────────────────────

TASKS = {
    "easy_salary": {
        "id": "easy_salary",
        "description": "Negotiate a starting salary. Wide ZOPA, generous rounds.",
        "difficulty": "easy",
        "buyer_budget_range": (70, 100),
        "seller_min_range": (50, 65),
        "max_rounds": 15,
        "starting_offer_bias": "low",
        "expected_score": 0.85,
    },
    "medium_vendor": {
        "id": "medium_vendor",
        "description": "Negotiate a vendor contract. Narrow ZOPA, moderate rounds.",
        "difficulty": "medium",
        "buyer_budget_range": (55, 75),
        "seller_min_range": (50, 70),
        "max_rounds": 10,
        "starting_offer_bias": "mid",
        "expected_score": 0.60,
    },
    "hard_acquisition": {
        "id": "hard_acquisition",
        "description": "Company acquisition negotiation. Very narrow ZOPA, few rounds.",
        "difficulty": "hard",
        "buyer_budget_range": (100, 110),
        "seller_min_range": (95, 108),
        "max_rounds": 6,
        "starting_offer_bias": "high",
        "expected_score": 0.35,
    },
}


# ─── Environment ─────────────────────────────────────────────────────────────

class NegotiationEnv:
    metadata = {
        "name": "AI Negotiation Environment",
        "version": "1.0.0",
        "author": "vinayhosur2604",
        "tasks": list(TASKS.keys()),
        "action_space": ["increase_small", "increase_large", "decrease_small",
                         "decrease_large", "accept", "reject", "counter_propose"],
        "reward_range": (-1.0, 1.0),
    }

    def __init__(self, task_id: str = "easy_salary", seed: Optional[int] = None):
        if task_id not in TASKS:
            raise ValueError(f"Unknown task_id '{task_id}'. Choose from: {list(TASKS.keys())}")
        self.task_id = task_id
        self.task = TASKS[task_id]
        self._rng = random.Random(seed)
        self._obs = None
        self._done = False
        self._episode_rewards = []
        self._start_time = time.time()
        self.buyer_budget = 0
        self.seller_min = 0
        self.current_offer = 0
        self.max_rounds = self.task["max_rounds"]
        self.round_number = 1

    def reset(self) -> Observation:
        t = self.task
        self.buyer_budget = self._rng.randint(*t["buyer_budget_range"])
        self.seller_min = self._rng.randint(*t["seller_min_range"])
        self.max_rounds = t["max_rounds"]
        self.round_number = 1
        self._done = False
        self._episode_rewards = []
        self._start_time = time.time()

        bias = t["starting_offer_bias"]
        if bias == "low":
            self.current_offer = self.seller_min - self._rng.randint(5, 15)
        elif bias == "high":
            self.current_offer = self.buyer_budget + self._rng.randint(5, 15)
        else:
            mid = (self.buyer_budget + self.seller_min) // 2
            self.current_offer = mid + self._rng.randint(-5, 5)

        self.current_offer = max(1, self.current_offer)
        self._obs = self._make_obs()
        return self._obs

    def step(self, action: Action):
        if self._done:
            raise RuntimeError("Episode is done. Call reset().")
        if self._obs is None:
            raise RuntimeError("Call reset() before step().")

        move = action.move
        reward_value, reason, partial = self._apply_move(move, action.counter_value)

        self.round_number += 1
        if self.round_number > self.max_rounds and not self._done:
            reward_value = -0.5
            reason = "Ran out of rounds without agreement"
            partial = self._progress()
            self._done = True

        reward = Reward(value=reward_value, reason=reason, partial_credit=partial)
        self._episode_rewards.append(reward_value)
        self._obs = self._make_obs(last_action=move)

        info = {
            "episode_rewards": list(self._episode_rewards),
            "elapsed_seconds": round(time.time() - self._start_time, 2),
            "buyer_budget": self.buyer_budget,
            "seller_min": self.seller_min,
        }
        return self._obs, reward, self._done, info

    def state(self) -> dict:
        return {
            "task_id": self.task_id,
            "buyer_budget": self.buyer_budget,
            "seller_min": self.seller_min,
            "current_offer": self.current_offer,
            "round_number": self.round_number,
            "max_rounds": self.max_rounds,
            "done": self._done,
            "episode_rewards": list(self._episode_rewards),
            "zopa_exists": self.seller_min <= self.buyer_budget,
        }

    def _apply_move(self, move, counter_value):
        prev = self.current_offer
        if move == "increase_small":
            self.current_offer += 5
            return self._step_reward(prev, "Small increase applied")
        elif move == "increase_large":
            self.current_offer += 15
            return self._step_reward(prev, "Large increase applied")
        elif move == "decrease_small":
            self.current_offer -= 5
            return self._step_reward(prev, "Small decrease applied")
        elif move == "decrease_large":
            self.current_offer -= 15
            return self._step_reward(prev, "Large decrease applied")
        elif move == "counter_propose":
            if counter_value is None:
                return -0.2, "counter_propose requires counter_value", self._progress()
            self.current_offer = counter_value
            return self._step_reward(prev, f"Counter-proposed {counter_value}")
        elif move == "accept":
            return self._handle_accept()
        elif move == "reject":
            self._done = True
            return -0.8, "Negotiation rejected outright", self._progress()
        else:
            return -0.1, f"Unknown move '{move}'", self._progress()

    def _handle_accept(self):
        self._done = True
        offer = self.current_offer
        if offer < self.seller_min:
            return -0.3, f"Accepted below seller minimum ({offer} < {self.seller_min})", 0.2
        if offer > self.buyer_budget:
            return -0.4, f"Accepted above buyer budget ({offer} > {self.buyer_budget})", 0.1
        zopa = self.buyer_budget - self.seller_min
        if zopa <= 0:
            return 1.0, "Deal struck despite zero/negative ZOPA!", 1.0
        efficiency = (self.buyer_budget - offer) / zopa
        score = 0.5 + 0.5 * efficiency
        round_eff = 1 - (self.round_number / self.max_rounds)
        final = min(1.0, score + 0.1 * round_eff)
        return final, f"Deal at {offer} (ZOPA {self.seller_min}–{self.buyer_budget})", final

    def _step_reward(self, prev, reason):
        progress = self._progress()
        was_in = self.seller_min <= prev <= self.buyer_budget
        now_in = self.seller_min <= self.current_offer <= self.buyer_budget
        if not was_in and now_in:
            r = 0.2
        elif was_in and not now_in:
            r = -0.15
        elif now_in:
            r = 0.05
        else:
            pd = self._dist_to_zopa(prev)
            cd = self._dist_to_zopa(self.current_offer)
            r = 0.1 * (pd - cd) / max(1, pd + cd)
        r = round(max(-1.0, min(1.0, r - 0.01)), 4)
        return r, reason, progress

    def _dist_to_zopa(self, offer):
        if offer < self.seller_min:
            return self.seller_min - offer
        if offer > self.buyer_budget:
            return offer - self.buyer_budget
        return 0

    def _progress(self):
        dist = self._dist_to_zopa(self.current_offer)
        w = max(1, self.buyer_budget - self.seller_min)
        return round(max(0.0, 1.0 - dist / (w + dist)), 4)

    def _make_obs(self, last_action=None):
        zone = "overlap" if self.seller_min <= self.buyer_budget else "no_overlap"
        return Observation(
            buyer_budget=self.buyer_budget,
            seller_min_price=self.seller_min,
            current_offer=self.current_offer,
            round_number=self.round_number,
            max_rounds=self.max_rounds,
            last_action=last_action,
            negotiation_zone=zone,
            progress_ratio=self._progress(),
            task_id=self.task_id,
        )


def grade_episode(episode_rewards, final_obs, done) -> float:
    if not episode_rewards:
        return 0.0
    final_reward = episode_rewards[-1]
    total = sum(episode_rewards)
    n = len(episode_rewards)
    norm_total = (total + n) / (2 * n + 1e-9)
    score = 0.7 * max(0.0, final_reward) + 0.3 * norm_total
    return round(min(1.0, max(0.0, score)), 4)
