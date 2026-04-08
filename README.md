---
title: AI Negotiation OpenEnv
emoji: 🤝
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
tags:
  - openenv
  - negotiation
  - reinforcement-learning
  - real-world
---

# 🤝 AI Negotiation — OpenEnv Environment

A **real-world negotiation simulation** where AI agents learn to reach
price agreements through offer/counter-offer dialogues.
Fully compliant with the [OpenEnv](https://openenv.ai) specification.

---

## 🌍 Environment Description

The environment simulates three real-world negotiation scenarios:

| Task | Scenario | Difficulty | Max Rounds | Expected Score |
|------|----------|------------|------------|----------------|
| `easy_salary`     | Salary negotiation — wide ZOPA   | Easy   | 15 | 0.85 |
| `medium_vendor`   | Vendor contract — narrow ZOPA    | Medium | 10 | 0.60 |
| `hard_acquisition`| Company acquisition — very tight | Hard   | 6  | 0.35 |

The agent plays the **buyer** role, making or responding to offers
while trying to close within both parties' acceptable range (ZOPA).

---

## 🔵 Observation Space

```python
class Observation(BaseModel):
    buyer_budget: int          # Max the buyer will pay
    seller_min_price: int      # Min the seller will accept
    current_offer: int         # Offer currently on the table
    round_number: int          # Current round (1-based)
    max_rounds: int            # Max rounds allowed
    last_action: str | None    # Previous move taken
    negotiation_zone: str      # "overlap" | "no_overlap"
    progress_ratio: float      # Distance to ZOPA (0.0–1.0)
    task_id: str               # Active task identifier
```

## 🟠 Action Space

```python
class Action(BaseModel):
    move: str           # One of 7 discrete moves
    counter_value: int | None   # Required for counter_propose
```

**Available moves:**

| Move | Effect |
|------|--------|
| `increase_small` | +5 to current offer |
| `increase_large` | +15 to current offer |
| `decrease_small` | −5 from current offer |
| `decrease_large` | −15 from current offer |
| `accept` | Accept current offer (ends episode) |
| `reject` | Reject and walk away (ends episode) |
| `counter_propose` | Set offer to any custom value |

## 🟢 Reward Function

| Situation | Reward |
|-----------|--------|
| Entering ZOPA | +0.20 |
| Staying in ZOPA | +0.05 |
| Leaving ZOPA | −0.15 |
| Moving toward ZOPA | 0.0 – +0.10 |
| Accepted in ZOPA | **+0.50 – +1.00** (buyer savings ratio) |
| Accepted out of ZOPA | −0.30 to −0.40 |
| Rejected outright | −0.80 |
| Round timeout | −0.50 |
| Per-step time penalty | −0.01 |

Partial progress is always returned, providing signal throughout the trajectory.

---

## 🚀 Setup & Usage

### Local (Python)

```bash
git clone https://huggingface.co/spaces/vinayhosur2604/AI_negotiation
cd AI_negotiation
pip install -r requirements.txt
uvicorn app:app --reload --port 7860
# Open http://localhost:7860
```

### Docker

```bash
docker build -t ai-negotiation .
docker run -p 7860:7860 ai-negotiation
```

### API Usage

```python
import requests

BASE = "http://localhost:7860"

# Reset
obs = requests.post(f"{BASE}/reset", json={
    "task_id": "easy_salary",
    "session_id": "agent1"
}).json()

# Step
result = requests.post(f"{BASE}/step", json={
    "session_id": "agent1",
    "move": "increase_small"
}).json()

print(result["observation"])
print(result["reward"])
print(result["done"])
```

### OpenEnv Python Interface

```python
from negotiation_env import NegotiationEnv, Action

env = NegotiationEnv(task_id="medium_vendor", seed=42)
obs = env.reset()

while True:
    action = Action(move="increase_small")
    obs, reward, done, info = env.step(action)
    print(f"Offer: {obs.current_offer} | Reward: {reward.value:.3f}")
    if done:
        break

state = env.state()
```

### Baseline Inference

```bash
# Heuristic agent (no API key needed)
python baseline.py --heuristic

# LLM agent
export OPENAI_API_KEY=sk-...
python baseline.py --model gpt-4o-mini --episodes 5
```

---

## 📊 Baseline Scores (Heuristic Agent, seed=0–4)

| Task | Score |
|------|-------|
| easy_salary | 0.842 |
| medium_vendor | 0.591 |
| hard_acquisition | 0.318 |
| **Overall** | **0.584** |

---

## ✅ OpenEnv Validation

```bash
curl http://localhost:7860/validate
# Returns: {"status": "ok", "tasks": {...}}
```

---

## 📁 File Structure

```
.
├── negotiation_env.py   # Core environment (OpenEnv spec)
├── app.py               # FastAPI HTTP server
├── dashboard.html       # Interactive web UI
├── baseline.py          # Baseline inference script
├── openenv.yaml         # OpenEnv metadata
├── requirements.txt
├── Dockerfile
└── README.md
```
