""""
===============================================================
Reads credentials from environment variables:
  API_BASE_URL   The API endpoint for the LLM  (default: https://api.openai.com/v1)
  MODEL_NAME     The model identifier           (default: Qwen/Qwen2.5-72B-Instruct)
  HF_TOKEN       Your Hugging Face / API key    (used as the API key)
Emits structured stdout logs in the required format:
  [START] task=<task_id> env=ai_negotiation model=<model_name>
  [STEP]  step=<n> action=<json> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>
"""

import os
import sys
import json

from negotiation_env import NegotiationEnv, Action, TASKS, grade_episode

# ── Credentials from environment variables ────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.environ.get("HF_TOKEN",     "")

# ── System prompt for the LLM agent ──────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert negotiation agent playing the BUYER role.
GOAL: Reach a deal where the agreed price is:
  - At or below the buyer's budget   (pay as LITTLE as possible)
  - At or above the seller's minimum (deal must be valid)
The ZOPA (Zone of Possible Agreement) is the range [seller_min_price, buyer_budget].
Your job is to find and close within the ZOPA quickly.
AVAILABLE MOVES:
  increase_small   +5 to current offer
  increase_large   +15 to current offer
  decrease_small   -5 from current offer
  decrease_large   -15 from current offer
  counter_propose  set offer to any value (must include counter_value)
  accept           finalise the deal at the current offer
  reject           walk away (only if negotiation_zone = "no_overlap")
STRATEGY:
1. Use counter_propose to jump directly to seller_min_price when possible.
2. Accept as soon as the offer is inside the ZOPA.
3. Never reject unless no deal is possible (no_overlap and no path forward).
4. Fewer rounds = higher score.
RESPOND ONLY with valid JSON, no other text:
{"move": "<move_name>", "counter_value": <integer or null>}
"""


def llm_agent(obs_dict: dict, client, model: str) -> Action:
    """Ask the LLM to choose the next action. Falls back to heuristic on failure."""
    user_msg = (
        "Current negotiation state:\n"
        + json.dumps(obs_dict, indent=2)
        + "\n\nChoose your next move. Reply ONLY with JSON."
    )
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.1,
            max_tokens=80,
        )
        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        data = json.loads(raw)
        move = data.get("move", "increase_small")
        counter = data.get("counter_value")
        if isinstance(counter, float):
            counter = int(counter)
        return Action(move=move, counter_value=counter)
    except Exception:
        return heuristic_agent(obs_dict)


def heuristic_agent(obs_dict: dict) -> Action:
    """Deterministic fallback agent — used when no API key is available."""
    offer      = obs_dict["current_offer"]
    budget     = obs_dict["buyer_budget"]
    seller_min = obs_dict["seller_min_price"]
    zone       = obs_dict["negotiation_zone"]
    rounds_left = obs_dict["max_rounds"] - obs_dict["round_number"]

    if zone == "overlap" and seller_min <= offer <= budget:
        if offer <= seller_min + 3 or rounds_left <= 1:
            return Action(move="accept")
        return Action(move="counter_propose", counter_value=max(seller_min, offer - 5))

    if offer < seller_min:
        gap = seller_min - offer
        if gap > 12:
            return Action(move="increase_large")
        if gap > 5:
            return Action(move="increase_small")
        return Action(move="counter_propose", counter_value=seller_min)

    if offer > budget:
        gap = offer - budget
        if gap > 12:
            return Action(move="decrease_large")
        if gap > 5:
            return Action(move="decrease_small")
        return Action(move="counter_propose", counter_value=budget)

    return Action(move="increase_small")


def run_task(task_id: str, agent_fn, model_name: str, seed: int = 42):
    """Run one episode for a task, emitting the required structured log lines."""
    env = NegotiationEnv(task_id=task_id, seed=seed)
    obs = env.reset()

    # ── [START] ──────────────────────────────────────────────────────────────
    print(f"[START] task={task_id} env=ai_negotiation model={model_name}", flush=True)

    done = False
    step_num = 0
    all_rewards = []
    last_reward = 0.0

    while not done:
        step_num += 1
        action = agent_fn(obs.model_dump())
        action_json = json.dumps({"move": action.move, "counter_value": action.counter_value})

        try:
            obs, reward, done, info = env.step(action)
            last_reward = reward.value
            all_rewards.append(round(reward.value, 4))
            error_field = "null"
        except Exception as e:
            error_field = str(e)
            done = True
            all_rewards.append(0.0)

        # ── [STEP] ───────────────────────────────────────────────────────────
        print(
            f"[STEP] step={step_num} "
            f"action={action_json} "
            f"reward={last_reward:.2f} "
            f"done={'true' if done else 'false'} "
            f"error={error_field}",
            flush=True,
        )

    # ── [END] ────────────────────────────────────────────────────────────────
    score   = grade_episode(all_rewards, obs, done)
    success = last_reward > 0.0
    rewards_str = ",".join(str(r) for r in all_rewards)

    print(
        f"[END] success={'true' if success else 'false'} "
        f"steps={step_num} "
        f"score={score:.3f} "
        f"rewards={rewards_str}",
        flush=True,
    )

    return score


def main():
    # ── Build the agent ───────────────────────────────────────────────────────
    if HF_TOKEN:
        try:
            from openai import OpenAI
            client   = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)
            agent_fn = lambda obs: llm_agent(obs, client, MODEL_NAME)
            model_label = MODEL_NAME
        except ImportError:
            agent_fn  = heuristic_agent
            model_label = "heuristic"
    else:
        agent_fn  = heuristic_agent
        model_label = "heuristic"

    # ── Run all 3 tasks ───────────────────────────────────────────────────────
    scores = []
    for task_id in TASKS:
        score = run_task(task_id, agent_fn, model_label, seed=42)
        scores.append(score)

    overall = sum(scores) / len(scores)
    print(f"overall_score={overall:.3f}", flush=True)


if __name__ == "__main__":
    main()
