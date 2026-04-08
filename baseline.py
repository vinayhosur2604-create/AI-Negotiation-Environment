"""
Baseline Inference Script
=========================
Runs an LLM agent (via OpenAI-compatible API) against all 3 tasks
and prints reproducible baseline scores.

Usage:
    export OPENAI_API_KEY=sk-...
    export OPENAI_BASE_URL=https://api.openai.com/v1   # optional
    python baseline.py

The script can also run a built-in heuristic agent (no API key needed):
    python baseline.py --heuristic
"""

import os
import json
import argparse
import random
from typing import Optional

from negotiation_env import NegotiationEnv, Action, TASKS, grade_episode


# ─── Heuristic Baseline Agent ─────────────────────────────────────────────────

def heuristic_agent(obs_dict: dict) -> Action:
    """
    Rule-based agent used as a no-API baseline.
    Strategy: move offer into ZOPA, then accept.
    """
    offer = obs_dict["current_offer"]
    buyer_budget = obs_dict["buyer_budget"]
    seller_min = obs_dict["seller_min_price"]
    zone = obs_dict["negotiation_zone"]
    rounds_left = obs_dict["max_rounds"] - obs_dict["round_number"]

    # If we're in the ZOPA and running out of time → accept
    if zone == "overlap" and seller_min <= offer <= buyer_budget:
        if rounds_left <= 2:
            return Action(move="accept")
        # Try to squeeze a bit more
        if offer > seller_min + 5:
            return Action(move="decrease_small")
        return Action(move="accept")

    # Offer too low — increase
    if offer < seller_min:
        gap = seller_min - offer
        if gap > 10:
            return Action(move="increase_large")
        return Action(move="increase_small")

    # Offer too high — decrease
    if offer > buyer_budget:
        gap = offer - buyer_budget
        if gap > 10:
            return Action(move="decrease_large")
        return Action(move="decrease_small")

    # In budget but below seller min — increase
    return Action(move="increase_small")


# ─── LLM Agent ────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a negotiation agent. You play the BUYER role.
Your goal: reach a deal where the price is within the buyer's budget AND above the seller's minimum.

Rules:
- You want to pay as LITTLE as possible (but the deal must succeed).
- Available moves: increase_small (+5), increase_large (+15), decrease_small (-5),
  decrease_large (-15), accept, reject, counter_propose (specify counter_value).
- Never reject unless there is truly no ZOPA (negotiation_zone = "no_overlap" and no path forward).

Respond ONLY with valid JSON: {"move": "<move>", "counter_value": <int or null>}
"""

def llm_agent(obs_dict: dict, client, model: str) -> Action:
    """Query an LLM to choose the next action."""
    prompt = f"""Current negotiation state:
{json.dumps(obs_dict, indent=2)}

Choose your next move. Respond ONLY with JSON like:
{{"move": "increase_small", "counter_value": null}}
"""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=100,
    )
    raw = response.choices[0].message.content.strip()
    try:
        data = json.loads(raw)
        return Action(move=data["move"], counter_value=data.get("counter_value"))
    except Exception:
        # Fallback to heuristic if LLM returns malformed output
        return heuristic_agent(obs_dict)


# ─── Runner ───────────────────────────────────────────────────────────────────

def run_task(task_id: str, agent_fn, n_episodes: int = 5, seed_offset: int = 0) -> dict:
    scores = []
    for ep in range(n_episodes):
        env = NegotiationEnv(task_id=task_id, seed=ep + seed_offset)
        obs = env.reset()
        done = False
        episode_rewards = []

        while not done:
            action = agent_fn(obs.model_dump())
            obs, reward, done, info = env.step(action)
            episode_rewards.append(reward.value)

        score = grade_episode(info["episode_rewards"], obs, done)
        scores.append(score)
        print(f"  Episode {ep+1}: score={score:.3f}  "
              f"deal={'✓' if reward.value > 0 else '✗'}  "
              f"rounds={obs.round_number-1}/{obs.max_rounds}")

    mean_score = sum(scores) / len(scores)
    return {
        "task_id": task_id,
        "difficulty": TASKS[task_id]["difficulty"],
        "n_episodes": n_episodes,
        "scores": scores,
        "mean_score": round(mean_score, 4),
        "expected_score": TASKS[task_id]["expected_score"],
    }


def main():
    parser = argparse.ArgumentParser(description="OpenEnv Negotiation Baseline")
    parser.add_argument("--heuristic", action="store_true", help="Use heuristic agent (no API key needed)")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model name")
    parser.add_argument("--episodes", type=int, default=5, help="Episodes per task")
    args = parser.parse_args()

    print("=" * 60)
    print("  AI Negotiation OpenEnv — Baseline Inference")
    print("=" * 60)

    if args.heuristic:
        print("  Agent: Heuristic (rule-based)")
        agent_fn = heuristic_agent
    else:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("  OPENAI_API_KEY not set. Falling back to heuristic agent.")
            agent_fn = heuristic_agent
        else:
            try:
                from openai import OpenAI
                base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
                client = OpenAI(api_key=api_key, base_url=base_url)
                print(f"  Agent: LLM ({args.model})")
                agent_fn = lambda obs: llm_agent(obs, client, args.model)
            except ImportError:
                print("  openai package not found. Falling back to heuristic agent.")
                agent_fn = heuristic_agent

    print()
    all_results = []
    for task_id in TASKS:
        print(f"── Task: {task_id} ({TASKS[task_id]['difficulty']}) ──")
        result = run_task(task_id, agent_fn, n_episodes=args.episodes)
        all_results.append(result)
        print(f"  Mean Score: {result['mean_score']:.4f}  "
              f"(expected ≈ {result['expected_score']})\n")

    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    overall = sum(r["mean_score"] for r in all_results) / len(all_results)
    for r in all_results:
        bar = "█" * int(r["mean_score"] * 20)
        print(f"  {r['task_id']:25s} {r['mean_score']:.4f}  {bar}")
    print(f"\n  Overall mean score: {overall:.4f}")
    print("=" * 60)

    # Save results
    with open("baseline_results.json", "w") as f:
        json.dump({"overall": overall, "tasks": all_results}, f, indent=2)
    print("  Results saved to baseline_results.json")


if __name__ == "__main__":
    main()
