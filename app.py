"""
OpenEnv HTTP API — AI Negotiation Environment
Flask-based (stdlib-compatible), exposes step()/reset()/state() over REST + dashboard.
"""
import json
import os
from flask import Flask, request, jsonify, send_file

from negotiation_env import NegotiationEnv, Action, TASKS, grade_episode

app = Flask(__name__)

_sessions = {}
_session_history = {}


def get_env(session_id):
    if session_id not in _sessions:
        return None, jsonify({"error": f"Session '{session_id}' not found. Call /reset first."}), 404
    return _sessions[session_id], None, None


@app.route("/", methods=["GET"])
def dashboard():
    return send_file("dashboard.html")


@app.route("/tasks", methods=["GET"])
def list_tasks():
    return jsonify({"tasks": TASKS})


@app.route("/reset", methods=["POST"])
def reset():
    body = request.get_json(force=True, silent=True) or {}
    task_id = body.get("task_id", "easy_salary")
    seed = body.get("seed", None)
    session_id = body.get("session_id", "default")

    if task_id not in TASKS:
        return jsonify({"error": f"Unknown task_id '{task_id}'"}), 400

    env = NegotiationEnv(task_id=task_id, seed=seed)
    obs = env.reset()
    _sessions[session_id] = env
    _session_history[session_id] = []

    return jsonify({
        "session_id": session_id,
        "observation": obs.model_dump(),
        "task": TASKS[task_id],
    })


@app.route("/step", methods=["POST"])
def step():
    body = request.get_json(force=True, silent=True) or {}
    session_id = body.get("session_id", "default")
    move = body.get("move")
    counter_value = body.get("counter_value")

    if not move:
        return jsonify({"error": "move is required"}), 400

    env, err_resp, code = get_env(session_id)
    if err_resp:
        return err_resp, code

    action = Action(move=move, counter_value=counter_value)
    try:
        obs, reward, done, info = env.step(action)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 400

    history = _session_history.get(session_id, [])
    history.append({
        "round": obs.round_number - 1,
        "move": move,
        "offer": obs.current_offer,
        "reward": reward.value,
        "reason": reward.reason,
    })
    _session_history[session_id] = history

    result = {
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info,
    }

    if done:
        result["episode_score"] = grade_episode(info["episode_rewards"], obs, done)
        result["history"] = history

    return jsonify(result)


@app.route("/state/<session_id>", methods=["GET"])
def state(session_id):
    env, err_resp, code = get_env(session_id)
    if err_resp:
        return err_resp, code
    return jsonify({
        "state": env.state(),
        "history": _session_history.get(session_id, []),
    })


@app.route("/history/<session_id>", methods=["GET"])
def history(session_id):
    return jsonify({"history": _session_history.get(session_id, [])})


@app.route("/validate", methods=["GET"])
def validate():
    results = {}
    for task_id in TASKS:
        env = NegotiationEnv(task_id=task_id, seed=42)
        obs = env.reset()
        action = Action(move="increase_small")
        obs2, reward, done, info = env.step(action)
        results[task_id] = {
            "reset_ok": True,
            "step_ok": True,
            "state_ok": bool(env.state()),
            "reward_in_range": -1.0 <= reward.value <= 1.0,
            "obs_fields": list(obs2.model_dump().keys()),
        }
    return jsonify({"status": "ok", "tasks": results})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)
