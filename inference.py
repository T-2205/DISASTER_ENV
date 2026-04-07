"""
inference.py — Baseline inference script for the OpenEnv hackathon.

REQUIRED by the competition spec. Must:
  - Use the OpenAI client
  - Read credentials from environment variables
  - Produce structured [START] / [STEP] / [END] stdout logs
  - Run against all 3 tasks and produce reproducible scores
  - Complete in under 20 minutes

Environment variables required:
    API_BASE_URL  — the LLM API endpoint
    MODEL_NAME    — model identifier (e.g. "meta-llama/Llama-3-8b-instruct")
    HF_TOKEN      — your Hugging Face API key

Usage:
    export API_BASE_URL="https://api-inference.huggingface.co/v1"
    export MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct"
    export HF_TOKEN="hf_your_token_here"
    python inference.py
"""

import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(__file__))

from disaster_env import DisasterEnv, FOOD, MEDICAL, RESCUE

# ── OpenAI client setup ───────────────────────────────────────────
try:
    from openai import OpenAI
except ImportError:
    print("pip install openai  ← run this first", file=sys.stderr)
    sys.exit(1)

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "meta-llama/Meta-Llama-3-8B-Instruct")
HF_TOKEN     = os.environ.get("HF_TOKEN",     "")

client = OpenAI(
    base_url = API_BASE_URL,
    api_key  = HF_TOKEN or "dummy",   # fallback for local testing
)

# ── Task definitions ──────────────────────────────────────────────
TASKS = [
    {"id": "easy",   "difficulty": "easy",   "max_steps": 20, "success_threshold": 0.5},
    {"id": "medium", "difficulty": "medium", "max_steps": 25, "success_threshold": 0.3},
    {"id": "hard",   "difficulty": "hard",   "max_steps": 30, "success_threshold": 0.15},
]


# ══════════════════════════════════════════════════════════════════
# LLM agent — asks the model what action to take
# ══════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are an emergency response coordinator AI.
You must allocate limited disaster relief resources across multiple affected zones.
Each zone has injured people needing medical aid, food shortages, and varying severity.
Some zones may be blocked and need rescue teams before other resources can help.

Your goal: maximise the total reward by helping the most critical zones first.

You will receive the current state and must respond with ONLY a JSON object:
{"resource_type": <0-2>, "zone_id": <int>, "quantity_index": <0-2>}

resource_type: 0=food, 1=medical, 2=rescue
quantity_index: 0=10 units, 1=20 units, 2=30 units

Rules:
- If any zone is rescue_blocked, send rescue there first
- Otherwise help the zone with highest severity or most injured/hungry
- Do not send food to zones with food_need=0
- Do not send medical to zones with injured=0"""


def build_state_prompt(env: DisasterEnv) -> str:
    """Convert current env state into a text prompt for the LLM."""
    lines = [
        f"Step {env.step_count}/{env.cfg['max_steps']} | Difficulty: {env.difficulty.upper()}",
        f"Stock — Food: {env.food_stock}  Medical: {env.med_stock}  Rescue: {env.resc_stock}",
        "",
        "Zone states:",
    ]
    for z in env.zones:
        blocked = " [BLOCKED]" if z.rescue_blocked else ""
        lines.append(
            f"  Zone {z.zone_id} ({z.name}){blocked}: "
            f"disaster={z.disaster_type}, severity={z.severity:.2f}, "
            f"injured={z.injured}, food_need={z.food_need}, pop={z.population}"
        )
    lines.append("")
    lines.append("Respond with JSON only: {\"resource_type\": X, \"zone_id\": Y, \"quantity_index\": Z}")
    return "\n".join(lines)


def llm_agent(env: DisasterEnv) -> tuple:
    """
    Call the LLM to pick an action. Falls back to rule-based if LLM fails.
    Returns action tuple (resource_type, zone_id, quantity_index).
    """
    try:
        prompt   = build_state_prompt(env)
        response = client.chat.completions.create(
            model       = MODEL_NAME,
            messages    = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            max_tokens  = 64,
            temperature = 0.0,   # deterministic
        )
        text = response.choices[0].message.content.strip()

        # Parse JSON response
        # Handle cases where model wraps in markdown
        if "```" in text:
            text = text.split("```")[1].replace("json", "").strip()

        data = json.loads(text)
        resource_type  = int(data["resource_type"])
        zone_id        = int(data["zone_id"])
        quantity_index = int(data["quantity_index"])

        # Validate bounds
        if resource_type not in (0, 1, 2):
            raise ValueError(f"Invalid resource_type: {resource_type}")
        if not (0 <= zone_id < len(env.zones)):
            raise ValueError(f"Invalid zone_id: {zone_id}")
        if quantity_index not in (0, 1, 2):
            raise ValueError(f"Invalid quantity_index: {quantity_index}")

        return (resource_type, zone_id, quantity_index)

    except Exception as e:
        # Fallback to rule-based agent if LLM fails
        return _rule_fallback(env)


def _rule_fallback(env: DisasterEnv) -> tuple:
    """Rule-based fallback in case LLM call fails."""
    for z in env.zones:
        if z.rescue_blocked:
            return (RESCUE, z.zone_id, 1)
    most_inj  = max(env.zones, key=lambda z: z.injured)
    most_food = max(env.zones, key=lambda z: z.food_need)
    if most_inj.injured >= most_food.food_need:
        return (MEDICAL, most_inj.zone_id, 2)
    return (FOOD, most_food.zone_id, 2)


# ══════════════════════════════════════════════════════════════════
# Episode runner with required log format
# ══════════════════════════════════════════════════════════════════

RESOURCE_NAMES = {0: "food", 1: "medical", 2: "rescue"}
QTY_MAP        = {0: 10, 1: 20, 2: 30}


def run_task(task: dict) -> dict:
    """
    Run one task and emit structured logs in the exact format required.

    Log format (MUST NOT deviate — judges parse this):
        [START] {"task_id": ..., "difficulty": ..., "max_steps": ...}
        [STEP]  {"step": ..., "action": ..., "reward": ..., "done": ...}
        [END]   {"task_id": ..., "score": ..., "passed": ..., "total_reward": ...}
    """
    task_id    = task["id"]
    difficulty = task["difficulty"]

    env   = DisasterEnv(difficulty)
    obs   = env.reset()

    # [START] log
    start_payload = {
        "task_id":    task_id,
        "difficulty": difficulty,
        "max_steps":  task["max_steps"],
        "num_zones":  len(env.zones),
        "timestamp":  time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    print(f"[START] {json.dumps(start_payload)}", flush=True)

    total_reward = 0.0
    done         = False

    while not done:
        action = llm_agent(env)
        resource_type, zone_id, qty_idx = action

        obs_dict, reward, done, info = env.step(action)
        total_reward += reward

        # [STEP] log
        step_payload = {
            "step":          env.step_count,
            "action":        {
                "resource_type":  resource_type,
                "resource_name":  RESOURCE_NAMES.get(resource_type, "unknown"),
                "zone_id":        zone_id,
                "zone_name":      env.zones[zone_id].name if zone_id < len(env.zones) else "unknown",
                "quantity_index": qty_idx,
                "units":          QTY_MAP.get(qty_idx, 0),
            },
            "reward":        round(reward, 4),
            "total_reward":  round(total_reward, 4),
            "done":          done,
            "stock": {
                "food":   env.food_stock,
                "med":    env.med_stock,
                "rescue": env.resc_stock,
            },
        }
        print(f"[STEP] {json.dumps(step_payload)}", flush=True)

    # Calculate final score (normalised to 0.0–1.0)
    score  = round(min(1.0, max(0.0, total_reward / (task["max_steps"] * 0.6))), 4)
    passed = score >= task["success_threshold"]
    zones_cleared = sum(1 for z in env.zones if z.injured == 0 and z.food_need == 0)

    # [END] log
    end_payload = {
        "task_id":      task_id,
        "score":        score,
        "passed":       passed,
        "total_reward": round(total_reward, 4),
        "steps_taken":  env.step_count,
        "zones_cleared": zones_cleared,
        "success_threshold": task["success_threshold"],
    }
    print(f"[END] {json.dumps(end_payload)}", flush=True)

    return end_payload


# ══════════════════════════════════════════════════════════════════
# Main — run all 3 tasks
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    results = []

    for task in TASKS:
        result = run_task(task)
        results.append(result)
        print(flush=True)   # blank line between tasks

    # Final summary
    all_passed   = all(r["passed"] for r in results)
    overall_score = round(sum(r["score"] for r in results) / len(results), 4)

    summary = {
        "overall_score": overall_score,
        "all_passed":    all_passed,
        "tasks":         results,
    }
    print(f"\n[SUMMARY] {json.dumps(summary)}", flush=True)

    sys.exit(0 if all_passed else 1)
