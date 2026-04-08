import os
import sys
import json

sys.path.insert(0, os.path.dirname(__file__))

from disaster_env import DisasterEnv, FOOD, MEDICAL, RESCUE

TASKS = [
    {"id": "easy",   "difficulty": "easy",   "max_steps": 20, "success_threshold": 0.5},
    {"id": "medium", "difficulty": "medium", "max_steps": 25, "success_threshold": 0.3},
    {"id": "hard",   "difficulty": "hard",   "max_steps": 30, "success_threshold": 0.15},
]

RESOURCE_NAMES = {0: "food", 1: "medical", 2: "rescue"}
QTY_MAP = {0: 10, 1: 20, 2: 30}

SYSTEM_PROMPT = """You are an emergency response AI.
Return JSON:
{"resource_type":0-2,"zone_id":int,"quantity_index":0-2}
"""

# ─────────────────────────────
def get_client():
    try:
        from openai import OpenAI

        client = OpenAI(
            base_url=os.environ["API_BASE_URL"],
            api_key=os.environ["API_KEY"],
        )

        model = os.environ.get(
            "MODEL_NAME",
            "meta-llama/Meta-Llama-3-8B-Instruct"
        )

        return client, model

    except Exception as e:
        print(f"[FATAL] client error: {e}", flush=True)
        return None, None


# ─────────────────────────────
def fallback(env):
    for z in env.zones:
        if z.rescue_blocked:
            return (RESCUE, z.zone_id, 1)

    most_inj = max(env.zones, key=lambda z: z.injured)
    most_food = max(env.zones, key=lambda z: z.food_need)

    if most_inj.injured >= most_food.food_need:
        return (MEDICAL, most_inj.zone_id, 2)

    return (FOOD, most_food.zone_id, 2)


# ─────────────────────────────
def llm_agent(env, client, model):
    if client is None:
        return fallback(env)

    try:
        prompt = str(env.state())

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=50,
            temperature=0.0,
        )

        text = response.choices[0].message.content.strip()

        if "```" in text:
            text = text.split("```")[1].replace("json", "").strip()

        data = json.loads(text)

        return (
            int(data["resource_type"]),
            int(data["zone_id"]),
            int(data["quantity_index"]),
        )

    except Exception:
        return fallback(env)


# ─────────────────────────────
def run_task(task, client, model):
    env = DisasterEnv(task["difficulty"])
    env.reset()

    rewards = []
    step_count = 0
    done = False

    print(f"[START] task={task['id']} env=disaster model={model}", flush=True)

    while not done:
        action = llm_agent(env, client, model)

        obs, reward, done, info = env.step(action)

        rewards.append(reward)
        step_count += 1

        action_str = f"{RESOURCE_NAMES[action[0]]}-{action[1]}-{QTY_MAP[action[2]]}"

        print(
            f"[STEP] step={step_count} action={action_str} "
            f"reward={reward:.2f} done={str(done).lower()} error=null",
            flush=True,
        )

    total_reward = sum(rewards)
    score = total_reward / (task["max_steps"] * 0.6)
    score = max(0.0, min(1.0, score))

    success = score >= task["success_threshold"]

    rewards_str = ",".join(f"{r:.2f}" for r in rewards)

    print(
        f"[END] success={str(success).lower()} steps={step_count} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ─────────────────────────────
if __name__ == "__main__":
    try:
        client, model = get_client()

        for task in TASKS:
            run_task(task, client, model)

    except Exception as e:
        print(f"[FATAL] {e}", flush=True)
