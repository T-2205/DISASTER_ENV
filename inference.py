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
QTY_MAP        = {0: 10, 1: 20, 2: 30}

SYSTEM_PROMPT = """You are an emergency response coordinator AI.
You allocate disaster relief resources across multiple zones.
Each zone has injured people, food shortages, and varying severity.
Blocked zones need rescue teams before other resources can help.

Respond ONLY with a valid JSON object, nothing else:
{"resource_type": <0|1|2>, "zone_id": <int>, "quantity_index": <0|1|2>}

resource_type: 0=food, 1=medical, 2=rescue
quantity_index: 0=10 units, 1=20 units, 2=30 units

Strategy: unblock zones first, then help highest severity zones."""

# Read ALL required env vars from the competition proxy
API_BASE_URL = os.environ.get("API_BASE_URL", "").strip()
MODEL_NAME   = os.environ.get("MODEL_NAME",   "meta-llama/Meta-Llama-3-8B-Instruct").strip()
HF_TOKEN     = os.environ.get("HF_TOKEN",     "").strip()

print(f"[DEBUG] API_BASE_URL={API_BASE_URL!r}", flush=True)
print(f"[DEBUG] MODEL_NAME={MODEL_NAME!r}", flush=True)
print(f"[DEBUG] HF_TOKEN={'SET' if HF_TOKEN else 'MISSING'}", flush=True)


def get_client():
    from openai import OpenAI

    if not API_BASE_URL:
        print("[FATAL] API_BASE_URL environment variable is not set", flush=True)
        sys.exit(1)

    if not HF_TOKEN:
        print("[FATAL] HF_TOKEN environment variable is not set", flush=True)
        sys.exit(1)

    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN,
    )

    print(f"[DEBUG] Client created -> base_url={API_BASE_URL}", flush=True)
    return client


def build_prompt(env):
    state = env.state()
    lines = [
        f"Step {state['step']}/{state['max_steps']} | Difficulty: {env.difficulty}",
        f"Stock -> Food:{state['food_stock']} Medical:{state['med_stock']} Rescue:{state['resc_stock']}",
        "",
        "Zone states:",
    ]
    for z in state["zones"]:
        blocked = " [BLOCKED - send rescue first]" if z["rescue_blocked"] else ""
        lines.append(
            f"  Zone {z['zone_id']} ({z['name']}){blocked}: "
            f"disaster={z['disaster_type']}, severity={z['severity']:.2f}, "
            f"injured={z['injured']}, food_need={z['food_need']}"
        )
    lines += [
        "",
        'Respond ONLY with JSON: {"resource_type": 0-2, "zone_id": int, "quantity_index": 0-2}',
    ]
    return "\n".join(lines)


def fallback(env):
    for z in env.zones:
        if z.rescue_blocked:
            return (RESCUE, z.zone_id, 1)
    most_inj  = max(env.zones, key=lambda z: z.injured)
    most_food = max(env.zones, key=lambda z: z.food_need)
    if most_inj.injured >= most_food.food_need:
        return (MEDICAL, most_inj.zone_id, 2)
    return (FOOD, most_food.zone_id, 2)


def llm_agent(env, client):
    prompt = build_prompt(env)
    print("[DEBUG] Calling LLM via competition proxy...", flush=True)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        max_tokens=64,
        temperature=0.0,
    )

    text = response.choices[0].message.content.strip()
    print(f"[DEBUG] LLM raw response: {text!r}", flush=True)

    if "```" in text:
        text = text.split("```")[1].replace("json", "").strip()

    data = json.loads(text)
    resource_type  = int(data["resource_type"])
    zone_id        = int(data["zone_id"])
    quantity_index = int(data["quantity_index"])

    num_zones = len(env.zones)
    if resource_type not in (0, 1, 2):
        raise ValueError(f"Invalid resource_type: {resource_type}")
    if not (0 <= zone_id < num_zones):
        raise ValueError(f"Invalid zone_id: {zone_id}")
    if quantity_index not in (0, 1, 2):
        raise ValueError(f"Invalid quantity_index: {quantity_index}")

    return (resource_type, zone_id, quantity_index)


def run_task(task, client):
    env = DisasterEnv(task["difficulty"])
    env.reset()

    rewards    = []
    step_count = 0
    done       = False

    print(
        f"[START] task={task['id']} env=disaster-resource-allocation model={MODEL_NAME}",
        flush=True,
    )

    while not done:
        try:
            action = llm_agent(env, client)
        except Exception as e:
            print(f"[WARN] LLM parse failed ({e}) -- using fallback", flush=True)
            action = fallback(env)

        resource_type, zone_id, quantity_index = action
        obs, reward, done, info = env.step(action)

        rewards.append(reward)
        step_count += 1

        action_str = f"{RESOURCE_NAMES[resource_type]}-{zone_id}-{QTY_MAP[quantity_index]}"

        print(
            f"[STEP] step={step_count} action={action_str} "
            f"reward={reward:.2f} done={str(done).lower()} error=null",
            flush=True,
        )

    total_reward = sum(rewards)
    score = max(0.0, min(1.0, total_reward / (task["max_steps"] * 0.6)))
    success = score >= task["success_threshold"]
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)

    print(
        f"[END] success={str(success).lower()} steps={step_count} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )

    return score


if __name__ == "__main__":
    client = get_client()

    all_scores = []
    for task in TASKS:
        score = run_task(task, client)
        all_scores.append(score)

    overall = sum(all_scores) / len(all_scores)
    print(f"[SUMMARY] overall_score={overall:.3f}", flush=True)
