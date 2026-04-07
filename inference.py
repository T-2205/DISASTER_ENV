"""
inference.py — OpenEnv-compliant inference script for Disaster Resource Allocation.

STDOUT FORMAT (mandatory):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

CRITICAL RULES followed here:
    - API_BASE_URL and API_KEY are read from environment variables injected by the validator.
    - OpenAI client is initialised with base_url=API_BASE_URL, api_key=API_KEY.
    - No hardcoded keys, no fallback to other providers.
    - The environment server runs locally on port 7860 (same container).
    - Every exception is caught; [END] is always emitted.
"""

import os
import sys
import json
import time
import urllib.request
import urllib.error
from typing import List, Optional

# ── Read validator-injected variables ──────────────────────────────────────
# The validator sets API_BASE_URL and API_KEY in the container environment.
# Do NOT provide fallbacks that would bypass the proxy.
API_BASE_URL: str = os.environ.get("API_BASE_URL", "")
API_KEY: str = os.environ.get("API_KEY", "") or os.environ.get("HF_TOKEN", "")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")

# Local environment server — same container, always localhost:7860
ENV_BASE_URL: str = "http://localhost:7860"

TASK_NAME: str = "medium"
BENCHMARK: str = "disaster-resource-allocation"
MAX_STEPS: int = 25
SUCCESS_THRESHOLD: float = 0.30   # medium task threshold from openenv.yaml


# ── Stdout helpers ──────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── Local environment HTTP helpers ─────────────────────────────────────────

def _http_post(path: str, body: Optional[dict] = None, retries: int = 5) -> dict:
    """POST to the local FastAPI server with retry logic."""
    url = ENV_BASE_URL + path
    data = json.dumps(body or {}).encode("utf-8")
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            if attempt == retries - 1:
                raise RuntimeError(f"POST {path} failed after {retries} attempts: {exc}") from exc
            time.sleep(2 ** attempt)


def _http_get(path: str) -> dict:
    url = ENV_BASE_URL + path
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def wait_for_server(max_wait: int = 60) -> bool:
    """Poll the health endpoint until the server is ready."""
    deadline = time.time() + max_wait
    while time.time() < deadline:
        try:
            _http_get("/health")
            return True
        except Exception:
            time.sleep(2)
    return False


def env_reset(difficulty: str = "medium") -> dict:
    return _http_post("/reset", {"difficulty": difficulty})


def env_step(action: int) -> dict:
    return _http_post("/step", {"action": action})


# ── LLM client ─────────────────────────────────────────────────────────────

def build_openai_client():
    """
    Build an OpenAI-compatible client using the validator-injected credentials.
    Raises a clear RuntimeError if the required env vars are missing.
    """
    from openai import OpenAI  # imported here so import errors are catchable

    if not API_BASE_URL:
        raise RuntimeError(
            "API_BASE_URL environment variable is not set. "
            "The validator should inject this — check your submission config."
        )
    if not API_KEY:
        raise RuntimeError(
            "API_KEY (or HF_TOKEN) environment variable is not set. "
            "The validator should inject this — check your submission config."
        )

    # Ensure base_url ends without trailing slash (openai SDK requirement)
    base = API_BASE_URL.rstrip("/")

    return OpenAI(base_url=base, api_key=API_KEY)


SYSTEM_PROMPT = """You are an emergency resource coordinator managing disaster relief.

You control resource allocation across multiple disaster zones.
Each round you must choose ONE action encoded as a single integer.

Action encoding:
    action = resource_type * (num_zones * 3) + zone_id * 3 + quantity_index
    resource_type: 0=food, 1=medical, 2=rescue
    quantity_index: 0=10 units, 1=20 units, 2=30 units

For medium difficulty: 5 zones, so action range is 0–44.

Strategy:
1. If any zone has rescue_blocked=true → send rescue (resource_type=2) to it with quantity_index=1
2. Else → send medical (resource_type=1) to zone with highest injured count, quantity_index=2
3. If no injured → send food (resource_type=0) to zone with highest food_need, quantity_index=2

Reply with ONLY a single integer between 0 and 44. No explanation."""


def choose_action_llm(client, observation: dict, step: int) -> int:
    """Ask the LLM to pick an action. Falls back to rule-based if LLM fails."""
    zones = observation.get("zones", [])

    # Build a concise state description
    zone_lines = []
    for z in zones:
        zone_lines.append(
            f"  Zone {z['zone_id']} ({z['name']}): injured={z['injured']}, "
            f"food_need={z['food_need']}, severity={z['severity']:.2f}, "
            f"rescue_blocked={z['rescue_blocked']}"
        )
    state_text = (
        f"Step {step}/{MAX_STEPS}\n"
        f"Stocks: food={observation.get('food_stock')}, "
        f"medical={observation.get('med_stock')}, "
        f"rescue={observation.get('resc_stock')}\n"
        f"Zones:\n" + "\n".join(zone_lines) + "\n\nChoose action (0-44):"
    )

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": state_text},
            ],
            max_tokens=10,
            temperature=0.0,
        )
        text = (response.choices[0].message.content or "").strip()
        # Extract first integer found in response
        import re
        nums = re.findall(r"\d+", text)
        if nums:
            action = int(nums[0])
            if 0 <= action <= 44:
                return action
    except Exception as exc:
        print(f"[DEBUG] LLM call failed at step {step}: {exc}", flush=True)

    # Fallback: rule-based logic
    return _rule_based_action(observation)


def _rule_based_action(observation: dict) -> int:
    """Deterministic fallback — does not use the LLM proxy."""
    zones = observation.get("zones", [])
    num_zones = max(len(zones), 1)

    # 1. Unblock any blocked zone
    for z in zones:
        if z.get("rescue_blocked"):
            zone_id = z["zone_id"]
            # rescue, zone_id, quantity_index=1 (20 units)
            return 2 * (num_zones * 3) + zone_id * 3 + 1

    # 2. Help most injured
    most_injured = max(zones, key=lambda z: z.get("injured", 0), default=None)
    if most_injured and most_injured.get("injured", 0) > 0:
        zone_id = most_injured["zone_id"]
        return 1 * (num_zones * 3) + zone_id * 3 + 2  # medical, 30 units

    # 3. Help most hungry
    most_hungry = max(zones, key=lambda z: z.get("food_need", 0), default=None)
    if most_hungry:
        zone_id = most_hungry["zone_id"]
        return 0 * (num_zones * 3) + zone_id * 3 + 2  # food, 30 units

    return 0


# ── Main episode loop ───────────────────────────────────────────────────────

def run_episode(client) -> None:
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        # Wait for the local server to be ready
        if not wait_for_server(max_wait=60):
            raise RuntimeError("Local environment server did not become ready within 60 seconds.")

        # Reset environment
        observation = env_reset(difficulty="medium")

        done = False
        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            # Choose action via LLM (with rule-based fallback)
            action = choose_action_llm(client, observation, step)
            action_str = str(action)

            error_msg = None
            reward = 0.0
            try:
                result = env_step(action)
                # result is a StepResult: {observation, reward, done, info}
                raw_reward = result.get("reward", {})
                if isinstance(raw_reward, dict):
                    reward = float(raw_reward.get("total", 0.0))
                else:
                    reward = float(raw_reward)

                done = bool(result.get("done", False))
                observation = result.get("observation", observation)

            except Exception as exc:
                error_msg = str(exc)[:120]
                done = True

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action_str, reward=reward, done=done, error=error_msg)

        # Score = average reward over episode (normalised to [0,1])
        score = sum(rewards) / len(rewards) if rewards else 0.0
        score = max(0.0, min(1.0, score))
        success = score >= SUCCESS_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", flush=True)
        # Emit a final step so the log is never empty
        if not rewards:
            log_step(step=1, action="0", reward=0.0, done=True, error=str(exc)[:120])
            rewards = [0.0]
            steps_taken = 1

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ── Entry point ─────────────────────────────────────────────────────────────

def main() -> None:
    try:
        client = build_openai_client()
    except RuntimeError as exc:
        # Can't build client — still must emit valid [START] and [END]
        log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)
        log_step(step=1, action="0", reward=0.0, done=True, error=str(exc)[:120])
        log_end(success=False, steps=1, score=0.0, rewards=[0.0])
        sys.exit(0)   # exit 0 so the validator sees clean output, not a crash
    except Exception as exc:
        log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)
        log_step(step=1, action="0", reward=0.0, done=True, error=f"import_error: {exc}"[:120])
        log_end(success=False, steps=1, score=0.0, rewards=[0.0])
        sys.exit(0)

    run_episode(client)


if __name__ == "__main__":
    main()
