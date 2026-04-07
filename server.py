"""
server.py — FastAPI server exposing the DisasterEnv as an HTTP API.

Fixed for OpenEnv validator compatibility:
- /reset accepts POST with NO body (difficulty defaults to "medium")
- /step accepts EITHER a single integer action OR the full tuple format
- All request bodies are fully optional
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Any
import uvicorn

from disaster_env import DisasterEnv
from models import Observation, Action, Reward, StepResult, GraderResult, ZoneState
from graders import EasyGrader, MediumGrader, HardGrader, rule_based_agent

app = FastAPI(
    title        = "Disaster Resource Allocation Environment",
    description  = "OpenEnv-compliant multi-zone disaster response simulation",
    version      = "1.0.0",
    docs_url     = "/docs",
    redoc_url    = "/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# Global environment instance
_env: Optional[DisasterEnv] = None
_last_action_info: dict = {}


# ── Request bodies — ALL fields optional ──────────────────────────

class ResetRequest(BaseModel):
    difficulty: Optional[str] = "medium"   # easy | medium | hard

class StepRequest(BaseModel):
    # Format 1: single integer action index (0-44) — used by OpenEnv validator
    action: Optional[int] = None
    # Format 2: tuple fields — used by human callers via /docs
    resource_type:  Optional[int] = None   # 0=food, 1=medical, 2=rescue
    zone_id:        Optional[int] = None   # target zone index
    quantity_index: Optional[int] = None   # 0=10, 1=20, 2=30 units

class GradeRequest(BaseModel):
    task_id:    Optional[str] = "medium"
    n_episodes: Optional[int] = 10


# ── Helpers ───────────────────────────────────────────────────────

def _env_to_observation(env: DisasterEnv) -> Observation:
    state = env.state()
    return Observation(
        zones=[
            ZoneState(
                zone_id        = z["zone_id"],
                name           = z["name"],
                disaster_type  = z["disaster_type"],
                population     = z["population"],
                injured        = z["injured"],
                food_need      = z["food_need"],
                rescue_blocked = z["rescue_blocked"],
                severity       = z["severity"],
            )
            for z in state["zones"]
        ],
        food_stock  = state["food_stock"],
        med_stock   = state["med_stock"],
        resc_stock  = state["resc_stock"],
        step        = state["step"],
        max_steps   = state["max_steps"],
        difficulty  = env.difficulty,
    )


def _decode_action(request: StepRequest, env: DisasterEnv):
    """
    Accept either:
      - a single integer (action index 0..num_actions-1)
      - resource_type + zone_id + quantity_index fields
    Returns tuple (resource_type, zone_id, quantity_index).
    """
    num_zones  = len(env.zones)
    qty_levels = 3  # 0=10, 1=20, 2=30

    # Format 1: single integer
    if request.action is not None:
        idx            = int(request.action)
        resource_type  = idx // (num_zones * qty_levels)
        remainder      = idx %  (num_zones * qty_levels)
        zone_id        = remainder // qty_levels
        quantity_index = remainder %  qty_levels
        return (resource_type, zone_id, quantity_index)

    # Format 2: explicit fields (default to 0 if missing)
    resource_type  = request.resource_type  if request.resource_type  is not None else 0
    zone_id        = request.zone_id        if request.zone_id        is not None else 0
    quantity_index = request.quantity_index if request.quantity_index is not None else 0
    return (resource_type, zone_id, quantity_index)


# ══════════════════════════════════════════════════════════════════
# Endpoints
# ══════════════════════════════════════════════════════════════════

@app.get("/", tags=["Health"])
def health_check():
    return {
        "status":      "ok",
        "environment": "disaster-resource-allocation",
        "version":     "1.0.0",
        "ready":       True,
    }


@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok"}


@app.post("/reset", tags=["OpenEnv API"])
def reset(request: Optional[ResetRequest] = None):
    """
    Start a fresh episode. Body is fully optional.
    POST with no body → difficulty defaults to 'medium'.
    """
    global _env

    difficulty = "medium"
    if request is not None and request.difficulty:
        difficulty = request.difficulty

    if difficulty not in ("easy", "medium", "hard"):
        raise HTTPException(400, f"difficulty must be easy/medium/hard, got: {difficulty}")

    _env = DisasterEnv(difficulty)
    _env.reset()

    try:
        return _env_to_observation(_env)
    except Exception:
        return _env.state()


@app.post("/step", tags=["OpenEnv API"])
def step(request: Optional[StepRequest] = None):
    """
    Apply one action. Body is optional — defaults to action=0.
    Accepts { "action": 17 } or { "resource_type": 1, "zone_id": 2, "quantity_index": 0 }.
    """
    global _env, _last_action_info

    # Auto-reset if needed
    if _env is None:
        _env = DisasterEnv("medium")
        _env.reset()

    if _env.done:
        _env = DisasterEnv(_env.difficulty)
        _env.reset()

    if request is None:
        request = StepRequest()

    try:
        action = _decode_action(request, _env)
        obs_dict, raw_reward, done, info = _env.step(action)
        _last_action_info = info
    except ValueError as e:
        raise HTTPException(400, str(e))

    RESOURCE_NAMES = {0: "food", 1: "medical", 2: "rescue"}
    resource_type, zone_id, quantity_index = action
    zones = _env.zones

    # Reward breakdown
    init_inj   = getattr(_env, "_initial_total_injured",  1)
    init_food  = getattr(_env, "_initial_total_food_need", 1)
    total_inj  = sum(z.injured   for z in zones)
    total_food = sum(z.food_need for z in zones)
    inj_cov    = 1.0 - (total_inj  / max(1, init_inj))
    food_cov   = 1.0 - (total_food / max(1, init_food))
    need_score = round(max(0, min(1, (inj_cov + food_cov) / 2)), 4)

    most_crit_id   = max(range(len(zones)), key=lambda i: zones[i].severity)
    priority_score = 1.0 if zone_id == most_crit_id else 0.3

    waste_penalty = 0.0
    if zone_id < len(zones):
        z = zones[zone_id]
        if resource_type == 0 and z.food_need == 0:                        waste_penalty = 0.1
        if resource_type == 1 and z.injured == 0:                          waste_penalty = 0.1
        if resource_type == 2 and not z.rescue_blocked and z.injured == 0: waste_penalty = 0.1

    try:
        reward_obj = Reward(
            total          = round(raw_reward, 4),
            need_score     = need_score,
            priority_score = round(priority_score, 4),
            waste_penalty  = round(waste_penalty, 4),
            resource_sent  = RESOURCE_NAMES.get(resource_type, "unknown"),
            zone_targeted  = zones[zone_id].name if zone_id < len(zones) else "unknown",
            units_sent     = info.get("quantity", 0),
        )
        return StepResult(
            observation = _env_to_observation(_env),
            reward      = reward_obj,
            done        = done,
            info        = info,
        )
    except Exception:
        return {
            "observation": _env.state(),
            "reward":      raw_reward,
            "done":        done,
            "info":        info,
        }


@app.get("/state", tags=["OpenEnv API"])
@app.post("/state", tags=["OpenEnv API"])
def state():
    """Return the current world state without advancing the episode."""
    if _env is None:
        raise HTTPException(400, "Call /reset first.")
    try:
        return _env_to_observation(_env)
    except Exception:
        return _env.state()


@app.get("/tasks", tags=["Tasks"])
def list_tasks():
    return {
        "tasks": [
            {
                "id":                "easy",
                "name":              "Basic Resource Allocation",
                "difficulty":        "easy",
                "num_zones":         3,
                "max_steps":         20,
                "success_threshold": 0.5,
                "description":       "3 zones, abundant resources, no escalation."
            },
            {
                "id":                "medium",
                "name":              "Triage Under Scarcity",
                "difficulty":        "medium",
                "num_zones":         5,
                "max_steps":         25,
                "success_threshold": 0.3,
                "description":       "5 zones, 5% escalation per step, 20% blocked zones."
            },
            {
                "id":                "hard",
                "name":              "Crisis Management",
                "difficulty":        "hard",
                "num_zones":         8,
                "max_steps":         30,
                "success_threshold": 0.15,
                "description":       "8 zones, 10% escalation, 40% blocked zones."
            }
        ]
    }


@app.post("/grade", response_model=GraderResult, tags=["Tasks"])
def grade(request: Optional[GradeRequest] = None):
    """Run the grader for a specific task and return the score."""
    if request is None:
        request = GradeRequest()

    grader_map = {
        "easy":   EasyGrader(),
        "medium": MediumGrader(),
        "hard":   HardGrader(),
    }
    task_id = request.task_id or "medium"
    if task_id not in grader_map:
        raise HTTPException(400, f"task_id must be easy/medium/hard, got: {task_id}")

    grader = grader_map[task_id]
    result = grader.run(rule_based_agent, n_episodes=request.n_episodes or 10)
    return result


# ══════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(f"\nStarting server on port {port}")
    print(f"API docs: http://localhost:{port}/docs\n")
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
