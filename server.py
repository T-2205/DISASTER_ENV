"""
server.py — FastAPI server exposing the DisasterEnv as an HTTP API.

This is required for Hugging Face Spaces deployment.
The server exposes reset(), step(), and state() as HTTP endpoints
so the automated validator and inference scripts can call them.

Endpoints:
    GET  /          — health check (returns 200)
    POST /reset     — start new episode, returns Observation
    POST /step      — apply action, returns StepResult
    GET  /state     — current world state
    GET  /tasks     — list all tasks with grader info
    POST /grade     — run a grader and return GraderResult

Run locally:
    python server.py

Run with uvicorn:
    uvicorn server:app --host 0.0.0.0 --port 7860
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
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

# Global environment instance (one per server process)
_env: Optional[DisasterEnv] = None
_last_action_info: dict = {}


# ── Request bodies ────────────────────────────────────────────────

class ResetRequest(BaseModel):
    difficulty: str = "medium"   # easy | medium | hard

class StepRequest(BaseModel):
    resource_type:  int           # 0=food, 1=medical, 2=rescue
    zone_id:        int           # target zone index
    quantity_index: int           # 0=10, 1=20, 2=30 units

class GradeRequest(BaseModel):
    task_id:    str = "medium"   # easy | medium | hard
    n_episodes: int = 10


# ── Helpers ───────────────────────────────────────────────────────

def _env_to_observation(env: DisasterEnv) -> Observation:
    """Convert internal env state to typed Observation model."""
    state = env.state()
    return Observation(
        zones = [
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


# ══════════════════════════════════════════════════════════════════
# Endpoints
# ══════════════════════════════════════════════════════════════════

@app.get("/", tags=["Health"])
def health_check():
    """Health check — automated validator pings this endpoint."""
    return {
        "status":      "ok",
        "environment": "disaster-resource-allocation",
        "version":     "1.0.0",
        "ready":       True,
    }


@app.post("/reset", response_model=Observation, tags=["OpenEnv API"])
def reset(request: ResetRequest):
    """
    Start a fresh episode.
    Resets all zones, refills resources, returns initial observation.
    """
    global _env
    if request.difficulty not in ("easy", "medium", "hard"):
        raise HTTPException(400, f"difficulty must be easy/medium/hard, got: {request.difficulty}")

    _env = DisasterEnv(request.difficulty)
    _env.reset()
    return _env_to_observation(_env)


@app.post("/step", response_model=StepResult, tags=["OpenEnv API"])
def step(request: StepRequest):
    """
    Apply one action and advance the world by one step.
    Returns new observation, reward breakdown, and done flag.
    """
    global _env, _last_action_info

    if _env is None:
        raise HTTPException(400, "Call /reset first to start an episode.")
    if _env.done:
        raise HTTPException(400, "Episode is over. Call /reset to start a new one.")

    try:
        action = (request.resource_type, request.zone_id, request.quantity_index)
        obs_dict, raw_reward, done, info = _env.step(action)
        _last_action_info = info
    except ValueError as e:
        raise HTTPException(400, str(e))

    # Build typed reward breakdown
    RESOURCE_NAMES = {0: "food", 1: "medical", 2: "rescue"}
    qty_map        = {0: 10, 1: 20, 2: 30}

    # Recalculate components for the response
    zones          = _env.zones
    init_inj       = _env._initial_total_injured
    init_food      = _env._initial_total_food_need
    total_inj      = sum(z.injured   for z in zones)
    total_food     = sum(z.food_need for z in zones)
    inj_cov        = 1.0 - (total_inj  / max(1, init_inj))
    food_cov       = 1.0 - (total_food / max(1, init_food))
    need_score     = round(max(0, min(1, (inj_cov + food_cov) / 2)), 4)
    most_crit_id   = max(range(len(zones)), key=lambda i: zones[i].severity)
    priority_score = 1.0 if request.zone_id == most_crit_id else 0.3
    waste_penalty  = 0.0
    z = zones[request.zone_id]
    if request.resource_type == 0 and z.food_need == 0:      waste_penalty = 0.1
    if request.resource_type == 1 and z.injured == 0:        waste_penalty = 0.1
    if request.resource_type == 2 and not z.rescue_blocked and z.injured == 0: waste_penalty = 0.1

    reward = Reward(
        total          = round(raw_reward, 4),
        need_score     = need_score,
        priority_score = round(priority_score, 4),
        waste_penalty  = round(waste_penalty, 4),
        resource_sent  = RESOURCE_NAMES.get(request.resource_type, "unknown"),
        zone_targeted  = zones[request.zone_id].name if request.zone_id < len(zones) else "unknown",
        units_sent     = info.get("quantity", 0),
    )

    return StepResult(
        observation = _env_to_observation(_env),
        reward      = reward,
        done        = done,
        info        = info,
    )


@app.get("/state", response_model=Observation, tags=["OpenEnv API"])
def state():
    """Return the current world state without advancing the episode."""
    if _env is None:
        raise HTTPException(400, "Call /reset first.")
    return _env_to_observation(_env)


@app.get("/tasks", tags=["Tasks"])
def list_tasks():
    """List all available tasks with grader info."""
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
                "description":       "5 zones, 5% escalation per step, 20% chance blocked zones."
            },
            {
                "id":                "hard",
                "name":              "Crisis Management",
                "difficulty":        "hard",
                "num_zones":         8,
                "max_steps":         30,
                "success_threshold": 0.15,
                "description":       "8 zones, 10% escalation, 40% blocked zones. Triage aggressively."
            }
        ]
    }


@app.post("/grade", response_model=GraderResult, tags=["Tasks"])
def grade(request: GradeRequest):
    """Run the grader for a specific task and return the score."""
    grader_map = {
        "easy":   EasyGrader(),
        "medium": MediumGrader(),
        "hard":   HardGrader(),
    }
    if request.task_id not in grader_map:
        raise HTTPException(400, f"task_id must be easy/medium/hard, got: {request.task_id}")

    grader = grader_map[request.task_id]
    result = grader.run(rule_based_agent, n_episodes=request.n_episodes)
    return result


# ══════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(f"\nStarting Disaster Resource Allocation server on port {port}")
    print(f"API docs: http://localhost:{port}/docs\n")
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
