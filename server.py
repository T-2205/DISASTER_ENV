import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
import threading

from disaster_env import DisasterEnv
from models import Observation, ZoneState
from graders import EasyGrader, MediumGrader, HardGrader, rule_based_agent

# ─────────────────────────────
# App setup
# ─────────────────────────────
app = FastAPI(
    title="Disaster Resource Allocation Environment",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_env: Optional[DisasterEnv] = None


# ─────────────────────────────
# 🔥 CRITICAL FIX: STARTUP EVENT
# ─────────────────────────────
@app.on_event("startup")
def startup_event():
    import threading

    def run_inference():
        try:
            print("[DEBUG] STARTING INFERENCE...", flush=True)

            import inference

            # 🔥 CRITICAL: manually call main logic
            client, model = inference.get_client()

            for task in inference.TASKS:
                inference.run_task(task, client, model)

        except Exception as e:
            print(f"[ERROR] Failed to run inference: {e}", flush=True)

    print("[DEBUG] STARTUP EVENT TRIGGERED", flush=True)

    threading.Thread(target=run_inference).start()


# ─────────────────────────────
# Request Models
# ─────────────────────────────
class ResetRequest(BaseModel):
    difficulty: Optional[str] = "medium"


class StepRequest(BaseModel):
    action: Optional[int] = None
    resource_type: Optional[int] = None
    zone_id: Optional[int] = None
    quantity_index: Optional[int] = None


class GradeRequest(BaseModel):
    task_id: Optional[str] = "medium"
    n_episodes: Optional[int] = 5


# ─────────────────────────────
# Helpers
# ─────────────────────────────
def _env_to_observation(env: DisasterEnv) -> Observation:
    state = env.state()
    return Observation(
        zones=[
            ZoneState(
                zone_id=z["zone_id"],
                name=z["name"],
                disaster_type=z["disaster_type"],
                population=z["population"],
                injured=z["injured"],
                food_need=z["food_need"],
                rescue_blocked=z["rescue_blocked"],
                severity=z["severity"],
            )
            for z in state["zones"]
        ],
        food_stock=state["food_stock"],
        med_stock=state["med_stock"],
        resc_stock=state["resc_stock"],
        step=state["step"],
        max_steps=state["max_steps"],
        difficulty=env.difficulty,
    )


def _decode_action(request: StepRequest, env: DisasterEnv):
    num_zones = len(env.zones)

    if request.action is not None:
        idx = int(request.action)
        resource = idx // (num_zones * 3)
        remainder = idx % (num_zones * 3)
        zone = remainder // 3
        qty = remainder % 3
        return (resource, zone, qty)

    return (
        request.resource_type or 0,
        request.zone_id or 0,
        request.quantity_index or 0,
    )


# ─────────────────────────────
# Endpoints
# ─────────────────────────────
@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/reset")
def reset(request: Optional[ResetRequest] = None):
    global _env
    difficulty = request.difficulty if request else "medium"

    _env = DisasterEnv(difficulty)
    _env.reset()
    return _env_to_observation(_env)


@app.post("/step")
def step(request: Optional[StepRequest] = None):
    global _env

    if _env is None:
        _env = DisasterEnv("medium")
        _env.reset()

    if request is None:
        request = StepRequest()

    action = _decode_action(request, _env)

    obs, reward, done, info = _env.step(action)

    return {
        "observation": _env.state(),
        "reward": reward,
        "done": done,
        "info": info,
    }


@app.get("/state")
def state():
    if _env is None:
        raise HTTPException(400, "Call /reset first")
    return _env.state()


@app.post("/grade")
def grade(request: Optional[GradeRequest] = None):
    if request is None:
        request = GradeRequest()

    graders = {
        "easy": EasyGrader(),
        "medium": MediumGrader(),
        "hard": HardGrader(),
    }

    grader = graders.get(request.task_id, MediumGrader())
    return grader.run(rule_based_agent, request.n_episodes)


# ─────────────────────────────
# ENTRY POINT
# ─────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("server:app", host="0.0.0.0", port=port)
