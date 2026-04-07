"""
models.py — Typed Pydantic models for the OpenEnv spec.

The OpenEnv specification requires that Observation, Action, and Reward
are defined as typed models. This makes the environment self-documenting
and allows the openenv validator to check compliance automatically.

These models describe exactly what flows in and out of the environment
at every step.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


# ══════════════════════════════════════════════════════════════════
# Zone state model — what one zone looks like
# ══════════════════════════════════════════════════════════════════

class ZoneState(BaseModel):
    """State of a single disaster zone."""

    zone_id: int = Field(..., description="Unique zone index (0-based)")
    name: str = Field(..., description="Human-readable name e.g. 'Zone A'")
    disaster_type: str = Field(
        ...,
        description="Type of disaster: earthquake | flood | hurricane | wildfire"
    )
    population: int = Field(..., ge=0, description="Total people in zone")
    injured: int = Field(..., ge=0, description="People needing medical aid")
    food_need: int = Field(..., ge=0, description="Food units still required")
    rescue_blocked: bool = Field(
        ...,
        description="True = zone inaccessible until rescue teams are sent"
    )
    severity: float = Field(
        ..., ge=0.0, le=1.0,
        description="Severity rating 0.0 (minor) to 1.0 (catastrophic)"
    )


# ══════════════════════════════════════════════════════════════════
# Observation model — what the AI sees each step
# ══════════════════════════════════════════════════════════════════

class Observation(BaseModel):
    """
    Full world state observation returned by reset() and step().

    The AI receives this every round and uses it to pick its next action.
    All numeric values are also available as a flat normalised float32
    vector via the gym_wrapper for neural network input.
    """

    zones: List[ZoneState] = Field(
        ...,
        description="State of every disaster zone in the current episode"
    )
    food_stock: int = Field(..., ge=0, description="Food units remaining in reserve")
    med_stock: int = Field(..., ge=0, description="Medical kits remaining in reserve")
    resc_stock: int = Field(..., ge=0, description="Rescue teams remaining in reserve")
    step: int = Field(..., ge=0, description="Current step number (0-indexed)")
    max_steps: int = Field(..., gt=0, description="Maximum steps in this episode")
    difficulty: str = Field(
        ...,
        description="Current difficulty level: easy | medium | hard"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "zones": [
                    {
                        "zone_id": 0,
                        "name": "Zone A",
                        "disaster_type": "earthquake",
                        "population": 500,
                        "injured": 80,
                        "food_need": 120,
                        "rescue_blocked": False,
                        "severity": 0.72
                    }
                ],
                "food_stock": 300,
                "med_stock": 150,
                "resc_stock": 20,
                "step": 0,
                "max_steps": 25,
                "difficulty": "medium"
            }
        }


# ══════════════════════════════════════════════════════════════════
# Action model — what the AI sends each step
# ══════════════════════════════════════════════════════════════════

class Action(BaseModel):
    """
    One action taken by the agent per step.

    Encodes which resource to send, to which zone, and how many units.
    The environment also accepts this as a plain tuple (resource_type, zone_id, quantity_index)
    for compatibility with standard RL libraries.
    """

    resource_type: int = Field(
        ..., ge=0, le=2,
        description="0=food, 1=medical, 2=rescue"
    )
    zone_id: int = Field(
        ..., ge=0,
        description="Target zone index (0 to num_zones-1)"
    )
    quantity_index: int = Field(
        ..., ge=0, le=2,
        description="0=10 units, 1=20 units, 2=30 units"
    )

    def to_tuple(self):
        """Convert to tuple format used by DisasterEnv.step()"""
        return (self.resource_type, self.zone_id, self.quantity_index)

    @classmethod
    def from_int(cls, action_int: int, num_zones: int) -> "Action":
        """
        Decode a flat integer action (from Discrete action space)
        back into (resource_type, zone_id, quantity_index).
        """
        qty_idx      = action_int % 3
        remainder    = action_int // 3
        zone_id      = remainder % num_zones
        resource_type = remainder // num_zones
        return cls(
            resource_type=resource_type,
            zone_id=zone_id,
            quantity_index=qty_idx
        )

    class Config:
        json_schema_extra = {
            "example": {
                "resource_type": 1,
                "zone_id": 2,
                "quantity_index": 2,
                "description": "Send 30 medical kits to Zone C"
            }
        }


# ══════════════════════════════════════════════════════════════════
# Reward model — the score returned after each step
# ══════════════════════════════════════════════════════════════════

class Reward(BaseModel):
    """
    Structured reward breakdown returned after each step.

    The total field is what RL algorithms use for training.
    The component fields are for human inspection and debugging.
    """

    total: float = Field(
        ..., ge=0.0, le=1.0,
        description="Final reward score for this step (0.0 to 1.0)"
    )
    need_score: float = Field(
        ..., ge=0.0, le=1.0,
        description="Fraction of total unmet need reduced (weight 0.6)"
    )
    priority_score: float = Field(
        ..., ge=0.0, le=1.0,
        description="Priority bonus for helping most critical zone (weight 0.3)"
    )
    waste_penalty: float = Field(
        ..., ge=0.0, le=0.1,
        description="Penalty for sending unneeded resources (up to 0.1)"
    )
    resource_sent: str = Field(
        ...,
        description="Which resource was sent: food | medical | rescue"
    )
    zone_targeted: str = Field(
        ...,
        description="Which zone received the resource"
    )
    units_sent: int = Field(
        ..., ge=0,
        description="Actual units delivered (may be less than requested if stock ran out)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "total": 0.72,
                "need_score": 0.65,
                "priority_score": 1.0,
                "waste_penalty": 0.0,
                "resource_sent": "medical",
                "zone_targeted": "Zone C",
                "units_sent": 30
            }
        }


# ══════════════════════════════════════════════════════════════════
# Step result model — everything returned by step()
# ══════════════════════════════════════════════════════════════════

class StepResult(BaseModel):
    """Full result of one environment step."""

    observation: Observation
    reward: Reward
    done: bool = Field(..., description="True if the episode has ended")
    info: dict = Field(
        default_factory=dict,
        description="Extra debug info: step count, stock levels, action taken"
    )


# ══════════════════════════════════════════════════════════════════
# Task grader model — scores an entire episode
# ══════════════════════════════════════════════════════════════════

class GraderResult(BaseModel):
    """Result from running a task grader on a completed episode."""

    task_id: str = Field(..., description="Task identifier: easy | medium | hard")
    score: float = Field(
        ..., ge=0.0, le=1.0,
        description="Final episode score (0.0 to 1.0)"
    )
    passed: bool = Field(
        ...,
        description="True if score >= success_threshold for this task"
    )
    steps_taken: int = Field(..., description="How many steps the episode ran")
    total_reward: float = Field(..., description="Sum of all step rewards")
    zones_cleared: int = Field(
        ...,
        description="Number of zones with injured=0 and food_need=0 at end"
    )
    details: dict = Field(
        default_factory=dict,
        description="Per-zone final state and other diagnostic info"
    )
