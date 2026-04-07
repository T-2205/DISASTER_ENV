"""
gym_wrapper.py — Wraps DisasterEnv in the Gymnasium interface.

Stable-Baselines3 (the RL library) expects environments to follow
a specific format called the Gymnasium API. Think of it like a
power adapter — your environment works fine, it just needs an
adapter so the AI library can plug into it.

What this file adds:
    - observation_space : tells the AI what the world looks like (numbers)
    - action_space      : tells the AI what moves are allowed
    - reset()           : matches Gymnasium's exact return format
    - step()            : matches Gymnasium's exact return format

Usage:
    from gym_wrapper import DisasterGymEnv
    env = DisasterGymEnv("medium")
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from disaster_env import DisasterEnv, FOOD, MEDICAL, RESCUE


class DisasterGymEnv(gym.Env):
    """
    Gymnasium-compatible wrapper around DisasterEnv.

    The AI sees the world as a flat list of numbers (a vector).
    Every zone's data + resource stocks gets squashed into one
    long array that the neural network can read.

    Observation vector layout (per zone, repeated for each zone):
        [injured, food_need, severity, rescue_blocked, disaster_type_encoded]
    Plus at the end:
        [food_stock, med_stock, resc_stock]

    Example for medium (5 zones):
        5 zones × 5 values = 25 numbers
        + 3 stock values
        = 28 numbers total
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, difficulty: str = "medium"):
        super().__init__()
        self.difficulty = difficulty

        # Create the underlying environment
        self._env = DisasterEnv(difficulty)
        self._env.reset()  # needed to know num_zones

        cfg       = self._env.cfg
        num_zones = cfg["num_zones"]

        # ----------------------------------------------------------
        # Action space — a single integer from 0 to (total_actions-1)
        # The AI picks one number and we decode it into (resource, zone, qty)
        # ----------------------------------------------------------
        self.total_actions = 3 * num_zones * len(cfg["quantity_options"])
        self.action_space  = spaces.Discrete(self.total_actions)

        # ----------------------------------------------------------
        # Observation space — a flat array of floats, all between 0 and 1
        # Normalising everything to 0–1 helps the neural network learn faster
        # ----------------------------------------------------------
        # 5 values per zone + 3 stock values
        obs_size = num_zones * 5 + 3
        self.observation_space = spaces.Box(
            low   = 0.0,
            high  = 1.0,
            shape = (obs_size,),
            dtype = np.float32,
        )

        # Lookup table: integer action → (resource, zone, qty_index)
        self._action_lookup = self._build_action_lookup()

        # Max values for normalisation (so 0–1 scaling works)
        self._max_population = 2000
        self._max_need       = 500
        self._max_stock      = max(
            cfg["initial_food"],
            cfg["initial_medical"],
            cfg["initial_rescue"],
        )

    # ------------------------------------------------------------------
    # Gymnasium API — reset()
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        """
        Start a new episode.
        Returns: (observation_array, info_dict)
        """
        super().reset(seed=seed)
        obs_dict = self._env.reset()
        return self._encode_obs(obs_dict), {}

    # ------------------------------------------------------------------
    # Gymnasium API — step()
    # ------------------------------------------------------------------

    def step(self, action: int):
        """
        Apply one action (an integer) to the environment.

        Returns:
            observation : np.array — new world state as numbers
            reward      : float    — score for this action
            terminated  : bool     — True if episode ended naturally
            truncated   : bool     — True if we hit max steps
            info        : dict     — extra debug info
        """
        # Decode integer → (resource_type, zone_id, quantity_index)
        decoded_action = self._action_lookup[int(action)]

        obs_dict, reward, done, info = self._env.step(decoded_action)

        # Gymnasium splits "done" into two booleans
        terminated = done and self._all_zones_clear()
        truncated  = done and not terminated

        return self._encode_obs(obs_dict), reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Gymnasium API — render()
    # ------------------------------------------------------------------

    def render(self):
        self._env.render()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _encode_obs(self, obs_dict: dict) -> np.ndarray:
        """
        Convert the state dictionary into a flat numpy array.
        All values are normalised to the range [0.0, 1.0].

        The neural network only understands numbers, not words like
        "earthquake" or "flood" — so we encode disaster types as integers.
        """
        DISASTER_ENCODING = {
            "earthquake": 0.25,
            "flood":      0.50,
            "hurricane":  0.75,
            "wildfire":   1.00,
        }

        values = []

        for zone in obs_dict["zones"]:
            values.append(zone["injured"]        / self._max_need)
            values.append(zone["food_need"]      / self._max_need)
            values.append(zone["severity"])                          # already 0–1
            values.append(1.0 if zone["rescue_blocked"] else 0.0)
            values.append(DISASTER_ENCODING.get(zone["disaster_type"], 0.0))

        # Resource stocks (normalised)
        values.append(obs_dict["food_stock"]  / self._max_stock)
        values.append(obs_dict["med_stock"]   / self._max_stock)
        values.append(obs_dict["resc_stock"]  / self._max_stock)

        return np.array(values, dtype=np.float32)

    def _build_action_lookup(self) -> dict:
        """
        Build a mapping from integer → (resource, zone, qty_index).

        Example for medium (5 zones, 3 qty options):
            0 → (FOOD,    zone_0, qty_0)
            1 → (FOOD,    zone_0, qty_1)
            2 → (FOOD,    zone_0, qty_2)
            3 → (FOOD,    zone_1, qty_0)
            ...
            44 → (RESCUE, zone_4, qty_2)
        """
        lookup   = {}
        num_zones = self._env.cfg["num_zones"]
        num_qtys  = len(self._env.cfg["quantity_options"])
        idx = 0

        for resource in [FOOD, MEDICAL, RESCUE]:
            for zone_id in range(num_zones):
                for qty_idx in range(num_qtys):
                    lookup[idx] = (resource, zone_id, qty_idx)
                    idx += 1

        return lookup

    def _all_zones_clear(self) -> bool:
        return all(
            z.injured == 0 and z.food_need == 0
            for z in self._env.zones
        )
