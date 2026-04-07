"""
disaster_env.py — The main environment class.

This is the "game engine" of your simulation.
It implements the OpenEnv API:
    - reset()  → start a fresh episode
    - step()   → apply one action, update the world
    - state()  → return what the AI currently sees

How a full episode works:
    env = DisasterEnv("medium")
    obs = env.reset()
    done = False

    while not done:
        action = (resource_type, zone_id, quantity_index)
        obs, reward, done, info = env.step(action)
        print(f"Reward: {reward:.2f}")
"""

from zone import Zone
from config import get_config


# Resource type constants — easier to read than magic numbers
FOOD    = 0
MEDICAL = 1
RESCUE  = 2

RESOURCE_NAMES = {FOOD: "food", MEDICAL: "medical", RESCUE: "rescue"}


class DisasterEnv:
    """
    Multi-Zone Disaster Resource Allocation Environment.

    Args:
        difficulty (str): "easy", "medium", or "hard"

    The action space is a tuple:
        (resource_type, zone_id, quantity_index)

        resource_type  : int — 0=food, 1=medical, 2=rescue
        zone_id        : int — which zone to send to (0 to num_zones-1)
        quantity_index : int — 0=10 units, 1=20 units, 2=30 units
    """

    def __init__(self, difficulty: str = "medium"):
        self.difficulty = difficulty
        self.cfg = get_config(difficulty)

        # These are set properly by reset() — just initialising here
        self.zones       = []
        self.food_stock  = 0
        self.med_stock   = 0
        self.resc_stock  = 0
        self.step_count  = 0
        self.done        = False

        # Track totals at the start of each episode (for reward calc)
        self._initial_total_injured   = 0
        self._initial_total_food_need = 0

    # ------------------------------------------------------------------
    # OpenEnv API — reset()
    # ------------------------------------------------------------------

    def reset(self) -> dict:
        """
        Start a brand new episode.

        - Creates fresh zones with random disaster types and needs
        - Refills all resources back to starting amounts
        - Resets step counter
        - Returns the initial state observation

        Returns:
            dict — the starting state (same format as state())
        """
        # 1. Create zones and randomize each one
        num_zones = self.cfg["num_zones"]
        self.zones = [Zone(i) for i in range(num_zones)]
        for zone in self.zones:
            zone.randomize(self.difficulty)

        # 2. Refill resource stocks
        self.food_stock  = self.cfg["initial_food"]
        self.med_stock   = self.cfg["initial_medical"]
        self.resc_stock  = self.cfg["initial_rescue"]

        # 3. Reset counters
        self.step_count = 0
        self.done = False

        # 4. Save initial totals for reward function
        self._initial_total_injured   = sum(z.injured   for z in self.zones)
        self._initial_total_food_need = sum(z.food_need for z in self.zones)

        return self.state()

    # ------------------------------------------------------------------
    # OpenEnv API — step()
    # ------------------------------------------------------------------

    def step(self, action: tuple) -> tuple:
        """
        Apply one action and advance the world by one round.

        Args:
            action: tuple of (resource_type, zone_id, quantity_index)
                    e.g. (0, 2, 1) = send 20 food to Zone C

        Returns:
            (observation, reward, done, info)

            observation : dict  — new state after action
            reward      : float — score for this action (0.0 to 1.0)
            done        : bool  — True if episode is over
            info        : dict  — extra debug info (not used by AI)
        """
        if self.done:
            raise RuntimeError("Episode is over. Call reset() to start again.")

        resource_type, zone_id, quantity_index = action

        # --- Validate action ---
        self._validate_action(resource_type, zone_id, quantity_index)

        # --- Get how many units this action sends ---
        quantity = self.cfg["quantity_options"][quantity_index]

        # --- Apply the action to the chosen zone ---
        reward = self._apply_action(resource_type, zone_id, quantity)

        # --- Escalate all zones (situation gets worse over time) ---
        escalation_rate = self.cfg["escalation_rate"]
        if escalation_rate > 0:
            for zone in self.zones:
                zone.escalate(escalation_rate)

        # --- Partial restock ---
        self._restock()

        # --- Advance step counter ---
        self.step_count += 1

        # --- Check if episode should end ---
        self.done = self._check_done()

        info = {
            "step":          self.step_count,
            "resource_sent": RESOURCE_NAMES[resource_type],
            "zone":          self.zones[zone_id].name,
            "quantity":      quantity,
            "food_stock":    self.food_stock,
            "med_stock":     self.med_stock,
            "resc_stock":    self.resc_stock,
        }

        return self.state(), reward, self.done, info

    # ------------------------------------------------------------------
    # OpenEnv API — state()
    # ------------------------------------------------------------------

    def state(self) -> dict:
        """
        Return what the AI currently sees — a full snapshot of the world.

        The AI gets:
        - The state of every zone (all their numbers)
        - Current resource stock levels
        - How many steps have passed

        Returns:
            dict — observation the AI uses to pick its next action
        """
        return {
            "zones":      [zone.to_dict() for zone in self.zones],
            "food_stock":  self.food_stock,
            "med_stock":   self.med_stock,
            "resc_stock":  self.resc_stock,
            "step":        self.step_count,
            "max_steps":   self.cfg["max_steps"],
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_action(self, resource_type: int, zone_id: int, quantity: int) -> float:
        """
        Send resources to a zone and calculate the reward.

        Steps:
        1. Check if we have enough stock — cap if not
        2. Deduct from stock
        3. Apply effect to the zone
        4. Calculate and return reward
        """
        zone = self.zones[zone_id]

        # --- Deduct stock (can't send more than we have) ---
        if resource_type == FOOD:
            actual = min(quantity, self.food_stock)
            self.food_stock -= actual
            zone.apply_food(actual)

        elif resource_type == MEDICAL:
            actual = min(quantity, self.med_stock)
            self.med_stock -= actual
            zone.apply_medical(actual)

        elif resource_type == RESCUE:
            actual = min(quantity, self.resc_stock)
            self.resc_stock -= actual
            zone.apply_rescue(actual)

        # --- Calculate reward for this action ---
        return self._calculate_reward(resource_type, zone_id, actual)

    def _calculate_reward(self, resource_type: int, zone_id: int, quantity_sent: int) -> float:
        """
        Score the action from 0.0 (terrible) to 1.0 (perfect).

        Three components:
            need_score     (60%) — how much total unmet need remains
            priority_score (30%) — did we help the most critical zone?
            waste_penalty  (10%) — did we send the wrong resource?

        Formula:
            reward = (need_score * 0.6) + (priority_score * 0.3) - waste_penalty
        """
        # --- Need score: what fraction of total need has been met? ---
        total_injured   = sum(z.injured   for z in self.zones)
        total_food_need = sum(z.food_need for z in self.zones)

        # Avoid division by zero
        inj_coverage  = 1.0 - (total_injured   / max(1, self._initial_total_injured))
        food_coverage = 1.0 - (total_food_need / max(1, self._initial_total_food_need))
        need_score = (inj_coverage + food_coverage) / 2.0
        need_score = max(0.0, min(1.0, need_score))  # clamp to [0, 1]

        # --- Priority score: was the most critical zone helped? ---
        most_critical_id = max(range(len(self.zones)), key=lambda i: self.zones[i].severity)
        priority_score = 1.0 if zone_id == most_critical_id else 0.3

        # --- Waste penalty: sent a resource the zone doesn't really need? ---
        zone = self.zones[zone_id]
        waste_penalty = 0.0

        if resource_type == FOOD and zone.food_need == 0:
            waste_penalty = 0.1   # Sent food to a zone with no food need

        elif resource_type == MEDICAL and zone.injured == 0:
            waste_penalty = 0.1   # Sent medicine to a zone with no injured

        elif resource_type == RESCUE and not zone.rescue_blocked and zone.injured == 0:
            waste_penalty = 0.1   # Sent rescue where it wasn't needed

        # --- Final score ---
        reward = (need_score * 0.6) + (priority_score * 0.3) - waste_penalty
        return round(max(0.0, min(1.0, reward)), 4)

    def _restock(self):
        """
        Partially refill resources each round (if config says so).
        Simulates aid supplies arriving over time.
        """
        rate = self.cfg["restock_rate"]
        if rate > 0:
            self.food_stock  += int(self.cfg["initial_food"]    * rate)
            self.med_stock   += int(self.cfg["initial_medical"] * rate)
            self.resc_stock  += int(self.cfg["initial_rescue"]  * rate)

    def _check_done(self) -> bool:
        """
        Is the episode over?

        Episode ends when:
        - We've hit the max number of steps, OR
        - All zones are fully helped (injured=0 and food_need=0)
        """
        if self.step_count >= self.cfg["max_steps"]:
            return True

        all_clear = all(z.injured == 0 and z.food_need == 0 for z in self.zones)
        return all_clear

    def _validate_action(self, resource_type: int, zone_id: int, quantity_index: int):
        """
        Raise a clear error if the action contains invalid values.
        Helps catch bugs early during development.
        """
        num_zones = len(self.zones)
        num_qtys  = len(self.cfg["quantity_options"])

        if resource_type not in (FOOD, MEDICAL, RESCUE):
            raise ValueError(f"resource_type must be 0, 1, or 2. Got: {resource_type}")

        if not (0 <= zone_id < num_zones):
            raise ValueError(f"zone_id must be 0–{num_zones - 1}. Got: {zone_id}")

        if not (0 <= quantity_index < num_qtys):
            raise ValueError(f"quantity_index must be 0–{num_qtys - 1}. Got: {quantity_index}")

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def render(self):
        """
        Print a human-readable summary of the current world state.
        Useful for debugging — call after each step to see what's happening.
        """
        print(f"\n{'='*55}")
        print(f"  Step {self.step_count}/{self.cfg['max_steps']}  |  Difficulty: {self.difficulty.upper()}")
        print(f"  Stock — Food: {self.food_stock}  Medical: {self.med_stock}  Rescue: {self.resc_stock}")
        print(f"{'='*55}")
        for zone in self.zones:
            print(f"  {zone}")
        print()

    def action_space_size(self) -> int:
        """
        Total number of possible actions.
        3 resources × num_zones × 3 quantity options
        """
        return 3 * len(self.zones) * len(self.cfg["quantity_options"])
