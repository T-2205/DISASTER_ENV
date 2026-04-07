"""
zone.py — Represents a single disaster-affected zone.

Each zone is like a patient file. It holds all the information
about one area: how many people are there, how bad the situation
is, and what resources are still needed.
"""

import random


# All possible disaster types that can be randomly assigned
DISASTER_TYPES = ["earthquake", "flood", "hurricane", "wildfire"]


class Zone:
    """
    A single disaster zone in the simulation.

    Attributes:
        zone_id      : int   — unique number (0, 1, 2, 3, 4)
        name         : str   — human-readable label e.g. "Zone A"
        disaster_type: str   — what kind of disaster hit
        population   : int   — total people in this zone
        injured      : int   — people who need medical aid
        food_need    : int   — food units still needed
        rescue_blocked: bool — True means rescue teams cannot enter yet
        severity     : float — 0.0 (minor) to 1.0 (catastrophic)

        # These store the STARTING values so reward can compare later
        initial_injured  : int
        initial_food_need: int
    """

    def __init__(self, zone_id: int):
        self.zone_id = zone_id
        self.name = f"Zone {chr(65 + zone_id)}"  # Zone A, Zone B, ...

        # These will be filled in by randomize()
        self.disaster_type = ""
        self.population = 0
        self.injured = 0
        self.food_need = 0
        self.rescue_blocked = False
        self.severity = 0.0

        # Starting snapshots (used by reward function)
        self.initial_injured = 0
        self.initial_food_need = 0

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def randomize(self, difficulty: str):
        """
        Randomly set this zone's state based on difficulty level.

        Easy   → smaller numbers, nothing blocked
        Medium → moderate numbers, some zones might be blocked
        Hard   → large numbers, blocked zones more likely
        """
        self.disaster_type = random.choice(DISASTER_TYPES)

        if difficulty == "easy":
            self.population   = random.randint(100, 400)
            self.injured      = random.randint(5,   50)
            self.food_need    = random.randint(20,  80)
            self.severity     = round(random.uniform(0.1, 0.4), 2)
            self.rescue_blocked = False

        elif difficulty == "medium":
            self.population   = random.randint(200, 800)
            self.injured      = random.randint(20,  150)
            self.food_need    = random.randint(50,  200)
            self.severity     = round(random.uniform(0.3, 0.7), 2)
            self.rescue_blocked = random.random() < 0.2  # 20% chance blocked

        elif difficulty == "hard":
            self.population   = random.randint(500, 2000)
            self.injured      = random.randint(100, 500)
            self.food_need    = random.randint(150, 400)
            self.severity     = round(random.uniform(0.6, 1.0), 2)
            self.rescue_blocked = random.random() < 0.4  # 40% chance blocked

        # Save starting values for the reward function
        self.initial_injured   = self.injured
        self.initial_food_need = self.food_need

    # ------------------------------------------------------------------
    # Resource application
    # ------------------------------------------------------------------

    def apply_food(self, units: int):
        """
        Reduce food_need by the given units.
        Cannot go below 0 (no such thing as negative hunger).
        """
        self.food_need = max(0, self.food_need - units)

    def apply_medical(self, kits: int):
        """
        Reduce injured count by the number of kits sent.
        Cannot go below 0.
        """
        self.injured = max(0, self.injured - kits)

    def apply_rescue(self, teams: int):
        """
        Rescue teams do two things:
        1. If zone is blocked, one team unblocks it (opens access).
        2. Remaining teams reduce the injured count slightly.
        """
        if self.rescue_blocked:
            self.rescue_blocked = False
            teams -= 1  # One team used to unblock

        if teams > 0:
            # Each rescue team saves ~5 injured people
            self.injured = max(0, self.injured - teams * 5)

    # ------------------------------------------------------------------
    # Escalation (used in medium/hard difficulty each step)
    # ------------------------------------------------------------------

    def escalate(self, rate: float):
        """
        Make the situation slightly worse each round.
        Simulates a disaster that keeps spreading if not helped.

        rate: float — how fast things get worse (e.g. 0.05 = 5% per round)
        """
        self.food_need = int(self.food_need * (1 + rate))
        self.injured   = int(self.injured   * (1 + rate * 0.5))

    # ------------------------------------------------------------------
    # State snapshot (what the AI sees)
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """
        Return a plain dictionary of this zone's current state.
        This is what gets passed to the AI as its observation.
        """
        return {
            "zone_id":        self.zone_id,
            "name":           self.name,
            "disaster_type":  self.disaster_type,
            "population":     self.population,
            "injured":        self.injured,
            "food_need":      self.food_need,
            "rescue_blocked": self.rescue_blocked,
            "severity":       self.severity,
        }

    def __repr__(self):
        blocked = " [BLOCKED]" if self.rescue_blocked else ""
        return (
            f"{self.name}{blocked} | {self.disaster_type} | "
            f"severity={self.severity} | injured={self.injured} | "
            f"food_need={self.food_need}"
        )
