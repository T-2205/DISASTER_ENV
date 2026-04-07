"""
config.py — Difficulty level settings for the disaster environment.

Think of this as the settings panel for your simulation.
Instead of writing separate classes for easy/medium/hard,
we store everything in one dictionary per level.

The DisasterEnv class reads from here when it starts up.
"""


CONFIGS = {

    # ----------------------------------------------------------------
    # EASY
    # Calm scenario. Great for debugging and first tests.
    # Few zones, lots of resources, situation doesn't get worse.
    # ----------------------------------------------------------------
    "easy": {
        "num_zones":         3,        # How many disaster zones exist

        # Starting stock of each resource
        "initial_food":      200,      # food units available
        "initial_medical":   100,      # medical kits available
        "initial_rescue":    15,       # rescue teams available

        # How many units per quantity choice (10 / 20 / 30)
        "quantity_options":  [10, 20, 30],

        # Escalation: does the situation get worse each round?
        "escalation_rate":   0.0,      # 0% = situation stays the same

        # Episode length: how many rounds before the game ends
        "max_steps":         20,

        # Partial restock: resources partially refill each round
        "restock_rate":      0.0,      # 0% = no restock (plenty to start)
    },

    # ----------------------------------------------------------------
    # MEDIUM
    # The main difficulty. 5 zones, moderate scarcity.
    # Needs grow slightly each round — act fast!
    # ----------------------------------------------------------------
    "medium": {
        "num_zones":         5,

        "initial_food":      300,
        "initial_medical":   150,
        "initial_rescue":    20,

        "quantity_options":  [10, 20, 30],

        "escalation_rate":   0.05,     # 5% worse each round

        "max_steps":         25,

        "restock_rate":      0.05,     # 5% of initial stock restocked each round
    },

    # ----------------------------------------------------------------
    # HARD
    # Chaos mode. 8 zones, severe scarcity.
    # Needs escalate fast. Some zones are blocked at the start.
    # The AI must triage — it cannot save everyone.
    # ----------------------------------------------------------------
    "hard": {
        "num_zones":         8,

        "initial_food":      250,      # Less stock, more zones = tough choices
        "initial_medical":   100,
        "initial_rescue":    12,

        "quantity_options":  [10, 20, 30],

        "escalation_rate":   0.10,     # 10% worse each round

        "max_steps":         30,

        "restock_rate":      0.02,     # Almost no restock — manage carefully
    },
}


def get_config(difficulty: str) -> dict:
    """
    Safely fetch a config by difficulty name.
    Raises a clear error if you mistype the difficulty.

    Usage:
        cfg = get_config("medium")
        print(cfg["num_zones"])  # → 5
    """
    if difficulty not in CONFIGS:
        raise ValueError(
            f"Unknown difficulty '{difficulty}'. "
            f"Choose from: {list(CONFIGS.keys())}"
        )
    return CONFIGS[difficulty]
