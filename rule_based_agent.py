"""
rule_based_agent.py — A simple hand-coded agent for testing.

This agent does NOT learn. It just follows a fixed set of rules:

    Rule 1: Find the zone with the highest severity
    Rule 2: Look at what that zone needs most
    Rule 3: Send that resource to that zone

Why build this before a real AI?
    - If this smart-but-dumb agent scores well → your env is working
    - If it scores badly → something is wrong with your reward function
    - It gives you a baseline score to beat with a real AI later

Usage:
    python rule_based_agent.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from disaster_env import DisasterEnv, FOOD, MEDICAL, RESCUE


class RuleBasedAgent:
    """
    A hand-coded agent that always helps the most critical zone.

    It follows simple triage logic — just like a real rescue coordinator
    would in the field: find the worst situation, fix the biggest need.
    """

    def pick_action(self, state: dict) -> tuple:
        """
        Look at the current world state and return the best action.

        Args:
            state: dict — the observation from env.state()

        Returns:
            tuple: (resource_type, zone_id, quantity_index)
        """
        zones = state["zones"]

        # --- Find the most critical zone ---
        # Priority order:
        #   1. Blocked zones (rescue teams needed to unlock)
        #   2. Highest injured count (medical emergency)
        #   3. Highest food need (hunger crisis)

        # First: is any zone blocked? Unblock it immediately.
        for zone in zones:
            if zone["rescue_blocked"]:
                return (RESCUE, zone["zone_id"], 1)  # send 20 rescue teams

        # Second: find zone with most injured
        most_injured_zone = max(zones, key=lambda z: z["injured"])

        # Third: find zone with most food need
        most_hungry_zone = max(zones, key=lambda z: z["food_need"])

        # Decide: which is more urgent — injured or hungry?
        if most_injured_zone["injured"] >= most_hungry_zone["food_need"]:
            # More injured people → send medical aid
            return (MEDICAL, most_injured_zone["zone_id"], 2)  # send 30 kits
        else:
            # More food need → send food
            return (FOOD, most_hungry_zone["zone_id"], 2)  # send 30 units


# ------------------------------------------------------------------
# Run a full episode and print a report
# ------------------------------------------------------------------

def run_episode(difficulty: str = "medium", verbose: bool = True):
    """
    Run one full episode with the rule-based agent.

    Args:
        difficulty : "easy", "medium", or "hard"
        verbose    : True = print every step, False = just final summary

    Returns:
        dict: summary of the episode (total reward, steps, etc.)
    """
    env   = DisasterEnv(difficulty)
    agent = RuleBasedAgent()

    obs  = env.reset()
    done = False

    total_reward = 0.0
    step_rewards = []

    RESOURCE_NAMES = {FOOD: "Food", MEDICAL: "Medical", RESCUE: "Rescue"}
    QUANTITY_MAP   = {0: 10, 1: 20, 2: 30}

    if verbose:
        print(f"\n{'='*58}")
        print(f"  Rule-Based Agent  |  Difficulty: {difficulty.upper()}")
        print(f"{'='*58}")
        env.render()

    while not done:
        # Agent picks action based on current state
        action = agent.pick_action(obs)
        resource_type, zone_id, qty_idx = action

        # Apply action to environment
        obs, reward, done, info = env.step(action)

        total_reward += reward
        step_rewards.append(reward)

        if verbose:
            zone_name = info["zone"]
            qty       = QUANTITY_MAP[qty_idx]
            res_name  = RESOURCE_NAMES[resource_type]
            print(f"  Step {info['step']:2d} | Sent {qty:2d} {res_name:<8s} → {zone_name}"
                  f"  | reward = {reward:.4f}"
                  f"  | stock F:{info['food_stock']} M:{info['med_stock']} R:{info['resc_stock']}")

    avg_reward = total_reward / len(step_rewards) if step_rewards else 0

    if verbose:
        print(f"\n{'─'*58}")
        print(f"  Episode finished after {len(step_rewards)} steps")
        print(f"  Total reward : {total_reward:.4f}")
        print(f"  Avg reward   : {avg_reward:.4f}")
        print(f"  Final zone states:")
        for zone in obs["zones"]:
            blocked = " [BLOCKED]" if zone["rescue_blocked"] else ""
            print(f"    {zone['name']}{blocked} | injured={zone['injured']} | food_need={zone['food_need']}")
        print(f"{'='*58}\n")

    return {
        "difficulty":   difficulty,
        "steps":        len(step_rewards),
        "total_reward": round(total_reward, 4),
        "avg_reward":   round(avg_reward, 4),
        "step_rewards": step_rewards,
    }


def compare_difficulties():
    """
    Run one episode on each difficulty and compare results side by side.
    This shows how the environment scales properly.
    """
    print("\n" + "="*58)
    print("  COMPARING ALL DIFFICULTY LEVELS")
    print("="*58)

    results = []
    for diff in ["easy", "medium", "hard"]:
        result = run_episode(diff, verbose=False)
        results.append(result)
        print(f"  {diff.upper():<8} | steps={result['steps']:<3} "
              f"| total={result['total_reward']:.4f} "
              f"| avg={result['avg_reward']:.4f}")

    print("="*58)
    print()

    best = max(results, key=lambda r: r["avg_reward"])
    print(f"  Highest avg reward: {best['difficulty'].upper()} ({best['avg_reward']:.4f})")
    print(f"  (Easy should score highest — that confirms difficulty scaling works)\n")


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    # Run a full verbose episode on medium difficulty
    run_episode("medium", verbose=True)

    # Then compare all 3 difficulties side by side
    compare_difficulties()
