"""
compare_agents.py — Compare 3 agents side by side and save a chart.

Agents compared:
    1. Random agent     — picks completely random actions
    2. Rule-based agent — follows fixed triage rules
    3. Trained AI       — your PPO agent

Produces:
    comparison_chart.png — saved in your project folder

Usage:
    python compare_agents.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import random
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("\npip install matplotlib   <- run this first\n")
    sys.exit(1)

from disaster_env import DisasterEnv, FOOD, MEDICAL, RESCUE
from gym_wrapper  import DisasterGymEnv

try:
    from stable_baselines3 import PPO
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("Warning: stable-baselines3 not installed — AI column skipped.\n")


# ── Detect which difficulty the saved model was trained on ──────────

def get_model_difficulty() -> str:
    """
    Look at the saved model's observation size and figure out
    which difficulty it was trained on.
    easy   = 3 zones = 3*5+3 = 18 values
    medium = 5 zones = 5*5+3 = 28 values
    hard   = 8 zones = 8*5+3 = 43 values
    """
    model_path = os.path.join(os.path.dirname(__file__), "disaster_agent")
    if not os.path.exists(model_path + ".zip"):
        return None

    model = PPO.load(model_path)
    size  = model.observation_space.shape[0]
    mapping = {18: "easy", 28: "medium", 43: "hard"}
    return mapping.get(size, None)


# ═══════════════════════════════════════════════════════════════════
# Agent 1 — Random
# ═══════════════════════════════════════════════════════════════════

def run_random_agent(difficulty: str, n_episodes: int = 20) -> list:
    rewards = []
    for _ in range(n_episodes):
        env   = DisasterEnv(difficulty)
        obs   = env.reset()
        done  = False
        total = 0.0
        while not done:
            resource = random.randint(0, 2)
            zone     = random.randint(0, len(env.zones) - 1)
            qty      = random.randint(0, 2)
            _, r, done, _ = env.step((resource, zone, qty))
            total += r
        rewards.append(total)
    return rewards


# ═══════════════════════════════════════════════════════════════════
# Agent 2 — Rule-based
# ═══════════════════════════════════════════════════════════════════

def run_rule_agent(difficulty: str, n_episodes: int = 20) -> list:
    rewards = []
    for _ in range(n_episodes):
        env   = DisasterEnv(difficulty)
        obs   = env.reset()
        done  = False
        total = 0.0
        while not done:
            action = None
            for z in env.zones:
                if z.rescue_blocked:
                    action = (RESCUE, z.zone_id, 1)
                    break
            if action is None:
                most_inj  = max(env.zones, key=lambda z: z.injured)
                most_food = max(env.zones, key=lambda z: z.food_need)
                if most_inj.injured >= most_food.food_need:
                    action = (MEDICAL, most_inj.zone_id, 2)
                else:
                    action = (FOOD, most_food.zone_id, 2)
            _, r, done, _ = env.step(action)
            total += r
        rewards.append(total)
    return rewards


# ═══════════════════════════════════════════════════════════════════
# Agent 3 — Trained PPO AI
# Only runs on the difficulty it was actually trained on
# ═══════════════════════════════════════════════════════════════════

def run_trained_agent(difficulty: str, model_diff: str, n_episodes: int = 20) -> list:
    if not SB3_AVAILABLE:
        return []

    # Only evaluate on the difficulty the model was trained for
    if difficulty != model_diff:
        return []

    model_path = os.path.join(os.path.dirname(__file__), "disaster_agent")
    model      = PPO.load(model_path)
    rewards    = []

    for _ in range(n_episodes):
        env      = DisasterGymEnv(model_diff)
        obs, _   = env.reset()
        done     = False
        total    = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, _ = env.step(int(action))
            total += r
            done   = terminated or truncated
        rewards.append(total)

    return rewards


# ═══════════════════════════════════════════════════════════════════
# Collect all results
# ═══════════════════════════════════════════════════════════════════

def collect_results(n_episodes: int = 20) -> dict:
    difficulties = ["easy", "medium", "hard"]
    model_diff   = get_model_difficulty()

    if model_diff:
        print(f"  Detected trained model was trained on: {model_diff.upper()}")
        print(f"  AI column will only show for {model_diff.upper()} difficulty.\n")
    else:
        print("  No trained model found — AI column will be empty.")
        print("  Run: python train_500k.py first, then retry.\n")

    results = {"random": {}, "rule_based": {}, "trained_ai": {}}

    for diff in difficulties:
        print(f"  Running {n_episodes} episodes on {diff.upper()}...")

        rnd  = run_random_agent(diff, n_episodes)
        rule = run_rule_agent(diff, n_episodes)
        ai   = run_trained_agent(diff, model_diff, n_episodes) if model_diff else []

        results["random"][diff]     = rnd
        results["rule_based"][diff] = rule
        results["trained_ai"][diff] = ai

        avg_rnd  = sum(rnd)  / len(rnd)  if rnd  else 0
        avg_rule = sum(rule) / len(rule) if rule else 0
        avg_ai   = sum(ai)   / len(ai)   if ai   else 0

        ai_str = f"  Trained AI={avg_ai:.3f}" if ai else "  Trained AI=N/A (wrong diff)"
        print(f"    Random={avg_rnd:.3f}  Rule-based={avg_rule:.3f}{ai_str}")

    return results


# ═══════════════════════════════════════════════════════════════════
# Plot
# ═══════════════════════════════════════════════════════════════════

def plot_results(results: dict, save_path: str = "comparison_chart.png"):
    difficulties = ["easy", "medium", "hard"]

    def stats(agent, diff):
        vals = results[agent].get(diff, [])
        if not vals:
            return None, None
        return float(np.mean(vals)), float(np.std(vals))

    agents  = ["random", "rule_based", "trained_ai"]
    labels  = ["Random agent", "Rule-based agent", "Trained AI (PPO)"]
    colors  = ["#B4B2A9", "#5DCAA5", "#7F77DD"]
    bar_w   = 0.22
    x       = np.arange(len(difficulties))

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#FAFAF8")
    ax.set_facecolor("#FAFAF8")

    for i, (agent, label, color) in enumerate(zip(agents, labels, colors)):
        means, stds, positions = [], [], []
        for j, diff in enumerate(difficulties):
            m, s = stats(agent, diff)
            if m is not None:
                means.append(m)
                stds.append(s)
                positions.append(x[j] + i * bar_w)

        if not means:
            continue

        bars = ax.bar(positions, means, bar_w, label=label, color=color,
                      yerr=stds, capsize=4,
                      error_kw=dict(elinewidth=1, ecolor="#888780"), zorder=3)

        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{mean:.2f}", ha="center", va="bottom",
                    fontsize=9, color="#444441")

    ax.set_xlabel("Difficulty level", fontsize=12, color="#444441", labelpad=8)
    ax.set_ylabel("Average total reward per episode", fontsize=12, color="#444441", labelpad=8)
    ax.set_title("Agent comparison — Random vs Rule-based vs Trained AI",
                 fontsize=14, color="#2C2C2A", pad=16, fontweight="normal")
    ax.set_xticks(x + bar_w)
    ax.set_xticklabels([d.capitalize() for d in difficulties], fontsize=11, color="#444441")
    ax.tick_params(axis="y", labelcolor="#888780", labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#D3D1C7")
    ax.spines["bottom"].set_color("#D3D1C7")
    ax.yaxis.grid(True, color="#E8E6DF", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=10, framealpha=0, labelcolor="#444441")

    plt.tight_layout()
    full_path = os.path.join(os.path.dirname(__file__), save_path)
    plt.savefig(full_path, dpi=150, bbox_inches="tight")
    print(f"\n  Chart saved -> {full_path}")
    return full_path


# ═══════════════════════════════════════════════════════════════════
# Summary table
# ═══════════════════════════════════════════════════════════════════

def print_summary(results: dict):
    difficulties = ["easy", "medium", "hard"]
    agents       = [("random", "Random"), ("rule_based", "Rule-based"), ("trained_ai", "Trained AI")]

    print(f"\n{'='*58}")
    print(f"  {'Agent':<18} {'Easy':>8} {'Medium':>8} {'Hard':>8}")
    print(f"  {'─'*18} {'─'*8} {'─'*8} {'─'*8}")

    for key, label in agents:
        row = []
        for diff in difficulties:
            vals = results[key].get(diff, [])
            avg  = sum(vals) / len(vals) if vals else None
            row.append(f"{avg:>8.3f}" if avg is not None else "     N/A")
        print(f"  {label:<18} {''.join(row)}")

    print(f"{'='*58}\n")


# ═══════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    N_EPISODES = 20

    print(f"\n{'='*58}")
    print(f"  Running agent comparison ({N_EPISODES} episodes each)...")
    print(f"{'='*58}\n")

    results = collect_results(N_EPISODES)
    print_summary(results)
    plot_results(results)

    print("  Done! Open comparison_chart.png in VS Code to see it.")
    print("  Use this chart in your competition presentation.\n")
