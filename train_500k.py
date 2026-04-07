"""
train.py — Train a real AI agent using Stable-Baselines3.

This is Step 6 — the real AI. Instead of hand-coded rules,
the agent now LEARNS by trial and error over thousands of episodes.

Algorithm: PPO (Proximal Policy Optimization)
    - One of the most reliable RL algorithms for beginners
    - Works well with small action spaces like ours (45 actions)
    - Doesn't need much tuning to get started

How it works in plain words:
    1. Agent tries random actions at first
    2. Gets reward scores back from the environment
    3. Slowly figures out which actions lead to higher rewards
    4. After enough episodes, it starts making smart decisions

Usage:
    python train.py

After training, a file called "disaster_agent.zip" is saved.
You can load it later and watch it play without retraining.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

# ── Check dependencies before anything else ──────────────────────────
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
    from stable_baselines3.common.monitor import Monitor
except ImportError:
    print("\n" + "="*58)
    print("  Missing library! Run this command first:")
    print()
    print("  pip install stable-baselines3 gymnasium")
    print()
    print("  Then run train.py again.")
    print("="*58 + "\n")
    sys.exit(1)

import numpy as np
from gym_wrapper import DisasterGymEnv


# ------------------------------------------------------------------
# Custom callback — prints a live progress report every N steps
# ------------------------------------------------------------------

class ProgressCallback(BaseCallback):
    """
    Prints training progress to the terminal every 2000 steps.
    Shows average reward so you can see the AI improving in real time.
    """

    def __init__(self, print_every: int = 2000, verbose: int = 0):
        super().__init__(verbose)
        self.print_every  = print_every
        self.episode_rewards = []
        self.current_ep_reward = 0.0

    def _on_step(self) -> bool:
        # Track reward for current episode
        reward = self.locals["rewards"][0]
        self.current_ep_reward += reward

        # Check if episode ended
        done = self.locals["dones"][0]
        if done:
            self.episode_rewards.append(self.current_ep_reward)
            self.current_ep_reward = 0.0

        # Print summary every N steps
        if self.num_timesteps % self.print_every == 0 and self.episode_rewards:
            recent = self.episode_rewards[-20:]  # last 20 episodes
            avg    = sum(recent) / len(recent)
            best   = max(self.episode_rewards)
            print(f"  Step {self.num_timesteps:>7,} | "
                  f"avg reward (last 20 eps): {avg:6.3f} | "
                  f"best ever: {best:6.3f}")

        return True  # return False to stop training early


# ------------------------------------------------------------------
# Training function
# ------------------------------------------------------------------

def train(difficulty: str = "medium", total_steps: int = 100_000):
    """
    Train a PPO agent on the disaster environment.

    Args:
        difficulty   : "easy", "medium", or "hard"
        total_steps  : how many environment steps to train for
                       - 50,000  = quick test (~2 min)
                       - 100,000 = decent agent (~4 min)
                       - 500,000 = well-trained agent (~20 min)

    Saves:
        disaster_agent.zip — the trained model you can reload later
        logs/              — training logs for TensorBoard (optional)
    """

    print(f"\n{'='*58}")
    print(f"  Training PPO agent  |  Difficulty: {difficulty.upper()}")
    print(f"  Total steps: {total_steps:,}")
    print(f"{'='*58}\n")

    # --- Create the environment ---
    # Monitor wraps it to track episode stats automatically
    env = Monitor(DisasterGymEnv(difficulty))

    # --- Create the PPO agent ---
    model = PPO(
        policy             = "MlpPolicy",   # Multi-layer perceptron (standard neural net)
        env                = env,
        learning_rate      = 3e-4,          # How fast the AI updates its knowledge
        n_steps            = 512,           # Steps collected before each update
        batch_size         = 64,            # How many samples per training batch
        n_epochs           = 10,            # How many passes over each batch
        gamma              = 0.99,          # How much future rewards matter (0–1)
        verbose            = 0,             # Suppress SB3's own logs (we have ours)
        tensorboard_log    = "./logs/",     # Optional: visualise in TensorBoard
    )

    # --- Attach our progress printer ---
    callback = ProgressCallback(print_every=2000)

    print("  Training started. Watch the reward go up!\n")
    print(f"  {'Step':>10} | {'Avg reward (last 20 eps)':^30} | {'Best ever':>10}")
    print(f"  {'─'*10}─{'─'*32}─{'─'*10}")

    # --- Train! ---
    model.learn(
        total_timesteps   = total_steps,
        callback          = callback,
        progress_bar      = False,
    )

    # --- Save the trained model ---
    model_path = "disaster_agent"
    model.save(model_path)
    print(f"\n  Model saved → {model_path}.zip")
    print(f"  Training complete!\n")

    return model, env


# ------------------------------------------------------------------
# Evaluation function — watch the trained agent play
# ------------------------------------------------------------------

def evaluate(model, difficulty: str = "medium", n_episodes: int = 5):
    """
    Load a trained model and watch it play several episodes.
    Compares its score against the rule-based agent baseline.

    Args:
        model      : trained PPO model (from train())
        difficulty : must match what the model was trained on
        n_episodes : how many episodes to evaluate
    """
    print(f"\n{'='*58}")
    print(f"  Evaluating trained agent  |  {n_episodes} episodes")
    print(f"{'='*58}\n")

    # Rule-based baseline (from Step 5) for comparison
    BASELINES = {"easy": 0.59, "medium": 0.16, "hard": 0.09}
    baseline  = BASELINES.get(difficulty, 0.0)

    all_rewards = []

    for ep in range(1, n_episodes + 1):
        env  = DisasterGymEnv(difficulty)
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        steps = 0

        while not done:
            # Model picks action (deterministic=True = no randomness)
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            steps     += 1
            done = terminated or truncated

        all_rewards.append(ep_reward)
        print(f"  Episode {ep}: total reward = {ep_reward:.4f}  ({steps} steps)")

    avg = sum(all_rewards) / len(all_rewards)
    print(f"\n  Average reward   : {avg:.4f}")
    print(f"  Rule-based baseline: {baseline:.4f}")

    if avg > baseline:
        improvement = ((avg - baseline) / baseline) * 100
        print(f"  AI beats the baseline by {improvement:.1f}%! The AI is learning.")
    else:
        print(f"  AI hasn't beaten the baseline yet — try training longer.")

    print(f"{'='*58}\n")


# ------------------------------------------------------------------
# Quick demo — load a saved model and watch one episode step by step
# ------------------------------------------------------------------

def watch_one_episode(model_path: str = "disaster_agent", difficulty: str = "medium"):
    """
    Load a saved model and print every decision it makes.
    Great for seeing exactly what the AI learned to do.
    """
    model = PPO.load(model_path)
    env   = DisasterGymEnv(difficulty)

    RESOURCE_NAMES = {0: "Food", 1: "Medical", 2: "Rescue"}
    QTY_MAP        = {0: 10, 1: 20, 2: 30}

    print(f"\n{'='*58}")
    print(f"  Watching trained agent play one episode")
    print(f"{'='*58}\n")

    obs, _ = env.reset()
    env.render()

    done = False
    total_reward = 0.0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))

        # Decode action for display
        decoded     = env._action_lookup[int(action)]
        res_name    = RESOURCE_NAMES[decoded[0]]
        zone_name   = env._env.zones[decoded[1]].name
        qty         = QTY_MAP[decoded[2]]
        total_reward += reward

        print(f"  Step {info['step']:2d} | Sent {qty} {res_name:<8} → {zone_name}"
              f"  | reward = {reward:.4f}")

        done = terminated or truncated

    print(f"\n  Total reward: {total_reward:.4f}")
    print(f"{'='*58}\n")


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    DIFFICULTY  = "medium"
    TOTAL_STEPS = 500_000  # increase to 500_000 for a better agent

    # 1. Train the agent
    model, env = train(DIFFICULTY, TOTAL_STEPS)

    # 2. Evaluate how well it learned
    evaluate(model, DIFFICULTY, n_episodes=5)

    # 3. Watch it play one full episode step by step
    watch_one_episode("disaster_agent", DIFFICULTY)
