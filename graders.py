"""
graders.py — Deterministic task graders for each difficulty level.

The OpenEnv spec requires each task to have a programmatic grader that:
  - Scores performance from 0.0 to 1.0
  - Is deterministic and reproducible
  - Has clear success/failure criteria

Each grader runs a full episode with a given agent and returns a GraderResult.
The graders use a fixed random seed so results are always reproducible.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import random
from models import GraderResult
from disaster_env import DisasterEnv, FOOD, MEDICAL, RESCUE


# ══════════════════════════════════════════════════════════════════
# Base grader — shared logic
# ══════════════════════════════════════════════════════════════════

class BaseGrader:
    """
    Base class for all task graders.

    Subclasses define:
        task_id           : str   — "easy" | "medium" | "hard"
        difficulty        : str   — matches DisasterEnv difficulty
        success_threshold : float — minimum score to "pass"
    """

    task_id           = "base"
    difficulty        = "easy"
    success_threshold = 0.5
    seed              = 42          # fixed seed = reproducible results

    def run(self, agent_fn, n_episodes: int = 10) -> GraderResult:
        """
        Run n_episodes with the given agent function and return a GraderResult.

        agent_fn: callable(env) -> action tuple
            A function that takes the current DisasterEnv and returns an action.

        The seed is fixed so every grader run with the same agent produces
        the same score — fully deterministic and reproducible.
        """
        random.seed(self.seed)

        all_rewards    = []
        all_steps      = []
        all_cleared    = []
        last_env_state = None

        for ep in range(n_episodes):
            env   = DisasterEnv(self.difficulty)
            obs   = env.reset()
            done  = False
            ep_reward = 0.0

            while not done:
                action = agent_fn(env)
                obs, reward, done, info = env.step(action)
                ep_reward += reward

            all_rewards.append(ep_reward)
            all_steps.append(env.step_count)
            cleared = sum(1 for z in env.zones if z.injured == 0 and z.food_need == 0)
            all_cleared.append(cleared)
            last_env_state = env

        avg_reward  = sum(all_rewards) / len(all_rewards)
        avg_steps   = int(sum(all_steps) / len(all_steps))
        avg_cleared = int(sum(all_cleared) / len(all_cleared))

        # Normalise score to [0.0, 1.0]
        score = round(min(1.0, max(0.0, avg_reward / self._max_possible_reward())), 4)

        details = {
            "avg_total_reward": round(avg_reward, 4),
            "avg_steps":        avg_steps,
            "n_episodes":       n_episodes,
            "final_zones": [
                {
                    "name":          z.name,
                    "injured_left":  z.injured,
                    "food_left":     z.food_need,
                    "cleared":       z.injured == 0 and z.food_need == 0
                }
                for z in (last_env_state.zones if last_env_state else [])
            ]
        }

        return GraderResult(
            task_id      = self.task_id,
            score        = score,
            passed       = score >= self.success_threshold,
            steps_taken  = avg_steps,
            total_reward = round(avg_reward, 4),
            zones_cleared= avg_cleared,
            details      = details,
        )

    def _max_possible_reward(self) -> float:
        """
        Theoretical maximum total reward for this difficulty.
        Used for normalising to [0, 1].
        Each step can score at most 1.0, so max = max_steps.
        We use a conservative 60% of max_steps as "excellent" baseline.
        """
        from config import get_config
        cfg = get_config(self.difficulty)
        return cfg["max_steps"] * 0.6


# ══════════════════════════════════════════════════════════════════
# Easy grader
# ══════════════════════════════════════════════════════════════════

class EasyGrader(BaseGrader):
    """
    Task: Basic Resource Allocation
    3 zones, abundant resources, no escalation.
    Expected: most agents can achieve 0.5+ with basic logic.
    """
    task_id           = "easy"
    difficulty        = "easy"
    success_threshold = 0.5


# ══════════════════════════════════════════════════════════════════
# Medium grader
# ══════════════════════════════════════════════════════════════════

class MediumGrader(BaseGrader):
    """
    Task: Triage Under Scarcity
    5 zones, moderate scarcity, 5% escalation per step.
    Requires learning to prioritise critical zones.
    Expected: rule-based ~0.3, trained AI ~0.5+
    """
    task_id           = "medium"
    difficulty        = "medium"
    success_threshold = 0.3


# ══════════════════════════════════════════════════════════════════
# Hard grader
# ══════════════════════════════════════════════════════════════════

class HardGrader(BaseGrader):
    """
    Task: Crisis Management
    8 zones, severe scarcity, 10% escalation, 40% blocked zones.
    Agent cannot save everyone — must triage aggressively.
    Expected: genuinely challenges frontier models.
    """
    task_id           = "hard"
    difficulty        = "hard"
    success_threshold = 0.15


# ══════════════════════════════════════════════════════════════════
# Standard agent functions for grader testing
# ══════════════════════════════════════════════════════════════════

def random_agent(env: DisasterEnv):
    """Picks a completely random action."""
    resource = random.randint(0, 2)
    zone     = random.randint(0, len(env.zones) - 1)
    qty      = random.randint(0, 2)
    return (resource, zone, qty)


def rule_based_agent(env: DisasterEnv):
    """Triage agent: unblock first, then help most critical need."""
    for z in env.zones:
        if z.rescue_blocked:
            return (RESCUE, z.zone_id, 1)
    most_inj  = max(env.zones, key=lambda z: z.injured)
    most_food = max(env.zones, key=lambda z: z.food_need)
    if most_inj.injured >= most_food.food_need:
        return (MEDICAL, most_inj.zone_id, 2)
    return (FOOD, most_food.zone_id, 2)


# ══════════════════════════════════════════════════════════════════
# Run all graders and print results
# ══════════════════════════════════════════════════════════════════

def run_all_graders(agent_fn=None, n_episodes: int = 10):
    """
    Run all 3 graders with the given agent and print a summary.
    Defaults to rule_based_agent if none provided.
    """
    if agent_fn is None:
        agent_fn = rule_based_agent

    graders = [EasyGrader(), MediumGrader(), HardGrader()]
    results = []

    print(f"\n{'='*55}")
    print(f"  Running all task graders ({n_episodes} episodes each)")
    print(f"{'='*55}")

    for grader in graders:
        print(f"  Grading {grader.task_id.upper()}...")
        result = grader.run(agent_fn, n_episodes)
        results.append(result)
        status = "PASS" if result.passed else "FAIL"
        print(f"    Score: {result.score:.4f}  [{status}]  "
              f"Zones cleared: {result.zones_cleared}")

    print(f"\n{'─'*55}")
    all_passed = all(r.passed for r in results)
    print(f"  Overall: {'ALL TASKS PASSED' if all_passed else 'SOME TASKS FAILED'}")
    print(f"{'='*55}\n")

    return results


if __name__ == "__main__":
    print("Testing with rule-based agent:")
    run_all_graders(rule_based_agent)

    print("Testing with random agent:")
    run_all_graders(random_agent)
