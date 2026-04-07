"""
test_env.py — Quick sanity check for the disaster environment.

Run this file to make sure everything is wired up correctly.
You should see a few rounds of the simulation printing to the screen.

Usage:
    python test_env.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from disaster_env import DisasterEnv, FOOD, MEDICAL, RESCUE


def test_basic_loop():
    print("TEST 1 — Basic loop (medium difficulty)")
    print("-" * 45)

    env = DisasterEnv("medium")
    obs = env.reset()

    print(f"Episode started. {len(obs['zones'])} zones created.")
    env.render()

    # Run 5 steps with a simple fixed action
    for step in range(5):
        # Always send 20 food to Zone 0 (dumb agent — just for testing)
        action = (FOOD, 0, 1)
        obs, reward, done, info = env.step(action)
        print(f"  Step {step+1}: sent {info['quantity']} {info['resource_sent']} "
              f"to {info['zone']} → reward = {reward:.4f}")
        if done:
            print("  Episode ended early (all zones clear).")
            break

    print("\nTEST 1 PASSED\n")


def test_all_difficulties():
    print("TEST 2 — All difficulty levels reset correctly")
    print("-" * 45)

    for diff in ["easy", "medium", "hard"]:
        env = DisasterEnv(diff)
        obs = env.reset()
        n = len(obs["zones"])
        space = env.action_space_size()
        print(f"  {diff:6s} → {n} zones, {space} possible actions")

    print("\nTEST 2 PASSED\n")


def test_blocked_zone():
    print("TEST 3 — Rescue unblocks a blocked zone")
    print("-" * 45)

    # Force a blocked zone for testing
    env = DisasterEnv("hard")
    obs = env.reset()

    # Manually block zone 0 to guarantee the test
    env.zones[0].rescue_blocked = True
    print(f"  Zone 0 blocked: {env.zones[0].rescue_blocked}")

    # Send rescue to zone 0
    action = (RESCUE, 0, 0)  # 10 rescue teams to zone 0
    obs, reward, done, info = env.step(action)
    print(f"  After rescue → Zone 0 blocked: {env.zones[0].rescue_blocked}")
    print(f"  Reward: {reward:.4f}")

    assert not env.zones[0].rescue_blocked, "Zone should be unblocked!"
    print("\nTEST 3 PASSED\n")


def test_invalid_action():
    print("TEST 4 — Invalid action raises clear error")
    print("-" * 45)

    env = DisasterEnv("easy")
    env.reset()

    try:
        env.step((5, 0, 0))  # resource_type=5 is invalid
        print("  ERROR: Should have raised ValueError!")
    except ValueError as e:
        print(f"  Correctly raised ValueError: {e}")

    print("\nTEST 4 PASSED\n")


if __name__ == "__main__":
    test_basic_loop()
    test_all_difficulties()
    test_blocked_zone()
    test_invalid_action()
    print("ALL TESTS PASSED!")
