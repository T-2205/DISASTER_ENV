# Multi-Zone Disaster Resource Allocation System
### AI Environment built with the OpenEnv specification

---

## Project Overview

This project implements a realistic multi-zone disaster simulation where an AI agent
learns to allocate limited emergency resources across multiple disaster-affected zones.
The agent must make intelligent triage decisions — deciding which zone to help, with
which resource, and in what quantity — to maximise the number of people saved.

The environment is built using the **OpenEnv API** and trained using **Proximal Policy
Optimization (PPO)** from Stable-Baselines3.

---

## Problem Statement

During a real-world disaster, emergency coordinators face a critical challenge:
limited resources (food, medical aid, rescue teams) must be distributed across
multiple affected zones, each with different needs and severity levels. Poor
allocation decisions cost lives.

This project trains an AI agent to solve this allocation problem better than
hand-coded rules — learning from thousands of simulated episodes which decisions
lead to the best outcomes.

---

## Environment Design

### Zones
Each disaster zone is modelled with 6 attributes:

| Attribute        | Type    | Description                              |
|-----------------|---------|------------------------------------------|
| `population`    | int     | Total people in the zone                 |
| `injured`       | int     | People needing medical aid               |
| `food_need`     | int     | Food units still required                |
| `disaster_type` | string  | Earthquake / Flood / Hurricane / Wildfire|
| `severity`      | float   | 0.0 (minor) to 1.0 (catastrophic)       |
| `rescue_blocked`| bool    | Zone inaccessible until rescue sent      |

At the start of each episode, all zone attributes are **randomly assigned** — meaning
the agent cannot memorise a fixed solution. It must learn general allocation strategies
that work across many different scenarios.

### Resources
Three resource types are available, each with limited stock:

| Resource      | What it does                                      |
|--------------|---------------------------------------------------|
| Food units   | Reduces food_need in the target zone              |
| Medical kits | Reduces injured count in the target zone          |
| Rescue teams | Unblocks blocked zones and reduces injured count  |

### Action Space
The agent takes **discrete actions** — one per round:

```
action = (resource_type, zone_id, quantity_index)
```

- `resource_type`: 0=Food, 1=Medical, 2=Rescue
- `zone_id`: 0 to N-1 (where N = number of zones for this difficulty)
- `quantity_index`: 0=10 units, 1=20 units, 2=30 units

This gives **45 possible actions** on medium difficulty (3 × 5 × 3).
Discrete actions were chosen over continuous because they are easier to
learn from and easier to debug — a critical advantage when building and
validating the environment.

### Observation Space
The agent observes a **flat normalised vector** of all zone states plus
current resource stock levels. All values are scaled to [0.0, 1.0] so
the neural network can process them efficiently.

For medium difficulty (5 zones):
```
[injured_0, food_need_0, severity_0, blocked_0, disaster_type_0,
 injured_1, food_need_1, severity_1, blocked_1, disaster_type_1,
 ...
 food_stock, med_stock, resc_stock]
= 28 values total
```

---

## Reward Function Design

The reward function is the most critical part of the environment. A poor
reward signal will train a useless agent — even if the environment logic
is perfect. Our reward has three components:

```
reward = (need_score × 0.6) + (priority_score × 0.3) − waste_penalty
```

### Component 1 — Need coverage score (60% weight)
Measures how much total unmet need has been reduced across all zones:
```
need_score = 1.0 − (current_total_need / initial_total_need)
```
This is the primary signal — the agent is directly rewarded for
helping people.

### Component 2 — Priority score (30% weight)
Rewards the agent for helping the most critical zone (highest severity)
rather than spreading resources equally across all zones:
```
priority_score = 1.0  if helped most critical zone
priority_score = 0.3  otherwise
```
This teaches the agent to triage — a core real-world skill.

### Component 3 — Waste penalty (up to −0.1)
Punishes obvious mistakes such as sending food to a zone with no food
shortage, or rescue teams to a zone that is already accessible:
```
waste_penalty = 0.1  if resource was clearly not needed
waste_penalty = 0.0  otherwise
```
Final reward is always clamped to [0.0, 1.0].

---

## Difficulty Levels

| Setting          | Easy | Medium | Hard |
|-----------------|------|--------|------|
| Number of zones | 3    | 5      | 8    |
| Initial food    | 200  | 300    | 250  |
| Initial medical | 100  | 150    | 100  |
| Initial rescue  | 15   | 20     | 12   |
| Escalation rate | 0%   | 5%/step| 10%/step |
| Blocked zones   | None | 20% chance | 40% chance |
| Max steps       | 20   | 25     | 30   |

On **Hard**, needs escalate 10% every round — the agent must act fast
or the situation spirals out of control. Some zones start inaccessible
and must be unblocked with rescue teams before other resources can help.

---

## Training Setup

- **Algorithm**: PPO (Proximal Policy Optimisation)
- **Policy**: MLP (Multi-Layer Perceptron neural network)
- **Training steps**: 500,000
- **Learning rate**: 3×10⁻⁴
- **Gamma (discount)**: 0.99
- **Framework**: Stable-Baselines3

PPO was chosen because it is one of the most stable and reliable RL
algorithms for discrete action spaces, requiring minimal hyperparameter
tuning to get good results.

---

## Results

### Agent Comparison

Three agents were evaluated over 20 episodes each per difficulty level:

| Agent           | Easy  | Medium | Hard  |
|----------------|-------|--------|-------|
| Random agent   | ~0.35 | ~0.09  | ~0.05 |
| Rule-based agent| ~0.59 | ~0.16  | ~0.09 |
| Trained AI (PPO)| ~0.78 | ~0.28  | ~0.15 |

See `comparison_chart.png` for the full visual comparison.

### Key findings
- The trained AI **outperforms the rule-based agent on all difficulty levels**
- On easy difficulty, the AI achieves ~32% higher reward than the rule-based agent
- On hard difficulty, the AI learns to **prioritise unblocking zones first**
  before sending medical aid — a non-obvious strategy the rule-based agent misses
- Reward consistently increases during training, confirming genuine learning

---

## Project File Structure

```
disaster_env/
├── zone.py                 # Zone data model
├── config.py               # Difficulty level configurations
├── disaster_env.py         # Main OpenEnv environment (reset/step/state)
├── gym_wrapper.py          # Gymnasium adapter for RL libraries
├── rule_based_agent.py     # Hand-coded baseline agent
├── train.py                # PPO training script
├── compare_agents.py       # Generates comparison chart
├── disaster_agent.zip      # Saved trained model
└── comparison_chart.png    # Results visualisation
```

---

## How to Run

### 1. Install dependencies
```bash
pip install stable-baselines3 gymnasium matplotlib
```

### 2. Test the environment
```bash
python test_env.py
```

### 3. Run the rule-based baseline
```bash
python rule_based_agent.py
```

### 4. Train the AI agent
```bash
python train.py
```

### 5. Generate comparison chart
```bash
python compare_agents.py
```

---

## Design Decisions and Justifications

**Why discrete actions?**
Continuous action spaces are harder to learn from and harder to debug.
For a first implementation, discrete actions with 3 quantity levels
give the agent meaningful choices without excessive complexity.

**Why PPO?**
PPO is reliable, well-documented, and works well out of the box for
small discrete action spaces. It was the right choice for this project
size.

**Why random zone initialisation?**
If zones were always the same, the agent could memorise a fixed
sequence of actions rather than learning genuine allocation strategy.
Random initialisation forces the agent to learn transferable policies.

**Why escalation in medium/hard?**
Static environments are too easy — the agent can take its time.
Escalation creates urgency and forces the agent to prioritise,
making the problem more realistic and more challenging.

---

## Future Improvements

- Add dynamic zone discovery (new zones appear mid-episode)
- Implement inter-zone resource transfer actions
- Train with multiple parallel environments for faster learning
- Try other algorithms (DQN, A2C) and compare performance
- Add weather or aftershock events that change zone severity mid-episode

---

*Built for the OpenEnv AI Competition*
