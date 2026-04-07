---
title: Disaster Resource Allocation Environment
emoji: 🚨
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
tags:
  - openenv
  - reinforcement-learning
  - disaster-response
  - resource-allocation
  - multi-agent
license: mit
---

# Disaster Resource Allocation — OpenEnv Environment

A multi-zone disaster response simulation where an AI agent learns to allocate
limited emergency resources across disaster-affected zones.

## Quick Start

```bash
# Reset environment (start new episode)
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"difficulty": "medium"}'

# Take a step (send 30 medical kits to zone 2)
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"resource_type": 1, "zone_id": 2, "quantity_index": 2}'

# Get current state
curl http://localhost:7860/state
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/`      | GET    | Health check |
| `/reset` | POST   | Start new episode |
| `/step`  | POST   | Apply action |
| `/state` | GET    | Current world state |
| `/tasks` | GET    | List all tasks |
| `/grade` | POST   | Run grader |
| `/docs`  | GET    | Interactive API docs |

## Environment Details

**Observation space:** Flat float32 vector — zone states + resource stocks
- Easy (3 zones): shape (18,)
- Medium (5 zones): shape (28,)
- Hard (8 zones): shape (43,)

**Action space:** Discrete integer encoding (resource, zone, quantity)
- Easy: 27 actions | Medium: 45 actions | Hard: 72 actions

**Reward:** 0.0–1.0 dense signal per step
- 60% need coverage score
- 30% priority bonus
- up to -10% waste penalty

## Tasks

| Task | Zones | Max Steps | Threshold |
|------|-------|-----------|-----------|
| Easy | 3 | 20 | 0.50 |
| Medium | 5 | 25 | 0.30 |
| Hard | 8 | 30 | 0.15 |

## Run Inference

```bash
export API_BASE_URL="https://api-inference.huggingface.co/v1"
export MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct"
export HF_TOKEN="hf_your_token"
python inference.py
```

## Baseline Scores

| Agent | Easy | Medium | Hard |
|-------|------|--------|------|
| Random | 0.35 | 0.09 | 0.05 |
| Rule-based | 0.59 | 0.16 | 0.09 |
| Trained PPO | — | 0.28 | — |
