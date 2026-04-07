# Competition Pitch Script
## Multi-Zone Disaster Resource Allocation AI

---

## YOUR 3-MINUTE PITCH — say this out loud, practise it

---

### Opening (20 seconds)

"Imagine a massive earthquake hits a city.
Five different zones are affected. Each zone has hundreds of injured people,
food shortages, and some zones are completely blocked off.
You have limited food, medicine, and rescue teams.
Who do you help first? How much do you send?
A wrong decision costs lives.
We built an AI that learns to answer that question — better than a human with fixed rules."

---

### What we built (40 seconds)

"We built a disaster simulation environment from scratch using the OpenEnv API.

The environment has:
- 5 disaster zones, each randomly hit by an earthquake, flood, hurricane, or wildfire
- 3 types of resources: food, medical kits, and rescue teams — all limited
- 3 difficulty levels: Easy, Medium, and Hard
- A reward function that scores every decision from 0.0 to 1.0

The AI agent — called PPO — starts knowing absolutely nothing.
It makes random decisions at first.
But after training on 500,000 simulated episodes,
it learns: unblock zones first, help the most critical zones,
don't waste medicine on zones that aren't injured.
It figures all of this out by itself, just from the reward signal."

---

### Results (30 seconds)

"We compared three agents:
- A random agent that picks actions blindly
- A rule-based agent we hand-coded with triage logic
- Our trained AI

On medium difficulty:
  Random scored 0.09
  Rule-based scored 0.16
  Our trained AI scored 0.28 — that's 75% better than random
  and 40% better than hand-coded rules.

[Show the comparison chart here]

The reward consistently increases during training — proof that genuine
learning is happening, not just luck."

---

### Why it matters (20 seconds)

"Real disaster response coordinators face exactly this problem —
not enough resources, too many zones, time pressure.
This project shows that reinforcement learning can learn allocation
strategies that outperform fixed human logic.
With more data and real-world zone information, this approach
could genuinely assist emergency coordinators."

---

### Closing (10 seconds)

"We built the environment, designed the reward function, trained the agent,
and validated the results — all from scratch.
Thank you."

---

---

## QUESTIONS JUDGES WILL ASK — your answers

---

**Q: What is PPO?**

"PPO stands for Proximal Policy Optimization.
It's a reinforcement learning algorithm — basically a method for training
an AI by trial and error. The AI tries an action, gets a reward score,
and slowly adjusts its behaviour to get higher scores next time.
PPO is one of the most reliable algorithms for this type of problem
because it learns steadily without making drastic jumps that break things."

---

**Q: Why did you choose discrete actions instead of continuous?**

"Discrete means the AI picks from a fixed list — send 10, 20, or 30 units.
Continuous would mean picking any number like 17.3.
We chose discrete because it's much easier for the AI to learn from —
there are only 45 possible actions, so the AI can explore all of them
quickly. It also makes debugging easier — you can see exactly what
the agent chose every single step."

---

**Q: How does your reward function work?**

"The reward has three parts.
60% comes from how much total unmet need we reduced across all zones —
so helping more people means a higher score.
30% is a priority bonus — if the AI helped the most critical zone
(the one with the highest severity), it gets extra reward.
This teaches triage — help the worst cases first.
The last part is a penalty — up to minus 10% — if the AI wasted
resources, like sending food to a zone that doesn't need food.
The final score is always between 0.0 and 1.0."

---

**Q: Why does the situation get worse each round?**

"That's the escalation feature. On medium difficulty, needs grow 5%
every round. On hard, 10%.
This makes the problem realistic — in a real disaster, if you don't
act fast, the situation gets worse.
It also forces the AI to prioritise rather than just slowly helping
everyone equally."

---

**Q: What would you improve with more time?**

"Three things.
First, add real geographic data — actual distances between zones
that affect how quickly resources can be delivered.
Second, compare more algorithms — we used PPO but DQN or A2C
might perform better on hard difficulty.
Third, add dynamic events — aftershocks or weather changes
mid-episode that force the agent to adapt."

---

**Q: Did you build all of this yourself?**

"Yes. We designed the environment from scratch — the zone model,
the resource system, the reward function, the difficulty levels.
We used Stable-Baselines3 as the RL library, which is like using
TensorFlow for deep learning — a standard professional tool.
But the actual environment and reward design is all our own work."

---

---

## THINGS TO REMEMBER ON PRESENTATION DAY

1. Open the demo dashboard BEFORE your presentation — have it ready to go
2. Show the comparison_chart.png on screen when you say the results
3. Speak slowly — judges are writing notes while you talk
4. If you don't know an answer, say: "That's a great question —
   we considered that but prioritised X instead because Y"
   Never say "I don't know" and stop. Always add what you DID do.
5. Make eye contact — don't read from notes the whole time
6. The demo dashboard is your biggest visual advantage — let it run
   during your presentation so judges can see the AI making decisions live

---

*Practise the opening 3 minutes out loud at least 5 times before the competition.*
