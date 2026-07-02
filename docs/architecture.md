# Architecture & Design Notes

## Problem as an MDP

A **Markov Decision Process** is defined by (S, A, R, T, γ):

| MDP Element | This Project |
|---|---|
| **S** — State space | (remaining jobs, machine load vector) |
| **A** — Action space | which job to schedule next |
| **R** — Reward | −1 per step; normalised makespan at terminal step |
| **T** — Transition | deterministic: job assigned to least-loaded machine |
| **γ** — Discount | 0.95 (future rewards slightly less valued) |

## Q-Learning Update Rule

```
Q(s, a)  ←  Q(s, a)  +  α · [r  +  γ · max_a' Q(s', a')  −  Q(s, a)]
                                 └──────────────────────────┘
                                   TD error (prediction error)
```

The agent is "off-policy" — it learns Q* (the optimal policy's values)
regardless of which actions it actually takes during training.

## State Representation

The state must be **hashable** for tabular Q-learning.

```python
state = (frozenset(remaining_jobs), tuple(rounded_machine_loads))
```

- `frozenset` captures which jobs remain (order-invariant, hashable)
- Machine loads are rounded to integers to keep the state space tractable
- Without rounding, continuous loads create infinitely many states

## ε-Greedy Exploration

```
With probability ε   →  random action  (explore)
With probability 1-ε →  argmax Q(s,·) (exploit)

ε(t) = max(ε_min,  ε(0) × decay^t)
```

Start: ε=1.0 (pure random — agent has no knowledge yet)
End:   ε→0.05 (mostly exploiting learned policy)

## Reward Design

Two-component reward:

1. **Step penalty**: −1 per step — discourages unnecessary actions and
   encourages compact schedules.

2. **Terminal reward**: −(makespan / lower_bound) — strongly penalises
   solutions far above the theoretical optimum. This normalisation makes
   the reward meaningful across different instance sizes.

## Why Tabular Q-learning (not DQN)?

- Dataset is small (6 jobs × 3 machines) → state space is manageable
- No function approximation overhead — the algorithm is fully transparent
- Demonstrates the foundational RL algorithm clearly
- For larger instances (50+ jobs), DQN or policy gradient methods are needed

## Baseline Comparison

| Baseline | Description |
|---|---|
| **Random** | Uniform random job order — weakest policy |
| **SPT** | Shortest Processing Time first — minimises avg completion time |
| **LPT** | Longest Processing Time first — good for makespan on parallel machines |
| **RL Agent** | Learned Q-policy — should match or beat SPT |

## File Responsibilities

| File | Role |
|---|---|
| `environment.py` | MDP definition: state, action, reward, transition |
| `agent.py` | Q-table, ε-greedy, Bellman update, save/load |
| `train.py` | Episode loop, logging, plotting |
| `evaluate.py` | Multi-trial comparison vs baselines |
| `visualise.py` | Gantt chart of the learned schedule |
