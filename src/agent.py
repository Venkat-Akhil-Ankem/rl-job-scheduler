"""
agent.py
--------
Q-learning agent for the job scheduling environment.

Q-LEARNING OVERVIEW
-------------------
Q-learning is a model-free, off-policy temporal difference (TD) algorithm.
It learns a Q-function: Q(s, a) = expected cumulative discounted reward
when taking action a in state s and then following the optimal policy.

The core update rule (Bellman equation):

  Q(s, a)  ←  Q(s, a)  +  α · [ r  +  γ · max_a' Q(s', a')  −  Q(s, a) ]
                                  └───────────────────┘
                                    TD target

where:
  α (alpha)   = learning rate      — how fast we update Q-values
  γ (gamma)   = discount factor    — how much we value future rewards
  ε (epsilon) = exploration rate   — probability of taking a random action

EXPLORATION STRATEGY: ε-GREEDY
-------------------------------
At each step, with probability ε the agent picks a random action (explore),
and with probability 1-ε it picks argmax_a Q(s, a) (exploit).

ε starts high (lots of exploration) and decays each episode, so the agent
gradually shifts from exploration to exploitation as it learns.

Q-TABLE IMPLEMENTATION
----------------------
Instead of a neural network (deep Q-learning / DQN), we use a Python
dictionary as the Q-table. This is the classic tabular Q-learning approach
and is appropriate for the small state spaces here.

  q_table[state][action] = estimated value of taking action in state
"""

import numpy as np
import json
from pathlib import Path
from typing import Optional


class QLearningAgent:
    """
    Tabular Q-learning agent with ε-greedy exploration.

    Parameters
    ----------
    n_jobs        : number of jobs (= number of possible actions)
    alpha         : learning rate ∈ (0, 1]
    gamma         : discount factor ∈ [0, 1]
    epsilon       : initial exploration rate ∈ [0, 1]
    epsilon_min   : minimum exploration rate (floor for decay)
    epsilon_decay : multiplicative decay applied after each episode
    seed          : random seed for reproducible exploration
    """

    def __init__(
        self,
        n_jobs: int,
        alpha:         float = 0.1,
        gamma:         float = 0.95,
        epsilon:       float = 1.0,
        epsilon_min:   float = 0.05,
        epsilon_decay: float = 0.995,
        seed: int = 42,
    ):
        self.n_jobs        = n_jobs
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng           = np.random.default_rng(seed)

        # Q-table: dict[state_tuple → dict[action_int → float]]
        # Lazily initialised: unseen states start with Q=0 for all actions
        self.q_table: dict = {}

        # Training history (for plotting)
        self.episode_rewards:   list = []
        self.episode_makespans: list = []
        self.epsilon_history:   list = []

    # ── Q-table access ─────────────────────────────────────────────────────────

    def _get_q_values(self, state: tuple) -> dict:
        """
        Return Q-values for all actions in a given state.
        Unseen states are initialised to 0.
        """
        if state not in self.q_table:
            self.q_table[state] = {j: 0.0 for j in range(self.n_jobs)}
        return self.q_table[state]

    def get_q_value(self, state: tuple, action: int) -> float:
        return self._get_q_values(state).get(action, 0.0)

    # ── Action selection ───────────────────────────────────────────────────────

    def choose_action(self, state: tuple, available_actions: list) -> int:
        """
        Select an action using ε-greedy policy.

        Parameters
        ----------
        state             : current environment state (hashable tuple)
        available_actions : list of valid actions (remaining job ids)

        Returns
        -------
        Chosen action (job index).
        """
        if not available_actions:
            raise ValueError("No available actions.")

        # Explore: random action
        if self.rng.random() < self.epsilon:
            return int(self.rng.choice(available_actions))

        # Exploit: best action among available ones
        q_vals = self._get_q_values(state)
        return max(available_actions, key=lambda a: q_vals.get(a, 0.0))

    def greedy_action(self, state: tuple, available_actions: list) -> int:
        """Pure greedy action (ε=0) — used at evaluation time."""
        q_vals = self._get_q_values(state)
        return max(available_actions, key=lambda a: q_vals.get(a, 0.0))

    # ── Q-update ───────────────────────────────────────────────────────────────

    def update(
        self,
        state:       tuple,
        action:      int,
        reward:      float,
        next_state:  tuple,
        done:        bool,
        next_available: list,
    ):
        """
        Apply the Q-learning update rule (Bellman equation).

        Q(s,a) ← Q(s,a) + α · [r + γ · max_a' Q(s',a') - Q(s,a)]

        When done=True, the future term γ·max Q(s',·) is 0 (no future).
        """
        current_q = self.get_q_value(state, action)

        if done or not next_available:
            td_target = reward
        else:
            next_q_vals  = self._get_q_values(next_state)
            best_next_q  = max(next_q_vals.get(a, 0.0) for a in next_available)
            td_target    = reward + self.gamma * best_next_q

        td_error = td_target - current_q
        self.q_table.setdefault(state, {j: 0.0 for j in range(self.n_jobs)})
        self.q_table[state][action] = current_q + self.alpha * td_error

    # ── Epsilon decay ──────────────────────────────────────────────────────────

    def decay_epsilon(self):
        """Decay exploration rate after each episode (called once per episode)."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ── Logging ────────────────────────────────────────────────────────────────

    def log_episode(self, total_reward: float, makespan: float):
        self.episode_rewards.append(total_reward)
        self.episode_makespans.append(makespan)
        self.epsilon_history.append(self.epsilon)

    # ── Save / load ────────────────────────────────────────────────────────────

    def save(self, path: str):
        """
        Save Q-table and hyperparameters to disk.
        Q-table keys contain frozensets, so we serialise them carefully.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        # Convert frozenset keys to serialisable format
        serialised = {}
        for state, q_vals in self.q_table.items():
            remaining, loads = state
            key = (sorted(remaining), list(loads))
            serialised[str(key)] = {str(k): v for k, v in q_vals.items()}

        data = {
            "hyperparams": {
                "n_jobs":        self.n_jobs,
                "alpha":         self.alpha,
                "gamma":         self.gamma,
                "epsilon":       self.epsilon,
                "epsilon_min":   self.epsilon_min,
                "epsilon_decay": self.epsilon_decay,
            },
            "q_table_size": len(self.q_table),
            "episodes_trained": len(self.episode_rewards),
            "q_table": serialised,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"💾 Agent saved → {path}  ({len(self.q_table)} states)")

    @classmethod
    def load(cls, path: str) -> "QLearningAgent":
        """Load a previously saved agent."""
        with open(path) as f:
            data = json.load(f)
        hp = data["hyperparams"]
        agent = cls(**hp)

        for key_str, q_vals in data["q_table"].items():
            key_list   = eval(key_str)          # [[remaining], [loads]]
            remaining  = frozenset(key_list[0])
            loads      = tuple(key_list[1])
            state      = (remaining, loads)
            agent.q_table[state] = {int(k): float(v) for k, v in q_vals.items()}

        print(f"✅ Agent loaded from {path}  ({len(agent.q_table)} states)")
        return agent

    # ── Statistics ─────────────────────────────────────────────────────────────

    @property
    def q_table_size(self) -> int:
        return len(self.q_table)

    def summarise(self):
        if not self.episode_makespans:
            print("Agent has not been trained yet.")
            return
        recent = self.episode_makespans[-100:]
        print(f"Q-table size:    {self.q_table_size:,} states")
        print(f"Episodes trained:{len(self.episode_makespans):,}")
        print(f"Current ε:       {self.epsilon:.4f}")
        print(f"Avg makespan (last 100 eps): {np.mean(recent):.2f}")
        print(f"Best makespan seen:          {min(self.episode_makespans):.2f}")


if __name__ == "__main__":
    # Quick sanity check
    agent = QLearningAgent(n_jobs=6)
    state = (frozenset({0, 1, 2, 3, 4, 5}), (0, 0, 0))
    available = [0, 1, 2, 3, 4, 5]
    action = agent.choose_action(state, available)
    print(f"Chosen action: {action}")

    next_state = (frozenset({1, 2, 3, 4, 5}), (3, 0, 0))
    agent.update(state, action, reward=-1.0, next_state=next_state,
                 done=False, next_available=[1, 2, 3, 4, 5])
    print(f"Q({action}): {agent.get_q_value(state, action):.4f}  (should be ~-0.1)")
    print("✅ Agent sanity check passed.")
