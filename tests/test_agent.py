"""
test_agent.py
-------------
Unit tests for the QLearningAgent.

Run:  pytest tests/ -v
"""

import sys
import json
import tempfile
import pytest
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from agent import QLearningAgent


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def agent():
    return QLearningAgent(n_jobs=4, alpha=0.1, gamma=0.9,
                          epsilon=1.0, epsilon_min=0.05,
                          epsilon_decay=0.99, seed=0)


STATE_A = (frozenset({0, 1, 2, 3}), (0, 0))
STATE_B = (frozenset({1, 2, 3}),    (3, 0))
AVAIL_A = [0, 1, 2, 3]
AVAIL_B = [1, 2, 3]


# ── Action selection ──────────────────────────────────────────────────────────

class TestActionSelection:

    def test_choose_action_returns_valid_action(self, agent):
        action = agent.choose_action(STATE_A, AVAIL_A)
        assert action in AVAIL_A

    def test_choose_action_only_from_available(self, agent):
        for _ in range(50):
            action = agent.choose_action(STATE_B, AVAIL_B)
            assert action in AVAIL_B, f"Invalid action {action} chosen"

    def test_greedy_action_in_available(self, agent):
        action = agent.greedy_action(STATE_A, AVAIL_A)
        assert action in AVAIL_A

    def test_empty_available_raises(self, agent):
        with pytest.raises((ValueError, Exception)):
            agent.choose_action(STATE_A, [])

    def test_explore_when_epsilon_one(self):
        """With ε=1.0, agent should always explore (random)."""
        agent = QLearningAgent(n_jobs=4, epsilon=1.0, seed=0)
        # Just verify it returns valid actions (randomness hard to assert deterministically)
        for _ in range(10):
            action = agent.choose_action(STATE_A, AVAIL_A)
            assert action in AVAIL_A

    def test_exploit_when_epsilon_zero(self):
        """With ε=0, agent always exploits Q-table."""
        agent = QLearningAgent(n_jobs=4, epsilon=0.0, seed=0)
        # Set one Q-value higher
        agent.q_table[STATE_A] = {0: 0.0, 1: 5.0, 2: 0.0, 3: 0.0}
        for _ in range(10):
            action = agent.choose_action(STATE_A, AVAIL_A)
            assert action == 1, "Should always choose action with highest Q"


# ── Q-update ──────────────────────────────────────────────────────────────────

class TestQUpdate:

    def test_q_value_initialised_to_zero(self, agent):
        q = agent.get_q_value(STATE_A, 0)
        assert q == 0.0

    def test_update_changes_q_value(self, agent):
        agent.update(STATE_A, 0, reward=-1.0, next_state=STATE_B,
                     done=False, next_available=AVAIL_B)
        q = agent.get_q_value(STATE_A, 0)
        assert q != 0.0, "Q-value should have changed after update"

    def test_bellman_update_direction(self, agent):
        """Negative reward should push Q-value below 0."""
        agent.update(STATE_A, 0, reward=-5.0, next_state=STATE_B,
                     done=True, next_available=[])
        q = agent.get_q_value(STATE_A, 0)
        assert q < 0.0

    def test_terminal_update_no_future(self, agent):
        """When done=True, future term should be 0."""
        # Manually set Q(B, 1) = 100 — should NOT affect terminal update from A
        agent.q_table[STATE_B] = {1: 100.0, 2: 100.0, 3: 100.0}
        agent.update(STATE_A, 0, reward=-1.0, next_state=STATE_B,
                     done=True, next_available=[])
        # Q(A,0) = 0 + 0.1 * (-1 - 0) = -0.1
        q = agent.get_q_value(STATE_A, 0)
        assert abs(q - (-0.1)) < 1e-9

    def test_multiple_updates_improve_q(self, agent):
        """Repeated updates with positive reward should push Q up."""
        for _ in range(20):
            agent.update(STATE_A, 2, reward=10.0, next_state=STATE_B,
                         done=True, next_available=[])
        q = agent.get_q_value(STATE_A, 2)
        assert q > 0


# ── Epsilon decay ─────────────────────────────────────────────────────────────

class TestEpsilonDecay:

    def test_epsilon_decreases_after_decay(self, agent):
        initial = agent.epsilon
        agent.decay_epsilon()
        assert agent.epsilon < initial

    def test_epsilon_never_below_min(self):
        agent = QLearningAgent(n_jobs=4, epsilon=1.0,
                               epsilon_min=0.1, epsilon_decay=0.5)
        for _ in range(100):
            agent.decay_epsilon()
        assert agent.epsilon >= 0.1

    def test_epsilon_reaches_floor(self):
        agent = QLearningAgent(n_jobs=4, epsilon=1.0,
                               epsilon_min=0.05, epsilon_decay=0.5)
        for _ in range(50):
            agent.decay_epsilon()
        assert abs(agent.epsilon - 0.05) < 1e-9


# ── Save / load ────────────────────────────────────────────────────────────────

class TestSaveLoad:

    def test_save_and_load_preserves_q_values(self, agent):
        agent.update(STATE_A, 1, reward=-2.0, next_state=STATE_B,
                     done=True, next_available=[])
        original_q = agent.get_q_value(STATE_A, 1)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        agent.save(path)
        loaded = QLearningAgent.load(path)
        loaded_q = loaded.get_q_value(STATE_A, 1)
        assert abs(original_q - loaded_q) < 1e-9

    def test_save_creates_file(self, agent):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        agent.save(path)
        assert Path(path).exists()

    def test_loaded_agent_same_hyperparams(self, agent):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        agent.save(path)
        loaded = QLearningAgent.load(path)
        assert loaded.n_jobs  == agent.n_jobs
        assert loaded.alpha   == agent.alpha
        assert loaded.gamma   == agent.gamma


# ── Logging ───────────────────────────────────────────────────────────────────

class TestLogging:

    def test_log_episode_records_data(self, agent):
        agent.log_episode(total_reward=-10.0, makespan=15.0)
        assert len(agent.episode_rewards)   == 1
        assert len(agent.episode_makespans) == 1
        assert agent.episode_rewards[0]    == -10.0
        assert agent.episode_makespans[0]  == 15.0

    def test_q_table_size_counts_states(self, agent):
        assert agent.q_table_size == 0
        agent.get_q_value(STATE_A, 0)   # triggers initialisation
        assert agent.q_table_size == 1
