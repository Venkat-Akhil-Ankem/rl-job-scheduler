"""
test_environment.py
-------------------
Unit tests for the JobSchedulingEnv RL environment.

Run:  pytest tests/ -v
"""

import sys
import pytest
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from environment import JobSchedulingEnv, random_policy, spt_policy, lpt_policy


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def env():
    """Small 4-job × 2-machine environment with fixed processing times."""
    times = np.array([3.0, 5.0, 2.0, 4.0])
    return JobSchedulingEnv(n_jobs=4, n_machines=2,
                            processing_times=times, seed=0)


@pytest.fixture
def env6():
    return JobSchedulingEnv(n_jobs=6, n_machines=3, seed=42)


# ── reset() ───────────────────────────────────────────────────────────────────

class TestReset:

    def test_returns_hashable_state(self, env):
        state = env.reset()
        hash(state)   # should not raise

    def test_all_jobs_remaining_after_reset(self, env):
        env.reset()
        assert env.remaining_jobs == {0, 1, 2, 3}

    def test_machine_loads_zero_after_reset(self, env):
        env.reset()
        assert np.all(env.machine_loads == 0)

    def test_schedule_empty_after_reset(self, env):
        env.reset()
        assert env.schedule == []

    def test_reset_twice_gives_same_state(self, env):
        s1 = env.reset()
        s2 = env.reset()
        assert s1 == s2


# ── step() ────────────────────────────────────────────────────────────────────

class TestStep:

    def test_step_removes_job_from_remaining(self, env):
        env.reset()
        env.step(0)
        assert 0 not in env.remaining_jobs

    def test_step_adds_entry_to_schedule(self, env):
        env.reset()
        env.step(2)
        assert len(env.schedule) == 1
        assert env.schedule[0]["job"] == 2

    def test_done_when_all_jobs_scheduled(self, env):
        env.reset()
        done = False
        for j in [0, 1, 2, 3]:
            _, _, done, _ = env.step(j)
        assert done

    def test_not_done_until_all_jobs_scheduled(self, env):
        env.reset()
        for j in [0, 1, 2]:
            _, _, done, _ = env.step(j)
            assert not done

    def test_info_contains_makespan_when_done(self, env):
        env.reset()
        info = {}
        for j in [0, 1, 2, 3]:
            _, _, done, info = env.step(j)
        assert "makespan" in info
        assert info["makespan"] > 0

    def test_step_reward_negative(self, env):
        env.reset()
        _, reward, done, _ = env.step(0)
        assert reward < 0

    def test_invalid_action_raises(self, env):
        env.reset()
        env.step(0)   # schedule job 0
        with pytest.raises(ValueError):
            env.step(0)   # try to schedule again

    def test_step_after_done_raises(self, env):
        env.reset()
        for j in [0, 1, 2, 3]:
            env.step(j)
        with pytest.raises(RuntimeError):
            env.step(0)

    def test_processing_time_used_correctly(self, env):
        env.reset()
        env.step(0)   # job 0 has processing time 3
        assert env.schedule[0]["proc"] == 3.0
        assert env.schedule[0]["end"]  == 3.0

    def test_machine_load_increases(self, env):
        env.reset()
        env.step(0)
        assert np.any(env.machine_loads > 0)

    def test_jobs_assigned_to_least_loaded_machine(self, env):
        env.reset()
        env.step(3)   # job 3 (proc=4) → machine 0
        env.step(1)   # job 1 (proc=5) → machine 1 (both free initially)
        # Machine loads should now be 4 and 5 (one each)
        loads = sorted(env.machine_loads)
        assert loads[0] in [4.0, 5.0]
        assert loads[1] in [4.0, 5.0]


# ── Makespan and lower bound ───────────────────────────────────────────────────

class TestMakespan:

    def test_makespan_positive_after_full_schedule(self, env):
        env.reset()
        for j in range(4):
            env.step(j)
        assert env.get_makespan() > 0

    def test_makespan_at_least_lower_bound(self, env):
        env.reset()
        lb = env.optimal_lower_bound
        for j in range(4):
            env.step(j)
        assert env.get_makespan() >= lb - 1e-9

    def test_lower_bound_positive(self, env):
        env.reset()
        assert env.optimal_lower_bound > 0


# ── Available actions ─────────────────────────────────────────────────────────

class TestAvailableActions:

    def test_all_jobs_available_at_start(self, env):
        env.reset()
        assert set(env.get_available_actions()) == {0, 1, 2, 3}

    def test_action_removed_after_step(self, env):
        env.reset()
        env.step(1)
        assert 1 not in env.get_available_actions()

    def test_no_actions_when_done(self, env):
        env.reset()
        for j in range(4):
            env.step(j)
        assert env.get_available_actions() == []


# ── Baseline policies ─────────────────────────────────────────────────────────

class TestBaselines:

    def test_spt_schedules_all_jobs(self, env6):
        env6.reset()
        spt_policy(env6)
        assert len(env6.schedule) == env6.n_jobs

    def test_lpt_schedules_all_jobs(self, env6):
        env6.reset()
        lpt_policy(env6)
        assert len(env6.schedule) == env6.n_jobs

    def test_random_schedules_all_jobs(self, env6):
        rng = np.random.default_rng(0)
        random_policy(env6, rng)
        assert len(env6.schedule) == env6.n_jobs

    def test_spt_shortest_job_scheduled_first(self, env):
        """SPT should schedule job 2 (proc=2) first."""
        env.reset()
        spt_policy(env)
        first_job = env.schedule[0]["job"]
        assert env.processing_times[first_job] == min(env.processing_times)
