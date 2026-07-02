"""
environment.py
--------------
Custom Gym-style Reinforcement Learning environment for job scheduling.

PROBLEM DEFINITION
------------------
We have:
  - n_jobs   : jobs, each with a known processing time p[j]
  - n_machines : identical parallel machines

The agent must assign every job to a machine, one job per step.
It chooses which job to schedule next; the job is placed on the
machine that becomes free earliest (greedy machine assignment).

Objective: minimise MAKESPAN — the time when the last machine finishes.

MARKOV DECISION PROCESS (MDP)
------------------------------
State  s  : (remaining_jobs_mask, machine_loads)
              remaining_jobs_mask : frozenset of job indices not yet scheduled
              machine_loads       : tuple of current load on each machine

Action a  : index of a job from the remaining jobs set

Reward r  :
  - At each step (except last): -1  (encourages finishing quickly)
  - At final step: -(makespan / optimal_lower_bound)
    This normalised terminal reward gives a strong signal about solution quality.

Episode ends when all jobs have been assigned.

STATE REPRESENTATION NOTE
--------------------------
For tabular Q-learning, the state must be hashable.
We represent it as a Python tuple:
  (frozenset_of_remaining_job_ids, tuple_of_machine_loads_rounded)

Rounding machine loads to integers keeps the state space tractable.
"""

import numpy as np
from typing import Optional


class JobSchedulingEnv:
    """
    Parallel machine scheduling environment for RL.

    Parameters
    ----------
    n_jobs       : number of jobs
    n_machines   : number of identical parallel machines
    processing_times : optional array of job processing times;
                       if None, randomly generated each reset()
    seed         : random seed for reproducibility
    """

    def __init__(
        self,
        n_jobs: int = 6,
        n_machines: int = 3,
        processing_times: Optional[np.ndarray] = None,
        seed: int = 42,
    ):
        self.n_jobs     = n_jobs
        self.n_machines = n_machines
        self.seed       = seed
        self.rng        = np.random.default_rng(seed)

        # Fixed processing times (same instance every episode) if provided
        self._fixed_times = processing_times

        # Will be set in reset()
        self.processing_times: np.ndarray = np.array([])
        self.machine_loads:    np.ndarray = np.array([])
        self.remaining_jobs:   set        = set()
        self.schedule:         list       = []    # list of (job, machine, start, end)
        self._done = False

    # ── Core MDP interface ─────────────────────────────────────────────────────

    def reset(self) -> tuple:
        """
        Reset the environment to a fresh episode.

        Returns
        -------
        Initial state tuple (hashable, for Q-table lookup).
        """
        if self._fixed_times is not None:
            self.processing_times = self._fixed_times.copy()
        else:
            # Random processing times in [1, 10]
            self.processing_times = self.rng.integers(1, 11, size=self.n_jobs).astype(float)

        self.machine_loads  = np.zeros(self.n_machines)
        self.remaining_jobs = set(range(self.n_jobs))
        self.schedule       = []
        self._done          = False

        return self._get_state()

    def step(self, action: int):
        """
        Execute action: assign job `action` to the earliest-free machine.

        Parameters
        ----------
        action : job index (must be in remaining_jobs)

        Returns
        -------
        next_state : hashable state tuple
        reward     : float
        done       : bool
        info       : dict with extra diagnostics
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() first.")
        if action not in self.remaining_jobs:
            raise ValueError(f"Job {action} is not available. Remaining: {self.remaining_jobs}")

        # Assign job to the machine with lowest current load (greedy machine pick)
        machine_idx = int(np.argmin(self.machine_loads))
        start_time  = self.machine_loads[machine_idx]
        end_time    = start_time + self.processing_times[action]

        self.machine_loads[machine_idx] = end_time
        self.remaining_jobs.remove(action)
        self.schedule.append({
            "job":     action,
            "machine": machine_idx,
            "start":   start_time,
            "end":     end_time,
            "proc":    self.processing_times[action],
        })

        done = len(self.remaining_jobs) == 0

        if done:
            makespan = float(np.max(self.machine_loads))
            # Normalised terminal reward: negative makespan relative to lower bound
            lower_bound = max(
                np.sum(self.processing_times) / self.n_machines,
                np.max(self.processing_times),
            )
            reward = -(makespan / lower_bound)
            self._done = True
            info = {"makespan": makespan, "lower_bound": lower_bound}
        else:
            reward = -1.0   # small step penalty
            info   = {}

        return self._get_state(), reward, done, info

    def get_available_actions(self) -> list:
        """Return list of job indices that can still be scheduled."""
        return sorted(self.remaining_jobs)

    def get_makespan(self) -> float:
        """Return current makespan (max machine load)."""
        return float(np.max(self.machine_loads))

    def get_processing_times(self) -> np.ndarray:
        return self.processing_times.copy()

    # ── State representation ───────────────────────────────────────────────────

    def _get_state(self) -> tuple:
        """
        Encode the current environment state as a hashable tuple.

        Format: (frozenset_of_remaining_jobs, rounded_machine_loads_tuple)

        Rounding machine loads to integers prevents state space explosion
        while retaining enough information for the agent to make good decisions.
        """
        rounded_loads = tuple(int(round(load)) for load in self.machine_loads)
        return (frozenset(self.remaining_jobs), rounded_loads)

    # ── Utility ────────────────────────────────────────────────────────────────

    def render(self) -> str:
        """Return a text representation of the current schedule."""
        lines = [f"Jobs: {self.n_jobs}  |  Machines: {self.n_machines}"]
        lines.append(f"Processing times: {self.processing_times.astype(int).tolist()}")
        lines.append(f"Machine loads:    {self.machine_loads.astype(int).tolist()}")
        lines.append(f"Remaining jobs:   {sorted(self.remaining_jobs)}")
        if self.schedule:
            lines.append(f"Makespan so far:  {self.get_makespan():.1f}")
        return "\n".join(lines)

    def clone(self) -> "JobSchedulingEnv":
        """Return a deep copy of the current environment state."""
        env = JobSchedulingEnv(
            n_jobs=self.n_jobs,
            n_machines=self.n_machines,
            processing_times=self.processing_times.copy(),
            seed=self.seed,
        )
        env.machine_loads  = self.machine_loads.copy()
        env.remaining_jobs = self.remaining_jobs.copy()
        env.schedule       = [s.copy() for s in self.schedule]
        env._done          = self._done
        return env

    @property
    def optimal_lower_bound(self) -> float:
        """Theoretical lower bound on makespan (not always achievable)."""
        return float(max(
            np.sum(self.processing_times) / self.n_machines,
            np.max(self.processing_times),
        ))


# ── Baseline policies (for comparison) ────────────────────────────────────────

def random_policy(env: JobSchedulingEnv, rng: np.random.Generator) -> list:
    """Schedule jobs in a uniformly random order."""
    env.reset()
    order = list(range(env.n_jobs))
    rng.shuffle(order)
    for job in order:
        env.step(job)
    return env.schedule.copy()


def spt_policy(env: JobSchedulingEnv) -> list:
    """
    Shortest Processing Time (SPT) greedy heuristic.
    Always schedules the job with the smallest processing time next.
    SPT minimises average completion time and is a strong baseline.
    """
    env.reset()
    order = np.argsort(env.processing_times)   # ascending = shortest first
    for job in order:
        env.step(int(job))
    return env.schedule.copy()


def lpt_policy(env: JobSchedulingEnv) -> list:
    """
    Longest Processing Time (LPT) greedy heuristic.
    Schedules longest jobs first — good for makespan on parallel machines.
    """
    env.reset()
    order = np.argsort(env.processing_times)[::-1]   # descending
    for job in order:
        env.step(int(job))
    return env.schedule.copy()


if __name__ == "__main__":
    # Quick demo
    env = JobSchedulingEnv(n_jobs=6, n_machines=3, seed=0)
    state = env.reset()
    print("Initial state:")
    print(env.render())
    print(f"\nProcessing times: {env.processing_times.astype(int).tolist()}")
    print(f"Lower bound: {env.optimal_lower_bound:.1f}")

    # Manually schedule jobs 0..5 in order
    done = False
    while not done:
        actions = env.get_available_actions()
        action  = actions[0]
        state, reward, done, info = env.step(action)

    print(f"\nFinal makespan: {env.get_makespan():.1f}")
    print(f"Schedule: {[(s['job'], s['machine']) for s in env.schedule]}")
