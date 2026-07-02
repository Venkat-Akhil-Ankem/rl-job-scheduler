"""
evaluate.py
-----------
Evaluates the trained RL agent and compares it against baselines:

  1. Random policy   — assigns jobs in a uniformly random order
  2. SPT heuristic   — Shortest Processing Time first
  3. LPT heuristic   — Longest Processing Time first
  4. RL agent        — trained Q-learning policy (greedy, ε=0)

Run over many random problem instances to get statistically robust results.

Run:
    python src/evaluate.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from environment import JobSchedulingEnv, random_policy, spt_policy, lpt_policy
from agent import QLearningAgent
from train import CONFIG, run_episode


def evaluate_policy(policy_fn, env: JobSchedulingEnv, n_trials: int, rng) -> list:
    """Run a policy n_trials times and return list of makespans."""
    makespans = []
    for _ in range(n_trials):
        # Randomise processing times each trial
        env._fixed_times = rng.integers(1, 11, size=env.n_jobs).astype(float)
        env.reset()
        policy_fn(env)
        makespans.append(env.get_makespan())
    return makespans


def evaluate_agent(agent: QLearningAgent, env: JobSchedulingEnv,
                   n_trials: int, rng) -> list:
    """Run the RL agent (greedy) n_trials times and return makespans."""
    makespans = []
    for _ in range(n_trials):
        env._fixed_times = rng.integers(1, 11, size=env.n_jobs).astype(float)
        _, makespan = run_episode(env, agent, train=False)
        makespans.append(makespan)
    return makespans


def print_comparison_table(results: dict, lower_bounds: list):
    """Pretty-print the comparison table."""
    lb_mean = np.mean(lower_bounds)

    print("\n" + "═" * 65)
    print("  POLICY COMPARISON RESULTS")
    print("═" * 65)
    print(f"  {'Policy':20s}  {'Mean':>7}  {'Std':>6}  {'Best':>6}  {'Gap%':>7}")
    print("─" * 65)

    for name, makespans in results.items():
        mean = np.mean(makespans)
        std  = np.std(makespans)
        best = np.min(makespans)
        gap  = (mean - lb_mean) / lb_mean * 100
        marker = " ← RL" if "RL Agent" in name else ""
        print(f"  {name:20s}  {mean:>7.2f}  {std:>6.2f}  {best:>6.1f}  {gap:>6.1f}%{marker}")

    print("─" * 65)
    print(f"  {'Lower Bound':20s}  {lb_mean:>7.2f}  {'—':>6}  {'—':>6}  {'0.0%':>7}")
    print("═" * 65)
    print()
    print("  Gap% = how far above the theoretical lower bound on average.")
    print("  Lower is better.\n")


def main(n_trials: int = 500, seed: int = 99):
    rng = np.random.default_rng(seed)

    # Check if trained agent exists
    agent_path = Path(CONFIG["agent_path"])
    if not agent_path.exists():
        print(f"❌ No trained agent found at {agent_path}")
        print("   Run  python src/train.py  first.")
        return

    # Load agent
    agent = QLearningAgent.load(str(agent_path))

    env = JobSchedulingEnv(
        n_jobs=CONFIG["n_jobs"],
        n_machines=CONFIG["n_machines"],
        seed=seed,
    )

    print(f"\n🔬 Evaluating over {n_trials} random instances...")
    print(f"   Jobs: {env.n_jobs}  |  Machines: {env.n_machines}")

    # Compute lower bounds for the same random instances
    lb_rng = np.random.default_rng(seed)
    lower_bounds = []
    for _ in range(n_trials):
        times = lb_rng.integers(1, 11, size=env.n_jobs).astype(float)
        lb = float(max(np.sum(times) / env.n_machines, np.max(times)))
        lower_bounds.append(lb)

    # Evaluate each policy on the same random instances
    def rand_fn(e):
        random_policy(e, np.random.default_rng(seed))
    def spt_fn(e): spt_policy(e)
    def lpt_fn(e): lpt_policy(e)

    results = {}

    rng1 = np.random.default_rng(seed)
    results["Random"] = evaluate_policy(
        lambda e: random_policy(e, rng1), env, n_trials, np.random.default_rng(seed)
    )

    results["SPT (greedy)"] = evaluate_policy(spt_policy, env, n_trials, np.random.default_rng(seed))
    results["LPT (greedy)"] = evaluate_policy(lpt_policy, env, n_trials, np.random.default_rng(seed))
    results["RL Agent (Q-learning)"] = evaluate_agent(agent, env, n_trials, np.random.default_rng(seed))

    print_comparison_table(results, lower_bounds)

    # Win-rate: how often does RL beat SPT?
    rl   = np.array(results["RL Agent (Q-learning)"])
    spt  = np.array(results["SPT (greedy)"])
    rand = np.array(results["Random"])

    rl_beats_spt  = (rl <= spt).mean()  * 100
    rl_beats_rand = (rl <= rand).mean() * 100

    print(f"  RL beats SPT on   {rl_beats_spt:.1f}% of instances")
    print(f"  RL beats Random on {rl_beats_rand:.1f}% of instances")
    print()


if __name__ == "__main__":
    main()
