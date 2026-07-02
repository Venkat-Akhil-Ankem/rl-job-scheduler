"""
train.py
--------
Training loop for the Q-learning job scheduler agent.

Steps:
  1. Create environment and agent
  2. Run N training episodes; update Q-table after every step
  3. Decay ε after every episode
  4. Print progress every 500 episodes
  5. Save trained agent → outputs/agent.json
  6. Plot learning curve → outputs/learning_curve.png

Run:
    python src/train.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from environment import JobSchedulingEnv
from agent import QLearningAgent


# ── Configuration ─────────────────────────────────────────────────────────────

CONFIG = {
    # Environment
    "n_jobs":     6,
    "n_machines": 3,
    "env_seed":   42,

    # Agent hyperparameters
    "alpha":         0.1,     # learning rate
    "gamma":         0.95,    # discount factor
    "epsilon":       1.0,     # initial exploration rate
    "epsilon_min":   0.05,    # floor for epsilon decay
    "epsilon_decay": 0.995,   # multiplicative decay per episode
    "agent_seed":    42,

    # Training
    "n_episodes":    5000,
    "print_every":   500,

    # Output
    "output_dir":   "outputs",
    "agent_path":   "outputs/agent.json",
    "plot_path":    "outputs/learning_curve.png",
}


# ── Training ───────────────────────────────────────────────────────────────────

def run_episode(env: JobSchedulingEnv, agent: QLearningAgent, train: bool = True):
    """
    Run a single episode.

    Parameters
    ----------
    env   : scheduling environment
    agent : Q-learning agent
    train : if True, update Q-table (training mode); else pure greedy

    Returns
    -------
    total_reward : sum of rewards collected in the episode
    makespan     : final makespan achieved
    """
    state     = env.reset()
    total_r   = 0.0
    makespan  = 0.0
    done      = False

    while not done:
        available = env.get_available_actions()

        if train:
            action = agent.choose_action(state, available)
        else:
            action = agent.greedy_action(state, available)

        next_state, reward, done, info = env.step(action)
        next_available = env.get_available_actions()

        if train:
            agent.update(
                state, action, reward, next_state, done, next_available
            )

        total_r  += reward
        state     = next_state

        if done:
            makespan = info["makespan"]

    return total_r, makespan


def train(config: dict = None) -> tuple:
    """
    Full training loop.

    Returns
    -------
    (agent, env) — trained agent and environment instance
    """
    if config is None:
        config = CONFIG

    Path(config["output_dir"]).mkdir(parents=True, exist_ok=True)

    env = JobSchedulingEnv(
        n_jobs=config["n_jobs"],
        n_machines=config["n_machines"],
        seed=config["env_seed"],
    )
    agent = QLearningAgent(
        n_jobs=config["n_jobs"],
        alpha=config["alpha"],
        gamma=config["gamma"],
        epsilon=config["epsilon"],
        epsilon_min=config["epsilon_min"],
        epsilon_decay=config["epsilon_decay"],
        seed=config["agent_seed"],
    )

    print("=" * 60)
    print("  🤖 RL Job Scheduler — Training")
    print("=" * 60)
    print(f"  Jobs: {config['n_jobs']}  |  Machines: {config['n_machines']}")
    print(f"  Episodes: {config['n_episodes']}  |  α={config['alpha']}  "
          f"γ={config['gamma']}  ε₀={config['epsilon']}")
    print()

    # Fix processing times so the agent learns one instance well
    env.reset()
    fixed_times = env.processing_times.copy()
    env._fixed_times = fixed_times
    lower_bound = env.optimal_lower_bound

    print(f"  Processing times: {fixed_times.astype(int).tolist()}")
    print(f"  Optimal lower bound: {lower_bound:.1f}")
    print()
    print(f"{'Episode':>8}  {'Avg Makespan':>13}  {'Best':>6}  {'ε':>6}  {'Q-states':>9}")
    print("─" * 60)

    t0 = time.time()

    for ep in range(1, config["n_episodes"] + 1):
        reward, makespan = run_episode(env, agent, train=True)
        agent.log_episode(reward, makespan)
        agent.decay_epsilon()

        if ep % config["print_every"] == 0 or ep == 1:
            recent = agent.episode_makespans[-config["print_every"]:]
            avg_ms = np.mean(recent)
            best   = min(agent.episode_makespans)
            print(f"{ep:>8}  {avg_ms:>13.2f}  {best:>6.1f}  "
                  f"{agent.epsilon:>6.3f}  {agent.q_table_size:>9,}")

    elapsed = time.time() - t0
    print(f"\n⏱  Training time: {elapsed:.1f}s")

    # Save agent
    agent.save(config["agent_path"])

    # Plot
    _plot_learning_curves(agent, config["plot_path"], lower_bound)

    # Final stats
    print()
    agent.summarise()

    return agent, env


# ── Plotting ───────────────────────────────────────────────────────────────────

def _smooth(values: list, window: int = 50) -> list:
    """Simple moving average for smoother curves."""
    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        result.append(np.mean(values[start:i + 1]))
    return result


def _plot_learning_curves(agent: QLearningAgent, save_path: str, lower_bound: float):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    episodes = list(range(1, len(agent.episode_makespans) + 1))
    smoothed_ms = _smooth(agent.episode_makespans, window=100)

    # ── Plot 1: Makespan over training ────────────────────────────────────────
    ax = axes[0]
    ax.plot(episodes, agent.episode_makespans, color="#b0c4de", alpha=0.3,
            linewidth=0.5, label="Per-episode")
    ax.plot(episodes, smoothed_ms, color="#2b5be0", linewidth=2,
            label="Smoothed (100-ep avg)")
    ax.axhline(lower_bound, color="green", linestyle="--", linewidth=1.5,
               label=f"Lower bound ({lower_bound:.1f})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Makespan")
    ax.set_title("Makespan During Training")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Plot 2: Cumulative reward ──────────────────────────────────────────────
    ax = axes[1]
    smoothed_r = _smooth(agent.episode_rewards, window=100)
    ax.plot(episodes, agent.episode_rewards, color="#f0a5a5", alpha=0.3,
            linewidth=0.5)
    ax.plot(episodes, smoothed_r, color="#c0392b", linewidth=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Episode Reward")
    ax.set_title("Episode Reward During Training")
    ax.grid(True, alpha=0.3)

    # ── Plot 3: Epsilon decay ─────────────────────────────────────────────────
    ax = axes[2]
    ax.plot(episodes, agent.epsilon_history, color="#f0a500", linewidth=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("ε (Exploration Rate)")
    ax.set_title("Epsilon Decay")
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)

    plt.suptitle("Q-Learning Training Progress — Job Scheduler", fontsize=13,
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 Learning curve saved → {save_path}")


if __name__ == "__main__":
    train()
