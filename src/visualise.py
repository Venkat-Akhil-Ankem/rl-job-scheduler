"""
visualise.py
------------
Generates a Gantt chart showing the schedule produced by:
  1. The trained RL agent
  2. The SPT baseline (for comparison)

Output: outputs/gantt_chart.png

Run:
    python src/visualise.py
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).parent))
from environment import JobSchedulingEnv, spt_policy
from agent import QLearningAgent
from train import CONFIG, run_episode


# ── Colour palette for jobs ───────────────────────────────────────────────────
JOB_COLOURS = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2",
    "#59a14f", "#edc948", "#b07aa1", "#ff9da7",
    "#9c755f", "#bab0ac",
]


def build_schedule(env: JobSchedulingEnv, agent: QLearningAgent = None) -> list:
    """
    Run one episode and return the schedule.
    If agent is None, uses SPT policy.
    """
    if agent is not None:
        run_episode(env, agent, train=False)
    else:
        spt_policy(env)
    return env.schedule.copy()


def draw_gantt(
    schedule: list,
    n_machines: int,
    processing_times: np.ndarray,
    title: str,
    ax: plt.Axes,
):
    """Draw a Gantt chart on the given axes."""
    makespan = max(s["end"] for s in schedule)

    for entry in schedule:
        job     = entry["job"]
        machine = entry["machine"]
        start   = entry["start"]
        proc    = entry["proc"]
        color   = JOB_COLOURS[job % len(JOB_COLOURS)]

        bar = mpatches.FancyBboxPatch(
            (start, machine - 0.35),
            proc, 0.70,
            boxstyle="round,pad=0.05",
            facecolor=color,
            edgecolor="white",
            linewidth=1.2,
        )
        ax.add_patch(bar)

        # Job label inside the bar
        ax.text(
            start + proc / 2, machine,
            f"J{job}\n({int(proc)})",
            ha="center", va="center",
            fontsize=7, fontweight="bold", color="white",
        )

    ax.set_xlim(0, makespan * 1.05)
    ax.set_ylim(-0.6, n_machines - 0.4)
    ax.set_yticks(range(n_machines))
    ax.set_yticklabels([f"Machine {i}" for i in range(n_machines)])
    ax.set_xlabel("Time")
    ax.set_title(f"{title}\nMakespan = {makespan:.1f}", fontweight="bold")
    ax.axvline(makespan, color="red", linestyle="--", linewidth=1.2, alpha=0.7,
               label=f"Makespan = {makespan:.1f}")
    ax.grid(True, axis="x", alpha=0.3)

    # Legend for jobs
    handles = [
        mpatches.Patch(color=JOB_COLOURS[j % len(JOB_COLOURS)], label=f"Job {j} (p={int(processing_times[j])})")
        for j in range(len(processing_times))
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=7, framealpha=0.8)


def main():
    agent_path = Path(CONFIG["agent_path"])
    if not agent_path.exists():
        print(f"❌ No trained agent at {agent_path}")
        print("   Run  python src/train.py  first.")
        return

    agent = QLearningAgent.load(str(agent_path))

    # Use the same fixed instance as training
    env = JobSchedulingEnv(
        n_jobs=CONFIG["n_jobs"],
        n_machines=CONFIG["n_machines"],
        seed=CONFIG["env_seed"],
    )
    env.reset()
    fixed_times = env.processing_times.copy()
    env._fixed_times = fixed_times

    print(f"Instance — processing times: {fixed_times.astype(int).tolist()}")
    print(f"Lower bound: {env.optimal_lower_bound:.1f}")

    # ── Build schedules ───────────────────────────────────────────────────────
    # RL agent
    env._fixed_times = fixed_times
    rl_schedule = build_schedule(env, agent=agent)
    rl_makespan = max(s["end"] for s in rl_schedule)

    # SPT baseline
    env._fixed_times = fixed_times
    spt_schedule = build_schedule(env, agent=None)
    spt_makespan = max(s["end"] for s in spt_schedule)

    print(f"RL Agent makespan: {rl_makespan:.1f}")
    print(f"SPT makespan:      {spt_makespan:.1f}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(12, 7))

    draw_gantt(rl_schedule,  CONFIG["n_machines"], fixed_times,
               "RL Agent (Q-learning)",  axes[0])
    draw_gantt(spt_schedule, CONFIG["n_machines"], fixed_times,
               "SPT Greedy Baseline",    axes[1])

    plt.suptitle(
        f"Job Scheduling — {CONFIG['n_jobs']} Jobs × {CONFIG['n_machines']} Machines",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()

    output_path = Path(CONFIG["output_dir"]) / "gantt_chart.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 Gantt chart saved → {output_path}")


if __name__ == "__main__":
    main()
