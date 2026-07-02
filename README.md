# 🤖 RL Job Scheduler

> A **Reinforcement Learning agent** that learns — from scratch, through trial
> and error — to schedule jobs on machines to minimise total completion time
> (makespan). Built with Q-learning and a custom Gym-style environment.
> No external RL library required.

---

## 🌟 What It Does

An agent observes the current state of a job shop (which jobs are waiting,
which machines are free) and decides which job to assign next. It receives a
**reward signal** based on how well it does, and gradually learns a
**policy** — a mapping from states to actions — that minimises makespan.

Over thousands of training episodes the agent improves from random decisions
to a learned scheduling strategy, which you can visualise as a Gantt chart.

---

## 🎯 Why This Problem?

Job scheduling is a classical NP-hard combinatorial optimisation problem.
Using RL to solve (or approximate solutions to) scheduling problems is an
active research area connecting OR and ML — see work by Zhang et al. (2020),
Kwon et al. (2021), and others on neural combinatorial optimisation.

This project demonstrates the **foundations** of that line of work:
building a scheduling environment, defining state/action/reward, and
training a Q-learning agent.

---

## 🏗️ Project Structure

```
rl-job-scheduler/
│
├── src/
│   ├── environment.py      # Custom Gym-style scheduling environment
│   ├── agent.py            # Q-learning agent (epsilon-greedy exploration)
│   ├── train.py            # Training loop + learning curve plots
│   ├── evaluate.py         # Compare RL agent vs random vs greedy baselines
│   └── visualise.py        # Gantt chart of the learned schedule
│
├── outputs/                # Saved Q-table, plots (auto-generated, git-ignored)
│
├── tests/
│   ├── test_environment.py # Unit tests for the RL environment
│   └── test_agent.py       # Unit tests for the Q-learning agent
│
├── notebooks/
│   └── walkthrough.ipynb   # End-to-end Jupyter notebook
│
├── docs/
│   └── architecture.md     # RL concepts and design decisions
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### 1. Install

```bash
git clone https://github.com/YOUR_USERNAME/rl-job-scheduler.git
cd rl-job-scheduler

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Train the Agent

```bash
python src/train.py
# Trains for 5000 episodes, prints progress every 500
# Saves Q-table → outputs/q_table.npy
# Saves learning curve → outputs/learning_curve.png
```

### 3. Evaluate & Compare Baselines

```bash
python src/evaluate.py
# Compares: RL agent vs Random policy vs Greedy (SPT) baseline
```

### 4. Visualise the Schedule

```bash
python src/visualise.py
# Generates a Gantt chart → outputs/gantt_chart.png
```

### 5. Run Tests

```bash
pytest tests/ -v
```

---

## 💡 How It Works — Full Flow

```
┌─────────────────────────────────────────────────────┐
│              RL Training Loop                        │
│                                                      │
│   For each episode:                                  │
│     env.reset()  →  initial state s                 │
│                                                      │
│     While not done:                                  │
│       agent.choose_action(s)  →  action a            │
│         (ε-greedy: explore or exploit Q-table)       │
│                                                      │
│       env.step(a)  →  next_state s', reward r, done │
│                                                      │
│       Q-update (Bellman equation):                   │
│       Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',·) - Q(s,a)]│
│                                                      │
│       s ← s'                                         │
│                                                      │
│     ε decays → agent exploits more over time         │
└─────────────────────────────────────────────────────┘
```

### The Environment (`environment.py`)

| Element | Definition |
|---|---|
| **State** | Tuple: (jobs remaining bitmask, machine availability times) |
| **Action** | Index of the next job to schedule (from available jobs) |
| **Reward** | −1 per step (encourages finishing fast); bonus at completion |
| **Episode end** | All jobs have been assigned to a machine |
| **Objective** | Minimise makespan (time when last machine finishes) |

### The Agent (`agent.py`)

| Concept | Implementation |
|---|---|
| **Q-table** | Dictionary mapping state → action values |
| **Exploration** | ε-greedy: random action with probability ε |
| **Exploitation** | argmax Q(s, ·) |
| **ε decay** | ε = max(ε_min, ε × decay_rate) after each episode |
| **Q-update** | Bellman equation with learning rate α and discount γ |

---

## 📊 Expected Results

On a 6-job × 3-machine instance after 5000 training episodes:

| Policy | Avg Makespan |
|---|---|
| Random | ~18–22 |
| Greedy (SPT) | ~13–15 |
| **RL Agent** | **~11–14** |

The RL agent learns to beat random quickly and typically matches or
beats the Shortest Processing Time (SPT) greedy heuristic.

---

## 🤖 RL Concepts Demonstrated

| Concept | Implementation |
|---|---|
| **Markov Decision Process** | Custom environment with S, A, R, T |
| **Q-learning** | Off-policy temporal difference learning |
| **Bellman equation** | Core Q-value update rule |
| **ε-greedy exploration** | Balancing explore vs exploit |
| **ε decay schedule** | Gradually shift from exploration to exploitation |
| **Discount factor γ** | Future reward weighting |
| **Episode-based training** | Reset environment, collect trajectory |
| **Baseline comparison** | Random and greedy policies for benchmarking |
| **Policy extraction** | greedy policy from trained Q-table |
| **Gantt chart visualisation** | Interpret the learned schedule |

---

## 📦 Tech Stack

- **Python 3.9+** — no RL framework needed
- **NumPy** — Q-table operations and environment logic
- **Matplotlib** — learning curves and Gantt charts
- **pytest** — unit testing

---

## 🔗 Related Research

This project is a simplified demonstration of ideas explored in:

- Zhang et al. (2020) — *Learning to Dispatch for Job Shop Scheduling via Deep RL*
- Kwon et al. (2021) — *POMO: Policy Optimisation with Multiple Optima*
- Nazari et al. (2018) — *RL for Solving the Vehicle Routing Problem*

---

## 📄 License

MIT — free to use, modify, and distribute.

---

## 👤 Author

Portfolio project demonstrating Reinforcement Learning fundamentals —
Q-learning, custom RL environments, exploration strategies, and the
application of RL to combinatorial optimisation scheduling problems.
