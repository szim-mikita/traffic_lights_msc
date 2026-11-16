import csv
import os
import random
from collections import defaultdict

import matplotlib.pyplot as plt
import multiprocessing
import numpy as np

from gym_envs.envs.state_traffic_env import (
    STATE_TRAFFIC_FIXED_GREEN_STATES,
    STATE_TRAFFIC_TRANSITION_SEQUENCES,
)
from gym_envs.envs.sumorl_env import SumoRLEnv
from train import ArrayFeaturizer, DQNAgent, DQNConfig


# -------------------------
# CONFIG (edit here)
# -------------------------
MODEL_PATHS = [
    "traffic_dqn_sumorl_l2_reward.pt"
    ]  # <-- add your model paths here
SIMULATION_TIME_SEC = 3600
TRAFFIC_SCALE = 8
DELTA_TIME = 5
RENDER_MODE = "console"
USE_GUI = False
SEED = 0
CSV_PATH = "policy_metrics_sumorl.csv"
SAVE_PLOT = True
SAVE_PLOT_DIR = "./plots"
SAVE_PLOT_FORMAT = "png"
SAVE_PLOT_NAME = "plot_sumorl_l2"  # if None, a name is generated automatically
EVAL_RUNS = 1
SUMO_NET_FILE = "sumo_files/SC_SU-102022_v01.net.xml"
SUMO_ROUTE_FILE = "sumo_files/dynamic_routes.rou.xml"

# Use None to include every available policy/model, or specify a list of names.
PLOT_POLICIES = None

# Include fixed policies in addition to model-based lines.
INCLUDE_CYCLIC_POLICY = True
INCLUDE_RANDOM_POLICY = True

# Add vertical lines at state switches for the chosen policies.
SHOW_STATE_SWITCH_LINES = False
SWITCH_LINE_POLICIES = ["Cyclic", "Random"]
SWITCH_LINE_ALPHA = 0.8
SWITCH_LINE_WIDTH = 0.8


class CyclicPolicy:
    def __init__(self):
        self.seq = [0, 0, 1, 1, 2, 2]
        self.i = 0

    def act(self, obs):
        action = self.seq[self.i % len(self.seq)]
        self.i += 1
        return action


class RandomPolicy:
    def __init__(self, action_dim):
        self.action_dim = action_dim

    def act(self, obs):
        return random.randrange(self.action_dim)


class DQNPolicy:
    def __init__(self, agent):
        self.agent = agent

    def act(self, obs):
        return self.agent.act(obs, eval_mode=True)


def build_sumorl_env():
    return SumoRLEnv(
        net_file=SUMO_NET_FILE,
        route_file=SUMO_ROUTE_FILE,
        num_seconds=SIMULATION_TIME_SEC,
        delta_time=DELTA_TIME,
        use_gui=USE_GUI,
        render_mode=None if RENDER_MODE == "console" else RENDER_MODE,
        ts_ids=["3"],
        fixed_green_states=STATE_TRAFFIC_FIXED_GREEN_STATES,
        transition_sequences=STATE_TRAFFIC_TRANSITION_SEQUENCES,
        single_agent=True,
        reward_fn="diff-waiting-time-l2",
        traffic_scale=TRAFFIC_SCALE,
    )


def build_agent(env, action_dim):
    obs, _ = env.reset()
    obs = np.asarray(obs, dtype=np.float32)
    input_dim = obs.size if hasattr(obs, "size") else len(obs)
    featurizer = ArrayFeaturizer(DQNConfig(), input_dim)
    return DQNAgent(action_dim=action_dim, featurizer=featurizer)


def get_model_name(model_path: str) -> str:
    return os.path.splitext(os.path.basename(model_path))[0]


def should_plot(name: str) -> bool:
    return PLOT_POLICIES is None or name in PLOT_POLICIES


def cars_waiting_from_info(info):
    if isinstance(info, dict):
        if "system_total_stopped" in info:
            return float(info["system_total_stopped"])
        if "system_total_waiting_time" in info:
            return float(info["system_total_waiting_time"])
    return 0.0


def avg_co2_from_env(env):
    conn = getattr(env, "traci", None) or getattr(env, "sumo", None)
    if conn is None:
        return 0.0

    try:
        edges = conn.edge.getIDList()
    except Exception:
        edges = []

    co2s = []
    for edge in edges:
        try:
            co2s.append(conn.edge.getCO2Emission(edge))
        except Exception:
            pass

    return float(np.mean(co2s)) if co2s else 0.0


def rollout_sumorl(env, policy):
    obs, _ = env.reset()
    ts_id = env.ts_ids[0] if hasattr(env, "ts_ids") and env.ts_ids else next(iter(env.traffic_signals.keys()))
    prev_phase = int(env.traffic_signals[ts_id].green_phase)

    xs = []
    ys = []
    rewards = []
    avg_speeds = []
    avg_co2s = []
    switch_times = []

    while True:
        action = policy.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)

        current_time = int(info.get("step", len(xs) * DELTA_TIME))
        current_phase = int(env.traffic_signals[ts_id].green_phase)

        if current_phase != prev_phase:
            switch_times.append(current_time)
            prev_phase = current_phase

        xs.append(current_time)
        ys.append(cars_waiting_from_info(info))
        rewards.append(float(reward))
        avg_speeds.append(float(info.get("system_mean_speed", 0.0)))
        avg_co2s.append(avg_co2_from_env(env))

        if done:
            break

    return (
        np.array(xs, dtype=np.int32),
        np.array(ys, dtype=np.float32),
        np.array(rewards, dtype=np.float32),
        np.array(avg_speeds, dtype=np.float32),
        np.array(avg_co2s, dtype=np.float32),
        np.array(switch_times, dtype=np.int32),
    )


def average_results(result_list):
    if not result_list:
        return None
    if len(result_list) == 1:
        return result_list[0]

    min_len = min(run[0].shape[0] for run in result_list)
    times = result_list[0][0][:min_len]
    waiting = np.mean(np.stack([run[1][:min_len] for run in result_list], axis=0), axis=0)
    rewards = np.mean(np.stack([run[2][:min_len] for run in result_list], axis=0), axis=0)
    speeds = np.mean(np.stack([run[3][:min_len] for run in result_list], axis=0), axis=0)
    co2s = np.mean(np.stack([run[4][:min_len] for run in result_list], axis=0), axis=0)
    return times, waiting, rewards, speeds, co2s, np.array([], dtype=np.int32)


def evaluate_policy(task):
    random.seed(task["seed"])
    np.random.seed(task["seed"])

    env = build_sumorl_env()
    try:
        if task["policy_type"] == "DQN":
            agent = build_agent(env, task["action_dim"])
            try:
                agent.load(task["model_path"])
            except Exception:
                pass
            policy = DQNPolicy(agent)
        elif task["policy_type"] == "Cyclic":
            policy = CyclicPolicy()
        elif task["policy_type"] == "Random":
            policy = RandomPolicy(task["action_dim"])
        else:
            raise ValueError(f"Unknown policy type: {task['policy_type']}")

        return task["name"], rollout_sumorl(env, policy)
    finally:
        env.close()


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    env = build_sumorl_env()
    action_dim = int(env.action_space.n)
    env.close()

    tasks = []
    for model_path in MODEL_PATHS:
        model_name = get_model_name(model_path)
        if should_plot(model_name):
            for run_index in range(EVAL_RUNS):
                tasks.append(
                    {
                        "name": model_name,
                        "policy_type": "DQN",
                        "model_path": model_path,
                        "action_dim": action_dim,
                        "seed": SEED + run_index,
                    }
                )

    if INCLUDE_CYCLIC_POLICY and should_plot("Cyclic"):
        for run_index in range(EVAL_RUNS):
            tasks.append(
                {
                    "name": "Cyclic",
                    "policy_type": "Cyclic",
                    "action_dim": action_dim,
                    "seed": SEED + run_index,
                }
            )
    if INCLUDE_RANDOM_POLICY and should_plot("Random"):
        for run_index in range(EVAL_RUNS):
            tasks.append(
                {
                    "name": "Random",
                    "policy_type": "Random",
                    "action_dim": action_dim,
                    "seed": SEED + run_index,
                }
            )

    results = {}
    if tasks:
        ctx = multiprocessing.get_context("spawn")
        num_workers = min(len(tasks), multiprocessing.cpu_count())
        grouped_results = defaultdict(list)
        with ctx.Pool(processes=num_workers) as pool:
            for name, result in pool.map(evaluate_policy, tasks):
                grouped_results[name].append(result)

        for name, result_list in grouped_results.items():
            results[name] = average_results(result_list)

    def write_policy_rows(writer, name, t, w, r, spd, co2):
        n = min(len(t), len(w), len(r), len(spd), len(co2))
        for i in range(n):
            writer.writerow([name, int(t[i]), float(w[i]), float(r[i]), float(spd[i]), float(co2[i])])

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["policy", "time_s", "waiting", "reward", "avg_speed", "avg_co2"])
        for name, (t, w, r, s, co2, _) in results.items():
            write_policy_rows(writer, name, t, w, r, s, co2)

    print(f"Saved CSV: {CSV_PATH} (averaged over {EVAL_RUNS} run{'s' if EVAL_RUNS != 1 else ''})")

    plt.figure(figsize=(10, 5))
    ax = plt.gca()
    color_map = {"DQN": "tab:blue", "Cyclic": "tab:orange", "Random": "tab:green"}

    for name, (t, w, _, _, _, switch_times) in results.items():
        linestyle = "-"
        color = color_map.get(name, None)
        ax.plot(t, w, label=name, linestyle=linestyle, color=color)

        if SHOW_STATE_SWITCH_LINES and name in SWITCH_LINE_POLICIES:
            for switch_time in switch_times:
                ax.axvline(
                    switch_time,
                    color=color,
                    alpha=SWITCH_LINE_ALPHA,
                    linewidth=SWITCH_LINE_WIDTH,
                    linestyle="--",
                )

    plt.xlabel("Simulation time (seconds)")
    plt.ylabel("Waiting / stopped vehicles")
    plt.title(f"SumORL evaluation (sim_time={SIMULATION_TIME_SEC}s, scale={TRAFFIC_SCALE}, avg_runs={EVAL_RUNS})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if SAVE_PLOT:
        os.makedirs(SAVE_PLOT_DIR, exist_ok=True)
        if SAVE_PLOT_NAME:
            save_name = f"{SAVE_PLOT_NAME}.{SAVE_PLOT_FORMAT}"
        else:
            labels = [name.replace(" ", "_") for name in results.keys()]
            short_label = "+".join(labels)
            save_name = f"plot_sumorl_{short_label}.{SAVE_PLOT_FORMAT}"
        save_path = os.path.join(SAVE_PLOT_DIR, save_name)
        plt.savefig(save_path, dpi=200)
        print(f"Saved plot: {save_path}")

    plt.show()


if __name__ == "__main__":
    main()
