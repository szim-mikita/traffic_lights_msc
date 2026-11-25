"""Train a Pareto-conditioned policy on a traffic-control MORL environment."""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import gymnasium as gym
import mo_gymnasium
import numpy as np
from morl_baselines.multi_policy.pcn.pcn import PCN

from gym_envs.envs.morl_env import (
    MO_REWARD_COMPONENTS,
    MO_REWARD_DEFAULT_SCALES,
    ExplorationActionSpace,
    TrafficMORLEnv,
)


CATALOGUE_SUMO_DIR = (
    Path(__file__).resolve().parent / "sumo_files" / "AI_agens_konyvtar"
)


class TrainingProgressWrapper(gym.Wrapper):
    """Report each training action interactively or at regular log intervals."""

    def __init__(self, env: gym.Env, total_timesteps: int):
        super().__init__(env)
        self.total_timesteps = total_timesteps
        self.steps = 0
        self.interactive = sys.stdout.isatty()
        self.report_interval = max(1, total_timesteps // 100)

    def step(self, action):
        result = self.env.step(action)
        self.steps += 1
        if self.interactive:
            print(
                f"\rTraining action steps: {self.steps}/{self.total_timesteps}",
                end="",
                flush=True,
            )
        elif (
            self.steps % self.report_interval == 0
            or self.steps >= self.total_timesteps
        ):
            print(f"Training action steps: {self.steps}/{self.total_timesteps}")
        return result

    def finish_progress(self) -> None:
        if self.interactive and self.steps:
            print()


def _epsilon_at_step(
    step: int, epsilon_start: float, epsilon_end: float, decay_steps: int
) -> float:
    fraction = min(step / decay_steps, 1.0)
    return epsilon_start + fraction * (epsilon_end - epsilon_start)


class ExploratoryPCN(PCN):
    """PCN with scheduled epsilon exploration during policy rollouts."""

    def __init__(
        self,
        epsilon_start: float,
        epsilon_end: float,
        epsilon_decay_steps: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.epsilon_schedule_steps = 0
        self.epsilon_exploration_actions = 0
        self.policy_actions = 0

    def _act(self, obs, desired_return, desired_horizon, eval_mode=False):
        if eval_mode:
            return super()._act(obs, desired_return, desired_horizon, eval_mode)

        if self.epsilon_schedule_steps == 0:
            self.epsilon_schedule_steps = self.global_step
        epsilon = _epsilon_at_step(
            self.epsilon_schedule_steps,
            self.epsilon_start,
            self.epsilon_end,
            self.epsilon_decay_steps,
        )
        self.epsilon_schedule_steps += 1
        if self.np_random.random() < epsilon:
            self.epsilon_exploration_actions += 1
            return self.env.action_space.sample()
        self.policy_actions += 1
        return super()._act(obs, desired_return, desired_horizon, eval_mode)


def run_train(
    env: gym.Env,
    total_timesteps: int,
    scaling_factor: np.ndarray,
    max_return: np.ndarray,
    ref_point: np.ndarray,
    initial_episodes: int,
    step_episodes: int,
    model_updates: int,
    batch_size: int,
    hidden_dim: int,
    learning_rate: float,
    gamma: float,
    seed: int,
    model_name: str = "PCN_model",
    results_dir: str | Path = "runs",
    epsilon_start: float = 0.2,
    epsilon_end: float = 0.02,
    epsilon_decay_steps: int | None = None,
    actuated_exploration_probability: float = 0.75,
    max_buffer_size: int = 100,
    num_points_pf: int = 100,
    num_eval_weights_for_eval: int = 50,
) -> PCN:
    """Train PCN, saving its policy and a compact training/evaluation record."""
    if total_timesteps <= 0:
        raise ValueError("total_timesteps must be positive.")
    if initial_episodes < 4:
        raise ValueError("PCN requires at least four initial episodes.")
    if not 0.0 <= epsilon_end <= epsilon_start <= 1.0:
        raise ValueError("epsilon values must satisfy 0 <= end <= start <= 1.")
    if not 0.0 <= actuated_exploration_probability <= 1.0:
        raise ValueError(
            "actuated_exploration_probability must be between 0 and 1."
        )
    if epsilon_decay_steps is None:
        epsilon_decay_steps = max(1, total_timesteps)
    if epsilon_decay_steps <= 0:
        raise ValueError("epsilon_decay_steps must be positive.")
    if max_buffer_size <= 0 or num_points_pf <= 0 or num_eval_weights_for_eval <= 0:
        raise ValueError("PCN buffer size and evaluation counts must be positive.")

    reward_dim = env.unwrapped.reward_space.shape[0]
    if scaling_factor.shape != (reward_dim + 1,):
        raise ValueError(
            "PCN scaling_factor must contain one value per reward objective plus one "
            f"for the desired horizon (expected {reward_dim + 1}, got {scaling_factor.size})."
        )
    if max_return.shape != (reward_dim,) or ref_point.shape != (reward_dim,):
        raise ValueError(f"max_return and ref_point must each contain {reward_dim} values.")

    progress_env = TrainingProgressWrapper(env, total_timesteps)
    if isinstance(progress_env.action_space, ExplorationActionSpace):
        progress_env.action_space.set_actuated_probability(
            actuated_exploration_probability
        )
        progress_env.action_space.reset_sample_counts()
    progress_env.action_space.seed(seed)
    agent = ExploratoryPCN(
        env=progress_env,
        scaling_factor=scaling_factor,
        learning_rate=learning_rate,
        gamma=gamma,
        batch_size=batch_size,
        hidden_dim=hidden_dim,
        log=False,
        seed=seed,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay_steps=epsilon_decay_steps,
    )
    try:
        agent.train(
            total_timesteps=total_timesteps,
            eval_env=env,
            ref_point=ref_point,
            num_er_episodes=initial_episodes,
            num_step_episodes=step_episodes,
            num_model_updates=model_updates,
            max_return=max_return,
            max_buffer_size=max_buffer_size,
            num_points_pf=num_points_pf,
            num_eval_weights_for_eval=num_eval_weights_for_eval,
        )
    finally:
        progress_env.finish_progress()
    agent.save(filename=model_name)
    evaluated_returns, replay_returns, return_distances = agent.evaluate(
        env, max_return, n=10
    )
    created_at = datetime.now(timezone.utc)
    results_path = (
        Path(results_dir)
        / created_at.date().isoformat()
        / f"{model_name}_results.json"
    )
    results_path.parent.mkdir(parents=True, exist_ok=True)
    action_space = progress_env.action_space
    results = {
        "created_at_utc": created_at.isoformat(),
        "model_path": str(Path("weights") / f"{model_name}.pt"),
        "environment": {
            "class": type(env.unwrapped).__name__,
            "traffic_light_id": getattr(env.unwrapped, "traffic_light_id", None),
            "simulation_time": getattr(env.unwrapped, "simulation_time", None),
            "random_traffic": getattr(env.unwrapped, "random_traffic", None),
            "route_file": (
                str(env.unwrapped.route_file)
                if getattr(env.unwrapped, "route_file", None) is not None
                else None
            ),
            "flow_range": (
                list(env.unwrapped.flow_range)
                if hasattr(env.unwrapped, "flow_range")
                else None
            ),
            "reward_components": list(env.unwrapped.reward_components),
            "reward_normalization": env.unwrapped.reward_normalization,
            "reward_scales": env.unwrapped.reward_scales.tolist(),
            "min_green_time": getattr(env.unwrapped, "min_green_time", None),
            "state_penalty_threshold": getattr(
                env.unwrapped, "state_penalty_threshold", None
            ),
            "state_penalty_steepness": getattr(
                env.unwrapped, "state_penalty_growth", None
            ),
            "actuated_max_green_time": getattr(
                env.unwrapped, "actuated_max_green_time", None
            ),
        },
        "training": {
            "algorithm": "PCN",
            "total_timesteps": total_timesteps,
            "seed": seed,
            "scaling_factor": scaling_factor.tolist(),
            "max_return": max_return.tolist(),
            "ref_point": ref_point.tolist(),
            "initial_episodes": initial_episodes,
            "step_episodes": step_episodes,
            "model_updates": model_updates,
            "batch_size": batch_size,
            "hidden_dim": hidden_dim,
            "learning_rate": learning_rate,
            "gamma": gamma,
            "epsilon_start": epsilon_start,
            "epsilon_end": epsilon_end,
            "epsilon_decay_steps": epsilon_decay_steps,
            "actuated_exploration_probability": (
                actuated_exploration_probability
            ),
            "random_phase_exploration_probability": (
                1.0 - actuated_exploration_probability
            ),
            "max_buffer_size": max_buffer_size,
            "num_points_pf": num_points_pf,
            "num_eval_weights_for_eval": num_eval_weights_for_eval,
            "epsilon_schedule_steps": agent.epsilon_schedule_steps,
            "epsilon_exploration_actions": agent.epsilon_exploration_actions,
            "policy_actions": agent.policy_actions,
            "actuated_exploration_actions": getattr(
                action_space, "actuated_samples", None
            ),
            "random_phase_exploration_actions": getattr(
                action_space, "random_phase_samples", None
            ),
        },
        "evaluation": {
            "episodes": len(evaluated_returns),
            "returns": np.asarray(evaluated_returns).tolist(),
            "replay_returns": np.asarray(replay_returns).tolist(),
            "return_distances": np.asarray(return_distances).tolist(),
        },
    }
    results_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Saved PCN model to weights/{model_name}.pt")
    print(f"Saved training results to {results_path}")
    return agent


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-timesteps", type=int, default=10_000)
    parser.add_argument("--num-seconds", type=int, default=3600)
    parser.add_argument(
        "--net-file",
        type=Path,
        default=CATALOGUE_SUMO_DIR / "mega_catalogue_final.net.xml",
    )
    parser.add_argument(
        "--logic-json-file",
        type=Path,
        default=CATALOGUE_SUMO_DIR / "traffic_lights.json",
    )
    parser.add_argument(
        "--detector-file",
        type=Path,
        default=CATALOGUE_SUMO_DIR / "mega_catalogue_final.detectors.add.xml",
    )
    parser.add_argument("--traffic-light-id", default=None)
    parser.add_argument("--route-file", type=Path, default=None)
    parser.add_argument(
        "--random-traffic",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--flow-range", type=int, nargs=2, default=(100, 600))
    parser.add_argument(
        "--reward-components",
        nargs="+",
        choices=MO_REWARD_COMPONENTS,
        default=["co2", "time_in_state_penalty"],
    )
    parser.add_argument(
        "--reward-normalization",
        choices=("sum", "average", "normalized_average"),
        default="normalized_average",
    )
    parser.add_argument("--reward-scales", type=float, nargs="+", default=None)
    parser.add_argument("--state-penalty-threshold", type=float, default=45.0)
    parser.add_argument(
        "--state-penalty-steepness",
        "--state-penalty-growth",
        dest="state_penalty_growth",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--actuated-exploration-probability", type=float, default=0.75
    )
    parser.add_argument("--actuated-max-green-time", type=int, default=60)
    parser.add_argument("--initial-episodes", type=int, default=4)
    parser.add_argument("--step-episodes", type=int, default=1)
    parser.add_argument("--model-updates", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--epsilon-start", type=float, default=0.2)
    parser.add_argument("--epsilon-end", type=float, default=0.02)
    parser.add_argument("--epsilon-decay-steps", type=int, default=None)
    parser.add_argument("--max-buffer-size", type=int, default=100)
    parser.add_argument("--num-points-pf", type=int, default=100)
    parser.add_argument("--num-eval-weights-for-eval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-name", default="PCN_model")
    parser.add_argument("--results-dir", type=Path, default=Path("runs"))
    parser.add_argument("--scaling-factor", type=float, nargs="+", default=None)
    parser.add_argument("--max-return", type=float, nargs="+", default=None)
    parser.add_argument("--ref-point", type=float, nargs="+", default=None)
    parser.add_argument("--render-mode", choices=["console", "gui"], default="console")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    reward_dim = len(args.reward_components)
    reward_scales = args.reward_scales
    if reward_scales is not None and len(reward_scales) != reward_dim:
        raise ValueError(
            f"--reward-scales needs {reward_dim} values for the selected objectives."
        )
    reward_scales = reward_scales or [
        MO_REWARD_DEFAULT_SCALES[name] for name in args.reward_components
    ]
    scaling_factor = args.scaling_factor or [1.0] * (reward_dim + 1)
    max_return = args.max_return or [100.0] * reward_dim
    ref_point = args.ref_point or [-100.0] * reward_dim
    for name, values in (
        ("--scaling-factor", scaling_factor),
        ("--max-return", max_return),
        ("--ref-point", ref_point),
    ):
        expected = reward_dim + 1 if name == "--scaling-factor" else reward_dim
        if len(values) != expected:
            raise ValueError(f"{name} needs {expected} values for the selected objectives.")

    base_env = TrafficMORLEnv(
        net_file=args.net_file,
        logic_json_file=args.logic_json_file,
        detector_file=args.detector_file,
        traffic_light_id=args.traffic_light_id,
        route_file=args.route_file,
        render_mode=args.render_mode,
        simulation_time=args.num_seconds,
        random_traffic=args.random_traffic,
        flow_range=args.flow_range,
        reward_components=args.reward_components,
        reward_normalization=args.reward_normalization,
        reward_scales=reward_scales,
        state_penalty_threshold=args.state_penalty_threshold,
        state_penalty_growth=args.state_penalty_growth,
        actuated_exploration_probability=args.actuated_exploration_probability,
        actuated_max_green_time=args.actuated_max_green_time,
    )
    env = mo_gymnasium.wrappers.MORecordEpisodeStatistics(
        gym.wrappers.FlattenObservation(base_env)
    )

    try:
        run_train(
            env=env,
            total_timesteps=args.total_timesteps,
            scaling_factor=np.asarray(scaling_factor, dtype=np.float32),
            max_return=np.asarray(max_return, dtype=np.float32),
            ref_point=np.asarray(ref_point, dtype=np.float32),
            initial_episodes=args.initial_episodes,
            step_episodes=args.step_episodes,
            model_updates=args.model_updates,
            batch_size=args.batch_size,
            hidden_dim=args.hidden_dim,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            seed=args.seed,
            model_name=args.model_name,
            results_dir=args.results_dir,
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
            epsilon_decay_steps=args.epsilon_decay_steps,
            actuated_exploration_probability=(
                args.actuated_exploration_probability
            ),
            max_buffer_size=args.max_buffer_size,
            num_points_pf=args.num_points_pf,
            num_eval_weights_for_eval=args.num_eval_weights_for_eval,
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
