"""Compare a trained PCN policy with the catalogue network's fixed-time plan."""

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import gymnasium as gym
import mo_gymnasium
import numpy as np
import torch
from morl_baselines.multi_policy.pcn.pcn import PCN

from gym_envs.envs.catalogue_morl_env import CatalogueTrafficMOEnv
from train_morl import CATALOGUE_SUMO_DIR


def comparison_objectives(components: tuple[str, ...]) -> tuple[str, ...]:
    """Return objectives available to both PCN and the native fixed-time plan."""
    shared = tuple(
        component for component in components if component != "time_in_state_penalty"
    )
    if not shared:
        raise ValueError(
            "The native fixed-time comparison requires at least one objective "
            "other than time_in_state_penalty."
        )
    return shared


def build_env(
    results: dict,
    simulation_time: int,
    flow_range: tuple[int, int],
    reward_components: tuple[str, ...] | None = None,
) -> gym.Env:
    environment = results["environment"]
    all_components = tuple(environment["reward_components"])
    components = reward_components or all_components
    component_indices = [all_components.index(component) for component in components]
    base_env = CatalogueTrafficMOEnv(
        net_file=CATALOGUE_SUMO_DIR / "mega_catalogue_final.net.xml",
        logic_json_file=CATALOGUE_SUMO_DIR / "traffic_lights.json",
        detector_file=CATALOGUE_SUMO_DIR / "mega_catalogue_final.detectors.add.xml",
        traffic_light_id=environment.get("traffic_light_id", "R3C2_C"),
        simulation_time=simulation_time,
        flow_range=flow_range,
        reward_components=components,
        reward_normalization=environment["reward_normalization"],
        reward_scales=[environment["reward_scales"][index] for index in component_indices],
    )
    return mo_gymnasium.wrappers.MORecordEpisodeStatistics(
        gym.wrappers.FlattenObservation(base_env)
    )


def run_episode(
    results: dict,
    model: PCN | None,
    seed: int,
    simulation_time: int,
    flow_range: tuple[int, int],
    desired_return: np.ndarray,
    desired_horizon: int,
    max_return: np.ndarray,
    program_id: str,
    capture_trace: bool = False,
    reward_components: tuple[str, ...] | None = None,
) -> tuple[np.ndarray, int, list[dict[str, int | str]] | None]:
    env = build_env(results, simulation_time, flow_range, reward_components)
    try:
        base_env = env.unwrapped
        base_env.record_signal_trace = capture_trace
        observation, _ = env.reset(seed=seed)
        if model is None:
            base_env.activate_fixed_time_program(program_id)

        episode_return = np.zeros(len(base_env.reward_components), dtype=np.float64)
        elapsed_seconds = 0
        remaining_return = desired_return.copy()
        remaining_horizon = np.float32(desired_horizon)
        done = False
        while not done:
            if model is None:
                observation, reward, terminated, truncated, info = (
                    env.unwrapped.step_fixed_time()
                )
            else:
                with torch.inference_mode():
                    action = model._act(
                        observation,
                        remaining_return,
                        remaining_horizon,
                        eval_mode=True,
                    )
                observation, reward, terminated, truncated, info = env.step(action)
                remaining_return = np.clip(
                    remaining_return - reward,
                    None,
                    max_return,
                )
                remaining_horizon = np.float32(max(remaining_horizon - 1, 1.0))
            episode_return += reward
            elapsed_seconds += info["elapsed_seconds"]
            done = bool(terminated or truncated)
        trace = list(base_env.signal_trace) if capture_trace else None
        return episode_return, elapsed_seconds, trace
    finally:
        env.close()


def compare(
    results_path: Path,
    episodes: int,
    first_seed: int,
    simulation_time: int | None,
    flow_range: tuple[int, int] | None,
    desired_horizon: int,
    output_dir: Path,
    prefix: str,
    program_id: str = "0",
    model_path: Path | None = None,
    trace_seed: int | None = None,
) -> tuple[Path, Path, Path, Path]:
    with results_path.open(encoding="utf-8") as file:
        results = json.load(file)

    training = results["training"]
    environment = results["environment"]
    simulation_time = (
        simulation_time or environment.get("simulation_time") or 60
    )
    flow_range = tuple(
        flow_range or environment.get("flow_range") or (100, 600)
    )
    components = tuple(results["environment"]["reward_components"])
    shared_components = comparison_objectives(components)
    shared_indices = [components.index(component) for component in shared_components]
    replay_returns = np.asarray(results["evaluation"]["replay_returns"], dtype=np.float32)
    if replay_returns.ndim != 2 or replay_returns.shape[1] != len(components):
        raise ValueError("Saved replay returns do not match the objective components.")
    desired_return = replay_returns.mean(axis=0)
    max_return = np.asarray(training["max_return"], dtype=np.float32)
    checkpoint = model_path or (results_path.parent.parent / results["model_path"])

    model_env = build_env(results, simulation_time, flow_range)
    try:
        model = PCN(
            env=model_env,
            scaling_factor=np.asarray(training["scaling_factor"], dtype=np.float32),
            learning_rate=training["learning_rate"],
            gamma=training["gamma"],
            batch_size=training["batch_size"],
            hidden_dim=training["hidden_dim"],
            log=False,
            seed=training["seed"],
        )
        model.model = torch.load(
            checkpoint,
            map_location=model.device,
            weights_only=False,
        )
        model.model.eval()
    finally:
        model_env.close()

    seeds = list(range(first_seed, first_seed + episodes))
    trace_seed = first_seed if trace_seed is None else trace_seed
    if trace_seed not in seeds:
        raise ValueError("trace_seed must be one of the evaluated episode seeds.")
    native_returns = []
    pcn_full_returns = []
    pcn_durations = []
    native_durations = []
    pcn_trace = None
    native_trace = None
    for seed in seeds:
        capture_trace = seed == trace_seed
        episode_return, duration, trace = run_episode(
            results,
            model,
            seed,
            simulation_time,
            flow_range,
            desired_return,
            desired_horizon,
            max_return,
            program_id,
            capture_trace=capture_trace,
        )
        pcn_full_returns.append(episode_return.tolist())
        pcn_durations.append(duration)
        if capture_trace:
            pcn_trace = trace
        episode_return, duration, trace = run_episode(
            results,
            None,
            seed,
            simulation_time,
            flow_range,
            desired_return,
            desired_horizon,
            max_return,
            program_id,
            capture_trace=capture_trace,
            reward_components=shared_components,
        )
        native_returns.append(episode_return.tolist())
        native_durations.append(duration)
        if capture_trace:
            native_trace = trace

    output_dir.mkdir(parents=True, exist_ok=True)
    pcn_path = output_dir / f"{prefix}_pcn_results.json"
    native_path = output_dir / f"{prefix}_native_fixed_time_results.json"
    csv_path = output_dir / f"{prefix}_paired.csv"
    trace_path = output_dir / f"{prefix}_state_trace.json"
    comparison_environment = dict(environment)
    comparison_environment["reward_components"] = list(shared_components)
    comparison_environment["reward_scales"] = [
        environment["reward_scales"][index] for index in shared_indices
    ]
    shared = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": comparison_environment,
        "comparison": {
            "source_results": str(results_path),
            "seeds": seeds,
            "trace_seed": trace_seed,
            "simulation_time": simulation_time,
            "flow_range": list(flow_range),
            "desired_return": desired_return.tolist(),
            "pcn_objectives": list(components),
            "shared_objectives": list(shared_components),
            "desired_horizon_steps": desired_horizon,
        },
    }
    pcn_result = {
        **shared,
        "policy": "PCN",
        "model_path": str(checkpoint),
        "policy_objectives": list(components),
        "evaluation": {
            "episodes": episodes,
            "seeds": seeds,
            "returns": [
                [episode[index] for index in shared_indices]
                for episode in pcn_full_returns
            ],
            "full_policy_returns": pcn_full_returns,
            "additional_objective_returns": {
                component: [
                    episode[index] for episode in pcn_full_returns
                ]
                for index, component in enumerate(components)
                if component not in shared_components
            },
            "elapsed_seconds": pcn_durations,
        },
    }
    native_result = {
        **shared,
        "policy": f"Native SUMO fixed-time (program {program_id})",
        "evaluation": {
            "episodes": episodes,
            "seeds": seeds,
            "returns": native_returns,
            "elapsed_seconds": native_durations,
        },
    }
    pcn_path.write_text(json.dumps(pcn_result, indent=2) + "\n", encoding="utf-8")
    native_path.write_text(
        json.dumps(native_result, indent=2) + "\n",
        encoding="utf-8",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        additional_components = [
            component for component in components if component not in shared_components
        ]
        writer.writerow(["seed", "policy", *shared_components, *additional_components])
        for index, seed in enumerate(seeds):
            writer.writerow(
                [
                    seed,
                    "PCN",
                    *[pcn_full_returns[index][i] for i in shared_indices],
                    *[
                        pcn_full_returns[index][components.index(component)]
                        for component in additional_components
                    ],
                ]
            )
            writer.writerow(
                [
                    seed,
                    "Native SUMO fixed-time",
                    *native_returns[index],
                    *["" for _ in additional_components],
                ]
            )
    trace_result = {
        "seed": trace_seed,
        "simulation_time": simulation_time,
        "traffic_light_id": environment.get("traffic_light_id", "R3C2_C"),
        "states": {
            "PCN": pcn_trace,
            f"Native SUMO fixed-time (program {program_id})": native_trace,
        },
    }
    trace_path.write_text(
        json.dumps(trace_result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Desired PCN return: {desired_return.tolist()}")
    print(f"Desired PCN horizon: {desired_horizon} actions")
    print(f"Compared objectives: {', '.join(shared_components)}")
    for component in additional_components:
        values = [episode[components.index(component)] for episode in pcn_full_returns]
        print(
            f"PCN-only objective {component}: "
            f"mean return {float(np.mean(values)):.4f}"
        )
    print(f"Saved PCN results: {pcn_path}")
    print(f"Saved native fixed-time results: {native_path}")
    print(f"Saved paired returns: {csv_path}")
    print(f"Saved state trace for seed {trace_seed}: {trace_path}")
    return pcn_path, native_path, csv_path, trace_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="Training result JSON for the PCN model.")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument(
        "--trace-seed",
        type=int,
        default=None,
        help="Seed to record state trajectories for (defaults to --seed).",
    )
    parser.add_argument(
        "--num-seconds",
        type=int,
        default=None,
        help="Defaults to the training result's episode duration, or 60 seconds.",
    )
    parser.add_argument(
        "--flow-range",
        type=int,
        nargs=2,
        default=None,
        help="Defaults to the training result's traffic range, or 100-600.",
    )
    parser.add_argument("--desired-horizon", type=int, default=7)
    parser.add_argument("--program-id", default="0")
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    parser.add_argument("--prefix", default="PCN_vs_native_fixed_time")
    args = parser.parse_args()
    if (
        args.episodes <= 0
        or (args.num_seconds is not None and args.num_seconds <= 0)
        or args.desired_horizon <= 0
    ):
        parser.error("--episodes, --num-seconds, and --desired-horizon must be positive.")
    trace_seed = args.trace_seed if args.trace_seed is not None else args.seed
    if not args.seed <= trace_seed < args.seed + args.episodes:
        parser.error("--trace-seed must be included in the evaluated seed range.")
    compare(
        args.results,
        args.episodes,
        args.seed,
        args.num_seconds,
        tuple(args.flow_range) if args.flow_range is not None else None,
        args.desired_horizon,
        args.output_dir,
        args.prefix,
        program_id=args.program_id,
        model_path=args.model_path,
        trace_seed=trace_seed,
    )


if __name__ == "__main__":
    main()
