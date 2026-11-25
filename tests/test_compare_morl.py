import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import compare_morl


class StubEnv(gym.Env):
    def __init__(self):
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(0, 1, shape=(1,), dtype=np.float32)
        self.reward_space = spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32)


class CompareMORLTests(unittest.TestCase):
    def test_checkpoint_path_is_resolved_from_project_root_for_dated_results(self):
        results = {
            "training": {
                "scaling_factor": [1.0, 1.0, 1.0],
                "learning_rate": 1e-3,
                "gamma": 1.0,
                "batch_size": 8,
                "hidden_dim": 16,
                "seed": 7,
                "max_return": [1.0, 1.0],
            },
            "environment": {
                "traffic_light_id": "R3C2_C",
                "reward_components": ["co2", "queue_length"],
                "reward_normalization": "average",
                "reward_scales": [10_000.0, 10.0],
                "simulation_time": 30,
                "flow_range": [600, 600],
            },
            "evaluation": {
                "replay_returns": [[-1.0, -0.5], [-0.8, -0.4]],
            },
            "model_path": "weights/catalogue_test.pt",
        }

        with tempfile.TemporaryDirectory() as directory:
            results_path = Path(directory) / "runs" / "2026-11-25" / "results.json"
            results_path.parent.mkdir(parents=True)
            results_path.write_text(json.dumps(results), encoding="utf-8")
            model_env = StubEnv()

            with (
                patch.object(compare_morl, "build_env", return_value=model_env),
                patch.object(compare_morl.PCN, "__init__", return_value=None) as init,
                patch.object(compare_morl.PCN, "device", "cpu", create=True),
                patch.object(compare_morl.torch, "load") as load,
            ):
                # The checkpoint is loaded before an episode environment is needed.
                with patch.object(compare_morl, "run_episode") as run_episode:
                    run_episode.return_value = (
                        np.asarray([-1.0, -0.5]),
                        30,
                        [{"time": 0, "state": "GG", "kind": "main"}],
                    )
                    compare_morl.compare(
                        results_path=results_path,
                        episodes=1,
                        first_seed=7,
                        simulation_time=30,
                        flow_range=(600, 600),
                        desired_horizon=7,
                        output_dir=Path(directory) / "output",
                        prefix="test",
                    )

            expected_checkpoint = (
                Path(compare_morl.__file__).resolve().parent
                / "weights/catalogue_test.pt"
            )
            self.assertEqual(load.call_args.args[0], expected_checkpoint)
            self.assertEqual(init.call_args.kwargs["env"], model_env)

    def test_comparison_excludes_only_agent_controlled_state_penalty(self):
        self.assertEqual(
            compare_morl.comparison_objectives(
                ("co2", "time_in_state_penalty", "queue_length")
            ),
            ("co2", "queue_length"),
        )

    def test_native_environment_uses_shared_objectives_and_scales(self):
        results = {
            "environment": {
                "traffic_light_id": "R3C2_C",
                "reward_components": [
                    "co2",
                    "queue_length",
                    "time_in_state_penalty",
                ],
                "reward_normalization": "normalized_average",
                "reward_scales": [10_000.0, 10.0, 1.0],
            }
        }
        with patch.object(compare_morl, "TrafficMORLEnv", return_value=StubEnv()) as make_env:
            compare_morl.build_env(
                results,
                simulation_time=900,
                flow_range=(100, 600),
                reward_components=("co2", "queue_length"),
            )

        self.assertEqual(
            make_env.call_args.kwargs["reward_components"],
            ("co2", "queue_length"),
        )
        self.assertEqual(
            make_env.call_args.kwargs["reward_scales"],
            [10_000.0, 10.0],
        )


if __name__ == "__main__":
    unittest.main()
