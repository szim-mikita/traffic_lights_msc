import unittest
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
