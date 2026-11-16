import unittest
from pathlib import Path
from unittest.mock import patch

import gymnasium as gym
import mo_gymnasium
import numpy as np
from morl_baselines.multi_policy.pcn.pcn import PCN
from gymnasium import spaces

from gym_envs.envs import state_traffic_env
from gym_envs.envs.state_traffic_env import StateTrafficEnv, StateTrafficMOEnv


class StateTrafficMOEnvTests(unittest.TestCase):
    def _reward_env(
        self,
        reward_components=("co2",),
        reward_normalization="average",
        reward_scales=None,
        env_type=StateTrafficMOEnv,
    ):
        env = object.__new__(env_type)
        env.edges = ["edge-0"]
        env.detectors = ["detector-0", "detector-1"]
        env.current_state = 0
        env.action_space = spaces.Discrete(3)
        env.current_step = 0
        env.time_limit = 100
        env.time_in_state = 45
        env.min_waiting_time = 5
        env.transitions = state_traffic_env.STATE_TRAFFIC_TRANSITION_SEQUENCES
        env.states = state_traffic_env.STATE_TRAFFIC_STATE_MAP
        env.observation_history = {"state": [0]}
        env.reward_components = tuple(reward_components)
        env.reward_normalization = reward_normalization
        env.reward_scales = np.asarray(
            reward_scales or [1.0] * len(reward_components), dtype=np.float32
        )
        env.state_penalty_threshold = 45.0
        env.state_penalty_growth = 0.1
        return env

    def test_available_reward_components_use_induction_loop_metrics(self):
        env = self._reward_env(reward_components=state_traffic_env.MO_REWARD_COMPONENTS)
        with (
            patch.object(state_traffic_env.traci.edge, "getCO2Emission", return_value=12000),
            patch.object(
                state_traffic_env.traci.inductionloop,
                "getLaneID",
                side_effect=["lane-0", "lane-1"],
            ),
            patch.object(
                state_traffic_env.traci.lane,
                "getLastStepVehicleIDs",
                side_effect=[["v0", "v1"], ["v2"]],
            ),
            patch.object(
                state_traffic_env.traci.vehicle,
                "getWaitingTime",
                side_effect=[2.0, 4.0, 6.0],
            ),
            patch.object(
                state_traffic_env.traci.vehicle,
                "getSpeed",
                side_effect=[0.0, 5.0, 0.0],
            ),
            patch.object(
                state_traffic_env.traci.inductionloop,
                "getLastStepVehicleNumber",
                side_effect=[2, 1],
            ),
            patch.object(
                state_traffic_env.traci.inductionloop,
                "getLastStepMeanSpeed",
                side_effect=[10.0, 16.0],
            ),
        ):
            metrics = env._get_available_metrics()
        with patch.object(env, "_get_available_metrics", return_value=metrics):
            reward = env._get_reward()

        self.assertEqual(
            metrics,
            {
                "co2": -12000,
                "throughput": 3.0,
                "average_speed": 12.0,
                "waiting_time": -4.0,
                "queue_length": -2.0,
                "time_in_state_penalty": 0.0,
            },
        )
        np.testing.assert_array_equal(
            reward,
            [-12000.0, 3.0, 12.0, -4.0, -2.0, 0.0],
        )

    def test_scalar_environment_reward_formula_is_preserved(self):
        env = self._reward_env(env_type=StateTrafficEnv)
        with patch.object(
            state_traffic_env.traci.edge, "getCO2Emission", return_value=2000
        ):
            reward = env._get_reward()

        self.assertAlmostEqual(reward, -0.2 + 1 - np.exp(0.45) - 10)

    def test_post_threshold_penalty_is_flat_until_45_seconds_then_exponential(self):
        env = self._reward_env(
            reward_components=("time_in_state_penalty",),
            reward_normalization="average",
        )
        with (
            patch.object(state_traffic_env.traci.edge, "getCO2Emission", return_value=0),
            patch.object(
                state_traffic_env.traci.inductionloop,
                "getLaneID",
                return_value="lane",
            ),
            patch.object(
                state_traffic_env.traci.lane,
                "getLastStepVehicleIDs",
                return_value=[],
            ),
            patch.object(
                state_traffic_env.traci.inductionloop,
                "getLastStepVehicleNumber",
                return_value=0,
            ),
            patch.object(
                state_traffic_env.traci.inductionloop,
                "getLastStepMeanSpeed",
                return_value=-1.0,
            ),
        ):
            self.assertEqual(float(env._get_reward()[0]), 0.0)
            env.time_in_state = 44
            self.assertEqual(float(env._get_reward()[0]), 0.0)
            env.time_in_state = 60
            self.assertAlmostEqual(
                float(env._get_reward()[0]), -np.expm1(1.5), delta=1e-6
            )

    def test_time_aggregation_modes_and_component_scales(self):
        expected_average = np.asarray([2.0, -4.0], dtype=np.float64)
        for mode, expected in (
            ("sum", [10.0, -20.0]),
            ("average", [2.0, -4.0]),
            ("normalized_average", [1.0, -2.0]),
        ):
            env = self._reward_env(
                reward_components=("throughput", "queue_length"),
                reward_normalization=mode,
                reward_scales=[2.0, 2.0],
            )
            env._skip_n_steps = lambda n, observe=True, reward=False: expected_average
            observation = {
                "mean_speed": np.zeros(2, dtype=np.float32),
                "occupancy": np.zeros(2, dtype=np.float32),
                "vehicles": np.zeros(2, dtype=np.float32),
                "state": 0,
                "time_in_state": np.float32(45),
            }
            env._get_obs = lambda: observation
            with patch.object(state_traffic_env.traci, "trafficlight"):
                actual, reward_vector, terminated, truncated, info = env.step(0)

            np.testing.assert_allclose(reward_vector, expected)
            self.assertIs(actual, observation)
            self.assertFalse(terminated)
            self.assertFalse(truncated)
            self.assertEqual(info["elapsed_seconds"], 5)

    def test_invalid_reward_configuration_fails_early(self):
        with (
            patch.object(state_traffic_env.sumolib, "checkBinary", return_value="/usr/bin/sumo"),
            patch.object(state_traffic_env.traci, "start"),
            patch.object(
                state_traffic_env.traci.inductionloop,
                "getIDList",
                return_value=["detector-0"],
            ),
            patch.object(state_traffic_env.traci.edge, "getIDList", return_value=[]),
        ):
            with self.assertRaisesRegex(ValueError, "Unknown reward components"):
                StateTrafficMOEnv(reward_components=("unknown",))

    def test_observations_are_per_detector_and_states_are_scalar_actions(self):
        with (
            patch.object(state_traffic_env.sumolib, "checkBinary", return_value="/usr/bin/sumo"),
            patch.object(state_traffic_env.traci, "start"),
            patch.object(
                state_traffic_env.traci.inductionloop,
                "getIDList",
                return_value=["detector-0", "detector-1"],
            ),
            patch.object(state_traffic_env.traci.edge, "getIDList", return_value=[]),
        ):
            base_env = StateTrafficMOEnv(simulation_time=60)

        try:
            self.assertEqual(base_env.action_space.n, 3)
            self.assertEqual(base_env.observation_space["mean_speed"].shape, (2,))
            self.assertEqual(base_env.observation_space["occupancy"].shape, (2,))
            self.assertEqual(base_env.observation_space["vehicles"].shape, (2,))
            self.assertEqual(base_env.observation_space["state"].n, 3)
            self.assertEqual(
                list(base_env.observation_space.spaces),
                ["mean_speed", "occupancy", "vehicles", "state", "time_in_state"],
            )
            env = mo_gymnasium.wrappers.MORecordEpisodeStatistics(
                gym.wrappers.FlattenObservation(base_env)
            )
            agent = PCN(
                env=env,
                scaling_factor=np.ones(3, dtype=np.float32),
                log=False,
                seed=1,
            )
            self.assertEqual(env.observation_space.shape, (10,))
            self.assertEqual(agent.action_dim, 3)
            self.assertEqual(agent.reward_dim, 2)
            self.assertTrue(Path(base_env.sumo_cmd[2]).is_file())
        finally:
            with patch.object(state_traffic_env.traci, "close"):
                base_env.close()

    def test_detector_observation_reads_loop_speed_occupancy_and_count(self):
        env = self._reward_env()
        env._reset_observation_history()
        with (
            patch.object(
                state_traffic_env.traci.inductionloop,
                "getLastStepMeanSpeed",
                side_effect=[8.0, 12.0],
            ),
            patch.object(
                state_traffic_env.traci.inductionloop,
                "getLastStepOccupancy",
                side_effect=[20.0, 40.0],
            ),
            patch.object(
                state_traffic_env.traci.inductionloop,
                "getLastStepVehicleNumber",
                side_effect=[1, 3],
            ),
        ):
            env._get_detector_data()
        env.time_in_state = 15
        obs = env._get_obs()
        np.testing.assert_array_equal(obs["mean_speed"], [8.0, 12.0])
        np.testing.assert_array_equal(obs["occupancy"], [20.0, 40.0])
        np.testing.assert_array_equal(obs["vehicles"], [1.0, 3.0])
        self.assertEqual(obs["state"], 0)
        self.assertEqual(obs["time_in_state"], 15.0)


if __name__ == "__main__":
    unittest.main()
