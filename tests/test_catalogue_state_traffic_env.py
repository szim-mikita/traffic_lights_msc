import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from gym_envs.envs import state_traffic_env
from gym_envs.envs.catalogue_state_traffic_env import (
    CatalogueStateTrafficEnv,
    CatalogueStateTrafficMOEnv,
)


CATALOGUE_PATH = (
    Path(__file__).resolve().parents[1]
    / "sumo_files"
    / "AI_agens_konyvtar"
    / "traffic_lights.json"
)


class CatalogueStateTrafficEnvTests(unittest.TestCase):
    def _start_env(self, env_type):
        with (
            patch.object(
                state_traffic_env.sumolib,
                "checkBinary",
                return_value="/usr/bin/sumo",
            ),
            patch.object(state_traffic_env.traci, "start"),
            patch.object(
                state_traffic_env.traci.inductionloop,
                "getIDList",
                return_value=[f"detector-{index}" for index in range(8)],
            ),
            patch.object(state_traffic_env.traci.edge, "getIDList", return_value=[]),
        ):
            return env_type(
                traffic_lights_file=CATALOGUE_PATH,
                traffic_light_id="R3C2_C",
                simulation_time=60,
            )

    def test_scalar_environment_uses_catalogue_states_and_transitions(self):
        env = self._start_env(CatalogueStateTrafficEnv)
        try:
            self.assertEqual(env.action_space.n, 12)
            self.assertEqual(env.observation_space["current_state"].n, 12)
            self.assertEqual(env.observation_space["occupancy"].shape, (8,))
            self.assertEqual(len(env.transitions), 132)

            env.observation_history["state"] = list(range(12))
            env.observation_history["occupancy"] = [[0.0] for _ in env.detectors]
            env.observation_history["vehicle_count"] = [[0] for _ in env.detectors]
            self.assertEqual(env._get_obs()["all_states_used"], 1)

            with (
                patch.object(
                    state_traffic_env.traci.trafficlight,
                    "setRedYellowGreenState",
                ) as set_state,
                patch.object(env, "_skip_n_steps", return_value=1.0),
                patch.object(env, "_get_obs", return_value={"all_states_used": 0}),
            ):
                _, _, _, _, _ = env.step(1)

            self.assertEqual(env.current_state, 1)
            self.assertEqual(
                set_state.call_args_list[-1].args,
                ("R3C2_C", env.states[1]),
            )
        finally:
            with patch.object(state_traffic_env.traci, "close"):
                env.close()

    def test_multi_objective_environment_uses_catalogue_action_space(self):
        env = self._start_env(CatalogueStateTrafficMOEnv)
        try:
            self.assertEqual(env.action_space.n, 12)
            self.assertEqual(env.observation_space["state"].n, 12)
            self.assertEqual(env.observation_space["mean_speed"].shape, (8,))

            with (
                patch.object(
                    state_traffic_env.traci.trafficlight,
                    "setRedYellowGreenState",
                ) as set_state,
                patch.object(
                    env,
                    "_skip_n_steps",
                    return_value=np.zeros(len(env.reward_components)),
                ),
            ):
                _, _, _, _, info = env.step(1)

            self.assertEqual(env.current_state, 1)
            self.assertEqual(info["elapsed_seconds"], 9)
            self.assertEqual(
                set_state.call_args_list[-1].args,
                ("R3C2_C", env.states[1]),
            )
        finally:
            with patch.object(state_traffic_env.traci, "close"):
                env.close()


if __name__ == "__main__":
    unittest.main()
