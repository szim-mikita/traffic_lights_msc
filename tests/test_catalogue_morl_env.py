import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch

import gymnasium as gym
import mo_gymnasium
import numpy as np
from morl_baselines.multi_policy.pcn.pcn import PCN

from gym_envs.envs import catalogue_morl_env


SUMO_ASSETS = Path(__file__).resolve().parents[1] / "sumo_files" / "AI_agens_konyvtar"


class CatalogueTrafficMOEnvTests(unittest.TestCase):
    def _make_env(self, **kwargs):
        simulation_time = kwargs.pop("simulation_time", 60)
        with patch.object(catalogue_morl_env.sumolib, "checkBinary", return_value="sumo"):
            return catalogue_morl_env.CatalogueTrafficMOEnv(
                net_file=SUMO_ASSETS / "mega_catalogue_final.net.xml",
                logic_json_file=SUMO_ASSETS / "traffic_lights.json",
                detector_file=SUMO_ASSETS / "mega_catalogue_final.detectors.add.xml",
                traffic_light_id="R3C2_C",
                simulation_time=simulation_time,
                **kwargs,
            )

    def test_catalogue_builds_discrete_morl_agent_and_random_routes(self):
        env = self._make_env()
        try:
            self.assertEqual(env.action_space.n, 12)
            self.assertEqual(env.observation_space["occupancy"].shape, (8,))
            self.assertEqual(env.reward_space.shape, (2,))
            self.assertEqual(env.agent.jid, "R3C2_C")

            route_file = env.generate_random_traffic(
                flow_range=(600, 600), duration=60, seed=7
            )
            trips = ET.parse(route_file).getroot().findall("trip")
            self.assertGreater(len(trips), 0)
            self.assertTrue(all(trip.get("departLane") is not None for trip in trips))

            wrapped_env = mo_gymnasium.wrappers.MORecordEpisodeStatistics(
                gym.wrappers.FlattenObservation(env)
            )
            agent = PCN(
                env=wrapped_env,
                scaling_factor=np.ones(3, dtype=np.float32),
                log=False,
                seed=7,
            )
            self.assertEqual(agent.action_dim, 12)
            self.assertEqual(agent.reward_dim, 2)
            self.assertEqual(wrapped_env.observation_space.shape, (26,))
        finally:
            env.close()

    def test_reset_and_step_return_gymnasium_morl_values(self):
        env = self._make_env(
            reward_components=("co2", "time_in_state_penalty"),
            reward_normalization="average",
        )
        with (
            patch.object(catalogue_morl_env.traci, "start"),
            patch.object(catalogue_morl_env.traci, "close"),
            patch.object(
                catalogue_morl_env.traci.trafficlight,
                "setRedYellowGreenState",
            ),
            patch.object(catalogue_morl_env.traci, "simulationStep"),
            patch.object(
                catalogue_morl_env.traci.simulation,
                "getMinExpectedNumber",
                return_value=1,
            ),
        ):
            try:
                observation, reset_info = env.reset(seed=3)
                self.assertEqual(reset_info["traffic_light_id"], "R3C2_C")
                self.assertTrue(env.observation_space.contains(observation))

                with patch.object(
                    env.agent,
                    "collect_measurements",
                    return_value={
                        "co2": -100.0,
                        "throughput": 1.0,
                        "average_speed": 5.0,
                        "waiting_time": -2.0,
                        "queue_length": -1.0,
                    },
                ):
                    observation, reward, terminated, truncated, info = env.step(1)

                self.assertTrue(env.observation_space.contains(observation))
                self.assertEqual(reward.shape, (2,))
                self.assertTrue(np.all(np.isfinite(reward)))
                self.assertFalse(terminated)
                self.assertFalse(truncated)
                self.assertEqual(
                    info["elapsed_seconds"],
                    sum(duration for duration, _ in env.agent.transitions[(0, 1)]) + 5,
                )
                self.assertEqual(info["measured_seconds"], 5)
                self.assertEqual(env.agent.current_logic_idx, 1)
            finally:
                env.close()

    def test_state_penalty_stays_finite_for_long_phases(self):
        env = self._make_env(
            reward_components=("time_in_state_penalty",),
            reward_normalization="average",
        )
        try:
            env.agent.phase_timer = 100_000

            reward = env._reward_for_metrics(
                {
                    "co2": 0.0,
                    "throughput": 0.0,
                    "average_speed": 0.0,
                    "waiting_time": 0.0,
                    "queue_length": 0.0,
                }
            )

            self.assertTrue(np.all(np.isfinite(reward)))
            self.assertEqual(reward.tolist(), [-catalogue_morl_env.MAX_STATE_PENALTY])
        finally:
            env.close()

    def test_step_does_not_exceed_episode_time_limit(self):
        env = self._make_env(simulation_time=8)
        with (
            patch.object(catalogue_morl_env.traci, "start"),
            patch.object(catalogue_morl_env.traci, "close"),
            patch.object(
                catalogue_morl_env.traci.trafficlight,
                "setRedYellowGreenState",
            ),
            patch.object(catalogue_morl_env.traci, "simulationStep"),
            patch.object(
                catalogue_morl_env.traci.simulation,
                "getMinExpectedNumber",
                return_value=1,
            ),
            patch.object(
                env.agent,
                "collect_measurements",
                return_value={
                    "co2": -100.0,
                    "throughput": 1.0,
                    "average_speed": 5.0,
                    "waiting_time": -2.0,
                    "queue_length": -1.0,
                },
            ),
        ):
            try:
                env.reset(seed=5)
                _, _, terminated, _, _ = env.step(0)
                _, _, terminated, _, info = env.step(0)

                self.assertTrue(terminated)
                self.assertEqual(info["elapsed_seconds"], 3)
                self.assertEqual(env.current_step, 8)
            finally:
                env.close()

    def test_terminal_transition_still_returns_measured_reward(self):
        env = self._make_env(simulation_time=2)
        with (
            patch.object(catalogue_morl_env.traci, "start"),
            patch.object(catalogue_morl_env.traci, "close"),
            patch.object(
                catalogue_morl_env.traci.trafficlight,
                "setRedYellowGreenState",
            ),
            patch.object(catalogue_morl_env.traci, "simulationStep"),
            patch.object(
                catalogue_morl_env.traci.simulation,
                "getMinExpectedNumber",
                return_value=1,
            ),
            patch.object(
                env.agent,
                "collect_measurements",
                return_value={
                    "co2": -100.0,
                    "throughput": 1.0,
                    "average_speed": 5.0,
                    "waiting_time": -2.0,
                    "queue_length": -1.0,
                },
            ),
        ):
            try:
                env.reset(seed=6)
                _, reward, terminated, _, info = env.step(1)

                self.assertTrue(terminated)
                self.assertEqual(info["elapsed_seconds"], 2)
                self.assertEqual(info["measured_seconds"], 2)
                self.assertTrue(np.all(np.isfinite(reward)))
                self.assertTrue(env.agent.is_transitioning)
            finally:
                env.close()

    def test_fixed_time_program_uses_native_signal_without_agent_overrides(self):
        env = self._make_env(reward_components=("co2", "queue_length"))
        with (
            patch.object(catalogue_morl_env.traci, "start"),
            patch.object(catalogue_morl_env.traci, "close"),
            patch.object(
                catalogue_morl_env.traci.trafficlight,
                "setRedYellowGreenState",
            ) as set_state,
            patch.object(
                catalogue_morl_env.traci.trafficlight, "setProgram"
            ) as set_program,
            patch.object(
                catalogue_morl_env.traci.trafficlight,
                "getProgram",
                return_value="0",
            ),
            patch.object(catalogue_morl_env.traci, "simulationStep"),
            patch.object(
                catalogue_morl_env.traci.simulation,
                "getMinExpectedNumber",
                return_value=1,
            ),
            patch.object(
                env.agent,
                "collect_measurements",
                return_value={
                    "co2": -100.0,
                    "throughput": 1.0,
                    "average_speed": 5.0,
                    "waiting_time": -2.0,
                    "queue_length": -1.0,
                },
            ),
        ):
            try:
                env.reset(seed=9)
                env.activate_fixed_time_program()
                self.assertEqual(
                    env._fixed_program_phase_kinds,
                    {
                        0: "main",
                        1: "transition",
                        2: "main",
                        3: "transition",
                        4: "transition",
                    },
                )
                _, reward, terminated, truncated, info = env.step_fixed_time()

                set_program.assert_called_once_with("R3C2_C", "0")
                self.assertEqual(set_state.call_count, 1)
                self.assertEqual(reward.shape, (2,))
                self.assertFalse(terminated)
                self.assertFalse(truncated)
                self.assertEqual(info["signal_program"], "0")
                self.assertEqual(info["elapsed_seconds"], 5)
                self.assertEqual(env.current_step, 5)
            finally:
                env.close()

    def test_signal_trace_records_actual_state_each_simulation_second(self):
        env = self._make_env(reward_components=("co2", "queue_length"))
        env.record_signal_trace = True
        main_state = env.agent.states[0]
        transition_state = env.agent.transitions[(0, 1)][0][1]
        signal_for_time = lambda: (
            main_state if env.current_step == 0 or env.current_step >= 5
            else transition_state
        )
        with (
            patch.object(catalogue_morl_env.traci, "start"),
            patch.object(catalogue_morl_env.traci, "close"),
            patch.object(
                catalogue_morl_env.traci.trafficlight,
                "setRedYellowGreenState",
            ),
            patch.object(
                catalogue_morl_env.traci.trafficlight, "setProgram"
            ),
            patch.object(
                catalogue_morl_env.traci.trafficlight,
                "getRedYellowGreenState",
                side_effect=lambda _jid: signal_for_time(),
            ),
            patch.object(
                catalogue_morl_env.traci.trafficlight,
                "getProgram",
                return_value="0",
            ),
            patch.object(catalogue_morl_env.traci, "simulationStep"),
            patch.object(
                catalogue_morl_env.traci.simulation,
                "getMinExpectedNumber",
                return_value=1,
            ),
            patch.object(
                env.agent,
                "collect_measurements",
                return_value={
                    "co2": -100.0,
                    "throughput": 1.0,
                    "average_speed": 5.0,
                    "waiting_time": -2.0,
                    "queue_length": -1.0,
                },
            ),
            patch.object(
                catalogue_morl_env.traci.lane,
                "getCO2Emission",
                return_value=0.0,
            ),
            patch.object(
                catalogue_morl_env.traci.lane,
                "getLastStepHaltingNumber",
                return_value=0,
            ),
            patch.object(
                catalogue_morl_env.traci.lane,
                "getLastStepVehicleIDs",
                return_value=[],
            ),
            patch.object(
                catalogue_morl_env.traci.inductionloop,
                "getLastStepVehicleNumber",
                return_value=0,
            ),
            patch.object(
                catalogue_morl_env.traci.inductionloop,
                "getLastStepMeanSpeed",
                return_value=0.0,
            ),
            patch.object(
                catalogue_morl_env.traci.inductionloop,
                "getLastStepOccupancy",
                return_value=0.0,
            ),
        ):
            try:
                env.reset(seed=10)
                env.step(1)

                self.assertEqual(
                    [(point["time"], point["kind"]) for point in env.signal_trace],
                    [
                        (0, "main"),
                        (1, "transition"),
                        (2, "transition"),
                        (3, "transition"),
                        (4, "transition"),
                        (5, "main"),
                        (6, "main"),
                        (7, "main"),
                        (8, "main"),
                        (9, "main"),
                    ],
                )
            finally:
                env.close()

    def test_gui_mode_starts_sumo_without_manual_play(self):
        env = self._make_env(render_mode="gui")
        try:
            args = env._sumo_args(Path("/tmp/routes.rou.xml"), 60, 7)
            self.assertIn("--start", args)
            self.assertIn(str(env._runtime_detector_file), args)
            runtime_detector = ET.parse(env._runtime_detector_file).getroot()
            self.assertTrue(
                all(
                    Path(detector.get("file")).parent
                    == Path(env._temporary_sumo_dir.name)
                    for detector in runtime_detector
                    if detector.tag == "inductionLoop"
                )
            )

            headless_env = self._make_env(render_mode="console")
            try:
                headless_args = headless_env._sumo_args(
                    Path("/tmp/routes.rou.xml"), 60, 7
                )
                self.assertNotIn("--start", headless_args)
            finally:
                headless_env.close()
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
