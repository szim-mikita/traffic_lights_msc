from collections import OrderedDict
from pathlib import Path
import shutil
import time
from typing import Sequence

import gymnasium as gym
import numpy as np
import sumolib
import traci
from gymnasium import spaces

STATE_TRAFFIC_STATE_MAP = {
    0: "ggrr",
    1: "rggr",
    2: "rrrg",
}

STATE_TRAFFIC_FIXED_GREEN_STATES = list(STATE_TRAFFIC_STATE_MAP.values())

STATE_TRAFFIC_TRANSITION_SEQUENCES = {
    (0, 1): [
        (3, "ygrr"),
        (2, "rgrr"),
        (2, "rgur"),
    ],
    (1, 0): [
        (3, "rgyr"),
        (3, "rgrr"),
        (2, "ugrr"),
    ],
    (1, 2): [
        (3, "ryyr"),
        (5, "rrrr"),
        (2, "rrru"),
    ],
    (2, 1): [
        (3, "rrry"),
        (1, "rrrr"),
        (2, "rrur"),
        (1, "rrgr"),
        (2, "rugr"),
    ],
    (2, 0): [
        (3, "rrry"),
        (3, "rrrr"),
        (1, "urrr"),
        (1, "uurr"),
        (1, "gurr"),
    ],
    (0, 2): [
        (3, "yyrr"),
        (1, "rrrr"),
        (2, "rrru"),
    ],
}

MO_REWARD_COMPONENTS = (
    "co2",
    "throughput",
    "average_speed",
    "waiting_time",
    "queue_length",
    "time_in_state_penalty",
)

MO_REWARD_DEFAULT_SCALES = {
    "co2": 10_000.0,
    "throughput": 1.0,
    "average_speed": 13.89,
    "waiting_time": 60.0,
    "queue_length": 10.0,
    "time_in_state_penalty": 1.0,
}


class StateTrafficEnv(gym.Env):
    def __init__(
        self,
        render_mode="console",
        simulation_time=3600,
        traffic_scale=1,
        sumo_config=None,
    ):
        super(StateTrafficEnv, self).__init__()
        self.render_mode = render_mode

        # define minimum time before next action can be taken (s)
        self.min_waiting_time = 5  

        self.action_space = spaces.Discrete(3)  # 3 main traffic light states
        self.states = STATE_TRAFFIC_STATE_MAP
        self.initial_state = 0  # starting with first state 
        self.current_state = self.initial_state
        
        self.current_step = 0
        self.time_limit = simulation_time
        self.time_in_state = 0

        # define observation space
        self.observation_space = spaces.Dict({
            "occupancy": spaces.Box(low=0.0, high=100.0, shape=(4,), dtype=np.float32),
            "vehicle_count": spaces.Box(low=0, high=7200, shape=(4,), dtype=np.int32),
            "current_state": spaces.Discrete(3),
            "time_in_state": spaces.Box(low=0, high=simulation_time, shape=(), dtype=np.int32),
            "all_states_used": spaces.Discrete(2),
        })

        self.observation_history = {
            "occupancy": [[] for _ in range(4)],
            "vehicle_count": [[] for _ in range(4)],
            "state": []
        }

        binary_name = "sumo" if self.render_mode == "console" else "sumo-gui"
        path_binary = shutil.which(binary_name)
        if path_binary:
            self.sumo_binary = path_binary
        else:
            self.sumo_binary = sumolib.checkBinary(binary_name)
        if not path_binary and not Path(self.sumo_binary).is_file():
            legacy_binary_paths = (
                Path("/opt/homebrew/Cellar/sumo/1.20.0/bin") / binary_name,
                Path("/home/nikita/sumo/bin") / binary_name,
            )
            for candidate in legacy_binary_paths:
                if candidate.is_file():
                    self.sumo_binary = str(candidate)
                    break
        default_sumo_config = (
            Path(__file__).resolve().parents[2] / "sumo_files" / "SmartCity.sumocfg"
        )
        self.sumo_config = Path(sumo_config).resolve() if sumo_config else default_sumo_config
        if not self.sumo_config.is_file():
            raise FileNotFoundError(f"SUMO configuration file not found: {self.sumo_config}")

        self.sumo_cmd = [
            self.sumo_binary,
            "-c",
            str(self.sumo_config),
            "--start",
            "-e", str(simulation_time),
            "--quit-on-end",
            "--scale", str(traffic_scale)
        ]

        print("Starting SUMO with command:", ' '.join(self.sumo_cmd))

        traci.start(self.sumo_cmd)

        # get TraCI ids
        self.detectors = traci.inductionloop.getIDList()
        self.edges = traci.edge.getIDList()
    def _translate_to_sumo_phase(self, state):
        """
        Translates the given state to the corresponding SUMO traffic light state string.
        :param state: The state to translate.
        :return: The corresponding SUMO traffic light state string.
        """
        traffic_light_idx_correspondence = {
            0: 2,
            1: 2,
            2: 3,
            3: 3,
            4: 0,
            5: 0,
        }
        mapped_phase = [state[traffic_light_idx_correspondence[i]] for i in range(6)]
        return mapped_phase
    
    def set_sumo_phase(self, state):
        """
        Sets the SUMO traffic light state to the given state.
        :param state: The state to set.
        """
        tls = traci.trafficlight.getIDList()    
        traci.trafficlight.setRedYellowGreenState(tls[3], self._translate_to_sumo_phase(state))


    def _get_obs(self):
        """
        Returns the observation for the environment.
        :return: The observation for the environment. Occupancy and vehicle count.
        """

        occupancy = []
        vehicle_count = []
        for idx in range(len(self.detectors)):
            try:
                occupancy.append(np.mean(self.observation_history["occupancy"][idx])) # possibly change to last value
            except Exception:
                occupancy.append(0.0)

            try:
                vehicle_count.append(np.mean(self.observation_history["vehicle_count"][idx])) # possibly change to last value
            except Exception:
                vehicle_count.append(0)

        obs = {
            "occupancy": np.array(occupancy, dtype=np.float32),
            "vehicle_count": np.array(vehicle_count, dtype=np.int32),
            "current_state": self.current_state,
            "time_in_state": self.time_in_state,
            "all_states_used": int(len(set(self.observation_history["state"][-(min(10, len(self.observation_history["state"]))):])) == 3)
        }

        return obs
    

    def _get_detector_data(self):
        """
        Returns the raw data from the detectors.
        :return: The raw data from the detectors.
        """

        for idx in range(len(self.detectors)):
            try:
                self.observation_history["occupancy"][idx].append(traci.inductionloop.getLastStepOccupancy(self.detectors[idx]))
            except Exception:
                self.observation_history["occupancy"][idx].append(0.0)

            try:
                self.observation_history["vehicle_count"][idx].append(traci.inductionloop.getLastStepVehicleNumber(self.detectors[idx]))
            except Exception:
                self.observation_history["vehicle_count"][idx].append(0)

        return self.observation_history
    

    def _reset_observation_history(self):
        """
        Resets the observation history.
        """
        self.observation_history["occupancy"] = [[] for _ in range(4)]
        self.observation_history["vehicle_count"] = [[] for _ in range(4)]


    def _get_reward_components(self):
        """
        Returns the components that make up the scalar reward.
        """
        co2_reward = 0.0
        for edge in self.edges:
            try:
                co2_reward -= traci.edge.getCO2Emission(edge) / 10000
            except Exception:
                pass

        timing_reward = 1 - np.exp(0.01 * self.time_in_state)
        coverage_reward = (
            -10.0
            if len(
                set(
                    self.observation_history["state"][
                        -(min(10, len(self.observation_history["state"]))):
                    ]
                )
            )
            < 3
            else 0.0
        )
        return co2_reward, timing_reward, coverage_reward

    def _get_reward(self):
        """
        Returns the reward for the current step.
        :return: The reward for the current step.
        """
        return sum(self._get_reward_components())


    def _get_info(self):
        """
        Returns the information for the environment.
        :return: The information for the environment.
        """
        pass


    def _skip_n_steps(self, n, observe=True, reward=False):
        """
        Steps the simulation n times, collecting detector data.
        :param n: The number of steps (seconds) to skip.
        """
        reward_total = 0.0
        for _ in range(n):
            traci.simulationStep()
            if observe:
                self._get_detector_data()
            if reward:
                reward_total += self._get_reward()
            self.current_step += 1
            self.time_in_state += 1
        return reward_total / n


    def reset(self, seed=None, options=None):
        """
        Resets the environment.
        :param seed: The seed for the environment.
        :return: The initial observation and info.
        """
        self._reset_observation_history()

        traci.load(self.sumo_cmd[1:])
        time.sleep(0.001)

        self.current_step = 0
        self.current_state = self.initial_state
        self.observation_history["state"] = [self.current_state]
        self.time_in_state = 0

        # set initial traffic light state
        self.set_sumo_phase(self.states[self.current_state])

        # step once to get initial observation
        self._skip_n_steps(1, observe=True)

        return self._get_obs(), {}
    

    def step(self, action):
        """
        Steps the environment by taking the given action.
        
        :param self: The environment instance.
        :param action: The action to take (0, 1, or 2).
        """

        self._reset_observation_history()

        if action not in [0, 1, 2]:
            raise ValueError("Invalid action. Action must be 0, 1, or 2.")
        
        # stay in the same state
        if action == self.current_state:
            reward = self._skip_n_steps(self.min_waiting_time, reward=True)
        else: 
            transition_key = (self.current_state, action)
            if transition_key in self.transitions:
                reward = 0.0
                for length, intermediate_state in self.transitions[transition_key]:
                    # set intermediate state
                    self.set_sumo_phase(intermediate_state)
                    # step simulation
                    reward += self._skip_n_steps(length, observe=True) # do not collect data during transitions (maybe)
                self.current_state = action
                self.observation_history["state"].append(self.current_state)
                self.time_in_state = 0
                self.set_sumo_phase(self.states[self.current_state])
                reward += self._skip_n_steps(self.min_waiting_time, reward=True)  # step mean waiting time collect data after transition
            else: # shouldn't happen if nothing goes extremely wrong
                raise ValueError(f"Invalid transition from state {self.current_state} to {action}.") 
            
        # get observation
        observation = self._get_obs()

        # check if done
        done = self.current_step >= self.time_limit
        # print(f"        [STEP] step={self.current_step} action={action} reward={reward:.3f} done={done}")

        return observation, reward, done, False, {}  # info is empty dict

    def close(self):
        # Always clean up
        traci.close()


class StateTrafficMOEnv(StateTrafficEnv):
    """State-based traffic control with configurable, induction-loop MORL signals."""

    def __init__(
        self,
        render_mode="console",
        simulation_time=3600,
        traffic_scale=1,
        sumo_config=None,
        reward_components: Sequence[str] = ("co2", "time_in_state_penalty"),
        reward_normalization: str = "normalized_average",
        reward_scales: Sequence[float] | None = None,
        state_penalty_threshold: float = 45.0,
        state_penalty_growth: float = 0.1,
    ):
        self.reward_components = tuple(reward_components)
        if not self.reward_components:
            raise ValueError("At least one reward component must be selected.")
        if len(set(self.reward_components)) != len(self.reward_components):
            raise ValueError("Reward components must not contain duplicates.")
        unknown_components = set(self.reward_components) - set(MO_REWARD_COMPONENTS)
        if unknown_components:
            raise ValueError(
                f"Unknown reward components: {sorted(unknown_components)}. "
                f"Choose from: {', '.join(MO_REWARD_COMPONENTS)}."
            )
        if reward_normalization not in ("sum", "average", "normalized_average"):
            raise ValueError(
                "reward_normalization must be 'sum', 'average', or 'normalized_average'."
            )
        self.reward_normalization = reward_normalization
        if reward_scales is None:
            reward_scales = [
                MO_REWARD_DEFAULT_SCALES[name] for name in self.reward_components
            ]
        self.reward_scales = np.asarray(reward_scales, dtype=np.float32)
        if self.reward_scales.shape != (len(self.reward_components),):
            raise ValueError(
                "reward_scales must contain exactly one value per selected reward component."
            )
        if not np.all(np.isfinite(self.reward_scales)) or np.any(self.reward_scales <= 0):
            raise ValueError("reward_scales must contain positive finite values.")

        if not np.isfinite(state_penalty_threshold) or state_penalty_threshold < 0:
            raise ValueError("state_penalty_threshold must be finite and non-negative.")
        if not np.isfinite(state_penalty_growth) or state_penalty_growth <= 0:
            raise ValueError("state_penalty_growth must be finite and positive.")
        self.state_penalty_threshold = state_penalty_threshold
        self.state_penalty_growth = state_penalty_growth

        super().__init__(
            render_mode=render_mode,
            simulation_time=simulation_time,
            traffic_scale=traffic_scale,
            sumo_config=sumo_config,
        )

        detector_count = len(self.detectors)
        self.observation_space = spaces.Dict(
            OrderedDict(
                [
                    (
                        "mean_speed",
                        spaces.Box(
                            low=0.0,
                            high=np.inf,
                            shape=(detector_count,),
                            dtype=np.float32,
                        ),
                    ),
                    (
                        "occupancy",
                        spaces.Box(
                            low=0.0,
                            high=100.0,
                            shape=(detector_count,),
                            dtype=np.float32,
                        ),
                    ),
                    (
                        "vehicles",
                        spaces.Box(
                            low=0.0,
                            high=np.inf,
                            shape=(detector_count,),
                            dtype=np.float32,
                        ),
                    ),
                    ("state", spaces.Discrete(len(self.states))),
                    (
                        "time_in_state",
                        spaces.Box(
                            low=0.0,
                            high=float(simulation_time),
                            shape=(),
                            dtype=np.float32,
                        ),
                    ),
                ]
            )
        )
        self._reset_observation_history()
        self.reward_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.reward_components),),
            dtype=np.float32,
        )

    def _reset_observation_history(self):
        self.observation_history = {
            "mean_speed": [[] for _ in self.detectors],
            "occupancy": [[] for _ in self.detectors],
            "vehicles": [[] for _ in self.detectors],
            "state": [],
        }

    def _get_detector_data(self):
        for idx, detector in enumerate(self.detectors):
            speed = traci.inductionloop.getLastStepMeanSpeed(detector)
            occupancy = traci.inductionloop.getLastStepOccupancy(detector)
            vehicles = traci.inductionloop.getLastStepVehicleNumber(detector)
            self.observation_history["mean_speed"][idx].append(max(0.0, speed))
            self.observation_history["occupancy"][idx].append(occupancy)
            self.observation_history["vehicles"][idx].append(vehicles)
        return self.observation_history

    def _get_obs(self):
        observation = {
            key: np.asarray(
                [np.mean(values) if values else 0.0 for values in self.observation_history[key]],
                dtype=np.float32,
            )
            for key in ("mean_speed", "occupancy", "vehicles")
        }
        observation["state"] = self.current_state
        observation["time_in_state"] = np.float32(self.time_in_state)
        return observation

    def _get_available_metrics(self):
        selected = set(self.reward_components)
        metrics = {}
        if "co2" in selected:
            metrics["co2"] = -sum(
                traci.edge.getCO2Emission(edge) for edge in self.edges
            )
        if "throughput" in selected or "average_speed" in selected:
            counts = np.asarray(
                [
                    traci.inductionloop.getLastStepVehicleNumber(detector)
                    for detector in self.detectors
                ],
                dtype=np.float64,
            )
            metrics["throughput"] = float(counts.sum())
            if "average_speed" in selected:
                speeds = np.asarray(
                    [
                        max(0.0, traci.inductionloop.getLastStepMeanSpeed(detector))
                        for detector in self.detectors
                    ],
                    dtype=np.float64,
                )
                metrics["average_speed"] = (
                    float(np.average(speeds, weights=counts))
                    if counts.sum() > 0
                    else 0.0
                )
        if "waiting_time" in selected or "queue_length" in selected:
            lanes = {
                traci.inductionloop.getLaneID(detector) for detector in self.detectors
            }
            vehicle_ids = [
                vehicle
                for lane in lanes
                for vehicle in traci.lane.getLastStepVehicleIDs(lane)
            ]
            if "waiting_time" in selected:
                waiting_times = [
                    traci.vehicle.getWaitingTime(vehicle) for vehicle in vehicle_ids
                ]
                metrics["waiting_time"] = -(
                    float(np.mean(waiting_times)) if waiting_times else 0.0
                )
            if "queue_length" in selected:
                metrics["queue_length"] = -float(
                    sum(
                        traci.vehicle.getSpeed(vehicle) < 0.1
                        for vehicle in vehicle_ids
                    )
                )
        if "time_in_state_penalty" in selected:
            metrics["time_in_state_penalty"] = -float(
                np.expm1(
                    self.state_penalty_growth
                    * max(0.0, self.time_in_state - self.state_penalty_threshold)
                )
            )
        return metrics

    def _get_reward(self):
        metrics = self._get_available_metrics()
        return np.asarray(
            [metrics[name] for name in self.reward_components], dtype=np.float32
        )

    def step(self, action):
        self._reset_observation_history()
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action!r}; expected a state in [0, 2].")

        total_reward = np.zeros(len(self.reward_components), dtype=np.float64)
        elapsed = 0

        if action != self.current_state:
            transition_key = (self.current_state, int(action))
            if transition_key not in self.transitions:
                raise ValueError(
                    f"Invalid transition from state {self.current_state} to {action}."
                )
            for length, intermediate_state in self.transitions[transition_key]:
                self.set_sumo_phase(intermediate_state)
                segment_reward = self._skip_n_steps(
                    length, observe=True, reward=True
                )
                total_reward += segment_reward * length
                elapsed += length

            self.current_state = int(action)
            self.observation_history["state"].append(self.current_state)
            self.time_in_state = 0
            self.set_sumo_phase(self.states[self.current_state])

        segment_reward = self._skip_n_steps(
            self.min_waiting_time, observe=True, reward=True
        )
        total_reward += segment_reward * self.min_waiting_time
        elapsed += self.min_waiting_time

        average_reward = total_reward / elapsed
        if self.reward_normalization == "sum":
            reward = total_reward
        elif self.reward_normalization == "normalized_average":
            reward = average_reward / self.reward_scales
        else:
            reward = average_reward
        reward = np.asarray(reward, dtype=np.float32)

        observation = self._get_obs()
        terminated = self.current_step >= self.time_limit
        info = {
            "elapsed_seconds": elapsed,
            "reward_components": dict(zip(self.reward_components, reward.tolist())),
        }
        return observation, reward, terminated, False, info