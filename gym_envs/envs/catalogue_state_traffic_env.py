"""State-based SUMO environments driven by a traffic-light JSON catalogue."""

from collections import OrderedDict

import numpy as np
import traci
from gymnasium import spaces

from .state_traffic_env import StateTrafficEnv, StateTrafficMOEnv
from .traffic_lights_parser import load_traffic_light_program


class _CatalogueStateMixin:
    def _load_catalogue(self, traffic_lights_file, traffic_light_id):
        program = load_traffic_light_program(
            traffic_lights_file, traffic_light_id=traffic_light_id
        )
        logic_ids = sorted(program.states)
        logic_to_action = {logic_id: action for action, logic_id in enumerate(logic_ids)}

        self.states = {
            logic_to_action[logic_id]: program.states[logic_id]
            for logic_id in logic_ids
        }
        self.transitions = {
            (logic_to_action[from_logic], logic_to_action[to_logic]): sequence
            for (from_logic, to_logic), sequence in program.transition_sequences.items()
        }
        self.traffic_light_id = program.traffic_light_id
        self.action_space = spaces.Discrete(len(self.states))

    def set_sumo_phase(self, state):
        traci.trafficlight.setRedYellowGreenState(self.traffic_light_id, state)

    def _reset_observation_history(self):
        if hasattr(self, "reward_components"):
            super()._reset_observation_history()
            return
        self.observation_history = {
            "occupancy": [[] for _ in self.detectors],
            "vehicle_count": [[] for _ in self.detectors],
            "state": [],
        }

    def _get_obs(self):
        observation = super()._get_obs()
        if "all_states_used" in observation:
            observation["all_states_used"] = int(
                len(set(self.observation_history["state"][-max(10, len(self.states)):]))
                == len(self.states)
            )
        return observation


class CatalogueStateTrafficEnv(_CatalogueStateMixin, StateTrafficEnv):
    """Scalar state environment using states and transitions from a JSON catalogue."""

    def __init__(self, traffic_lights_file, traffic_light_id=None, **kwargs):
        super().__init__(**kwargs)
        self._load_catalogue(traffic_lights_file, traffic_light_id)
        detector_count = len(self.detectors)
        self.observation_space = spaces.Dict(
            {
                "occupancy": spaces.Box(
                    low=0.0, high=100.0, shape=(detector_count,), dtype=np.float32
                ),
                "vehicle_count": spaces.Box(
                    low=0, high=7200, shape=(detector_count,), dtype=np.int32
                ),
                "current_state": spaces.Discrete(len(self.states)),
                "time_in_state": spaces.Box(
                    low=0, high=self.time_limit, shape=(), dtype=np.int32
                ),
                "all_states_used": spaces.Discrete(2),
            }
        )
        self._reset_observation_history()

    def _get_reward_components(self):
        co2_reward, timing_reward, _ = super()._get_reward_components()
        used_all_states = (
            len(set(self.observation_history["state"][-max(10, len(self.states)):]))
            == len(self.states)
        )
        coverage_reward = 0.0 if used_all_states else -10.0
        return co2_reward, timing_reward, coverage_reward

    def step(self, action):
        self._reset_observation_history()
        if not self.action_space.contains(action):
            raise ValueError(
                f"Invalid action {action!r}; expected a state in "
                f"[0, {len(self.states) - 1}]."
            )
        action = int(action)

        if action == self.current_state:
            reward = self._skip_n_steps(self.min_waiting_time, reward=True)
        else:
            transition_key = (self.current_state, action)
            if transition_key not in self.transitions:
                raise ValueError(
                    f"Invalid transition from state {self.current_state} to {action}."
                )
            reward = 0.0
            for length, intermediate_state in self.transitions[transition_key]:
                self.set_sumo_phase(intermediate_state)
                reward += self._skip_n_steps(length, observe=True)
            self.current_state = action
            self.observation_history["state"].append(self.current_state)
            self.time_in_state = 0
            self.set_sumo_phase(self.states[self.current_state])
            reward += self._skip_n_steps(self.min_waiting_time, reward=True)

        observation = self._get_obs()
        done = self.current_step >= self.time_limit
        return observation, reward, done, False, {}


class CatalogueStateTrafficMOEnv(_CatalogueStateMixin, StateTrafficMOEnv):
    """Multi-objective state environment using a traffic-light JSON catalogue."""

    def __init__(self, traffic_lights_file, traffic_light_id=None, **kwargs):
        super().__init__(**kwargs)
        self._load_catalogue(traffic_lights_file, traffic_light_id)
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
                            high=float(self.time_limit),
                            shape=(),
                            dtype=np.float32,
                        ),
                    ),
                ]
            )
        )
