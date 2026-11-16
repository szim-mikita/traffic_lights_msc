"""MORL traffic control using catalogue SUMO assets and generated traffic."""

from __future__ import annotations

import json
import os
import random
import tempfile
import xml.etree.ElementTree as ET
from collections import OrderedDict
from pathlib import Path
from typing import Sequence

import gymnasium as gym
import numpy as np
import sumolib
import traci
from gymnasium import spaces

from .traffic_lights_parser import load_traffic_light_program

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

MAX_STATE_PENALTY = 100.0


class TrafficAgent:
    """Per-junction controller and measurement state, adapted to catalogue phases."""

    def __init__(
        self,
        junction_id: str,
        states: dict[int, str],
        transitions: dict[tuple[int, int], list[tuple[int, str]]],
        detectors: Sequence[str],
        incoming_lanes: Sequence[str],
        measure_during_transition: bool = False,
    ):
        self.jid = junction_id
        self.states = states
        self.transitions = transitions
        self.detectors = list(detectors)
        self.incoming_lanes = list(incoming_lanes)
        self.measure_during_transition = measure_during_transition
        self.action_space = spaces.Discrete(len(states))
        detector_count = max(1, len(self.detectors))
        self.observation_space = spaces.Dict(
            OrderedDict(
                [
                    (
                        "phase",
                        spaces.Box(
                            low=0,
                            high=len(states) - 1,
                            shape=(1,),
                            dtype=np.int32,
                        ),
                    ),
                    (
                        "phase_timer",
                        spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                    ),
                    (
                        "occupancy",
                        spaces.Box(
                            low=0.0,
                            high=1.0,
                            shape=(detector_count,),
                            dtype=np.float32,
                        ),
                    ),
                    (
                        "flow",
                        spaces.Box(
                            low=0.0,
                            high=np.inf,
                            shape=(detector_count,),
                            dtype=np.float32,
                        ),
                    ),
                    (
                        "speed",
                        spaces.Box(
                            low=0.0,
                            high=1.0,
                            shape=(detector_count,),
                            dtype=np.float32,
                        ),
                    ),
                ]
            )
        )
        self.reset_logic()
        self.reset_step_metrics()

    def reset_logic(self) -> None:
        self.current_logic_idx = min(self.states)
        self.target_logic_idx = self.current_logic_idx
        self.transition_queue: list[tuple[int, str]] = []
        self.transition_cursor = 0
        self.transition_steps_remaining = 0
        self.is_transitioning = False
        self.phase_timer = 0

    def reset_step_metrics(self) -> None:
        self.steps_measured = 0
        self.detector_flow = np.zeros(len(self.detectors), dtype=np.float64)
        self.detector_occupancy = np.zeros(len(self.detectors), dtype=np.float64)
        self.detector_speed = np.zeros(len(self.detectors), dtype=np.float64)
        self._last_metrics = {
            "co2": 0.0,
            "throughput": 0.0,
            "average_speed": 0.0,
            "waiting_time": 0.0,
            "queue_length": 0.0,
        }

    def set_target_phase(self, target_idx: int) -> None:
        if target_idx == self.current_logic_idx:
            self.target_logic_idx = target_idx
            return
        key = (self.current_logic_idx, target_idx)
        if key not in self.transitions:
            raise ValueError(
                f"Invalid transition from state {self.current_logic_idx} to {target_idx}."
            )
        self.target_logic_idx = target_idx
        self.phase_timer = 0
        self.transition_queue = self.transitions[key]
        self.transition_cursor = 0
        self.transition_steps_remaining = 0
        self.is_transitioning = True

    def update_logic(self) -> bool:
        """Advance one simulated second and return whether it is transition time."""
        if not self.is_transitioning:
            traci.trafficlight.setRedYellowGreenState(
                self.jid, self.states[self.current_logic_idx]
            )
            self.phase_timer += 1
            return False

        if self.transition_steps_remaining:
            self.transition_steps_remaining -= 1
            return True

        if self.transition_cursor < len(self.transition_queue):
            duration, state = self.transition_queue[self.transition_cursor]
            traci.trafficlight.setRedYellowGreenState(self.jid, state)
            self.transition_cursor += 1
            self.transition_steps_remaining = duration - 1
            return True

        self.is_transitioning = False
        self.current_logic_idx = self.target_logic_idx
        self.phase_timer = 0
        traci.trafficlight.setRedYellowGreenState(
            self.jid, self.states[self.current_logic_idx]
        )
        self.phase_timer += 1
        return False

    def collect_measurements(self) -> dict[str, float]:
        vehicle_counts = []
        speeds = []
        occupancy = []
        halting = []
        for detector in self.detectors:
            count = traci.inductionloop.getLastStepVehicleNumber(detector)
            speed = max(0.0, traci.inductionloop.getLastStepMeanSpeed(detector))
            vehicle_counts.append(count)
            speeds.append(speed)
            occupancy.append(
                traci.inductionloop.getLastStepOccupancy(detector) / 100.0
            )
            halting.append(count if speed < 0.1 and count > 0 else 0)

        lane_vehicle_ids = [
            vehicle_id
            for lane in self.incoming_lanes
            for vehicle_id in traci.lane.getLastStepVehicleIDs(lane)
        ]
        self._last_metrics = {
            "co2": -sum(
                traci.lane.getCO2Emission(lane) for lane in self.incoming_lanes
            ),
            "throughput": float(
                sum(vehicle_counts) - sum(halting)
            ),
            "average_speed": (
                float(np.average(speeds, weights=vehicle_counts))
                if sum(vehicle_counts) > 0
                else 0.0
            ),
            "waiting_time": -(
                float(
                    np.mean(
                        [
                            traci.vehicle.getWaitingTime(vehicle_id)
                            for vehicle_id in lane_vehicle_ids
                        ]
                    )
                )
                if lane_vehicle_ids
                else 0.0
            ),
            "queue_length": -float(
                sum(
                    traci.lane.getLastStepHaltingNumber(lane)
                    for lane in self.incoming_lanes
                )
            ),
        }
        self.steps_measured += 1
        if self.detectors:
            self.detector_flow += vehicle_counts
            self.detector_speed += speeds
            self.detector_occupancy += occupancy
        return self._last_metrics.copy()

    def get_observation(self) -> dict[str, np.ndarray]:
        detector_count = len(self.detectors)
        flow = np.zeros(detector_count, dtype=np.float32)
        speed = np.zeros(detector_count, dtype=np.float32)
        occupancy = np.zeros(detector_count, dtype=np.float32)
        if self.steps_measured and detector_count:
            flow = (self.detector_flow / self.steps_measured).astype(np.float32)
            speed = np.minimum(
                self.detector_speed / self.steps_measured / 14.0, 1.0
            ).astype(np.float32)
            occupancy = (
                self.detector_occupancy / self.steps_measured
            ).astype(np.float32)
        if not detector_count:
            flow = speed = occupancy = np.zeros(1, dtype=np.float32)
        return {
            "phase": np.asarray([self.current_logic_idx], dtype=np.int32),
            "phase_timer": np.asarray(
                [min(self.phase_timer / 120.0, 1.0)], dtype=np.float32
            ),
            "occupancy": occupancy,
            "flow": flow,
            "speed": speed,
        }


class TrafficMORLEnv(gym.Env):
    """Single-policy MORL environment for a junction in a SUMO catalogue network.

    The SUMO inputs are the network, traffic-light catalogue, and detector
    additional file. A ``TrafficAgent`` is created for each catalogue light
    present in the network;
    ``traffic_light_id`` selects the discrete action/reward interface exposed
    to single-agent MORL algorithms such as PCN.
    """

    metadata = {"render_modes": ["console", "gui"]}

    def __init__(
        self,
        net_file: str | Path,
        logic_json_file: str | Path,
        detector_file: str | Path,
        traffic_light_id: str | None = None,
        route_file: str | Path | None = None,
        render_mode: str = "console",
        simulation_time: int = 3600,
        random_traffic: bool = True,
        flow_range: tuple[int, int] = (100, 600),
        min_green_time: int = 5,
        measure_during_transition: bool = False,
        reward_components: Sequence[str] = ("co2", "time_in_state_penalty"),
        reward_normalization: str = "normalized_average",
        reward_scales: Sequence[float] | None = None,
        state_penalty_threshold: float = 45.0,
        state_penalty_growth: float = 0.1,
    ):
        super().__init__()
        if render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render_mode: {render_mode!r}.")
        if simulation_time <= 0 or min_green_time <= 0:
            raise ValueError("simulation_time and min_green_time must be positive.")
        if reward_normalization not in ("sum", "average", "normalized_average"):
            raise ValueError(
                "reward_normalization must be 'sum', 'average', or 'normalized_average'."
            )
        if not reward_components or len(set(reward_components)) != len(reward_components):
            raise ValueError("reward_components must be non-empty and unique.")
        unknown = set(reward_components) - set(MO_REWARD_COMPONENTS)
        if unknown:
            raise ValueError(
                f"Unknown reward components: {sorted(unknown)}. "
                f"Choose from: {', '.join(MO_REWARD_COMPONENTS)}."
            )
        if not np.isfinite(state_penalty_threshold) or state_penalty_threshold < 0:
            raise ValueError("state_penalty_threshold must be finite and non-negative.")
        if not np.isfinite(state_penalty_growth) or state_penalty_growth <= 0:
            raise ValueError("state_penalty_growth must be finite and positive.")

        self.net_file = Path(net_file).resolve()
        self.logic_json_file = Path(logic_json_file).resolve()
        self.detector_file = Path(detector_file).resolve()
        for path in (self.net_file, self.logic_json_file, self.detector_file):
            if not path.is_file():
                raise FileNotFoundError(f"SUMO input file not found: {path}")
        self._temporary_sumo_dir = tempfile.TemporaryDirectory(
            prefix="catalogue_sumo_"
        )
        self._runtime_detector_file = self._make_runtime_detector_file()

        self.render_mode = render_mode
        self.simulation_time = int(simulation_time)
        self.time_limit = self.simulation_time
        self.random_traffic = random_traffic
        self.flow_range = tuple(map(int, flow_range))
        if (
            len(self.flow_range) != 2
            or self.flow_range[0] <= 0
            or self.flow_range[1] < self.flow_range[0]
        ):
            raise ValueError("flow_range must be a positive (minimum, maximum) pair.")
        self.min_green_time = int(min_green_time)
        self.measure_during_transition = measure_during_transition
        self._fixed_time_program_active = False
        self._fixed_program_phase_kinds: dict[int, str] = {}
        self.record_signal_trace = False
        self.signal_trace: list[dict[str, int | str]] = []
        self.reward_components = tuple(reward_components)
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
        self.state_penalty_threshold = float(state_penalty_threshold)
        self.state_penalty_growth = float(state_penalty_growth)

        program = load_traffic_light_program(
            self.logic_json_file, traffic_light_id=traffic_light_id
        )
        self.traffic_light_id = program.traffic_light_id
        logic_ids = sorted(program.states)
        self.logic_to_action = {
            logic_id: action for action, logic_id in enumerate(logic_ids)
        }
        states = {
            self.logic_to_action[logic_id]: program.states[logic_id]
            for logic_id in logic_ids
        }
        transitions = {
            (
                self.logic_to_action[from_logic],
                self.logic_to_action[to_logic],
            ): sequence
            for (from_logic, to_logic), sequence in program.transition_sequences.items()
        }
        self._catalogue_main_signal_states = set(states.values())
        self._catalogue_transition_signal_states = {
            state
            for sequence in transitions.values()
            for _, state in sequence
        }

        self._net = sumolib.net.readNet(str(self.net_file))
        self.agents: dict[str, TrafficAgent] = {}
        self._incoming_lanes: dict[str, list[str]] = {}
        detector_lanes = {
            detector.get("id"): detector.get("lane")
            for detector in ET.parse(self.detector_file).getroot()
            if detector.tag == "inductionLoop"
        }
        all_catalogue_lights = self._read_catalogue_light_ids()
        for jid in all_catalogue_lights:
            node = self._net.getNode(jid)
            if node is None:
                if jid == self.traffic_light_id:
                    raise ValueError(
                        f"Traffic light {jid!r} is not a junction in {self.net_file}."
                    )
                continue
            incoming_lanes = sorted(
                lane.getID()
                for edge in node.getIncoming()
                if not edge.getID().startswith(":")
                for lane in edge.getLanes()
            )
            self._incoming_lanes[jid] = incoming_lanes
            if jid == self.traffic_light_id:
                agent_states = states
                agent_transitions = transitions
            else:
                other = load_traffic_light_program(
                    self.logic_json_file, traffic_light_id=jid
                )
                other_ids = sorted(other.states)
                mapping = {state_id: index for index, state_id in enumerate(other_ids)}
                agent_states = {
                    mapping[state_id]: other.states[state_id] for state_id in other_ids
                }
                agent_transitions = {
                    (mapping[source], mapping[target]): sequence
                    for (source, target), sequence in other.transition_sequences.items()
                }
            self.agents[jid] = TrafficAgent(
                jid,
                agent_states,
                agent_transitions,
                sorted(
                    detector_id
                    for detector_id, lane_id in detector_lanes.items()
                    if lane_id in incoming_lanes
                ),
                incoming_lanes,
                measure_during_transition=measure_during_transition,
            )

        self.agent = self.agents[self.traffic_light_id]
        self.action_space = self.agent.action_space
        self.observation_space = self.agent.observation_space
        self.reward_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.reward_components),),
            dtype=np.float32,
        )
        self.route_file = Path(route_file).resolve() if route_file else None
        if not self.random_traffic and self.route_file is None:
            raise ValueError(
                "route_file is required when random_traffic is disabled."
            )
        if (
            not self.random_traffic
            and self.route_file is not None
            and not self.route_file.is_file()
        ):
            raise FileNotFoundError(f"Route file not found: {self.route_file}")
        self._temporary_route_file: Path | None = None
        self.is_running = False
        self.current_step = 0
        self.time_in_state = 0
        self.sumo_binary = sumolib.checkBinary(
            "sumo-gui" if render_mode == "gui" else "sumo"
        )

    def _read_catalogue_light_ids(self) -> list[str]:
        with self.logic_json_file.open(encoding="utf-8") as file:
            light_ids = list(json.load(file))
        if self.traffic_light_id not in light_ids:
            raise ValueError(
                f"Traffic light {self.traffic_light_id!r} is absent from the catalogue."
            )
        return light_ids

    def _make_runtime_detector_file(self) -> Path:
        """Keep SUMO detector outputs out of the checked-in asset directory."""
        tree = ET.parse(self.detector_file)
        runtime_dir = Path(self._temporary_sumo_dir.name)
        for detector in tree.getroot():
            output_file = detector.get("file")
            if output_file:
                detector.set("file", str(runtime_dir / Path(output_file).name))
        runtime_file = runtime_dir / self.detector_file.name
        tree.write(runtime_file, encoding="utf-8", xml_declaration=True)
        return runtime_file

    def _make_route_file(self) -> Path:
        if self.route_file is not None:
            return self.route_file
        descriptor, filename = tempfile.mkstemp(suffix=".rou.xml", prefix="catalogue_")
        os.close(descriptor)
        self._temporary_route_file = Path(filename)
        return self._temporary_route_file

    def generate_random_traffic(
        self,
        flow_range: tuple[int, int] | None = None,
        duration: int | None = None,
        seed: int | None = None,
    ) -> Path:
        """Write randomized per-incoming-lane trips for the selected junction."""
        min_flow, max_flow = flow_range or self.flow_range
        min_flow = max(100, min(int(min_flow), 600))
        max_flow = max(min_flow, min(int(max_flow), 600))
        duration = int(duration or self.simulation_time)
        rng = random.Random(seed)
        node = self._net.getNode(self.traffic_light_id)
        trips = []
        for incoming_edge in node.getIncoming():
            edge_id = incoming_edge.getID()
            if edge_id.startswith(":"):
                continue
            for lane in incoming_edge.getLanes():
                targets = {
                    connection.getToLane().getEdge().getID()
                    for connection in lane.getOutgoing()
                    if connection.getToLane() is not None
                    and not connection.getToLane().getEdge().getID().startswith(":")
                }
                if not targets:
                    continue
                flow = rng.randint(min_flow, max_flow)
                vehicle_count = int(flow * duration / 3600.0)
                if vehicle_count <= 0:
                    continue
                average_gap = duration / vehicle_count
                for vehicle_index in range(vehicle_count):
                    depart = vehicle_index * average_gap + rng.uniform(
                        0.0, average_gap * 0.5
                    )
                    trips.append(
                        (
                            min(depart, duration - 1.0),
                            edge_id,
                            lane.getIndex(),
                            rng.choice(sorted(targets)),
                        )
                    )
        trips.sort(key=lambda trip: trip[0])
        routes = ET.Element("routes")
        ET.SubElement(
            routes,
            "vType",
            {
                "id": "car",
                "accel": "0.8",
                "decel": "4.5",
                "sigma": "0.5",
                "length": "5",
                "minGap": "2.5",
                "maxSpeed": "16.67",
                "guiShape": "passenger",
            },
        )
        for index, (depart, from_edge, lane_index, to_edge) in enumerate(trips):
            ET.SubElement(
                routes,
                "trip",
                {
                    "id": f"veh_{index}",
                    "type": "car",
                    "depart": f"{depart:.2f}",
                    "from": from_edge,
                    "to": to_edge,
                    "departLane": str(lane_index),
                },
            )
        route_path = self._make_route_file()
        route_path.parent.mkdir(parents=True, exist_ok=True)
        ET.ElementTree(routes).write(route_path, encoding="utf-8", xml_declaration=True)
        return route_path

    def _sumo_args(self, route_file: Path, duration: int, seed: int) -> list[str]:
        args = [
            "-n",
            str(self.net_file),
            "-r",
            str(route_file),
            "-a",
            str(self._runtime_detector_file),
            "--end",
            str(duration),
            "--seed",
            str(seed),
            "--no-step-log",
            "true",
            "--waiting-time-memory",
            "10000",
            "--ignore-route-errors",
            "true",
            "--no-warnings",
            "true",
            "--xml-validation",
            "never",
        ]
        if self.render_mode == "gui":
            args.append("--start")
        return args

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        duration = int(options.get("traffic_duration", self.simulation_time))
        if duration <= 0:
            raise ValueError("traffic_duration must be positive.")
        self.time_limit = duration
        flow_range = tuple(options.get("flow_range", self.flow_range))
        traffic_seed = int(self.np_random.integers(1, 2**31 - 1))
        if self.random_traffic:
            route_file = self.generate_random_traffic(
                flow_range=flow_range, duration=duration, seed=traffic_seed
            )
        elif self.route_file is not None:
            route_file = self.route_file
        else:
            raise ValueError(
                "route_file is required when random_traffic is disabled."
            )

        sumo_args = self._sumo_args(
            route_file,
            duration,
            int(self.np_random.integers(1, 2**31 - 1)),
        )
        if self.is_running:
            traci.load(sumo_args)
        else:
            traci.start([self.sumo_binary, *sumo_args])
            self.is_running = True
        self.current_step = 0
        self.time_in_state = 0
        self.agent.reset_logic()
        self._fixed_time_program_active = False
        self.signal_trace = []
        for agent in self.agents.values():
            agent.reset_step_metrics()
        traci.trafficlight.setRedYellowGreenState(
            self.traffic_light_id,
            self.agent.states[self.agent.current_logic_idx],
        )
        self._record_signal_state()
        return self.agent.get_observation(), {
            "traffic_light_id": self.traffic_light_id,
            "ready": True,
        }

    def _step_seconds(
        self,
        seconds: int,
        include_transition_measurements: bool,
        update_signal: bool = True,
    ) -> tuple[np.ndarray, int]:
        reward_sum = np.zeros(len(self.reward_components), dtype=np.float64)
        measured_seconds = 0
        for _ in range(seconds):
            transitioning = self.agent.update_logic() if update_signal else False
            traci.simulationStep()
            self.current_step += 1
            self.time_in_state = self.agent.phase_timer
            self._record_signal_state()
            if include_transition_measurements or not transitioning:
                metrics = self.agent.collect_measurements()
                reward_sum += self._reward_for_metrics(metrics)
                measured_seconds += 1
        return reward_sum, measured_seconds

    def _record_signal_state(self) -> None:
        if self.record_signal_trace:
            signal_state = traci.trafficlight.getRedYellowGreenState(
                self.traffic_light_id
            )
            if self._fixed_time_program_active:
                phase = traci.trafficlight.getPhase(self.traffic_light_id)
                kind = self._fixed_program_phase_kinds.get(phase, "other")
            elif signal_state in self._catalogue_main_signal_states:
                kind = "main"
            elif signal_state in self._catalogue_transition_signal_states:
                kind = "transition"
            else:
                kind = "other"
            self.signal_trace.append(
                {
                    "time": self.current_step,
                    "state": signal_state,
                    "kind": kind,
                }
            )

    def _reward_for_metrics(self, metrics: dict[str, float]) -> np.ndarray:
        values = dict(metrics)
        penalty_exponent = min(
            self.state_penalty_growth
            * max(0.0, self.agent.phase_timer - self.state_penalty_threshold),
            np.log1p(MAX_STATE_PENALTY),
        )
        values["time_in_state_penalty"] = -min(
            float(np.expm1(penalty_exponent)),
            MAX_STATE_PENALTY,
        )
        return np.asarray(
            [values[name] for name in self.reward_components], dtype=np.float64
        )

    def step(self, action):
        if not self.is_running:
            raise RuntimeError("SUMO simulation is not running; call reset() first.")
        if self.current_step >= self.time_limit:
            raise RuntimeError("Episode is over; call reset() before stepping again.")
        if not self.action_space.contains(action):
            raise ValueError(
                f"Invalid action {action!r}; expected an action in "
                f"[0, {self.action_space.n - 1}]."
            )

        self.agent.reset_step_metrics()
        total_reward = np.zeros(len(self.reward_components), dtype=np.float64)
        measured_seconds = 0
        elapsed_seconds = 0
        if (
            int(action) != self.agent.current_logic_idx
            and self.agent.phase_timer < self.min_green_time
        ):
            initial_green_seconds = min(
                self.min_green_time - self.agent.phase_timer,
                self.time_limit - self.current_step,
            )
            initial_reward, initial_measured = self._step_seconds(
                initial_green_seconds,
                include_transition_measurements=True,
            )
            total_reward += initial_reward
            measured_seconds += initial_measured
            elapsed_seconds += initial_green_seconds

        if self.current_step < self.time_limit:
            self.agent.set_target_phase(int(action))
        if self.agent.is_transitioning:
            full_transition_seconds = sum(
                duration for duration, _ in self.agent.transition_queue
            )
            transition_seconds = min(
                full_transition_seconds, self.time_limit - self.current_step
            )
            transition_reward, transition_measured = self._step_seconds(
                transition_seconds,
                include_transition_measurements=(
                    self.measure_during_transition
                    or transition_seconds == self.time_limit - self.current_step
                ),
            )
            total_reward += transition_reward
            measured_seconds += transition_measured
            elapsed_seconds += transition_seconds

        green_seconds = min(
            self.min_green_time, self.time_limit - self.current_step
        )
        if green_seconds:
            green_reward, green_measured = self._step_seconds(
                green_seconds, include_transition_measurements=True
            )
            total_reward += green_reward
            measured_seconds += green_measured
            elapsed_seconds += green_seconds
        if measured_seconds == 0:
            raise RuntimeError("No simulation seconds were measured for this action.")

        average_reward = total_reward / measured_seconds
        if self.reward_normalization == "sum":
            reward = total_reward
        elif self.reward_normalization == "normalized_average":
            reward = average_reward / self.reward_scales
        else:
            reward = average_reward
        reward = np.asarray(reward, dtype=np.float32)
        terminated = self.current_step >= self.time_limit
        if not terminated:
            terminated = traci.simulation.getMinExpectedNumber() <= 0
        info = {
            "elapsed_seconds": elapsed_seconds,
            "measured_seconds": measured_seconds,
            "traffic_light_id": self.traffic_light_id,
            "reward_components": dict(
                zip(self.reward_components, reward.tolist())
            ),
        }
        return self.agent.get_observation(), reward, bool(terminated), False, info

    def activate_fixed_time_program(self, program_id: str = "0") -> None:
        """Return the selected light to a SUMO-defined signal program."""
        if not self.is_running:
            raise RuntimeError("SUMO simulation is not running; call reset() first.")
        if "time_in_state_penalty" in self.reward_components:
            raise ValueError(
                "The time_in_state_penalty objective is only defined for "
                "agent-controlled phases."
            )
        self._fixed_program_phase_kinds = {}
        for logic in ET.parse(self.net_file).getroot().findall("tlLogic"):
            if (
                logic.get("id") == self.traffic_light_id
                and logic.get("programID") == program_id
            ):
                for phase_index, phase in enumerate(logic.findall("phase")):
                    state = phase.get("state", "")
                    is_transition = (
                        any(signal in state for signal in "yYuU")
                        or not any(signal in state for signal in "gG")
                    )
                    self._fixed_program_phase_kinds[phase_index] = (
                        "transition" if is_transition else "main"
                    )
                break
        if not self._fixed_program_phase_kinds:
            raise ValueError(
                f"Signal program {program_id!r} for {self.traffic_light_id!r} "
                f"was not found in {self.net_file}."
            )
        traci.trafficlight.setProgram(self.traffic_light_id, program_id)
        self._fixed_time_program_active = True
        if self.record_signal_trace:
            self.signal_trace = []
            self._record_signal_state()

    def step_fixed_time(self):
        """Advance SUMO's active signal program for one action interval."""
        if not self.is_running:
            raise RuntimeError("SUMO simulation is not running; call reset() first.")
        if not self._fixed_time_program_active:
            raise RuntimeError(
                "Activate a SUMO signal program before calling step_fixed_time()."
            )
        if self.current_step >= self.time_limit:
            raise RuntimeError("Episode is over; call reset() before stepping again.")

        self.agent.reset_step_metrics()
        seconds = min(self.min_green_time, self.time_limit - self.current_step)
        reward_sum, measured_seconds = self._step_seconds(
            seconds,
            include_transition_measurements=True,
            update_signal=False,
        )
        average_reward = reward_sum / measured_seconds
        if self.reward_normalization == "sum":
            reward = reward_sum
        elif self.reward_normalization == "normalized_average":
            reward = average_reward / self.reward_scales
        else:
            reward = average_reward
        reward = np.asarray(reward, dtype=np.float32)
        terminated = self.current_step >= self.time_limit
        if not terminated:
            terminated = traci.simulation.getMinExpectedNumber() <= 0
        info = {
            "elapsed_seconds": seconds,
            "measured_seconds": measured_seconds,
            "traffic_light_id": self.traffic_light_id,
            "signal_program": traci.trafficlight.getProgram(self.traffic_light_id),
            "reward_components": dict(
                zip(self.reward_components, reward.tolist())
            ),
        }
        return self.agent.get_observation(), reward, bool(terminated), False, info

    def close(self) -> None:
        if self.is_running:
            traci.close()
            self.is_running = False
        if self._temporary_route_file is not None:
            self._temporary_route_file.unlink(missing_ok=True)
            self._temporary_route_file = None
        self._temporary_sumo_dir.cleanup()
