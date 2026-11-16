"""Load traffic-light states and transition sequences from a JSON catalogue."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union


TransitionKey = Tuple[int, int]
TransitionStep = Tuple[int, str]


@dataclass(frozen=True)
class TrafficLightProgram:
    traffic_light_id: str
    states: Dict[int, str]
    transition_sequences: Dict[TransitionKey, List[TransitionStep]]


def load_traffic_light_program(
    json_file: Union[str, Path], traffic_light_id: Optional[str] = None
) -> TrafficLightProgram:
    """Convert green phases and indexed intermediate phases into environment data.

    Green phase ``logic_idx`` values identify states. Each entry in
    ``transitions`` is an ordered list of phase-array indices; those phase
    objects supply the intermediate signal strings and durations.
    """
    with Path(json_file).open(encoding="utf-8") as file:
        catalogue = json.load(file)

    if traffic_light_id is None:
        traffic_light_id = next(iter(catalogue))
    light = catalogue[traffic_light_id]
    phases = {phase["index"]: phase for phase in light["phases"]}

    states = {
        phase["logic_idx"]: phase["state"]
        for phase in light["phases"]
        if phase["type"] == "green"
    }
    transitions = {}
    for transition, phase_indices in light["transitions"].items():
        from_logic, to_logic = (int(state) for state in transition.split("->"))
        transitions[(from_logic, to_logic)] = [
            (int(phases[index]["duration"]), phases[index]["state"])
            for index in phase_indices
        ]

    return TrafficLightProgram(traffic_light_id, states, transitions)
