import unittest
from pathlib import Path

from gym_envs.envs.traffic_lights_parser import load_traffic_light_program


CATALOGUE_PATH = (
    Path(__file__).resolve().parents[1]
    / "sumo_files"
    / "AI_agens_konyvtar"
    / "traffic_lights.json"
)


class TrafficLightsParserTests(unittest.TestCase):
    def test_parses_states_and_ordered_transition_phases(self):
        program = load_traffic_light_program(CATALOGUE_PATH, "R3C2_C")

        self.assertEqual(len(program.states), 12)
        self.assertEqual(program.states[0], "GrrGrGGrrrrGGGG")
        self.assertEqual(len(program.transition_sequences), 132)
        self.assertEqual(
            program.transition_sequences[(0, 1)],
            [
                (2, "GrrGrGGrrrrGyGG"),
                (1, "GrrGrGGrurrGyGG"),
                (1, "GrrGrGGrurrGrGG"),
            ],
        )
        self.assertEqual(
            sum(map(len, program.transition_sequences.values())),
            358,
        )


if __name__ == "__main__":
    unittest.main()
