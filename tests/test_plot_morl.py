import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from plot_morl import load_results, pareto_mask, plot_results, plot_state_trace


class PlotMORLTests(unittest.TestCase):
    def test_pareto_mask_keeps_only_non_dominated_returns(self):
        returns = np.asarray(
            [
                [1.0, 1.0],
                [2.0, 0.5],
                [0.5, 2.0],
                [0.5, 0.5],
            ]
        )

        np.testing.assert_array_equal(
            pareto_mask(returns),
            np.asarray([True, True, True, False]),
        )

    def test_load_results_reads_evaluated_returns_and_objectives(self):
        data = {
            "environment": {"reward_components": ["co2", "queue_length"]},
            "evaluation": {"returns": [[-1.0, -0.5], [-2.0, -0.25]]},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample_results.json"
            path.write_text(json.dumps(data), encoding="utf-8")

            name, components, returns = load_results(path)

        self.assertEqual(name, "sample")
        self.assertEqual(components, ("co2", "queue_length"))
        np.testing.assert_array_equal(returns, data["evaluation"]["returns"])

    def test_plot_results_writes_both_plot_files(self):
        data = {
            "environment": {"reward_components": ["co2", "queue_length"]},
            "evaluation": {
                "returns": [
                    [-1.0, -0.5],
                    [-0.8, -0.7],
                    [-0.5, -0.9],
                ]
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "sample_results.json"
            input_path.write_text(json.dumps(data), encoding="utf-8")

            pair_path, distribution_path = plot_results(
                [input_path], root / "plots", "test"
            )

            self.assertTrue(pair_path.is_file())
            self.assertGreater(pair_path.stat().st_size, 0)
            self.assertTrue(distribution_path.is_file())
            self.assertGreater(distribution_path.stat().st_size, 0)

    def test_plot_state_trace_writes_trajectory_figure(self):
        trace = {
            "seed": 100,
            "simulation_time": 3,
            "states": {
                "PCN": [
                    {"time": 0, "state": "GGrr", "kind": "main"},
                    {"time": 1, "state": "yyrr", "kind": "transition"},
                    {"time": 2, "state": "rrGG", "kind": "main"},
                    {"time": 3, "state": "rrGG", "kind": "main"},
                ],
                "Native": [
                    {"time": 0, "state": "GGrr", "kind": "main"},
                    {"time": 1, "state": "GGrr", "kind": "main"},
                    {"time": 2, "state": "yyrr", "kind": "transition"},
                    {"time": 3, "state": "rrGG", "kind": "main"},
                ],
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace_path = root / "trace.json"
            trace_path.write_text(json.dumps(trace), encoding="utf-8")

            figure_path = plot_state_trace(trace_path, root / "states.png")

            self.assertTrue(figure_path.is_file())
            self.assertGreater(figure_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
