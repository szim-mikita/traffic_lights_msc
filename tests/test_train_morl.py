import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import train_morl


class StubMORLEnv(gym.Env):
    def __init__(self):
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(0, 1, shape=(1,), dtype=np.float32)
        self.reward_space = spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32)
        self.reward_components = ("co2", "queue_length")
        self.reward_normalization = "normalized_average"
        self.reward_scales = np.asarray([10_000.0, 10.0], dtype=np.float32)

    def step(self, action):
        return action, 0.0, False, False, {}


class FakePCN:
    def __init__(self, **_kwargs):
        pass

    def train(self, **_kwargs):
        pass

    def save(self, filename):
        self.filename = filename

    def evaluate(self, _env, _max_return, n):
        returns = [np.asarray([-1.0, -0.5], dtype=np.float32)] * min(n, 2)
        replay_returns = [np.asarray([-0.8, -0.4], dtype=np.float32)] * len(returns)
        distances = np.asarray([0.2] * len(returns), dtype=np.float32)
        return returns, replay_returns, distances


class TrainMORLTests(unittest.TestCase):
    def test_training_progress_counts_actions_and_preserves_step_result(self):
        env = StubMORLEnv()
        progress_env = train_morl.TrainingProgressWrapper(env, total_timesteps=2)
        progress_env.interactive = False
        output = io.StringIO()

        with redirect_stdout(output):
            result = progress_env.step(1)

        self.assertEqual(result[0], 1)
        self.assertFalse(result[2])
        self.assertEqual(progress_env.steps, 1)
        self.assertIn("Training action steps: 1/2", output.getvalue())

    def test_run_train_saves_model_and_evaluation_results(self):
        env = StubMORLEnv()
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(train_morl, "PCN", FakePCN):
                train_morl.run_train(
                    env=env,
                    total_timesteps=16,
                    scaling_factor=np.ones(3, dtype=np.float32),
                    max_return=np.ones(2, dtype=np.float32),
                    ref_point=-np.ones(2, dtype=np.float32),
                    initial_episodes=4,
                    step_episodes=1,
                    model_updates=1,
                    batch_size=8,
                    hidden_dim=16,
                    learning_rate=1e-3,
                    gamma=1.0,
                    seed=7,
                    model_name="catalogue_test",
                    results_dir=directory,
                )

            results_path = Path(directory) / "catalogue_test_results.json"
            results = json.loads(results_path.read_text(encoding="utf-8"))
            self.assertEqual(
                results["environment"]["reward_components"],
                ["co2", "queue_length"],
            )
            self.assertEqual(results["evaluation"]["episodes"], 2)
            self.assertEqual(results["evaluation"]["returns"][0], [-1.0, -0.5])


if __name__ == "__main__":
    unittest.main()
