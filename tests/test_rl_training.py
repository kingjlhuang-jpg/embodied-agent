import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "demos"))

from rl_training import RobotArmEnv, SUCCESS_DISTANCE  # noqa: E402


class RobotArmEnvironmentTest(unittest.TestCase):
    def test_reset_and_step_follow_gymnasium_api(self):
        env = RobotArmEnv(max_steps=2)
        try:
            observation, info = env.reset(seed=123)
            self.assertEqual(observation.shape, (13,))
            self.assertEqual(observation.dtype, np.float32)
            self.assertTrue(env.observation_space.contains(observation))
            self.assertIn("distance", info)

            result = env.step(np.zeros(7, dtype=np.float32))
            self.assertEqual(len(result), 5)
            self.assertEqual(result[0].shape, (13,))
            self.assertIsInstance(result[1], float)
            self.assertIsInstance(result[2], bool)
            self.assertIsInstance(result[3], bool)
        finally:
            env.close()

    def test_ik_teacher_reaches_fixed_targets_within_two_centimeters(self):
        env = RobotArmEnv()
        successes = 0
        try:
            for episode in range(10):
                env.reset(seed=8_000 + episode)
                for _ in range(env.max_steps):
                    _, _, terminated, truncated, info = env.step(
                        env.get_ik_action())
                    if terminated or truncated:
                        break
                successes += info["distance"] < SUCCESS_DISTANCE
        finally:
            env.close()

        self.assertGreaterEqual(successes, 9)


if __name__ == "__main__":
    unittest.main()
