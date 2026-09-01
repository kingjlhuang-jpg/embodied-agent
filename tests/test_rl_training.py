import sys
import unittest
from pathlib import Path

import numpy as np
import pybullet as p


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "demos"))

from grasp_training import (  # noqa: E402
    GRASP_SUCCESS_HEIGHT,
    RobotGraspEnv,
)
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


class RobotGraspEnvironmentTest(unittest.TestCase):
    def test_grasp_environment_has_physical_cube_and_gripper_action(self):
        env = RobotGraspEnv(max_steps=2)
        try:
            observation, info = env.reset(seed=123)
            self.assertEqual(observation.shape, (22,))
            self.assertEqual(env.action_space.shape, (8,))
            self.assertTrue(env.observation_space.contains(observation))
            mass = p.getDynamicsInfo(
                env.cube_id, -1, physicsClientId=env.client)[0]
            collision = p.getCollisionShapeData(
                env.cube_id, -1, physicsClientId=env.client)
            self.assertGreater(mass, 0.0)
            self.assertTrue(collision)
            self.assertIn("lift_height", info)
            result = env.step(np.array([0.0] * 7 + [-1.0], dtype=np.float32))
            self.assertEqual(len(result), 5)
            self.assertEqual(result[0].shape, (22,))
        finally:
            env.close()

    def test_teacher_closes_gripper_and_lifts_cube(self):
        env = RobotGraspEnv()
        successes = 0
        try:
            for episode in range(6):
                env.reset(seed=9_000 + episode)
                for _ in range(env.max_steps):
                    _, _, terminated, truncated, info = env.step(
                        env.get_ik_action())
                    if terminated or truncated:
                        break
                successes += (
                    info["grasped"]
                    and info["lift_height"] >= GRASP_SUCCESS_HEIGHT
                )
        finally:
            env.close()

        self.assertGreaterEqual(successes, 5)


if __name__ == "__main__":
    unittest.main()
