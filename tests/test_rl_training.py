import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pybullet as p


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "demos"))

from grasp_training import (  # noqa: E402
    GRASP_SUCCESS_HEIGHT,
    GraspTrainingConfig,
    RobotGraspEnv,
    build_grasp_td3,
    grasp_checkpoint_score,
    warm_start_grasp_model,
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
    def test_robust_config_uses_separate_curriculum_and_large_evaluation(self):
        config = GraspTrainingConfig.robust(seed=7)
        self.assertEqual(config.seed, 7)
        self.assertEqual(config.randomization_curriculum, (0.35, 0.70, 1.0))
        self.assertEqual(config.final_eval_episodes, 1_000)

    def test_checkpoint_score_keeps_partial_grasp_progress(self):
        base = {
            "success_rate": 0.0,
            "grasp_rate": 0.0,
            "bilateral_contact_rate": 0.0,
            "mean_lift_height_m": 0.0,
            "approach_rate": 0.7,
        }
        contact_policy = {
            **base,
            "grasp_rate": 0.4,
            "bilateral_contact_rate": 0.5,
            "mean_lift_height_m": 0.012,
            "approach_rate": 1.0,
        }
        self.assertGreater(
            grasp_checkpoint_score(contact_policy),
            grasp_checkpoint_score(base),
        )

    def test_grasp_environment_has_physical_cube_and_gripper_action(self):
        env = RobotGraspEnv(max_steps=2)
        try:
            observation, info = env.reset(seed=123)
            self.assertEqual(observation.shape, (24,))
            self.assertEqual(env.action_space.shape, (8,))
            self.assertTrue(env.observation_space.contains(observation))
            self.assertEqual(env.get_demonstration_phase(), 0)
            self.assertEqual(observation[-2], 0.0)
            self.assertEqual(observation[-1], 0.0)
            mass = p.getDynamicsInfo(
                env.cube_id, -1, physicsClientId=env.client)[0]
            collision = p.getCollisionShapeData(
                env.cube_id, -1, physicsClientId=env.client)
            self.assertGreater(mass, 0.0)
            self.assertTrue(collision)
            self.assertIn("lift_height", info)
            self.assertEqual(info["randomization"], 0.0)
            self.assertAlmostEqual(info["cube_size"], 0.05)
            self.assertAlmostEqual(info["cube_mass"], 0.03)
            self.assertEqual(info["action_delay_steps"], 0)
            result = env.step(np.array([0.0] * 7 + [-1.0], dtype=np.float32))
            self.assertEqual(len(result), 5)
            self.assertEqual(result[0].shape, (24,))
        finally:
            env.close()

    def test_domain_randomization_is_seeded_and_within_ranges(self):
        first = RobotGraspEnv(randomization=1.0, max_steps=2)
        second = RobotGraspEnv(randomization=1.0, max_steps=2)
        try:
            first_observation, first_info = first.reset(seed=321)
            second_observation, second_info = second.reset(seed=321)
            for key in (
                "cube_size",
                "cube_mass",
                "cube_friction",
                "cube_yaw",
                "action_delay_steps",
            ):
                self.assertEqual(first_info[key], second_info[key])
            np.testing.assert_allclose(first_observation, second_observation)
            self.assertGreaterEqual(first_info["cube_size"], 0.04)
            self.assertLessEqual(first_info["cube_size"], 0.06)
            self.assertGreaterEqual(first_info["cube_mass"], 0.02)
            self.assertLessEqual(first_info["cube_mass"], 0.08)
            self.assertGreaterEqual(first_info["cube_friction"], 0.8)
            self.assertLessEqual(first_info["cube_friction"], 3.5)
            self.assertLessEqual(abs(first_info["cube_yaw"]), np.pi / 4)
            self.assertIn(first_info["action_delay_steps"], (0, 1, 2))
        finally:
            first.close()
            second.close()

    def test_robust_observation_can_include_two_action_commands(self):
        env = RobotGraspEnv(observe_action_history=True, max_steps=2)
        try:
            observation, _ = env.reset(seed=654)
            self.assertEqual(observation.shape, (40,))
            np.testing.assert_array_equal(observation[-16:], 0.0)
            command = np.linspace(-1.0, 1.0, 8, dtype=np.float32)
            observation, *_ = env.step(command)
            np.testing.assert_allclose(observation[-8:], command)
        finally:
            env.close()

    def test_policy_expansion_preserves_actions_and_ignores_new_history(self):
        source_env = RobotGraspEnv(max_steps=2)
        target_env = RobotGraspEnv(
            observe_action_history=True, max_steps=2)
        try:
            source = build_grasp_td3(source_env, seed=11)
            target = build_grasp_td3(target_env, seed=12)
            with tempfile.TemporaryDirectory() as directory:
                model_path = Path(directory) / "source_policy.zip"
                source.save(model_path)
                warm_start_grasp_model(target, model_path)

            generator = np.random.default_rng(42)
            observations = generator.normal(size=(16, 24)).astype(np.float32)
            histories = generator.uniform(
                -1.0, 1.0, size=(16, 16)).astype(np.float32)
            expanded = np.concatenate([observations, histories], axis=1)
            source_actions, _ = source.predict(
                observations, deterministic=True)
            target_actions, _ = target.predict(expanded, deterministic=True)
            np.testing.assert_array_equal(source_actions, target_actions)
        finally:
            source_env.close()
            target_env.close()

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
                if info["grasped"]:
                    self.assertEqual(env.get_demonstration_phase(), 2)
                    observation = env._get_observation()
                    self.assertEqual(observation[-2], 1.0)
                    self.assertGreater(observation[-1], 0.0)
        finally:
            env.close()

        self.assertGreaterEqual(successes, 5)

    def test_opening_after_grasp_terminates_as_failure(self):
        env = RobotGraspEnv()
        try:
            env.reset(seed=9_100)
            for _ in range(env.max_steps):
                _, _, terminated, truncated, info = env.step(
                    env.get_ik_action())
                if info["grasped"] or terminated or truncated:
                    break
            self.assertTrue(info["grasped"])
            open_action = np.array(
                [0.0] * 7 + [-1.0], dtype=np.float32)
            _, reward, terminated, _, info = env.step(open_action)
            self.assertTrue(terminated)
            self.assertFalse(info["is_success"])
            self.assertLess(reward, 0.0)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
