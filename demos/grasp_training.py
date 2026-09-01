"""IK-guided TD3+BC training for contact-rich Kuka cube grasping.

The inverse-kinematics teacher handles the long-horizon arm motion while the
policy learns the complete 8-dimensional action: seven arm joints plus one
continuous gripper command.  A grasp is latched only after both finger groups
touch the dynamic cube, which keeps the simplified simulator stable enough for
fast local reinforcement-learning experiments.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import torch
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise

from rl_training import (
    Demonstrations,
    GuidedTD3,
    add_to_replay_buffer,
    collect_demonstrations,
    pretrain_actor,
    warmup_critic,
)


ARM_ACTION_SCALE = 0.08
TABLE_TOP_Z = 0.58
CUBE_HALF_EXTENT = 0.025
CUBE_REST_Z = TABLE_TOP_Z + CUBE_HALF_EXTENT
GRASP_SUCCESS_HEIGHT = 0.035
GRASP_HOLD_STEPS = 8
GRASP_DISTANCE = 0.030
SAFE_CLOSE_DISTANCE = 0.035
FINE_ALIGNMENT_DISTANCE = 0.075
GRIPPER_OPEN_ANGLE = 0.30
GRIPPER_CLOSED_ANGLE = -0.02
GRASP_CENTER_Z_OFFSET = 0.005
TEACHER_LIFT_HEIGHT = 0.08
GRASP_BC_ACTION_WEIGHTS = np.array([1.0] * 7 + [4.0], dtype=np.float32)

GRASP_MODEL_PATH = Path("grasp_policy.zip")
BEST_GRASP_MODEL_PATH = Path("best_grasp_policy.zip")
GRASP_METRICS_PATH = Path("grasp_training_metrics.json")


@dataclass(frozen=True)
class GraspTrainingConfig:
    """Training budget for the contact-rich grasping task."""

    seed: int = 20260901
    expert_steps: int = 30_000
    bc_epochs: int = 25
    dagger_iterations: int = 3
    dagger_steps: int = 6_000
    dagger_epochs: int = 8
    critic_warmup_steps: int = 3_000
    rl_phase_steps: tuple[int, ...] = (15_000, 25_000, 40_000)
    curriculum: tuple[float, ...] = (1.00, 1.00, 1.00)
    eval_episodes: int = 50
    final_eval_episodes: int = 100
    target_success_rate: float = 0.95

    @classmethod
    def quick(cls, seed: int = 20260901) -> "GraspTrainingConfig":
        """Small smoke-test configuration; it is not a convergence claim."""

        return cls(
            seed=seed,
            expert_steps=1_500,
            bc_epochs=2,
            dagger_iterations=1,
            dagger_steps=750,
            dagger_epochs=1,
            critic_warmup_steps=75,
            rl_phase_steps=(750,),
            curriculum=(0.35,),
            eval_episodes=10,
            final_eval_episodes=20,
            target_success_rate=1.0,
        )


class RobotGraspEnv(gym.Env):
    """Kuka iiwa + WSG50 task: approach, close, lift, and hold a cube."""

    metadata = {"render_modes": ["human"], "render_fps": 60}

    ARM_JOINTS = tuple(range(7))
    END_EFFECTOR_LINK = 6
    GRIPPER_BASE_LINK = 7
    LEFT_FINGER_LINKS = (8, 9, 10)
    RIGHT_FINGER_LINKS = (11, 12, 13)
    READY_JOINTS = np.array([
        0.82809855,
        0.21385830,
        1.90508482,
        0.93096502,
        0.23793220,
        -2.09400000,
        2.86660703,
    ])

    def __init__(
        self,
        render_mode: str | None = None,
        *,
        render: bool | None = None,
        difficulty: float = 1.0,
        max_steps: int = 300,
    ):
        super().__init__()
        if render is not None:
            render_mode = "human" if render else None
        if render_mode not in (None, "human"):
            raise ValueError(f"unsupported render_mode: {render_mode}")

        self.render_mode = render_mode
        self.difficulty = float(np.clip(difficulty, 0.0, 1.0))
        self.max_steps = int(max_steps)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(8,), dtype=np.float32)
        self.observation_space = spaces.Box(
            -5.0, 5.0, shape=(24,), dtype=np.float32)

        mode = p.GUI if render_mode == "human" else p.DIRECT
        self.client = p.connect(mode)
        if self.client < 0:
            raise RuntimeError("failed to connect to PyBullet")
        p.setAdditionalSearchPath(
            pybullet_data.getDataPath(), physicsClientId=self.client)
        if render_mode == "human":
            p.resetDebugVisualizerCamera(
                1.4, 45, -32, [0.42, 0.0, 0.62],
                physicsClientId=self.client,
            )

        self.robot_id = -1
        self.cube_id = -1
        self.grasp_constraint: int | None = None
        self.cube_start_pos = np.zeros(3, dtype=np.float64)
        self.lift_target_pos = np.zeros(3, dtype=np.float64)
        self.step_count = 0
        self.hold_steps = 0
        self.previous_distance = 0.0
        self.previous_cube_height = CUBE_REST_Z
        self.previous_action = np.zeros(8, dtype=np.float32)
        self.gripper_command = -1.0
        self.joint_lower = np.full(7, -np.pi, dtype=np.float64)
        self.joint_upper = np.full(7, np.pi, dtype=np.float64)
        self._ik_joints: list[int] = []
        self._ik_lower: list[float] = []
        self._ik_upper: list[float] = []
        self._ik_ranges: list[float] = []
        self._downward_orientation = p.getQuaternionFromEuler(
            [0.0, -math.pi, 0.0])

    @property
    def is_grasped(self) -> bool:
        return self.grasp_constraint is not None

    def set_difficulty(self, difficulty: float) -> None:
        self.difficulty = float(np.clip(difficulty, 0.0, 1.0))

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        del options

        p.resetSimulation(physicsClientId=self.client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        p.setTimeStep(1.0 / 240.0, physicsClientId=self.client)
        p.setPhysicsEngineParameter(
            numSolverIterations=150, physicsClientId=self.client)
        p.loadURDF("plane.urdf", physicsClientId=self.client)

        platform_collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[0.32, 0.32, 0.025],
            physicsClientId=self.client,
        )
        platform_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.32, 0.32, 0.025],
            rgbaColor=[0.55, 0.55, 0.58, 1.0],
            physicsClientId=self.client,
        )
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=platform_collision,
            baseVisualShapeIndex=platform_visual,
            basePosition=[0.42, 0.0, TABLE_TOP_Z - 0.025],
            physicsClientId=self.client,
        )

        self.robot_id = p.loadSDF(
            "kuka_iiwa/kuka_with_gripper2.sdf",
            physicsClientId=self.client,
        )[0]
        for joint, position in enumerate(self.READY_JOINTS):
            p.resetJointState(
                self.robot_id,
                joint,
                float(position),
                physicsClientId=self.client,
            )
            info = p.getJointInfo(
                self.robot_id, joint, physicsClientId=self.client)
            self.joint_lower[joint] = info[8]
            self.joint_upper[joint] = info[9]
        self._reset_gripper(opened=True)
        self._prepare_ik_limits()

        for link in self.LEFT_FINGER_LINKS + self.RIGHT_FINGER_LINKS:
            p.changeDynamics(
                self.robot_id,
                link,
                lateralFriction=3.0,
                spinningFriction=0.15,
                rollingFriction=0.01,
                physicsClientId=self.client,
            )

        cube_x = 0.40 + self.difficulty * self.np_random.uniform(-0.055, 0.055)
        cube_y = self.difficulty * self.np_random.uniform(-0.075, 0.075)
        cube_position = np.array([cube_x, cube_y, CUBE_REST_Z])
        cube_collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[CUBE_HALF_EXTENT] * 3,
            physicsClientId=self.client,
        )
        cube_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[CUBE_HALF_EXTENT] * 3,
            rgbaColor=[0.9, 0.05, 0.05, 1.0],
            physicsClientId=self.client,
        )
        self.cube_id = p.createMultiBody(
            baseMass=0.03,
            baseCollisionShapeIndex=cube_collision,
            baseVisualShapeIndex=cube_visual,
            basePosition=cube_position,
            physicsClientId=self.client,
        )
        p.changeDynamics(
            self.cube_id,
            -1,
            lateralFriction=2.5,
            spinningFriction=0.1,
            rollingFriction=0.001,
            restitution=0.0,
            physicsClientId=self.client,
        )

        self.grasp_constraint = None
        self.step_count = 0
        self.hold_steps = 0
        self.gripper_command = -1.0
        self.previous_action.fill(0.0)
        for _ in range(20):
            self._hold_ready_pose()
            self._set_gripper(-1.0)
            p.stepSimulation(physicsClientId=self.client)

        self.cube_start_pos = self._get_cube_pos()
        self.lift_target_pos = self.cube_start_pos + np.array(
            [0.0, 0.0, TEACHER_LIFT_HEIGHT])
        self.previous_distance = self._distance_to_cube()
        self.previous_cube_height = float(self.cube_start_pos[2])
        info = self._info(self.previous_distance)
        return self._get_observation(), info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != self.action_space.shape:
            raise ValueError(f"expected action shape (8,), got {action.shape}")
        action = np.clip(action, -1.0, 1.0)

        released_cube = False
        if self.is_grasped and action[7] < -0.5:
            self._release_cube()
            released_cube = True

        joint_positions = self._get_joint_positions()
        joint_targets = joint_positions + action[:7] * ARM_ACTION_SCALE
        joint_targets = np.clip(joint_targets, self.joint_lower, self.joint_upper)
        for joint, target in enumerate(joint_targets):
            p.setJointMotorControl2(
                self.robot_id,
                joint,
                p.POSITION_CONTROL,
                targetPosition=float(target),
                force=300,
                maxVelocity=1.5,
                physicsClientId=self.client,
            )
        self._set_gripper(float(action[7]))

        for _ in range(12):
            p.stepSimulation(physicsClientId=self.client)

        left_contact, right_contact = self._finger_contacts()
        newly_grasped = False
        if (
            not self.is_grasped
            and action[7] > 0.35
            and left_contact
            and right_contact
            and self._distance_to_cube() < 0.055
        ):
            self._attach_cube()
            newly_grasped = True

        self.step_count += 1
        distance = self._distance_to_cube()
        cube_height = float(self._get_cube_pos()[2])
        lift_height = max(0.0, cube_height - float(self.cube_start_pos[2]))
        approach_progress = self.previous_distance - distance
        lift_progress = cube_height - self.previous_cube_height
        smoothness_cost = np.square(action - self.previous_action).sum()

        # Reward stages: approach < bilateral contact < lift < stable hold.
        reward = (
            8.0 * approach_progress
            - 0.15 * distance
            - 0.0015 * np.square(action[:7]).sum()
            - 0.001 * smoothness_cost
        )
        reward += 0.08 * (int(left_contact) + int(right_contact))
        premature_close = (
            max(float(action[7]), 0.0)
            * max(distance - SAFE_CLOSE_DISTANCE, 0.0)
            / 0.05
        )
        reward -= 0.25 * premature_close
        if left_contact and right_contact:
            reward += 0.35
        if newly_grasped:
            reward += 4.0
        if self.is_grasped:
            reward += 0.20
            reward += 0.60 * min(
                lift_height / GRASP_SUCCESS_HEIGHT, 1.0)
        reward += 45.0 * max(lift_progress, -0.005)
        if lift_height >= 0.02:
            reward += 1.0

        stable_lift = self.is_grasped and lift_height >= GRASP_SUCCESS_HEIGHT
        self.hold_steps = self.hold_steps + 1 if stable_lift else 0
        grasp_succeeded = self.hold_steps >= GRASP_HOLD_STEPS
        if released_cube:
            reward -= 8.0
        terminated = grasp_succeeded or released_cube
        if grasp_succeeded:
            reward += 25.0
        truncated = self.step_count >= self.max_steps

        self.previous_distance = distance
        self.previous_cube_height = cube_height
        self.previous_action = action.copy()
        return (
            self._get_observation(),
            float(reward),
            bool(terminated),
            bool(truncated),
            self._info(distance),
        )

    def get_ik_action(self) -> np.ndarray:
        """Scripted teacher: align open fingers, close on contact, then lift."""

        cube_position = self._get_cube_pos()
        end_effector_position = self._get_ee_pos()
        distance = self._distance_to_cube()
        if self.is_grasped:
            # Closed-loop Cartesian lift: move the flange by the cube's
            # remaining displacement instead of relying on a fixed tool offset.
            target_position = (
                end_effector_position
                + self.lift_target_pos
                - cube_position
            )
            gripper_action = 1.0
        else:
            desired_grasp_center = cube_position + np.array(
                [0.0, 0.0, GRASP_CENTER_Z_OFFSET])
            target_position = (
                end_effector_position
                + desired_grasp_center
                - self._get_grasp_center()
            )
            gripper_action = 1.0 if distance < GRASP_DISTANCE else -1.0

        rest_poses = [
            p.getJointState(
                self.robot_id, joint, physicsClientId=self.client)[0]
            for joint in self._ik_joints
        ]
        solution = p.calculateInverseKinematics(
            self.robot_id,
            self.END_EFFECTOR_LINK,
            target_position,
            self._downward_orientation,
            lowerLimits=self._ik_lower,
            upperLimits=self._ik_upper,
            jointRanges=self._ik_ranges,
            restPoses=rest_poses,
            maxNumIterations=200,
            residualThreshold=1e-5,
            physicsClientId=self.client,
        )
        current = self._get_joint_positions()
        delta = np.asarray(solution[:7]) - current
        arm_action = np.clip(delta / ARM_ACTION_SCALE, -1.0, 1.0)
        return np.concatenate([
            arm_action,
            np.array([gripper_action], dtype=np.float64),
        ]).astype(np.float32)

    def get_demonstration_phase(self) -> int:
        """Return approach/alignment/lift phase for balanced imitation batches."""

        if self.is_grasped:
            return 2
        if self._distance_to_cube() < FINE_ALIGNMENT_DISTANCE:
            return 1
        return 0

    def _prepare_ik_limits(self) -> None:
        self._ik_joints = []
        self._ik_lower = []
        self._ik_upper = []
        self._ik_ranges = []
        for joint in range(
            p.getNumJoints(self.robot_id, physicsClientId=self.client)
        ):
            info = p.getJointInfo(
                self.robot_id, joint, physicsClientId=self.client)
            if info[2] == p.JOINT_FIXED:
                continue
            lower, upper = float(info[8]), float(info[9])
            if upper < lower:
                lower, upper = -math.pi, math.pi
            self._ik_joints.append(joint)
            self._ik_lower.append(lower)
            self._ik_upper.append(upper)
            self._ik_ranges.append(upper - lower)

    def _hold_ready_pose(self) -> None:
        for joint, target in enumerate(self.READY_JOINTS):
            p.setJointMotorControl2(
                self.robot_id,
                joint,
                p.POSITION_CONTROL,
                targetPosition=float(target),
                force=300,
                physicsClientId=self.client,
            )

    def _reset_gripper(self, *, opened: bool) -> None:
        angle = GRIPPER_OPEN_ANGLE if opened else GRIPPER_CLOSED_ANGLE
        for joint, position in (
            (7, 0.0),
            (8, -angle),
            (10, 0.0),
            (11, angle),
            (13, 0.0),
        ):
            p.resetJointState(
                self.robot_id,
                joint,
                targetValue=float(position),
                physicsClientId=self.client,
            )

    def _set_gripper(self, command: float) -> None:
        self.gripper_command = float(np.clip(command, -1.0, 1.0))
        closure = 0.5 * (self.gripper_command + 1.0)
        angle = (
            GRIPPER_OPEN_ANGLE
            + closure * (GRIPPER_CLOSED_ANGLE - GRIPPER_OPEN_ANGLE)
        )
        p.setJointMotorControl2(
            self.robot_id,
            7,
            p.POSITION_CONTROL,
            targetPosition=0.0,
            force=100,
            physicsClientId=self.client,
        )
        for joint, target in ((8, -angle), (11, angle), (10, 0.0), (13, 0.0)):
            p.setJointMotorControl2(
                self.robot_id,
                joint,
                p.POSITION_CONTROL,
                targetPosition=float(target),
                force=180,
                maxVelocity=1.0,
                physicsClientId=self.client,
            )

    def _finger_contacts(self) -> tuple[bool, bool]:
        left = any(
            p.getContactPoints(
                bodyA=self.robot_id,
                bodyB=self.cube_id,
                linkIndexA=link,
                physicsClientId=self.client,
            )
            for link in self.LEFT_FINGER_LINKS
        )
        right = any(
            p.getContactPoints(
                bodyA=self.robot_id,
                bodyB=self.cube_id,
                linkIndexA=link,
                physicsClientId=self.client,
            )
            for link in self.RIGHT_FINGER_LINKS
        )
        return bool(left), bool(right)

    def _attach_cube(self) -> None:
        parent_state = p.getLinkState(
            self.robot_id,
            self.GRIPPER_BASE_LINK,
            computeForwardKinematics=True,
            physicsClientId=self.client,
        )
        parent_position, parent_orientation = parent_state[4], parent_state[5]
        cube_position, cube_orientation = p.getBasePositionAndOrientation(
            self.cube_id, physicsClientId=self.client)
        inverse_parent = p.invertTransform(parent_position, parent_orientation)
        relative_position, relative_orientation = p.multiplyTransforms(
            inverse_parent[0],
            inverse_parent[1],
            cube_position,
            cube_orientation,
        )
        self.grasp_constraint = p.createConstraint(
            parentBodyUniqueId=self.robot_id,
            parentLinkIndex=self.GRIPPER_BASE_LINK,
            childBodyUniqueId=self.cube_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=relative_position,
            childFramePosition=[0, 0, 0],
            parentFrameOrientation=relative_orientation,
            childFrameOrientation=[0, 0, 0, 1],
            physicsClientId=self.client,
        )
        p.changeConstraint(
            self.grasp_constraint,
            maxForce=300,
            physicsClientId=self.client,
        )

    def _release_cube(self) -> None:
        if self.grasp_constraint is not None:
            p.removeConstraint(
                self.grasp_constraint, physicsClientId=self.client)
            self.grasp_constraint = None
            self.hold_steps = 0

    def _get_joint_positions(self) -> np.ndarray:
        return np.array([
            p.getJointState(
                self.robot_id, joint, physicsClientId=self.client)[0]
            for joint in self.ARM_JOINTS
        ])

    def _get_gripper_opening(self) -> float:
        left = p.getJointState(
            self.robot_id, 8, physicsClientId=self.client)[0]
        right = p.getJointState(
            self.robot_id, 11, physicsClientId=self.client)[0]
        return float(np.clip((right - left) / (2 * GRIPPER_OPEN_ANGLE), 0.0, 1.0))

    def _get_grasp_center(self) -> np.ndarray:
        tips = [
            p.getLinkState(
                self.robot_id,
                link,
                computeForwardKinematics=True,
                physicsClientId=self.client,
            )[0]
            for link in (10, 13)
        ]
        return np.mean(np.asarray(tips), axis=0)

    def _get_ee_pos(self) -> np.ndarray:
        state = p.getLinkState(
            self.robot_id,
            self.END_EFFECTOR_LINK,
            computeForwardKinematics=True,
            physicsClientId=self.client,
        )
        return np.asarray(state[0])

    def _get_cube_pos(self) -> np.ndarray:
        position, _ = p.getBasePositionAndOrientation(
            self.cube_id, physicsClientId=self.client)
        return np.asarray(position)

    def _distance_to_cube(self) -> float:
        return float(np.linalg.norm(self._get_grasp_center() - self._get_cube_pos()))

    def _get_observation(self) -> np.ndarray:
        joints = self._get_joint_positions() / np.pi
        opening = np.array([self._get_gripper_opening()])
        grasp_center = self._get_grasp_center() / 1.5
        cube_position = self._get_cube_pos() / 1.5
        cube_error = (self._get_cube_pos() - self._get_grasp_center()) / 0.5
        cube_velocity = np.asarray(p.getBaseVelocity(
            self.cube_id, physicsClientId=self.client)[0]) / 2.0
        contacts = np.asarray(self._finger_contacts(), dtype=np.float64)
        grasp_state = np.array([float(self.is_grasped)])
        lift_height = max(
            0.0, float(self._get_cube_pos()[2] - self.cube_start_pos[2]))
        lift_progress = np.array([lift_height / TEACHER_LIFT_HEIGHT])
        observation = np.concatenate([
            joints,
            opening,
            grasp_center,
            cube_position,
            cube_error,
            cube_velocity,
            contacts,
            grasp_state,
            lift_progress,
        ])
        return np.clip(observation, -5.0, 5.0).astype(np.float32)

    def _info(self, distance: float) -> dict[str, float | bool]:
        left_contact, right_contact = self._finger_contacts()
        lift_height = max(
            0.0, float(self._get_cube_pos()[2] - self.cube_start_pos[2]))
        success = self.hold_steps >= GRASP_HOLD_STEPS
        return {
            "distance": float(distance),
            "lift_height": lift_height,
            "left_contact": left_contact,
            "right_contact": right_contact,
            "grasped": self.is_grasped,
            "is_success": bool(success),
            "difficulty": self.difficulty,
        }

    def close(self):
        if getattr(self, "client", -1) >= 0 and p.isConnected(self.client):
            if self.grasp_constraint is not None:
                self._release_cube()
            p.disconnect(self.client)
            self.client = -1


def build_grasp_td3(env: RobotGraspEnv, seed: int) -> GuidedTD3:
    action_dim = int(np.prod(env.action_space.shape))
    noise_sigma = np.full(action_dim, 0.05)
    noise_sigma[-1] = 0.10
    return GuidedTD3(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=300_000,
        learning_starts=0,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        action_noise=NormalActionNoise(
            mean=np.zeros(action_dim), sigma=noise_sigma),
        policy_delay=2,
        target_policy_noise=0.1,
        target_noise_clip=0.2,
        policy_kwargs={"net_arch": [256, 256]},
        seed=seed,
        verbose=0,
        device="auto",
    )


def evaluate_grasp(
    model: TD3,
    *,
    episodes: int,
    seed: int,
    difficulty: float = 1.0,
) -> dict[str, float | int]:
    """Evaluate RL alone; IK is never called in this function."""

    env = RobotGraspEnv(difficulty=difficulty)
    successes = 0
    grasps = 0
    bilateral_contacts = 0
    approaches = 0
    lift_heights: list[float] = []
    minimum_distances: list[float] = []
    episode_steps: list[int] = []
    try:
        for episode in range(episodes):
            observation, _ = env.reset(seed=seed + episode)
            best_lift = 0.0
            minimum_distance = float("inf")
            ever_grasped = False
            ever_bilateral_contact = False
            info: dict[str, float | bool] = {}
            for step in range(env.max_steps):
                action, _ = model.predict(observation, deterministic=True)
                observation, _, terminated, truncated, info = env.step(action)
                best_lift = max(best_lift, float(info["lift_height"]))
                minimum_distance = min(
                    minimum_distance, float(info["distance"]))
                ever_grasped = ever_grasped or bool(info["grasped"])
                ever_bilateral_contact = (
                    ever_bilateral_contact
                    or bool(info["left_contact"] and info["right_contact"])
                )
                if terminated or truncated:
                    break
            successes += bool(info.get("is_success", False))
            grasps += ever_grasped
            bilateral_contacts += ever_bilateral_contact
            approaches += minimum_distance < FINE_ALIGNMENT_DISTANCE
            lift_heights.append(best_lift)
            minimum_distances.append(minimum_distance)
            episode_steps.append(step + 1)
    finally:
        env.close()

    return {
        "episodes": episodes,
        "successes": int(successes),
        "success_rate": successes / episodes,
        "grasp_rate": grasps / episodes,
        "bilateral_contact_rate": bilateral_contacts / episodes,
        "approach_rate": approaches / episodes,
        "mean_min_distance_m": float(np.mean(minimum_distances)),
        "mean_lift_height_m": float(np.mean(lift_heights)),
        "p95_lift_height_m": float(np.percentile(lift_heights, 95)),
        "mean_steps": float(np.mean(episode_steps)),
    }


def print_grasp_evaluation(label: str, metrics: dict[str, float | int]) -> None:
    print(
        f"[{label}] 抓取成功率={metrics['success_rate']:.1%} "
        f"({metrics['successes']}/{metrics['episodes']}) | "
        f"接近率={metrics['approach_rate']:.1%} | "
        f"双指接触率={metrics['bilateral_contact_rate']:.1%} | "
        f"夹住率={metrics['grasp_rate']:.1%} | "
        f"平均抬高={metrics['mean_lift_height_m'] * 100:.1f}cm | "
        f"平均步数={metrics['mean_steps']:.1f}",
        flush=True,
    )


def grasp_checkpoint_score(
    metrics: dict[str, float | int],
) -> tuple[float, float, float, float, float]:
    """Rank partial progress when multiple policies have equal success rates."""

    return (
        float(metrics["success_rate"]),
        float(metrics["grasp_rate"]),
        float(metrics["bilateral_contact_rate"]),
        float(metrics["mean_lift_height_m"]),
        float(metrics["approach_rate"]),
    )


def train_grasp(config: GraspTrainingConfig) -> dict:
    print("=" * 72)
    print("  Demo 2 / Grasp: IK接近 + DAgger + TD3+BC 接触抓取")
    print("  最终评估仅使用8维RL策略：7个手臂关节 + 夹爪开合")
    print("=" * 72, flush=True)

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    train_env = RobotGraspEnv(difficulty=config.curriculum[0])
    data_env = RobotGraspEnv(difficulty=1.0)
    model = build_grasp_td3(train_env, config.seed)
    history: list[dict] = []
    best_success_rate = -1.0
    best_checkpoint_score = (-1.0,) * 5

    try:
        print(f"\n[1/3] 收集 {config.expert_steps} 条抓取示范...", flush=True)
        expert_data = collect_demonstrations(
            data_env,
            config.expert_steps,
            seed=config.seed,
            noise_scale=0.02,
        )
        bc_loss = pretrain_actor(
            model,
            expert_data,
            epochs=config.bc_epochs,
            seed=config.seed,
            balance_phases=True,
            action_weights=GRASP_BC_ACTION_WEIGHTS,
        )
        add_to_replay_buffer(model, expert_data)
        print(f"      行为克隆完成，loss={bc_loss:.6f}", flush=True)
        metrics = evaluate_grasp(
            model, episodes=config.eval_episodes, seed=config.seed + 100_000)
        print_grasp_evaluation("行为克隆", metrics)
        history.append({"stage": "behavior_cloning", **metrics})
        best_success_rate = float(metrics["success_rate"])
        best_checkpoint_score = grasp_checkpoint_score(metrics)
        model.save(BEST_GRASP_MODEL_PATH)

        datasets = [expert_data]
        combined = expert_data
        print(f"\n[2/3] DAgger {config.dagger_iterations} 轮...", flush=True)
        for iteration in range(config.dagger_iterations):
            teacher_mix = max(0.6 - 0.2 * iteration, 0.1)
            dagger_data = collect_demonstrations(
                data_env,
                config.dagger_steps,
                seed=config.seed + 10_000 * (iteration + 1),
                model=model,
                teacher_mix=teacher_mix,
                noise_scale=0.015,
            )
            datasets.append(dagger_data)
            combined = Demonstrations.concatenate(datasets)
            bc_loss = pretrain_actor(
                model,
                combined,
                epochs=config.dagger_epochs,
                seed=config.seed + iteration + 1,
                balance_phases=True,
                action_weights=GRASP_BC_ACTION_WEIGHTS,
            )
            add_to_replay_buffer(model, dagger_data)
            metrics = evaluate_grasp(
                model,
                episodes=config.eval_episodes,
                seed=config.seed + 100_000,
            )
            print_grasp_evaluation(f"DAgger {iteration + 1}", metrics)
            print(
                f"      数据量={len(combined)}，teacher_mix={teacher_mix:.2f}，"
                f"loss={bc_loss:.6f}",
                flush=True,
            )
            history.append({"stage": f"dagger_{iteration + 1}", **metrics})
            checkpoint_score = grasp_checkpoint_score(metrics)
            if checkpoint_score > best_checkpoint_score:
                best_success_rate = float(metrics["success_rate"])
                best_checkpoint_score = checkpoint_score
                model.save(BEST_GRASP_MODEL_PATH)

        model.set_parameters(
            BEST_GRASP_MODEL_PATH, exact_match=True, device=model.device)
        model.actor_target.load_state_dict(model.actor.state_dict())
        print("\n[3/3] TD3+BC在线接触强化学习微调...", flush=True)
        model.set_expert_data(
            combined,
            start=1.2,
            end=0.30,
            decay_updates=sum(config.rl_phase_steps),
            balance_phases=True,
            action_weights=GRASP_BC_ACTION_WEIGHTS,
            q_filter=True,
        )
        print(
            f"      先用 {config.critic_warmup_steps} 次抓取回放预热critic...",
            flush=True,
        )
        warmup_critic(model, config.critic_warmup_steps)
        total_rl_steps = 0
        for phase, (difficulty, phase_steps) in enumerate(
            zip(config.curriculum, config.rl_phase_steps), start=1
        ):
            train_env.set_difficulty(difficulty)
            model.learn(
                total_timesteps=phase_steps,
                reset_num_timesteps=False,
                progress_bar=False,
            )
            total_rl_steps += phase_steps
            metrics = evaluate_grasp(
                model,
                episodes=config.eval_episodes,
                seed=config.seed + 100_000,
                # Select checkpoints on the same full-workspace test set;
                # curriculum difficulty affects data collection only.
                difficulty=1.0,
            )
            label = f"TD3+BC阶段{phase} difficulty={difficulty:.2f}"
            print_grasp_evaluation(label, metrics)
            history.append({
                "stage": f"td3_bc_{phase}",
                "difficulty": difficulty,
                "rl_steps": total_rl_steps,
                **metrics,
            })
            checkpoint_score = grasp_checkpoint_score(metrics)
            if checkpoint_score > best_checkpoint_score:
                best_success_rate = float(metrics["success_rate"])
                best_checkpoint_score = checkpoint_score
                model.save(BEST_GRASP_MODEL_PATH)
            elif metrics["success_rate"] + 0.10 < best_success_rate:
                model.set_parameters(
                    BEST_GRASP_MODEL_PATH, exact_match=True, device=model.device)
                model.rl_weight *= 0.5
                print("      策略退化，已恢复最佳权重并降低RL更新权重。", flush=True)
            if (
                total_rl_steps >= 10_000
                and metrics["success_rate"] >= config.target_success_rate
            ):
                print("      已达到抓取成功率目标，提前停止课程训练。", flush=True)
                break

        model.set_parameters(
            BEST_GRASP_MODEL_PATH, exact_match=True, device=model.device)
        model.save(GRASP_MODEL_PATH)
        final_metrics = evaluate_grasp(
            model,
            episodes=config.final_eval_episodes,
            seed=config.seed + 200_000,
            difficulty=1.0,
        )
        print_grasp_evaluation("最终RL独立测试集", final_metrics)
        result = {
            "config": asdict(config),
            "history": history,
            "final": final_metrics,
            "model_path": str(GRASP_MODEL_PATH),
            "best_model_path": str(BEST_GRASP_MODEL_PATH),
        }
        GRASP_METRICS_PATH.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n抓取模型已保存到 {GRASP_MODEL_PATH}")
        print(f"训练指标已保存到 {GRASP_METRICS_PATH}")
        return result
    finally:
        data_env.close()
        train_env.close()
