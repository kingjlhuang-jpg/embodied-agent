"""IK-guided TD3+BC training for contact-rich Kuka cube grasping.

The inverse-kinematics teacher handles the long-horizon arm motion while the
policy learns the complete 8-dimensional action: seven arm joints plus one
continuous gripper command.  The baseline can use a finite-force grasp assist;
physical mode disables it completely and requires sustained bilateral contact,
bounded motor force, and friction to lift a heavier dynamic cube.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
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
    collect_failure_demonstrations,
    pretrain_actor,
    warmup_critic,
)


ARM_ACTION_SCALE = 0.08
TABLE_TOP_Z = 0.58
CUBE_HALF_EXTENT = 0.025
CUBE_REST_Z = TABLE_TOP_Z + CUBE_HALF_EXTENT
GRASP_SUCCESS_HEIGHT = 0.035
GRASP_HOLD_STEPS = 8
PHYSICAL_GRASP_CONFIRM_STEPS = 3
PHYSICAL_CONTACT_LOSS_STEPS = 2
GRASP_DISTANCE = 0.030
SAFE_CLOSE_DISTANCE = 0.035
FINE_ALIGNMENT_DISTANCE = 0.075
GRIPPER_OPEN_ANGLE = 0.30
GRIPPER_CLOSED_ANGLE = -0.02
GRASP_CENTER_Z_OFFSET = 0.005
TEACHER_LIFT_HEIGHT = 0.08
BASE_OBSERVATION_SIZE = 24
ACTION_HISTORY_STEPS = 2
GRASP_BC_ACTION_WEIGHTS = np.array([1.0] * 7 + [4.0], dtype=np.float32)

GRASP_MODEL_PATH = Path("grasp_policy.zip")
BEST_GRASP_MODEL_PATH = Path("best_grasp_policy.zip")
GRASP_METRICS_PATH = Path("grasp_training_metrics.json")
ROBUST_GRASP_MODEL_PATH = Path("robust_grasp_policy.zip")
BEST_ROBUST_GRASP_MODEL_PATH = Path("best_robust_grasp_policy.zip")
ROBUST_GRASP_METRICS_PATH = Path("robust_grasp_training_metrics.json")
PHYSICAL_GRASP_MODEL_PATH = Path("physical_grasp_policy.zip")
BEST_PHYSICAL_GRASP_MODEL_PATH = Path("best_physical_grasp_policy.zip")
PHYSICAL_GRASP_METRICS_PATH = Path("physical_grasp_training_metrics.json")
POSE_PHYSICAL_GRASP_MODEL_PATH = Path("pose_physical_grasp_policy.zip")
BEST_POSE_PHYSICAL_GRASP_MODEL_PATH = Path(
    "best_pose_physical_grasp_policy.zip")
POSE_PHYSICAL_GRASP_METRICS_PATH = Path(
    "pose_physical_grasp_training_metrics.json")


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
    randomization_curriculum: tuple[float, ...] = (0.0, 0.0, 0.0)
    pose_randomization_curriculum: tuple[float, ...] = (0.0, 0.0, 0.0)
    eval_episodes: int = 50
    final_eval_episodes: int = 100
    target_success_rate: float = 0.95
    rl_weight: float = 0.01
    bc_start: float = 1.2
    bc_end: float = 0.30
    hold_coefficient: float = 1.0
    anchor_coefficient: float = 0.0
    actor_learning_rate: float | None = None
    grasp_constraint_force: float = 300.0
    failure_only_dagger: bool = False

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
            randomization_curriculum=(0.0,),
            pose_randomization_curriculum=(0.0,),
            eval_episodes=10,
            final_eval_episodes=20,
            target_success_rate=1.0,
        )

    @classmethod
    def robust(cls, seed: int = 20260901) -> "GraspTrainingConfig":
        """Full-domain training and a 1,000-episode held-out evaluation."""

        return cls(
            seed=seed,
            expert_steps=20_000,
            bc_epochs=0,
            dagger_iterations=3,
            dagger_steps=4_000,
            dagger_epochs=0,
            critic_warmup_steps=2_000,
            rl_phase_steps=(10_000, 15_000, 25_000),
            curriculum=(1.0, 1.0, 1.0),
            randomization_curriculum=(0.35, 0.70, 1.0),
            pose_randomization_curriculum=(0.0, 0.0, 0.0),
            eval_episodes=75,
            final_eval_episodes=1_000,
            target_success_rate=0.95,
            rl_weight=0.002,
            bc_start=0.20,
            bc_end=0.05,
            hold_coefficient=5.0,
            anchor_coefficient=20.0,
            actor_learning_rate=1e-6,
        )

    @classmethod
    def quick_robust(cls, seed: int = 20260901) -> "GraspTrainingConfig":
        """Smoke-test the randomized pipeline without claiming convergence."""

        return cls(
            seed=seed,
            expert_steps=2_000,
            bc_epochs=0,
            dagger_iterations=1,
            dagger_steps=1_000,
            dagger_epochs=0,
            critic_warmup_steps=100,
            rl_phase_steps=(1_000,),
            curriculum=(1.0,),
            randomization_curriculum=(0.35,),
            pose_randomization_curriculum=(0.0,),
            eval_episodes=10,
            final_eval_episodes=20,
            target_success_rate=1.0,
            rl_weight=0.002,
            bc_start=0.20,
            bc_end=0.05,
            hold_coefficient=5.0,
            anchor_coefficient=20.0,
            actor_learning_rate=1e-6,
        )

    @classmethod
    def physical(cls, seed: int = 20260901) -> "GraspTrainingConfig":
        """Train a heavy cube grasp using only finger contact and friction."""

        return cls(
            seed=seed,
            expert_steps=20_000,
            bc_epochs=0,
            dagger_iterations=3,
            dagger_steps=4_000,
            dagger_epochs=0,
            critic_warmup_steps=2_000,
            rl_phase_steps=(10_000, 15_000, 25_000),
            curriculum=(1.0, 1.0, 1.0),
            randomization_curriculum=(0.0, 0.0, 0.0),
            pose_randomization_curriculum=(0.0, 0.0, 0.0),
            eval_episodes=75,
            final_eval_episodes=1_000,
            target_success_rate=0.90,
            rl_weight=0.002,
            bc_start=0.50,
            bc_end=0.10,
            hold_coefficient=3.0,
            anchor_coefficient=20.0,
            actor_learning_rate=1e-7,
            grasp_constraint_force=0.0,
            failure_only_dagger=True,
        )

    @classmethod
    def quick_physical(cls, seed: int = 20260901) -> "GraspTrainingConfig":
        """Smoke-test the constraint-free physical grasp pipeline."""

        return cls(
            seed=seed,
            expert_steps=2_000,
            bc_epochs=0,
            dagger_iterations=1,
            dagger_steps=1_000,
            dagger_epochs=0,
            critic_warmup_steps=100,
            rl_phase_steps=(1_000,),
            curriculum=(1.0,),
            randomization_curriculum=(0.0,),
            pose_randomization_curriculum=(0.0,),
            eval_episodes=10,
            final_eval_episodes=20,
            target_success_rate=1.0,
            rl_weight=0.002,
            bc_start=0.50,
            bc_end=0.10,
            hold_coefficient=3.0,
            anchor_coefficient=20.0,
            actor_learning_rate=1e-7,
            grasp_constraint_force=0.0,
            failure_only_dagger=True,
        )

    @classmethod
    def physical_pose(cls, seed: int = 20260901) -> "GraspTrainingConfig":
        """Train friction grasping over wider XY positions and cube yaws."""

        return replace(
            cls.physical(seed),
            # Keep one continuous optimization trajectory while progressively
            # expanding the pose distribution.  Earlier versions evaluated
            # every partial-curriculum stage on the full workspace and rolled
            # it back immediately, so later stages repeatedly restarted from
            # the fixed-pose warm start.
            curriculum=(1.0, 1.0, 1.0, 1.0),
            randomization_curriculum=(0.0, 0.0, 0.0, 0.0),
            pose_randomization_curriculum=(0.50, 0.75, 1.0, 1.0),
            rl_phase_steps=(15_000, 25_000, 35_000, 50_000),
            eval_episodes=200,
            target_success_rate=0.70,
            actor_learning_rate=2e-6,
            anchor_coefficient=2.0,
        )

    @classmethod
    def quick_physical_pose(cls, seed: int = 20260901) -> "GraspTrainingConfig":
        """Smoke-test the randomized-pose physical grasp pipeline."""

        return replace(
            cls.quick_physical(seed),
            pose_randomization_curriculum=(0.50,),
        )


class RobotGraspEnv(gym.Env):
    """Kuka iiwa + WSG50 task with assisted or friction-only grasping."""

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
        randomization: float = 0.0,
        pose_randomization: float = 0.0,
        observe_action_history: bool = False,
        observe_physical_residual: bool = False,
        observe_cube_yaw: bool = False,
        grasp_constraint_force: float = 300.0,
        max_steps: int = 300,
    ):
        super().__init__()
        if render is not None:
            render_mode = "human" if render else None
        if render_mode not in (None, "human"):
            raise ValueError(f"unsupported render_mode: {render_mode}")

        self.render_mode = render_mode
        self.difficulty = float(np.clip(difficulty, 0.0, 1.0))
        self.randomization = float(np.clip(randomization, 0.0, 1.0))
        self.pose_randomization = float(
            np.clip(pose_randomization, 0.0, 1.0))
        self.observe_action_history = bool(observe_action_history)
        self.observe_physical_residual = bool(observe_physical_residual)
        self.observe_cube_yaw = bool(observe_cube_yaw)
        self.grasp_constraint_force = max(float(grasp_constraint_force), 0.0)
        self.max_steps = int(max_steps)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(8,), dtype=np.float32)
        observation_size = BASE_OBSERVATION_SIZE
        if self.observe_action_history:
            observation_size += ACTION_HISTORY_STEPS * self.action_space.shape[0]
        if self.observe_physical_residual:
            observation_size += BASE_OBSERVATION_SIZE
        if self.observe_cube_yaw:
            observation_size += 2
        self.observation_space = spaces.Box(
            -5.0, 5.0, shape=(observation_size,), dtype=np.float32)

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
        self.physical_grasped = False
        self.grasp_contact_steps = 0
        self.contact_loss_steps = 0
        self.cube_start_pos = np.zeros(3, dtype=np.float64)
        self.lift_target_pos = np.zeros(3, dtype=np.float64)
        self.step_count = 0
        self.hold_steps = 0
        self.previous_distance = 0.0
        self.previous_yaw_error = 0.0
        self.previous_cube_height = CUBE_REST_Z
        self.previous_action = np.zeros(8, dtype=np.float32)
        self.command_history = np.zeros(
            (ACTION_HISTORY_STEPS, 8), dtype=np.float32)
        self.action_buffer: list[np.ndarray] = []
        self.action_delay_steps = 0
        self.action_noise_std = np.zeros(8, dtype=np.float32)
        self.observation_bias = np.zeros(20, dtype=np.float32)
        self.observation_noise_std = 0.0
        self.cube_half_extent = CUBE_HALF_EXTENT
        self.cube_mass = 0.03
        self.cube_friction = 2.5
        self.cube_yaw = 0.0
        self.finger_friction = 3.0
        self.gripper_motor_force = 180.0
        self.motor_force = 300.0
        self.motor_velocity = 1.5
        self.gripper_command = -1.0
        self.teacher_closing = False
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
        return self.grasp_constraint is not None or self.physical_grasped

    @property
    def uses_grasp_constraint(self) -> bool:
        return self.grasp_constraint_force > 0.0

    def set_difficulty(self, difficulty: float) -> None:
        self.difficulty = float(np.clip(difficulty, 0.0, 1.0))

    def set_randomization(self, randomization: float) -> None:
        """Set domain-randomization severity for subsequent resets."""

        self.randomization = float(np.clip(randomization, 0.0, 1.0))

    def set_pose_randomization(self, pose_randomization: float) -> None:
        """Set object position/yaw randomization for subsequent resets."""

        self.pose_randomization = float(
            np.clip(pose_randomization, 0.0, 1.0))

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        del options
        self._sample_domain_parameters()

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
                lateralFriction=self.finger_friction,
                spinningFriction=0.15,
                rollingFriction=0.01,
                physicsClientId=self.client,
            )

        cube_x_span = 0.055 + 0.035 * self.pose_randomization
        cube_y_span = 0.075 + 0.035 * self.pose_randomization
        cube_x = 0.40 + self.difficulty * self.np_random.uniform(
            -cube_x_span, cube_x_span)
        cube_y = self.difficulty * self.np_random.uniform(
            -cube_y_span, cube_y_span)
        cube_position = np.array([
            cube_x,
            cube_y,
            TABLE_TOP_Z + self.cube_half_extent,
        ])
        cube_collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[self.cube_half_extent] * 3,
            physicsClientId=self.client,
        )
        cube_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[self.cube_half_extent] * 3,
            rgbaColor=[0.9, 0.05, 0.05, 1.0],
            physicsClientId=self.client,
        )
        self.cube_id = p.createMultiBody(
            baseMass=self.cube_mass,
            baseCollisionShapeIndex=cube_collision,
            baseVisualShapeIndex=cube_visual,
            basePosition=cube_position,
            baseOrientation=p.getQuaternionFromEuler([0.0, 0.0, self.cube_yaw]),
            physicsClientId=self.client,
        )
        p.changeDynamics(
            self.cube_id,
            -1,
            lateralFriction=self.cube_friction,
            spinningFriction=0.1,
            rollingFriction=0.001,
            restitution=0.0,
            physicsClientId=self.client,
        )

        self.grasp_constraint = None
        self.physical_grasped = False
        self.grasp_contact_steps = 0
        self.contact_loss_steps = 0
        self.step_count = 0
        self.hold_steps = 0
        self.gripper_command = -1.0
        self.teacher_closing = False
        self.previous_action.fill(0.0)
        self.command_history.fill(0.0)
        self.action_buffer = [
            np.zeros(8, dtype=np.float32)
            for _ in range(self.action_delay_steps)
        ]
        for _ in range(20):
            self._hold_ready_pose()
            self._set_gripper(-1.0)
            p.stepSimulation(physicsClientId=self.client)

        self.cube_start_pos = self._get_cube_pos()
        self.lift_target_pos = self.cube_start_pos + np.array(
            [0.0, 0.0, TEACHER_LIFT_HEIGHT])
        self.previous_distance = self._distance_to_cube()
        self.previous_yaw_error = abs(self._get_gripper_cube_yaw_error())
        self.previous_cube_height = float(self.cube_start_pos[2])
        info = self._info(self.previous_distance)
        return self._get_observation(), info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != self.action_space.shape:
            raise ValueError(f"expected action shape (8,), got {action.shape}")
        action = np.clip(action, -1.0, 1.0)
        self.command_history[:-1] = self.command_history[1:]
        self.command_history[-1] = action
        self.action_buffer.append(action.copy())
        action = self.action_buffer.pop(0)
        if np.any(self.action_noise_std):
            action = np.clip(
                action + self.np_random.normal(
                    0.0, self.action_noise_std, size=action.shape),
                -1.0,
                1.0,
            ).astype(np.float32)

        released_cube = False
        if self.is_grasped and action[7] < -0.5:
            self._clear_grasp_state()
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
                force=self.motor_force,
                maxVelocity=self.motor_velocity,
                physicsClientId=self.client,
            )
        self._set_gripper(float(action[7]))

        for _ in range(12):
            p.stepSimulation(physicsClientId=self.client)

        left_contact, right_contact = self._finger_contacts()
        newly_grasped = False
        if self.uses_grasp_constraint:
            if (
                not self.is_grasped
                and action[7] > 0.35
                and left_contact
                and right_contact
                and self._distance_to_cube() < 0.055
            ):
                self._attach_cube()
                newly_grasped = True
        else:
            newly_grasped, contact_released = self._update_physical_grasp(
                left_contact,
                right_contact,
                float(action[7]),
            )
            released_cube = released_cube or contact_released

        self.step_count += 1
        distance = self._distance_to_cube()
        cube_height = float(self._get_cube_pos()[2])
        lift_height = max(0.0, cube_height - float(self.cube_start_pos[2]))
        approach_progress = self.previous_distance - distance
        lift_progress = cube_height - self.previous_cube_height
        smoothness_cost = np.square(action - self.previous_action).sum()
        yaw_error = self._get_gripper_cube_yaw_error()
        yaw_error_abs = abs(yaw_error)
        yaw_progress = self.previous_yaw_error - yaw_error_abs
        yaw_alignment = 0.5 * (math.cos(4.0 * yaw_error) + 1.0)
        alignment_gate = max(0.0, 1.0 - distance / FINE_ALIGNMENT_DISTANCE)

        # Reward stages: approach < bilateral contact < lift < stable hold.
        reward = (
            8.0 * approach_progress
            - 0.15 * distance
            - 0.0015 * np.square(action[:7]).sum()
            - 0.001 * smoothness_cost
        )
        reward += 0.08 * (int(left_contact) + int(right_contact))
        if self.pose_randomization > 0.0 and not self.is_grasped:
            # Reward reducing the square-symmetric yaw error instead of
            # paying an absolute alignment bonus that can be exploited by
            # waiting near the cube.  Closing while misaligned is explicitly
            # discouraged so the policy learns a pre-grasp rotation phase.
            reward += 10.0 * yaw_progress
            reward -= (
                0.80
                * max(float(action[7]), 0.0)
                * alignment_gate
                * (1.0 - yaw_alignment)
            )
        premature_close = (
            max(float(action[7]), 0.0)
            * max(distance - SAFE_CLOSE_DISTANCE, 0.0)
            / 0.05
        )
        reward -= 0.25 * premature_close
        if left_contact and right_contact:
            reward += 0.35
            if self.pose_randomization > 0.0:
                reward += 0.40 * yaw_alignment
        if newly_grasped:
            reward += 4.0
        if self.is_grasped:
            reward += 0.20
            reward += 0.60 * min(
                lift_height / GRASP_SUCCESS_HEIGHT, 1.0)
        reward += 45.0 * max(lift_progress, -0.005)
        if lift_height >= 0.02:
            reward += 1.0

        stable_lift = (
            self.is_grasped
            and left_contact
            and right_contact
            and lift_height >= GRASP_SUCCESS_HEIGHT
        )
        self.hold_steps = self.hold_steps + 1 if stable_lift else 0
        grasp_succeeded = self.hold_steps >= GRASP_HOLD_STEPS
        if released_cube:
            reward -= 8.0
        terminated = grasp_succeeded or released_cube
        if grasp_succeeded:
            reward += 25.0
        truncated = self.step_count >= self.max_steps

        self.previous_distance = distance
        self.previous_yaw_error = yaw_error_abs
        self.previous_cube_height = cube_height
        self.previous_action = action.copy()
        return (
            self._get_observation(),
            float(reward),
            bool(terminated),
            bool(truncated),
            self._info(distance),
        )

    def get_ik_action(
        self,
        grasp_offset: np.ndarray | None = None,
    ) -> np.ndarray:
        """Scripted teacher: align open fingers, close on contact, then lift."""

        if grasp_offset is None:
            grasp_offset = np.zeros(3, dtype=np.float64)
        else:
            grasp_offset = np.asarray(grasp_offset, dtype=np.float64)
            if grasp_offset.shape != (3,):
                raise ValueError("grasp_offset must have shape (3,)")
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
                [0.0, 0.0, GRASP_CENTER_Z_OFFSET]) + grasp_offset
            target_position = (
                end_effector_position
                + desired_grasp_center
                - self._get_grasp_center()
            )
            yaw_ready = (
                self.pose_randomization <= 0.0
                or abs(self._get_gripper_cube_yaw_error())
                < math.radians(8.0)
            )
            if distance < GRASP_DISTANCE and yaw_ready:
                self.teacher_closing = True
            elif distance > 2.0 * GRASP_DISTANCE or not yaw_ready:
                self.teacher_closing = False
            gripper_action = 1.0 if self.teacher_closing else -1.0

        rest_poses = [
            p.getJointState(
                self.robot_id, joint, physicsClientId=self.client)[0]
            for joint in self._ik_joints
        ]
        target_orientation = self._downward_orientation
        if self.pose_randomization > 0.0:
            # Strictly vertical IK approaches a wrist singularity at the far
            # edge of this Kuka workspace.  Gradually tilt toward the robot by
            # at most 20 degrees only beyond x=0.43 m, which preserves the
            # central grasp while producing reachable labels at the edge.
            tilt_blend = float(np.clip(
                (cube_position[0] - 0.43) / 0.03, 0.0, 1.0))
            downward_orientation = p.getQuaternionFromEuler([
                0.0,
                -math.pi - math.radians(20.0) * tilt_blend,
                0.0,
            ])
            # A square has four equivalent face-normal directions.  Command
            # only the nearest correction in [-45, 45] degrees instead of
            # rotating the wrist through the cube's full visual yaw.
            yaw_correction = self._wrap_square_yaw(self.cube_yaw)
            yaw_orientation = p.getQuaternionFromEuler(
                [0.0, 0.0, yaw_correction])
            target_orientation = p.multiplyTransforms(
                [0.0, 0.0, 0.0],
                yaw_orientation,
                [0.0, 0.0, 0.0],
                downward_orientation,
            )[1]
        solution = p.calculateInverseKinematics(
            self.robot_id,
            self.END_EFFECTOR_LINK,
            target_position,
            target_orientation,
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

    def _sample_domain_parameters(self) -> None:
        """Sample one reproducible physics/sensor domain for this episode."""

        severity = self.randomization

        def blend(base: float, low: float, high: float) -> float:
            sampled = float(self.np_random.uniform(low, high))
            return (1.0 - severity) * base + severity * sampled

        self.cube_half_extent = blend(CUBE_HALF_EXTENT, 0.020, 0.030)
        if self.uses_grasp_constraint:
            self.cube_mass = blend(0.03, 0.02, 0.08)
            self.cube_friction = blend(2.5, 0.8, 3.5)
            self.finger_friction = blend(3.0, 1.5, 4.0)
            self.gripper_motor_force = 180.0
        else:
            # A 5 cm solid object is much heavier than the original 30 g
            # training cube.  Moderate rubber/plastic friction and bounded
            # finger force make off-centre pinches slip naturally.
            self.cube_mass = blend(0.20, 0.10, 0.30)
            self.cube_friction = blend(0.55, 0.35, 0.80)
            self.finger_friction = blend(1.00, 0.70, 1.30)
            self.gripper_motor_force = blend(60.0, 45.0, 75.0)
        if self.pose_randomization > 0.0:
            # Sample a full visual rotation while curriculum severity controls
            # only the physically unique residual within one 90-degree square
            # symmetry sector.  At severity 1 this is uniform over [-pi, pi].
            raw_yaw = float(self.np_random.uniform(-math.pi, math.pi))
            quadrant = round(raw_yaw / (math.pi / 2.0)) * (math.pi / 2.0)
            residual = self._wrap_square_yaw(raw_yaw)
            self.cube_yaw = float(
                (quadrant + self.pose_randomization * residual + math.pi)
                % (2.0 * math.pi)
                - math.pi
            )
        else:
            self.cube_yaw = severity * float(
                self.np_random.uniform(-math.pi / 4, math.pi / 4))
        self.motor_force = blend(300.0, 240.0, 360.0)
        self.motor_velocity = blend(1.5, 1.2, 1.8)

        max_delay = int(math.ceil(2.0 * severity)) if severity > 0 else 0
        self.action_delay_steps = int(
            self.np_random.integers(0, max_delay + 1))
        self.action_noise_std = np.array(
            [0.012 * severity] * 7 + [0.025 * severity],
            dtype=np.float32,
        )
        self.observation_bias = self.np_random.normal(
            0.0, 0.0015 * severity, size=20).astype(np.float32)
        self.observation_noise_std = 0.0015 * severity

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
                force=self.motor_force,
                maxVelocity=self.motor_velocity,
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
                force=self.gripper_motor_force,
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
            maxForce=self.grasp_constraint_force,
            physicsClientId=self.client,
        )

    def _update_physical_grasp(
        self,
        left_contact: bool,
        right_contact: bool,
        gripper_action: float,
    ) -> tuple[bool, bool]:
        """Track sustained bilateral contact without attaching the cube."""

        was_grasped = self.physical_grasped
        valid_contact = (
            gripper_action > 0.10
            and left_contact
            and right_contact
            and self._distance_to_cube() < 0.055
        )
        if valid_contact:
            self.grasp_contact_steps += 1
            self.contact_loss_steps = 0
            if self.grasp_contact_steps >= PHYSICAL_GRASP_CONFIRM_STEPS:
                self.physical_grasped = True
        else:
            self.grasp_contact_steps = 0
            if self.physical_grasped:
                self.contact_loss_steps += 1
                if (
                    gripper_action < -0.5
                    or self.contact_loss_steps >= PHYSICAL_CONTACT_LOSS_STEPS
                ):
                    self.physical_grasped = False
                    self.contact_loss_steps = 0
        return (not was_grasped and self.physical_grasped), (
            was_grasped and not self.physical_grasped)

    def _clear_grasp_state(self) -> None:
        self._release_cube()
        self.physical_grasped = False
        self.grasp_contact_steps = 0
        self.contact_loss_steps = 0

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

    def _get_gripper_cube_yaw_error(self) -> float:
        """Signed jaw-to-cube yaw error, respecting square symmetry."""

        tips = [
            np.asarray(p.getLinkState(
                self.robot_id,
                link,
                computeForwardKinematics=True,
                physicsClientId=self.client,
            )[0])
            for link in (10, 13)
        ]
        jaw_axis = tips[1] - tips[0]
        gripper_yaw = math.atan2(float(jaw_axis[1]), float(jaw_axis[0]))
        return self._wrap_square_yaw(gripper_yaw - self.cube_yaw)

    @staticmethod
    def _wrap_square_yaw(yaw: float) -> float:
        """Wrap an angle into the unique [-45, 45) degree square sector."""

        period = math.pi / 2.0
        return float((yaw + period / 2.0) % period - period / 2.0)

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
        if self.observation_noise_std > 0:
            observation[:20] += self.observation_bias
            observation[:20] += self.np_random.normal(
                0.0, self.observation_noise_std, size=20)
        if self.observe_action_history:
            observation = np.concatenate([
                observation,
                self.command_history.reshape(-1),
            ])
        if self.observe_physical_residual or self.observe_cube_yaw:
            interaction_gate = float(
                self._distance_to_cube() < FINE_ALIGNMENT_DISTANCE)
        if self.observe_physical_residual:
            observation = np.concatenate([
                observation,
                observation[:BASE_OBSERVATION_SIZE] * interaction_gate,
            ])
        if self.observe_cube_yaw:
            yaw_error = self._get_gripper_cube_yaw_error()
            observation = np.concatenate([
                observation,
                np.array([
                    math.sin(4.0 * yaw_error),
                    math.cos(4.0 * yaw_error),
                ]),
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
            "randomization": self.randomization,
            "cube_size": 2.0 * self.cube_half_extent,
            "cube_mass": self.cube_mass,
            "cube_friction": self.cube_friction,
            "cube_yaw": self.cube_yaw,
            "gripper_cube_yaw_error": self._get_gripper_cube_yaw_error(),
            "cube_x": float(self.cube_start_pos[0]),
            "cube_y": float(self.cube_start_pos[1]),
            "action_delay_steps": self.action_delay_steps,
            "pose_randomization": self.pose_randomization,
            "finger_friction": self.finger_friction,
            "gripper_motor_force": self.gripper_motor_force,
            "grasp_constraint_force": self.grasp_constraint_force,
            "constraint_active": self.grasp_constraint is not None,
        }

    def close(self):
        if getattr(self, "client", -1) >= 0 and p.isConnected(self.client):
            if self.grasp_constraint is not None:
                self._clear_grasp_state()
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


def warm_start_grasp_model(
    model: GuidedTD3,
    source_path: Path,
) -> None:
    """Expand a grasp policy with zeroed appended inputs without drift."""

    source = GuidedTD3.load(source_path, device=model.device)
    source_observation_size = int(source.observation_space.shape[0])
    target_observation_size = int(model.observation_space.shape[0])
    action_size = int(model.action_space.shape[0])
    if source_observation_size == target_observation_size:
        model.set_parameters(source_path, exact_match=True, device=model.device)
        return
    if target_observation_size <= source_observation_size:
        raise ValueError(
            "unsupported grasp-policy observation expansion: "
            f"{source_observation_size} -> {target_observation_size}"
        )

    def expand_actor(source_actor, target_actor) -> None:
        source_state = source_actor.state_dict()
        target_state = target_actor.state_dict()
        for key, source_value in source_state.items():
            if key == "mu.0.weight":
                target_state[key].zero_()
                target_state[key][:, :source_observation_size].copy_(source_value)
            else:
                target_state[key].copy_(source_value)
        target_actor.load_state_dict(target_state)

    def expand_critic(source_critic, target_critic) -> None:
        source_state = source_critic.state_dict()
        target_state = target_critic.state_dict()
        first_layers = {"qf0.0.weight", "qf1.0.weight"}
        for key, source_value in source_state.items():
            if key in first_layers:
                target_state[key].zero_()
                target_state[key][:, :source_observation_size].copy_(
                    source_value[:, :source_observation_size])
                target_state[key][
                    :, target_observation_size:target_observation_size + action_size
                ].copy_(source_value[
                    :, source_observation_size:source_observation_size + action_size
                ])
            else:
                target_state[key].copy_(source_value)
        target_critic.load_state_dict(target_state)

    expand_actor(source.actor, model.actor)
    expand_actor(source.actor_target, model.actor_target)
    expand_critic(source.critic, model.critic)
    expand_critic(source.critic_target, model.critic_target)


def evaluate_grasp(
    model: TD3,
    *,
    episodes: int,
    seed: int,
    difficulty: float = 1.0,
    randomization: float = 0.0,
    pose_randomization: float = 0.0,
    grasp_constraint_force: float = 300.0,
) -> dict[str, float | int]:
    """Evaluate RL alone; IK is never called in this function."""

    observation_size = int(model.observation_space.shape[0])
    action_history_size = ACTION_HISTORY_STEPS * int(model.action_space.shape[0])
    env = RobotGraspEnv(
        difficulty=difficulty,
        randomization=randomization,
        pose_randomization=pose_randomization,
        observe_action_history=(
            observation_size
            in (
                BASE_OBSERVATION_SIZE + action_history_size,
                2 * BASE_OBSERVATION_SIZE + action_history_size,
                2 * BASE_OBSERVATION_SIZE + action_history_size + 2,
            )
        ),
        observe_physical_residual=(
            observation_size
            in (
                2 * BASE_OBSERVATION_SIZE,
                2 * BASE_OBSERVATION_SIZE + 2,
                2 * BASE_OBSERVATION_SIZE + action_history_size,
                2 * BASE_OBSERVATION_SIZE + action_history_size + 2,
            )
        ),
        observe_cube_yaw=observation_size in (
            2 * BASE_OBSERVATION_SIZE + 2,
            2 * BASE_OBSERVATION_SIZE + action_history_size + 2,
        ),
        grasp_constraint_force=grasp_constraint_force,
    )
    successes = 0
    grasps = 0
    bilateral_contacts = 0
    approaches = 0
    lift_heights: list[float] = []
    minimum_distances: list[float] = []
    episode_steps: list[int] = []
    cube_sizes: list[float] = []
    cube_masses: list[float] = []
    cube_frictions: list[float] = []
    cube_x_positions: list[float] = []
    cube_y_positions: list[float] = []
    cube_yaws: list[float] = []
    action_delays: list[int] = []
    episode_successes: list[bool] = []
    constraint_activations = 0
    try:
        for episode in range(episodes):
            observation, reset_info = env.reset(seed=seed + episode)
            cube_sizes.append(float(reset_info["cube_size"]))
            cube_masses.append(float(reset_info["cube_mass"]))
            cube_frictions.append(float(reset_info["cube_friction"]))
            cube_x_positions.append(float(reset_info["cube_x"]))
            cube_y_positions.append(float(reset_info["cube_y"]))
            cube_yaws.append(float(reset_info["cube_yaw"]))
            action_delays.append(int(reset_info["action_delay_steps"]))
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
                constraint_activations += bool(info["constraint_active"])
                if terminated or truncated:
                    break
            successes += bool(info.get("is_success", False))
            episode_successes.append(bool(info.get("is_success", False)))
            grasps += ever_grasped
            bilateral_contacts += ever_bilateral_contact
            approaches += minimum_distance < FINE_ALIGNMENT_DISTANCE
            lift_heights.append(best_lift)
            minimum_distances.append(minimum_distance)
            episode_steps.append(step + 1)
    finally:
        env.close()

    result: dict[str, float | int] = {
        "episodes": episodes,
        "successes": int(successes),
        "success_rate": successes / episodes,
        "grasp_rate": grasps / episodes,
        "drop_rate": (grasps - successes) / episodes,
        "bilateral_contact_rate": bilateral_contacts / episodes,
        "approach_rate": approaches / episodes,
        "mean_min_distance_m": float(np.mean(minimum_distances)),
        "mean_lift_height_m": float(np.mean(lift_heights)),
        "p95_lift_height_m": float(np.percentile(lift_heights, 95)),
        "mean_steps": float(np.mean(episode_steps)),
        "randomization": randomization,
        "pose_randomization": pose_randomization,
        "grasp_constraint_force": grasp_constraint_force,
        "constraint_activations": int(constraint_activations),
        "cube_size_min_m": float(np.min(cube_sizes)),
        "cube_size_max_m": float(np.max(cube_sizes)),
        "cube_mass_min_kg": float(np.min(cube_masses)),
        "cube_mass_max_kg": float(np.max(cube_masses)),
        "cube_friction_min": float(np.min(cube_frictions)),
        "cube_friction_max": float(np.max(cube_frictions)),
        "cube_x_min_m": float(np.min(cube_x_positions)),
        "cube_x_max_m": float(np.max(cube_x_positions)),
        "cube_y_min_m": float(np.min(cube_y_positions)),
        "cube_y_max_m": float(np.max(cube_y_positions)),
        "cube_yaw_min_rad": float(np.min(cube_yaws)),
        "cube_yaw_max_rad": float(np.max(cube_yaws)),
        "action_delay_max_steps": int(np.max(action_delays)),
    }
    delay_array = np.asarray(action_delays)
    success_array = np.asarray(episode_successes, dtype=np.float32)
    for delay in range(3):
        delay_mask = delay_array == delay
        result[f"delay_{delay}_episodes"] = int(delay_mask.sum())
        result[f"delay_{delay}_success_rate"] = (
            float(success_array[delay_mask].mean())
            if bool(delay_mask.any())
            else 0.0
        )
    return result


def print_grasp_evaluation(label: str, metrics: dict[str, float | int]) -> None:
    print(
        f"[{label}] 抓取成功率={metrics['success_rate']:.1%} "
        f"({metrics['successes']}/{metrics['episodes']}) | "
        f"接近率={metrics['approach_rate']:.1%} | "
        f"双指接触率={metrics['bilateral_contact_rate']:.1%} | "
        f"夹住率={metrics['grasp_rate']:.1%} | "
        f"掉落率={metrics['drop_rate']:.1%} | "
        f"平均抬高={metrics['mean_lift_height_m'] * 100:.1f}cm | "
        f"平均步数={metrics['mean_steps']:.1f}",
        flush=True,
    )
    if float(metrics.get("randomization", 0.0)) > 0.0:
        print(
            "      延迟分层："
            f"0步={metrics['delay_0_success_rate']:.1%} "
            f"({metrics['delay_0_episodes']}轮) | "
            f"1步={metrics['delay_1_success_rate']:.1%} "
            f"({metrics['delay_1_episodes']}轮) | "
            f"2步={metrics['delay_2_success_rate']:.1%} "
            f"({metrics['delay_2_episodes']}轮)",
            flush=True,
        )
    if float(metrics.get("grasp_constraint_force", 300.0)) <= 0.0:
        print(
            "      纯物理证据：固定约束激活="
            f"{metrics['constraint_activations']}步 | "
            f"方块质量={metrics['cube_mass_min_kg'] * 1000:.0f}g | "
            f"摩擦系数={metrics['cube_friction_min']:.2f}",
            flush=True,
        )
    if float(metrics.get("pose_randomization", 0.0)) > 0.0:
        print(
            "      姿态范围："
            f"x=[{metrics['cube_x_min_m']:.3f}, "
            f"{metrics['cube_x_max_m']:.3f}]m | "
            f"y=[{metrics['cube_y_min_m']:.3f}, "
            f"{metrics['cube_y_max_m']:.3f}]m | "
            f"yaw=[{math.degrees(metrics['cube_yaw_min_rad']):.1f}, "
            f"{math.degrees(metrics['cube_yaw_max_rad']):.1f}]°",
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
    physical = config.grasp_constraint_force <= 0.0
    robust = max(config.randomization_curriculum, default=0.0) > 0.0
    pose_randomized = max(
        config.pose_randomization_curriculum, default=0.0) > 0.0
    if physical and pose_randomized:
        model_path = POSE_PHYSICAL_GRASP_MODEL_PATH
        best_model_path = BEST_POSE_PHYSICAL_GRASP_MODEL_PATH
        metrics_path = POSE_PHYSICAL_GRASP_METRICS_PATH
    elif physical:
        model_path = PHYSICAL_GRASP_MODEL_PATH
        best_model_path = BEST_PHYSICAL_GRASP_MODEL_PATH
        metrics_path = PHYSICAL_GRASP_METRICS_PATH
    elif robust:
        model_path = ROBUST_GRASP_MODEL_PATH
        best_model_path = BEST_ROBUST_GRASP_MODEL_PATH
        metrics_path = ROBUST_GRASP_METRICS_PATH
    else:
        model_path = GRASP_MODEL_PATH
        best_model_path = BEST_GRASP_MODEL_PATH
        metrics_path = GRASP_METRICS_PATH
    evaluation_randomization = 1.0 if robust else 0.0
    evaluation_pose_randomization = 1.0 if pose_randomized else 0.0
    print("=" * 72)
    mode = (
        "纯物理摩擦抓取"
        if physical
        else "域随机化鲁棒抓取" if robust else "基线抓取"
    )
    print(f"  Demo 2 / Grasp: IK + DAgger + TD3+BC {mode}")
    print("  最终评估仅使用8维RL策略：7个手臂关节 + 夹爪开合")
    print("=" * 72, flush=True)

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    train_env = RobotGraspEnv(
        difficulty=config.curriculum[0],
        randomization=config.randomization_curriculum[0],
        pose_randomization=config.pose_randomization_curriculum[0],
        observe_action_history=robust,
        observe_physical_residual=physical,
        observe_cube_yaw=physical and pose_randomized,
        grasp_constraint_force=config.grasp_constraint_force,
    )
    data_env = RobotGraspEnv(
        difficulty=1.0,
        randomization=evaluation_randomization,
        pose_randomization=evaluation_pose_randomization,
        observe_action_history=robust,
        observe_physical_residual=physical,
        observe_cube_yaw=physical and pose_randomized,
        grasp_constraint_force=config.grasp_constraint_force,
    )
    model = build_grasp_td3(train_env, config.seed)
    history: list[dict] = []
    best_success_rate = -1.0
    best_checkpoint_score = (-1.0,) * 5

    try:
        warm_start_path = (
            POSE_PHYSICAL_GRASP_MODEL_PATH
            if physical and pose_randomized and POSE_PHYSICAL_GRASP_MODEL_PATH.exists()
            else PHYSICAL_GRASP_MODEL_PATH
            if physical and pose_randomized and PHYSICAL_GRASP_MODEL_PATH.exists()
            else GRASP_MODEL_PATH
        )
        if (robust or physical) and warm_start_path.exists():
            warm_start_grasp_model(model, warm_start_path)
            metrics = evaluate_grasp(
                model,
                episodes=config.eval_episodes,
                seed=config.seed + 100_000,
                randomization=evaluation_randomization,
                pose_randomization=evaluation_pose_randomization,
                grasp_constraint_force=config.grasp_constraint_force,
            )
            print_grasp_evaluation("基线模型热启动", metrics)
            history.append({"stage": "warm_start", **metrics})
            best_success_rate = float(metrics["success_rate"])
            best_checkpoint_score = grasp_checkpoint_score(metrics)
            model.save(best_model_path)

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
            model,
            episodes=config.eval_episodes,
            seed=config.seed + 100_000,
            randomization=evaluation_randomization,
            pose_randomization=evaluation_pose_randomization,
            grasp_constraint_force=config.grasp_constraint_force,
        )
        print_grasp_evaluation("行为克隆", metrics)
        history.append({"stage": "behavior_cloning", **metrics})
        checkpoint_score = grasp_checkpoint_score(metrics)
        if checkpoint_score > best_checkpoint_score:
            best_success_rate = float(metrics["success_rate"])
            best_checkpoint_score = checkpoint_score
            model.save(best_model_path)

        datasets = [expert_data]
        combined = expert_data
        print(f"\n[2/3] DAgger {config.dagger_iterations} 轮...", flush=True)
        for iteration in range(config.dagger_iterations):
            model.set_parameters(
                best_model_path, exact_match=True, device=model.device)
            model.actor_target.load_state_dict(model.actor.state_dict())
            teacher_mix = max(0.6 - 0.2 * iteration, 0.1)
            if config.failure_only_dagger:
                dagger_data = collect_failure_demonstrations(
                    data_env,
                    config.dagger_steps,
                    seed=config.seed + 10_000 * (iteration + 1),
                    model=model,
                    noise_scale=0.0,
                )
                teacher_mix_label = "failure-only"
            else:
                dagger_data = collect_demonstrations(
                    data_env,
                    config.dagger_steps,
                    seed=config.seed + 10_000 * (iteration + 1),
                    model=model,
                    teacher_mix=teacher_mix,
                    noise_scale=0.015,
                )
                teacher_mix_label = f"{teacher_mix:.2f}"
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
                randomization=evaluation_randomization,
                pose_randomization=evaluation_pose_randomization,
                grasp_constraint_force=config.grasp_constraint_force,
            )
            print_grasp_evaluation(f"DAgger {iteration + 1}", metrics)
            print(
                f"      数据量={len(combined)}，teacher_mix={teacher_mix_label}，"
                f"loss={bc_loss:.6f}",
                flush=True,
            )
            history.append({"stage": f"dagger_{iteration + 1}", **metrics})
            checkpoint_score = grasp_checkpoint_score(metrics)
            if checkpoint_score > best_checkpoint_score:
                best_success_rate = float(metrics["success_rate"])
                best_checkpoint_score = checkpoint_score
                model.save(best_model_path)

        model.set_parameters(
            best_model_path, exact_match=True, device=model.device)
        model.actor_target.load_state_dict(model.actor.state_dict())
        print("\n[3/3] TD3+BC在线接触强化学习微调...", flush=True)
        model.set_expert_data(
            combined,
            start=config.bc_start,
            end=config.bc_end,
            decay_updates=sum(config.rl_phase_steps),
            balance_phases=True,
            action_weights=GRASP_BC_ACTION_WEIGHTS,
            # The pose teacher has already been validated by physical rollout.
            # Do not let an immature critic reject its pre-grasp yaw labels.
            q_filter=not pose_randomized,
            hold_state_index=22,
            hold_action_index=7,
            hold_target=1.0,
            hold_coefficient=config.hold_coefficient,
            anchor_coefficient=config.anchor_coefficient,
            actor_trainable_suffix_start=(
                BASE_OBSERVATION_SIZE
                if (robust or physical) and not pose_randomized
                else None
            ),
        )
        model.rl_weight = config.rl_weight
        model.actor_learning_rate = config.actor_learning_rate
        print(
            f"      先用 {config.critic_warmup_steps} 次抓取回放预热critic...",
            flush=True,
        )
        warmup_critic(model, config.critic_warmup_steps)
        total_rl_steps = 0
        for phase, (
            difficulty,
            randomization,
            pose_randomization,
            phase_steps,
        ) in enumerate(
            zip(
                config.curriculum,
                config.randomization_curriculum,
                config.pose_randomization_curriculum,
                config.rl_phase_steps,
            ),
            start=1,
        ):
            train_env.set_difficulty(difficulty)
            train_env.set_randomization(randomization)
            train_env.set_pose_randomization(pose_randomization)
            model.learn(
                total_timesteps=phase_steps,
                reset_num_timesteps=False,
                progress_bar=False,
            )
            total_rl_steps += phase_steps
            action_drift = model.reference_action_drift()
            metrics = evaluate_grasp(
                model,
                episodes=config.eval_episodes,
                seed=config.seed + 100_000,
                # Select checkpoints on the same full-workspace test set;
                # curriculum difficulty affects data collection only.
                difficulty=1.0,
                randomization=evaluation_randomization,
                pose_randomization=evaluation_pose_randomization,
                grasp_constraint_force=config.grasp_constraint_force,
            )
            label = (
                f"TD3+BC阶段{phase} difficulty={difficulty:.2f} "
                f"randomization={randomization:.2f} "
                f"pose_randomization={pose_randomization:.2f}"
            )
            print_grasp_evaluation(label, metrics)
            print(
                "      基线动作漂移："
                f"mean={action_drift['mean']:.4f} | "
                f"P95={action_drift['p95']:.4f} | "
                f"max={action_drift['max']:.4f}",
                flush=True,
            )
            history.append({
                "stage": f"td3_bc_{phase}",
                "difficulty": difficulty,
                "randomization": randomization,
                "pose_randomization": pose_randomization,
                "rl_steps": total_rl_steps,
                "reference_action_drift": action_drift,
                **metrics,
            })
            checkpoint_score = grasp_checkpoint_score(metrics)
            checkpoint_improved = checkpoint_score > best_checkpoint_score
            if checkpoint_improved:
                best_success_rate = float(metrics["success_rate"])
                best_checkpoint_score = checkpoint_score
                model.save(best_model_path)
            elif not pose_randomized:
                model.set_parameters(
                    best_model_path, exact_match=True, device=model.device)
                model.rl_weight *= 0.5
                print(
                    "      未超过最佳检查点，已恢复最佳权重并降低RL更新权重。",
                    flush=True,
                )
            else:
                print(
                    "      当前阶段尚未超过全范围最佳检查点；"
                    "保留课程权重进入下一姿态阶段。",
                    flush=True,
                )
            if (
                total_rl_steps >= 10_000
                and checkpoint_improved
                and metrics["success_rate"] >= config.target_success_rate
            ):
                print("      已达到抓取成功率目标，提前停止课程训练。", flush=True)
                break

        model.set_parameters(
            best_model_path, exact_match=True, device=model.device)
        model.save(model_path)
        final_metrics = evaluate_grasp(
            model,
            episodes=config.final_eval_episodes,
            seed=config.seed + 200_000,
            difficulty=1.0,
            randomization=evaluation_randomization,
            pose_randomization=evaluation_pose_randomization,
            grasp_constraint_force=config.grasp_constraint_force,
        )
        print_grasp_evaluation("最终RL独立测试集", final_metrics)
        result = {
            "config": asdict(config),
            "history": history,
            "final": final_metrics,
            "model_path": str(model_path),
            "best_model_path": str(best_model_path),
        }
        metrics_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n抓取模型已保存到 {model_path}")
        print(f"训练指标已保存到 {metrics_path}")
        return result
    finally:
        data_env.close()
        train_env.close()
