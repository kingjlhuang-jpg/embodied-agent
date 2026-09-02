"""IK-guided reinforcement learning for the Kuka reaching demos.

The training pipeline has three stages: behaviour cloning from an inverse-
kinematics teacher, DAgger collection on policy-visited states, and TD3+BC
fine-tuning. IK is used only during training; evaluation uses RL alone.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import torch
import torch.nn.functional as F
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.utils import polyak_update


SUCCESS_DISTANCE = 0.02
ACTION_SCALE = 0.1
MODEL_PATH = Path("trained_policy.zip")
BEST_MODEL_PATH = Path("best_policy.zip")
METRICS_PATH = Path("training_metrics.json")


@dataclass(frozen=True)
class TrainingConfig:
    """Training sizes chosen for a fast local demonstration."""

    seed: int = 20260901
    expert_steps: int = 20_000
    bc_epochs: int = 20
    dagger_iterations: int = 3
    dagger_steps: int = 4_000
    dagger_epochs: int = 6
    critic_warmup_steps: int = 2_000
    rl_phase_steps: tuple[int, ...] = (10_000, 15_000, 25_000)
    curriculum: tuple[float, ...] = (1.00, 1.00, 1.00)
    eval_episodes: int = 100
    target_success_rate: float = 0.90

    @classmethod
    def quick(cls, seed: int = 20260901) -> "TrainingConfig":
        """Small configuration for smoke tests, not final convergence."""

        return cls(
            seed=seed,
            expert_steps=1_000,
            bc_epochs=2,
            dagger_iterations=1,
            dagger_steps=500,
            dagger_epochs=1,
            critic_warmup_steps=50,
            rl_phase_steps=(500,),
            curriculum=(1.0,),
            eval_episodes=10,
            target_success_rate=1.0,
        )


@dataclass
class Demonstrations:
    """Expert labels plus real transitions for replay-buffer warm-up."""

    observations: np.ndarray
    teacher_actions: np.ndarray
    executed_actions: np.ndarray
    next_observations: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    phases: np.ndarray

    def __len__(self) -> int:
        return len(self.observations)

    @classmethod
    def concatenate(cls, datasets: Iterable["Demonstrations"]) -> "Demonstrations":
        items = list(datasets)
        return cls(*(
            np.concatenate([getattr(item, field) for item in items], axis=0)
            for field in cls.__dataclass_fields__
        ))


class RobotArmEnv(gym.Env):
    """Gymnasium environment for Kuka end-effector reaching."""

    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(
        self,
        render_mode: str | None = None,
        *,
        render: bool | None = None,
        difficulty: float = 1.0,
        max_steps: int = 200,
    ):
        super().__init__()
        if render is not None:
            render_mode = "human" if render else None
        if render_mode not in (None, "human"):
            raise ValueError(f"unsupported render_mode: {render_mode}")

        self.render_mode = render_mode
        self.difficulty = float(np.clip(difficulty, 0.0, 1.0))
        self.max_steps = max_steps
        self.action_space = spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float32)
        self.observation_space = spaces.Box(
            -2.0, 2.0, shape=(13,), dtype=np.float32)

        mode = p.GUI if render_mode == "human" else p.DIRECT
        self.client = p.connect(mode)
        if self.client < 0:
            raise RuntimeError("failed to connect to PyBullet")
        p.setAdditionalSearchPath(
            pybullet_data.getDataPath(), physicsClientId=self.client)
        if render_mode == "human":
            p.resetDebugVisualizerCamera(
                1.5, 45, -30, [0.5, 0, 0.3], physicsClientId=self.client)

        self.robot_id = -1
        self.target_pos = np.zeros(3, dtype=np.float64)
        self.step_count = 0
        self.previous_distance = 0.0
        self.previous_action = np.zeros(7, dtype=np.float32)
        self.joint_lower = np.full(7, -np.pi, dtype=np.float64)
        self.joint_upper = np.full(7, np.pi, dtype=np.float64)

    def set_difficulty(self, difficulty: float) -> None:
        """Set curriculum difficulty for targets generated on future resets."""

        self.difficulty = float(np.clip(difficulty, 0.0, 1.0))

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        del options

        p.resetSimulation(physicsClientId=self.client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        p.setTimeStep(1.0 / 240.0, physicsClientId=self.client)
        p.loadURDF("plane.urdf", physicsClientId=self.client)
        self.robot_id = p.loadURDF(
            "kuka_iiwa/model.urdf",
            [0, 0, 0],
            useFixedBase=True,
            physicsClientId=self.client,
        )

        for joint in range(7):
            info = p.getJointInfo(
                self.robot_id, joint, physicsClientId=self.client)
            self.joint_lower[joint] = info[8]
            self.joint_upper[joint] = info[9]

        full_target = np.array([
            0.4 + self.np_random.uniform(-0.15, 0.15),
            self.np_random.uniform(-0.2, 0.2),
            0.1 + self.np_random.uniform(0, 0.3),
        ])
        initial_ee = self._get_ee_pos()
        self.target_pos = (
            initial_ee + self.difficulty * (full_target - initial_ee))

        visual = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=0.03,
            rgbaColor=[0, 1, 0, 0.7],
            physicsClientId=self.client,
        )
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=visual,
            basePosition=self.target_pos,
            physicsClientId=self.client,
        )

        self.step_count = 0
        self.previous_action.fill(0.0)
        self.previous_distance = self._distance_to_target()
        return self._get_observation(), self._info(self.previous_distance)

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != self.action_space.shape:
            raise ValueError(f"expected action shape (7,), got {action.shape}")
        action = np.clip(action, -1.0, 1.0)

        joint_positions = self._get_joint_positions()
        joint_targets = joint_positions + action * ACTION_SCALE
        joint_targets = np.clip(joint_targets, self.joint_lower, self.joint_upper)
        for joint, target in enumerate(joint_targets):
            p.setJointMotorControl2(
                self.robot_id,
                joint,
                p.POSITION_CONTROL,
                targetPosition=float(target),
                force=200,
                physicsClientId=self.client,
            )

        for _ in range(10):
            p.stepSimulation(physicsClientId=self.client)

        self.step_count += 1
        distance = self._distance_to_target()
        progress = self.previous_distance - distance
        smoothness_cost = np.square(action - self.previous_action).sum()
        reward = (
            20.0 * progress
            - 0.1 * distance
            - 0.002 * np.square(action).sum()
            - 0.001 * smoothness_cost
        )
        terminated = distance < SUCCESS_DISTANCE
        if terminated:
            reward += 10.0
        truncated = self.step_count >= self.max_steps

        self.previous_distance = distance
        self.previous_action = action.copy()
        return (
            self._get_observation(),
            float(reward),
            bool(terminated),
            bool(truncated),
            self._info(distance),
        )

    def get_ik_action(self) -> np.ndarray:
        """Return the normalized one-step action chosen by the IK teacher."""

        current = self._get_joint_positions()
        ranges = np.maximum(self.joint_upper - self.joint_lower, 1e-3)
        solution = p.calculateInverseKinematics(
            self.robot_id,
            6,
            self.target_pos,
            lowerLimits=self.joint_lower.tolist(),
            upperLimits=self.joint_upper.tolist(),
            jointRanges=ranges.tolist(),
            restPoses=current.tolist(),
            maxNumIterations=100,
            residualThreshold=1e-5,
            physicsClientId=self.client,
        )
        delta = np.asarray(solution[:7]) - current
        return np.clip(delta / ACTION_SCALE, -1.0, 1.0).astype(np.float32)

    def _get_joint_positions(self) -> np.ndarray:
        return np.array([
            p.getJointState(
                self.robot_id, joint, physicsClientId=self.client)[0]
            for joint in range(7)
        ])

    def _get_ee_pos(self) -> np.ndarray:
        state = p.getLinkState(
            self.robot_id, 6, physicsClientId=self.client)
        return np.asarray(state[0])

    def _distance_to_target(self) -> float:
        return float(np.linalg.norm(self._get_ee_pos() - self.target_pos))

    def _get_observation(self) -> np.ndarray:
        joints = self._get_joint_positions() / np.pi
        ee_pos = self._get_ee_pos() / 1.5
        target_error = (self.target_pos - self._get_ee_pos()) / 1.5
        observation = np.concatenate([joints, ee_pos, target_error])
        return np.clip(observation, -2.0, 2.0).astype(np.float32)

    def _info(self, distance: float) -> dict[str, float | bool]:
        return {
            "distance": float(distance),
            "is_success": bool(distance < SUCCESS_DISTANCE),
            "difficulty": self.difficulty,
        }

    def close(self):
        if getattr(self, "client", -1) >= 0 and p.isConnected(self.client):
            p.disconnect(self.client)
            self.client = -1


class GuidedTD3(TD3):
    """TD3+BC-style updates that keep the actor near its IK teacher."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._expert_observations: torch.Tensor | None = None
        self._expert_actions: torch.Tensor | None = None
        self._expert_phase_indices: list[torch.Tensor] = []
        self._bc_action_weights: torch.Tensor | None = None
        self._reference_actions: torch.Tensor | None = None
        self._q_filter = False
        self._hold_state_index: int | None = None
        self._hold_action_index: int | None = None
        self._hold_target = 1.0
        self._hold_coefficient = 0.0
        self._anchor_coefficient = 0.0
        self._actor_trainable_suffix_start: int | None = None
        self.actor_learning_rate: float | None = None
        self._bc_start = 1.0
        self._bc_end = 0.25
        self._bc_decay_updates = 50_000
        self.rl_weight = 0.01

    def set_expert_data(
        self,
        demonstrations: Demonstrations,
        *,
        start: float = 1.0,
        end: float = 0.25,
        decay_updates: int = 50_000,
        balance_phases: bool = False,
        action_weights: np.ndarray | None = None,
        q_filter: bool = False,
        hold_state_index: int | None = None,
        hold_action_index: int | None = None,
        hold_target: float = 1.0,
        hold_coefficient: float = 0.0,
        anchor_coefficient: float = 0.0,
        actor_trainable_suffix_start: int | None = None,
    ) -> None:
        self._expert_observations = torch.as_tensor(
            demonstrations.observations, device=self.device)
        self._expert_actions = torch.as_tensor(
            demonstrations.teacher_actions, device=self.device)
        self._expert_phase_indices = []
        if balance_phases:
            for phase in np.unique(demonstrations.phases):
                indices = np.flatnonzero(demonstrations.phases == phase)
                self._expert_phase_indices.append(torch.as_tensor(
                    indices, dtype=torch.long, device=self.device))
        if action_weights is None:
            action_weights = np.ones(
                demonstrations.teacher_actions.shape[1], dtype=np.float32)
        weights = np.asarray(action_weights, dtype=np.float32)
        if weights.shape != (demonstrations.teacher_actions.shape[1],):
            raise ValueError("action_weights must match the action dimension")
        self._bc_action_weights = torch.as_tensor(
            weights / max(float(weights.mean()), 1e-6), device=self.device)
        self._q_filter = bool(q_filter)
        if (hold_state_index is None) != (hold_action_index is None):
            raise ValueError(
                "hold_state_index and hold_action_index must be set together")
        if hold_state_index is not None:
            if not 0 <= hold_state_index < demonstrations.observations.shape[1]:
                raise ValueError("hold_state_index is outside the observation")
            if not 0 <= hold_action_index < demonstrations.teacher_actions.shape[1]:
                raise ValueError("hold_action_index is outside the action")
        self._hold_state_index = hold_state_index
        self._hold_action_index = hold_action_index
        self._hold_target = float(hold_target)
        self._hold_coefficient = max(float(hold_coefficient), 0.0)
        self._anchor_coefficient = max(float(anchor_coefficient), 0.0)
        if (
            actor_trainable_suffix_start is not None
            and not 0 <= actor_trainable_suffix_start < demonstrations.observations.shape[1]
        ):
            raise ValueError(
                "actor_trainable_suffix_start is outside the observation")
        self._actor_trainable_suffix_start = actor_trainable_suffix_start
        self._reference_actions = None
        if self._anchor_coefficient > 0.0:
            with torch.no_grad():
                self._reference_actions = self.actor(
                    self._expert_observations).detach().clone()
        self._bc_start = start
        self._bc_end = end
        self._bc_decay_updates = max(decay_updates, 1)

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        if self._expert_observations is None or self._expert_actions is None:
            raise RuntimeError("expert data must be attached before TD3+BC training")

        self.policy.set_training_mode(True)
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])
        if self.actor_learning_rate is not None:
            for parameter_group in self.actor.optimizer.param_groups:
                parameter_group["lr"] = self.actor_learning_rate
        actor_losses = []
        critic_losses = []
        bc_losses = []
        hold_losses = []
        anchor_losses = []
        q_filter_fractions = []

        for _ in range(gradient_steps):
            self._n_updates += 1
            replay_data = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env)
            discounts = (
                replay_data.discounts
                if replay_data.discounts is not None
                else self.gamma
            )
            with torch.no_grad():
                noise = replay_data.actions.clone().normal_(
                    0, self.target_policy_noise)
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                next_actions = (
                    self.actor_target(replay_data.next_observations) + noise
                ).clamp(-1, 1)
                next_q_values = torch.cat(
                    self.critic_target(
                        replay_data.next_observations, next_actions),
                    dim=1,
                ).min(dim=1, keepdim=True).values
                target_q_values = (
                    replay_data.rewards
                    + (1 - replay_data.dones) * discounts * next_q_values
                )

            current_q_values = self.critic(
                replay_data.observations, replay_data.actions)
            critic_loss = sum(
                F.mse_loss(current_q, target_q_values)
                for current_q in current_q_values
            )
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()
            critic_losses.append(critic_loss.item())

            if self._n_updates % self.policy_delay == 0:
                policy_actions = self.actor(replay_data.observations)
                q_values = self.critic.q1_forward(
                    replay_data.observations, policy_actions)
                q_scale = q_values.abs().mean().detach().clamp_min(1e-3)
                rl_loss = -self.rl_weight * q_values.mean() / q_scale

                if self._expert_phase_indices:
                    samples_per_phase = int(np.ceil(
                        batch_size / len(self._expert_phase_indices)))
                    sampled_groups = []
                    for phase_indices in self._expert_phase_indices:
                        offsets = torch.randint(
                            len(phase_indices),
                            (samples_per_phase,),
                            device=self.device,
                        )
                        sampled_groups.append(phase_indices[offsets])
                    expert_indices = torch.cat(sampled_groups)[:batch_size]
                    expert_indices = expert_indices[
                        torch.randperm(batch_size, device=self.device)]
                else:
                    expert_indices = torch.randint(
                        len(self._expert_observations),
                        (batch_size,),
                        device=self.device,
                    )
                expert_observations = self._expert_observations[expert_indices]
                expert_actions = self._expert_actions[expert_indices]
                reference_actions = (
                    self._reference_actions[expert_indices]
                    if self._reference_actions is not None
                    else None
                )
                expert_predictions = self.actor(
                    expert_observations)
                element_losses = F.smooth_l1_loss(
                    expert_predictions, expert_actions, reduction="none")
                if self._bc_action_weights is not None:
                    element_losses = element_losses * self._bc_action_weights
                sample_losses = element_losses.mean(dim=1)
                if self._q_filter:
                    with torch.no_grad():
                        teacher_q = torch.cat(
                            self.critic(expert_observations, expert_actions),
                            dim=1,
                        ).min(dim=1).values
                        policy_q = torch.cat(
                            self.critic(
                                expert_observations,
                                expert_predictions.detach(),
                            ),
                            dim=1,
                        ).min(dim=1).values
                        q_mask = (teacher_q >= policy_q).float()
                    bc_loss = (
                        sample_losses * q_mask
                    ).sum() / q_mask.sum().clamp_min(1.0)
                    q_filter_fractions.append(q_mask.mean().item())
                else:
                    bc_loss = sample_losses.mean()
                progress = min(self._n_updates / self._bc_decay_updates, 1.0)
                bc_coefficient = (
                    self._bc_start
                    + progress * (self._bc_end - self._bc_start)
                )
                hold_loss = torch.zeros((), device=self.device)
                if (
                    self._hold_state_index is not None
                    and self._hold_action_index is not None
                    and self._hold_coefficient > 0.0
                ):
                    replay_hold_mask = (
                        replay_data.observations[:, self._hold_state_index] > 0.5)
                    expert_hold_mask = (
                        expert_observations[:, self._hold_state_index] > 0.5)
                    # Q-filtering is useful during approach, but an immature
                    # critic can incorrectly reject the teacher precisely
                    # after contact.  Keep an unfiltered imitation anchor for
                    # the complete lift action in known grasp states.
                    if bool(expert_hold_mask.any()):
                        expert_hold_losses = F.smooth_l1_loss(
                            expert_predictions[expert_hold_mask],
                            expert_actions[expert_hold_mask],
                            reduction="none",
                        )
                        if self._bc_action_weights is not None:
                            expert_hold_losses = (
                                expert_hold_losses * self._bc_action_weights)
                        hold_loss = hold_loss + expert_hold_losses.mean()

                    # Replay states are not labelled by IK, but closing the
                    # gripper is always safe after the grasp latch is active.
                    if bool(replay_hold_mask.any()):
                        hold_predictions = policy_actions[
                            replay_hold_mask, self._hold_action_index]
                        hold_targets = torch.full_like(
                            hold_predictions, self._hold_target)
                        hold_loss = hold_loss + F.smooth_l1_loss(
                            hold_predictions, hold_targets)
                anchor_loss = torch.zeros((), device=self.device)
                if (
                    reference_actions is not None
                    and self._anchor_coefficient > 0.0
                ):
                    anchor_elements = F.smooth_l1_loss(
                        expert_predictions, reference_actions, reduction="none")
                    if self._bc_action_weights is not None:
                        anchor_elements = anchor_elements * self._bc_action_weights
                    anchor_loss = anchor_elements.mean()
                actor_loss = (
                    rl_loss
                    + bc_coefficient * bc_loss
                    + self._hold_coefficient * hold_loss
                    + self._anchor_coefficient * anchor_loss
                )

                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                if self._actor_trainable_suffix_start is not None:
                    first_weight = self.actor.mu[0].weight
                    for parameter in self.actor.parameters():
                        if parameter.grad is None:
                            continue
                        if parameter is first_weight:
                            parameter.grad[
                                :, :self._actor_trainable_suffix_start
                            ].zero_()
                        else:
                            parameter.grad.zero_()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
                self.actor.optimizer.step()
                actor_losses.append(actor_loss.item())
                bc_losses.append(bc_loss.item())
                hold_losses.append(hold_loss.item())
                anchor_losses.append(anchor_loss.item())

                polyak_update(
                    self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(
                    self.actor.parameters(), self.actor_target.parameters(), self.tau)
                polyak_update(
                    self.critic_batch_norm_stats,
                    self.critic_batch_norm_stats_target,
                    1.0,
                )
                polyak_update(
                    self.actor_batch_norm_stats,
                    self.actor_batch_norm_stats_target,
                    1.0,
                )

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/critic_loss", float(np.mean(critic_losses)))
        if actor_losses:
            self.logger.record("train/actor_loss", float(np.mean(actor_losses)))
            self.logger.record("train/ik_bc_loss", float(np.mean(bc_losses)))
            self.logger.record("train/rl_weight", self.rl_weight)
            self.logger.record(
                "train/hold_action_loss", float(np.mean(hold_losses)))
            self.logger.record(
                "train/reference_anchor_loss", float(np.mean(anchor_losses)))
            if q_filter_fractions:
                self.logger.record(
                    "train/q_filter_fraction",
                    float(np.mean(q_filter_fractions)),
                )

    def reference_action_drift(self) -> dict[str, float]:
        """Measure actor drift from the policy captured before online RL."""

        if self._expert_observations is None or self._reference_actions is None:
            return {"mean": 0.0, "p95": 0.0, "max": 0.0}
        differences = []
        with torch.no_grad():
            for start in range(0, len(self._expert_observations), 2_048):
                observations = self._expert_observations[start:start + 2_048]
                references = self._reference_actions[start:start + 2_048]
                differences.append(
                    (self.actor(observations) - references).abs().cpu())
        values = torch.cat(differences).numpy()
        return {
            "mean": float(np.mean(values)),
            "p95": float(np.percentile(values, 95)),
            "max": float(np.max(values)),
        }

    def _excluded_save_params(self) -> list[str]:
        return super()._excluded_save_params() + [
            "_expert_observations",
            "_expert_actions",
            "_expert_phase_indices",
            "_bc_action_weights",
            "_reference_actions",
        ]


def collect_demonstrations(
    env: gym.Env,
    steps: int,
    *,
    seed: int,
    model: TD3 | None = None,
    teacher_mix: float = 1.0,
    noise_scale: float = 0.03,
) -> Demonstrations:
    """Collect teacher labels, optionally on states visited by a learned policy."""

    rng = np.random.default_rng(seed)
    observations: list[np.ndarray] = []
    teacher_actions: list[np.ndarray] = []
    executed_actions: list[np.ndarray] = []
    next_observations: list[np.ndarray] = []
    rewards: list[float] = []
    dones: list[bool] = []
    phases: list[int] = []

    episode = 0
    observation, _ = env.reset(seed=seed)
    while len(observations) < steps:
        get_phase = getattr(env, "get_demonstration_phase", None)
        phase = int(get_phase()) if get_phase is not None else 0
        teacher_action = env.get_ik_action()
        if model is None:
            policy_action = teacher_action
        else:
            policy_action, _ = model.predict(observation, deterministic=True)
        action = (
            teacher_mix * teacher_action
            + (1.0 - teacher_mix) * np.asarray(policy_action)
        )
        action = np.clip(
            action + rng.normal(
                0.0, noise_scale, size=env.action_space.shape),
            -1.0,
            1.0,
        )
        next_observation, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        observations.append(observation)
        teacher_actions.append(teacher_action)
        executed_actions.append(action.astype(np.float32))
        next_observations.append(next_observation)
        rewards.append(reward)
        dones.append(done)
        phases.append(phase)

        observation = next_observation
        if done:
            episode += 1
            observation, _ = env.reset(seed=seed + episode)

    return Demonstrations(
        observations=np.asarray(observations, dtype=np.float32),
        teacher_actions=np.asarray(teacher_actions, dtype=np.float32),
        executed_actions=np.asarray(executed_actions, dtype=np.float32),
        next_observations=np.asarray(next_observations, dtype=np.float32),
        rewards=np.asarray(rewards, dtype=np.float32),
        dones=np.asarray(dones, dtype=np.float32),
        phases=np.asarray(phases, dtype=np.int64),
    )


def collect_failure_demonstrations(
    env: gym.Env,
    steps: int,
    *,
    seed: int,
    model: TD3,
    noise_scale: float = 0.0,
) -> Demonstrations:
    """Label policy-only trajectories and retain only failed episodes."""

    rng = np.random.default_rng(seed)
    retained: list[Demonstrations] = []
    retained_steps = 0
    episode = 0
    while retained_steps < steps:
        observation, _ = env.reset(seed=seed + episode)
        episode_observations = []
        episode_teacher_actions = []
        episode_executed_actions = []
        episode_next_observations = []
        episode_rewards = []
        episode_dones = []
        episode_phases = []
        info: dict[str, float | bool] = {}
        for _ in range(env.max_steps):
            get_phase = getattr(env, "get_demonstration_phase", None)
            phase = int(get_phase()) if get_phase is not None else 0
            teacher_action = env.get_ik_action()
            policy_action, _ = model.predict(observation, deterministic=True)
            executed_action = np.clip(
                np.asarray(policy_action)
                + rng.normal(0.0, noise_scale, size=env.action_space.shape),
                -1.0,
                1.0,
            ).astype(np.float32)
            next_observation, reward, terminated, truncated, info = env.step(
                executed_action)
            done = terminated or truncated
            episode_observations.append(observation)
            episode_teacher_actions.append(teacher_action)
            episode_executed_actions.append(executed_action)
            episode_next_observations.append(next_observation)
            episode_rewards.append(reward)
            episode_dones.append(done)
            episode_phases.append(phase)
            observation = next_observation
            if done:
                break

        if not bool(info.get("is_success", False)):
            episode_data = Demonstrations(
                observations=np.asarray(
                    episode_observations, dtype=np.float32),
                teacher_actions=np.asarray(
                    episode_teacher_actions, dtype=np.float32),
                executed_actions=np.asarray(
                    episode_executed_actions, dtype=np.float32),
                next_observations=np.asarray(
                    episode_next_observations, dtype=np.float32),
                rewards=np.asarray(episode_rewards, dtype=np.float32),
                dones=np.asarray(episode_dones, dtype=np.float32),
                phases=np.asarray(episode_phases, dtype=np.int64),
            )
            retained.append(episode_data)
            retained_steps += len(episode_data)
        episode += 1

    combined = Demonstrations.concatenate(retained)
    return Demonstrations(*(
        getattr(combined, field)[:steps]
        for field in Demonstrations.__dataclass_fields__
    ))


def pretrain_actor(
    model: TD3,
    demonstrations: Demonstrations,
    *,
    epochs: int,
    batch_size: int = 256,
    seed: int,
    balance_phases: bool = False,
    action_weights: np.ndarray | None = None,
) -> float:
    """Behaviour-clone the deterministic TD3 actor from IK labels."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    rng = np.random.default_rng(seed)
    observations = torch.as_tensor(
        demonstrations.observations, device=model.device)
    teacher_actions = torch.as_tensor(
        demonstrations.teacher_actions, device=model.device)
    if action_weights is None:
        action_weights = np.ones(teacher_actions.shape[1], dtype=np.float32)
    weights_array = np.asarray(action_weights, dtype=np.float32)
    if weights_array.shape != (teacher_actions.shape[1],):
        raise ValueError("action_weights must match the action dimension")
    action_weight_tensor = torch.as_tensor(
        weights_array / max(float(weights_array.mean()), 1e-6),
        device=model.device,
    )
    phase_groups = [
        np.flatnonzero(demonstrations.phases == phase)
        for phase in np.unique(demonstrations.phases)
    ]
    final_loss = 0.0

    model.actor.train(True)
    for _ in range(epochs):
        if balance_phases and len(phase_groups) > 1:
            samples_per_phase = int(np.ceil(
                len(demonstrations) / len(phase_groups)))
            sampled = np.concatenate([
                rng.choice(group, samples_per_phase, replace=True)
                for group in phase_groups
            ])[:len(demonstrations)]
            permutation = torch.as_tensor(
                sampled[rng.permutation(len(sampled))],
                dtype=torch.long,
            )
        else:
            permutation = torch.randperm(
                len(demonstrations), generator=generator)
        total_loss = 0.0
        batches = 0
        for start in range(0, len(demonstrations), batch_size):
            indices = permutation[start:start + batch_size].to(model.device)
            predicted = model.actor(observations[indices])
            loss = (
                F.smooth_l1_loss(
                    predicted, teacher_actions[indices], reduction="none")
                * action_weight_tensor
            ).mean()
            model.actor.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.actor.parameters(), 1.0)
            model.actor.optimizer.step()
            total_loss += loss.item()
            batches += 1
        final_loss = total_loss / max(batches, 1)
    if hasattr(model, "actor_target"):
        model.actor_target.load_state_dict(model.actor.state_dict())
    return final_loss


def add_to_replay_buffer(model: TD3, demonstrations: Demonstrations) -> None:
    """Warm TD3's replay buffer with transitions seen during imitation."""

    for index in range(len(demonstrations)):
        done = bool(demonstrations.dones[index])
        model.replay_buffer.add(
            demonstrations.observations[index][None, :],
            demonstrations.next_observations[index][None, :],
            demonstrations.executed_actions[index][None, :],
            np.array([demonstrations.rewards[index]], dtype=np.float32),
            np.array([done], dtype=np.float32),
            [{"TimeLimit.truncated": False}],
        )


def warmup_critic(model: GuidedTD3, gradient_steps: int, batch_size: int = 256) -> None:
    """Fit TD3 critics on demonstrations before allowing actor updates."""

    model.policy.set_training_mode(True)
    for gradient_step in range(gradient_steps):
        replay_data = model.replay_buffer.sample(batch_size)
        with torch.no_grad():
            next_actions = model.actor_target(replay_data.next_observations)
            next_q_values = torch.cat(
                model.critic_target(replay_data.next_observations, next_actions),
                dim=1,
            ).min(dim=1, keepdim=True).values
            targets = (
                replay_data.rewards
                + (1.0 - replay_data.dones) * model.gamma * next_q_values
            )

        current_q_values = model.critic(
            replay_data.observations, replay_data.actions)
        loss = 0.5 * sum(
            F.mse_loss(current_q, targets) for current_q in current_q_values)
        model.critic.optimizer.zero_grad()
        loss.backward()
        model.critic.optimizer.step()
        polyak_update(
            model.critic.parameters(), model.critic_target.parameters(), model.tau)


def evaluate(
    model: TD3,
    *,
    episodes: int,
    seed: int,
    difficulty: float = 1.0,
) -> dict[str, float | int]:
    """Evaluate RL alone on a deterministic set of random targets."""

    env = RobotArmEnv(difficulty=difficulty)
    distances: list[float] = []
    steps: list[int] = []
    try:
        for episode in range(episodes):
            observation, _ = env.reset(seed=seed + episode)
            info: dict[str, float | bool] = {}
            for step in range(env.max_steps):
                action, _ = model.predict(observation, deterministic=True)
                observation, _, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
            distances.append(float(info["distance"]))
            steps.append(step + 1)
    finally:
        env.close()

    distance_array = np.asarray(distances)
    step_array = np.asarray(steps)
    successes = int(np.sum(distance_array < SUCCESS_DISTANCE))
    return {
        "episodes": episodes,
        "successes": successes,
        "success_rate": successes / episodes,
        "mean_distance_m": float(distance_array.mean()),
        "p95_distance_m": float(np.percentile(distance_array, 95)),
        "mean_steps": float(step_array.mean()),
    }


def print_evaluation(label: str, metrics: dict[str, float | int]) -> None:
    print(
        f"[{label}] 2cm成功率={metrics['success_rate']:.1%} "
        f"({metrics['successes']}/{metrics['episodes']}) | "
        f"平均误差={metrics['mean_distance_m'] * 100:.2f}cm | "
        f"P95={metrics['p95_distance_m'] * 100:.2f}cm | "
        f"平均步数={metrics['mean_steps']:.1f}",
        flush=True,
    )


def build_td3(env: RobotArmEnv, seed: int) -> GuidedTD3:
    return GuidedTD3(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=200_000,
        learning_starts=0,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        action_noise=NormalActionNoise(
            mean=np.zeros(7), sigma=np.full(7, 0.05)),
        policy_delay=2,
        target_policy_noise=0.1,
        target_noise_clip=0.2,
        policy_kwargs={"net_arch": [128, 128]},
        seed=seed,
        verbose=0,
        device="auto",
    )


def train(config: TrainingConfig) -> dict:
    print("=" * 72)
    print("  Demo 2: IK示范 + DAgger + TD3+BC 快速收敛训练")
    print("  最终评估只使用RL策略，不调用IK")
    print("=" * 72, flush=True)

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    train_env = RobotArmEnv(difficulty=config.curriculum[0])
    data_env = RobotArmEnv(difficulty=1.0)
    model = build_td3(train_env, config.seed)
    history: list[dict] = []
    best_success_rate = -1.0

    try:
        print(f"\n[1/3] 收集 {config.expert_steps} 条IK示范...", flush=True)
        expert_data = collect_demonstrations(
            data_env,
            config.expert_steps,
            seed=config.seed,
            noise_scale=0.03,
        )
        bc_loss = pretrain_actor(
            model,
            expert_data,
            epochs=config.bc_epochs,
            seed=config.seed,
        )
        add_to_replay_buffer(model, expert_data)
        print(f"      行为克隆完成，loss={bc_loss:.6f}", flush=True)
        metrics = evaluate(
            model, episodes=config.eval_episodes, seed=config.seed + 100_000)
        print_evaluation("行为克隆", metrics)
        history.append({"stage": "behavior_cloning", **metrics})
        best_success_rate = float(metrics["success_rate"])
        model.save(BEST_MODEL_PATH)

        datasets = [expert_data]
        combined = expert_data
        print(f"\n[2/3] DAgger {config.dagger_iterations} 轮...", flush=True)
        for iteration in range(config.dagger_iterations):
            teacher_mix = max(0.5 - 0.25 * iteration, 0.0)
            dagger_data = collect_demonstrations(
                data_env,
                config.dagger_steps,
                seed=config.seed + 10_000 * (iteration + 1),
                model=model,
                teacher_mix=teacher_mix,
                noise_scale=0.02,
            )
            datasets.append(dagger_data)
            combined = Demonstrations.concatenate(datasets)
            bc_loss = pretrain_actor(
                model,
                combined,
                epochs=config.dagger_epochs,
                seed=config.seed + iteration + 1,
            )
            add_to_replay_buffer(model, dagger_data)
            metrics = evaluate(
                model,
                episodes=config.eval_episodes,
                seed=config.seed + 100_000,
            )
            print_evaluation(f"DAgger {iteration + 1}", metrics)
            print(
                f"      数据量={len(combined)}，teacher_mix={teacher_mix:.2f}，"
                f"loss={bc_loss:.6f}",
                flush=True,
            )
            history.append({"stage": f"dagger_{iteration + 1}", **metrics})
            if metrics["success_rate"] > best_success_rate:
                best_success_rate = float(metrics["success_rate"])
                model.save(BEST_MODEL_PATH)

        # Behaviour cloning changes the actor after TD3 creates its target copy.
        # Synchronize them before critic warm-up and online RL.
        model.actor_target.load_state_dict(model.actor.state_dict())
        print("\n[3/3] TD3+BC在线强化学习微调...", flush=True)
        model.set_expert_data(
            combined,
            start=1.0,
            end=0.25,
            decay_updates=sum(config.rl_phase_steps),
        )
        print(
            f"      先用 {config.critic_warmup_steps} 次专家回放预热critic...",
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
            metrics = evaluate(
                model,
                episodes=config.eval_episodes,
                seed=config.seed + 100_000,
            )
            label = f"TD3+BC阶段{phase} difficulty={difficulty:.2f}"
            print_evaluation(label, metrics)
            history.append({
                "stage": f"td3_bc_{phase}",
                "difficulty": difficulty,
                "rl_steps": total_rl_steps,
                **metrics,
            })
            if metrics["success_rate"] > best_success_rate:
                best_success_rate = float(metrics["success_rate"])
                model.save(BEST_MODEL_PATH)
            elif metrics["success_rate"] + 0.05 < best_success_rate:
                model.set_parameters(
                    BEST_MODEL_PATH, exact_match=True, device=model.device)
                model.rl_weight *= 0.25
                print(
                    "      RL策略出现明显退化，已恢复最佳权重并降低RL更新权重。",
                    flush=True,
                )
            if (
                total_rl_steps >= 5_000
                and metrics["success_rate"] >= config.target_success_rate
            ):
                print("      已达到目标成功率，提前停止课程训练。", flush=True)
                break

        # Always deploy the best fixed-evaluation checkpoint, never a regressed
        # final online-RL update.
        model.set_parameters(BEST_MODEL_PATH, exact_match=True, device=model.device)
        model.save(MODEL_PATH)
        final_metrics = evaluate(
            model, episodes=config.eval_episodes, seed=config.seed + 200_000)
        print_evaluation("最终独立测试集", final_metrics)
        result = {
            "config": asdict(config),
            "history": history,
            "final": final_metrics,
            "model_path": str(MODEL_PATH),
            "best_model_path": str(BEST_MODEL_PATH),
        }
        METRICS_PATH.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n模型已保存到 {MODEL_PATH}")
        print(f"训练指标已保存到 {METRICS_PATH}")
        return result
    finally:
        data_env.close()
        train_env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Kuka reaching or grasping with IK-guided TD3+BC.")
    parser.add_argument(
        "--task",
        choices=("reach", "grasp"),
        default="reach",
        help="keep the 7-DoF reaching baseline or train the 8-DoF grasp task",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="run a small smoke test instead of convergence training")
    parser.add_argument(
        "--robust", action="store_true",
        help="train grasping with physics, sensor, and latency randomization")
    parser.add_argument(
        "--physical", action="store_true",
        help="train grasping with finger contact/friction and no fixed constraint")
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument(
        "--check-env", action="store_true",
        help="validate the Gymnasium environment and exit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.robust and args.task != "grasp":
        raise SystemExit("--robust is only available with --task grasp")
    if args.physical and args.task != "grasp":
        raise SystemExit("--physical is only available with --task grasp")
    if args.physical and args.robust:
        raise SystemExit(
            "combine physical contact and domain randomization in a later stage; "
            "choose either --physical or --robust")
    if args.task == "grasp":
        from grasp_training import (  # Imported lazily to avoid a cycle.
            GraspTrainingConfig,
            RobotGraspEnv,
            train_grasp,
        )

        if args.check_env:
            env = RobotGraspEnv(
                randomization=1.0 if args.robust else 0.0,
                observe_action_history=args.robust,
                observe_physical_residual=args.physical,
                grasp_constraint_force=0.0 if args.physical else 300.0,
            )
            try:
                check_env(env, warn=True)
                print("Grasping Gymnasium environment check passed.")
            finally:
                env.close()
            return
        if args.physical:
            config = (
                GraspTrainingConfig.quick_physical(args.seed)
                if args.quick
                else GraspTrainingConfig.physical(args.seed)
            )
        elif args.robust:
            config = (
                GraspTrainingConfig.quick_robust(args.seed)
                if args.quick
                else GraspTrainingConfig.robust(args.seed)
            )
        else:
            config = (
                GraspTrainingConfig.quick(args.seed)
                if args.quick
                else GraspTrainingConfig(seed=args.seed)
            )
        train_grasp(config)
        return
    if args.check_env:
        env = RobotArmEnv()
        try:
            check_env(env, warn=True)
            print("Gymnasium environment check passed.")
        finally:
            env.close()
        return
    config = TrainingConfig.quick(args.seed) if args.quick else TrainingConfig(
        seed=args.seed)
    train(config)


if __name__ == "__main__":
    main()
