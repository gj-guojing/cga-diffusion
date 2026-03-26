"""
Bimanual Franka Panda environment for Isaac Lab.
Two Franka robots arranged for cooperative manipulation tasks.
"""

from __future__ import annotations

import torch
from typing import ClassVar

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR


@configclass
class BimanualFrankaSceneCfg(InteractiveSceneCfg):
    """Configuration for a bimanual Franka scene."""

    # Ground plane
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # Left Franka robot
    robot_left = ArticulationCfg(
        prim_path="/World/envs/env_.*/RobotLeft",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(-0.45, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),
            joint_pos={
                "panda_joint1": 0.5,
                "panda_joint2": -0.569,
                "panda_joint3": 0.0,
                "panda_joint4": -2.810,
                "panda_joint5": 0.0,
                "panda_joint6": 2.5,
                "panda_joint7": 0.741,
                "panda_finger_joint.*": 0.04,
            },
        ),
        actuators={
            "panda_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                effort_limit_sim=87.0,
                stiffness=80.0,
                damping=4.0,
            ),
            "panda_forearm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                effort_limit_sim=12.0,
                stiffness=80.0,
                damping=4.0,
            ),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint.*"],
                effort_limit_sim=200.0,
                stiffness=2e3,
                damping=1e2,
            ),
        },
    )

    # Right Franka robot
    robot_right = ArticulationCfg(
        prim_path="/World/envs/env_.*/RobotRight",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.45, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),
            joint_pos={
                "panda_joint1": -0.5,
                "panda_joint2": -0.569,
                "panda_joint3": 0.0,
                "panda_joint4": -2.810,
                "panda_joint5": 0.0,
                "panda_joint6": 2.5,
                "panda_joint7": 0.741,
                "panda_finger_joint.*": 0.04,
            },
        ),
        actuators={
            "panda_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                effort_limit_sim=87.0,
                stiffness=80.0,
                damping=4.0,
            ),
            "panda_forearm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                effort_limit_sim=12.0,
                stiffness=80.0,
                damping=4.0,
            ),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint.*"],
                effort_limit_sim=200.0,
                stiffness=2e3,
                damping=1e2,
            ),
        },
    )

    # Lights
    light = sim_utils.DomeLightCfg(
        prim_path="/World/Light",
        intensity=2000.0,
        color=(0.75, 0.75, 0.75),
    )


@configclass
class BimanualFrankaEnvCfg(DirectRLEnvCfg):
    """Configuration for the bimanual Franka environment."""

    # Env settings
    episode_length_s = 10.0
    decimation = 2
    num_actions = 18  # 9 + 9 for both arms (7 joints + 2 grippers)
    num_observations = 46  # Observation space size
    num_states = 0

    # Simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # Scene
    scene: BimanualFrankaSceneCfg = BimanualFrankaSceneCfg(
        num_envs=1,
        env_spacing=2.5,
        replicate_physics=False,
        clone_in_fabric=False,
    )


class BimanualFrankaEnv(DirectRLEnv):
    """Bimanual Franka environment for cooperative manipulation tasks.

    This environment provides a basic setup with two Franka Panda robots
    arranged facing each other, suitable for cooperative manipulation tasks.
    """

    cfg: BimanualFrankaEnvCfg

    def __init__(self, cfg: BimanualFrankaEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        print(f"[BimanualFrankaEnv] Initialized with {self.num_envs} environments")

    def _setup_scene(self):
        """Setup the scene with both robots."""
        # Add robots to the scene
        self._robot_left = Articulation(self.cfg.scene.robot_left)
        self._robot_right = Articulation(self.cfg.scene.robot_right)
        self.scene.articulations["robot_left"] = self._robot_left
        self.scene.articulations["robot_right"] = self._robot_right

        # Setup terrain
        self.cfg.scene.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.scene.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.scene.terrain.class_type(self.cfg.scene.terrain)

        # Clone environments
        self.scene.clone_environments(copy_from_source=False)

        # Filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.scene.terrain.prim_path])

        # Add lights
        light_cfg = self.cfg.scene.light
        light_cfg.func("/World/Light", light_cfg)

        # Store joint indices for convenience
        self._left_arm_dof_idx = self._robot_left.find_joints("panda_joint.*")[0]
        self._right_arm_dof_idx = self._robot_right.find_joints("panda_joint.*")[0]

        # Get end-effector body indices
        self._left_ee_idx = self._robot_left.find_bodies("panda_link7")[0][0]
        self._right_ee_idx = self._robot_right.find_bodies("panda_link7")[0][0]

        print("[BimanualFrankaEnv] Scene setup complete")

    def _pre_physics_step(self, actions: torch.Tensor):
        """Process actions before physics step."""
        self.actions = actions.clone()
        # Split actions for left and right robots
        actions_left = actions[:, :9]
        actions_right = actions[:, 9:]

        # Get joint limits
        left_lower = self._robot_left.data.soft_joint_pos_limits[0, :, 0]
        left_upper = self._robot_left.data.soft_joint_pos_limits[0, :, 1]
        right_lower = self._robot_right.data.soft_joint_pos_limits[0, :, 0]
        right_upper = self._robot_right.data.soft_joint_pos_limits[0, :, 1]

        # Compute targets
        dt = self.cfg.sim.dt * self.cfg.decimation
        self._left_targets = self._robot_left.data.joint_pos + actions_left * dt
        self._right_targets = self._robot_right.data.joint_pos + actions_right * dt

        # Clamp to joint limits
        self._left_targets = torch.clamp(self._left_targets, left_lower, left_upper)
        self._right_targets = torch.clamp(self._right_targets, right_lower, right_upper)

    def _apply_action(self):
        """Apply actions to the simulation."""
        self._robot_left.set_joint_position_target(self._left_targets)
        self._robot_right.set_joint_position_target(self._right_targets)

    def _get_observations(self) -> dict:
        """Get observations from the environment."""
        # Left arm observations
        left_joint_pos = self._robot_left.data.joint_pos
        left_joint_vel = self._robot_left.data.joint_vel
        left_ee_pos = self._robot_left.data.body_pos_w[:, self._left_ee_idx]
        left_ee_quat = self._robot_left.data.body_quat_w[:, self._left_ee_idx]

        # Right arm observations
        right_joint_pos = self._robot_right.data.joint_pos
        right_joint_vel = self._robot_right.data.joint_vel
        right_ee_pos = self._robot_right.data.body_pos_w[:, self._right_ee_idx]
        right_ee_quat = self._robot_right.data.body_quat_w[:, self._right_ee_idx]

        # Combine all observations
        obs = torch.cat(
            [
                left_joint_pos,
                left_joint_vel,
                left_ee_pos,
                left_ee_quat,
                right_joint_pos,
                right_joint_vel,
                right_ee_pos,
                right_ee_quat,
            ],
            dim=-1,
        )

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """Get rewards - placeholder for cooperative tasks."""
        # Simple reward for staying alive
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get done flags."""
        # Episode length termination
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        terminated = torch.zeros_like(truncated)
        return terminated, truncated

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset specified environments."""
        if env_ids is None:
            env_ids = self._robot_left._ALL_INDICES

        super()._reset_idx(env_ids)

        # Reset left robot
        left_default_pos = self._robot_left.data.default_joint_pos[env_ids]
        left_default_vel = torch.zeros_like(left_default_pos)
        self._robot_left.write_joint_state_to_sim(
            left_default_pos, left_default_vel, env_ids=env_ids
        )

        # Reset right robot
        right_default_pos = self._robot_right.data.default_joint_pos[env_ids]
        right_default_vel = torch.zeros_like(right_default_pos)
        self._robot_right.write_joint_state_to_sim(
            right_default_pos, right_default_vel, env_ids=env_ids
        )

    # Helper methods for accessing robot states
    def get_left_ee_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get left end-effector position and quaternion."""
        pos = self._robot_left.data.body_pos_w[:, self._left_ee_idx]
        quat = self._robot_left.data.body_quat_w[:, self._left_ee_idx]
        return pos, quat

    def get_right_ee_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get right end-effector position and quaternion."""
        pos = self._robot_right.data.body_pos_w[:, self._right_ee_idx]
        quat = self._robot_right.data.body_quat_w[:, self._right_ee_idx]
        return pos, quat
