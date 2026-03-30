#!/usr/bin/env python3
"""
Bimanual GAFRO Impedance Control Demo for Isaac Lab.
Implements Paper 1's cooperative impedance control using geometric algebra.

This version uses impedance control (torque) instead of position control.
"""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Bimanual GAFRO Impedance Control Demo")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers

from isaaclab_assets import FRANKA_PANDA_CFG
from isaaclab.actuators import ImplicitActuatorCfg

# Import GAFRO controller
import sys
sys.path.insert(0, '/home/jing/GitRepo/cga-diffusion')
from controllers.gafro_controller import BimanualGAFROController
from utils.gafro_utils import BimanualCooperativeSpace


def design_scene():
    """Designs the scene with two Franka robots for GAFRO control."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)

    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # Left Franka robot - create LOW stiffness config for torque control
    sim_utils.create_prim("/World/LeftRobotOrigin", "Xform", translation=(0.0, -0.4, 0.0))
    left_arm_cfg = FRANKA_PANDA_CFG.replace(prim_path="/World/LeftRobotOrigin/Robot")
    # Set stiffness=0 for impedance/torque control (no damping - let impedance control handle it)
    left_arm_cfg.actuators["panda_shoulder"].stiffness = 0.0
    left_arm_cfg.actuators["panda_shoulder"].damping = 0.0
    left_arm_cfg.actuators["panda_forearm"].stiffness = 0.0
    left_arm_cfg.actuators["panda_forearm"].damping = 0.0
    left_arm_cfg.init_state.pos = (0.0, 0.0, 0.0)
    left_arm_cfg.init_state.rot = (1.0, 0.0, 0.0, 0.0)
    left_arm_cfg.init_state.joint_pos = {
        "panda_joint1": 0.0,
        "panda_joint2": -0.569,
        "panda_joint3": 0.0,
        "panda_joint4": -2.5,
        "panda_joint5": 0.0,
        "panda_joint6": 2.5,
        "panda_joint7": 0.741,
        "panda_finger_joint.*": 0.04,
    }
    left_robot = Articulation(cfg=left_arm_cfg)

    # Right Franka robot - create LOW stiffness config for torque control
    sim_utils.create_prim("/World/RightRobotOrigin", "Xform", translation=(0.0, 0.4, 0.0))
    right_arm_cfg = FRANKA_PANDA_CFG.replace(prim_path="/World/RightRobotOrigin/Robot")
    # Set stiffness=0 for impedance/torque control (no damping - let impedance control handle it)
    right_arm_cfg.actuators["panda_shoulder"].stiffness = 0.0
    right_arm_cfg.actuators["panda_shoulder"].damping = 0.0
    right_arm_cfg.actuators["panda_forearm"].stiffness = 0.0
    right_arm_cfg.actuators["panda_forearm"].damping = 0.0
    right_arm_cfg.init_state.pos = (0.0, 0.0, 0.0)
    right_arm_cfg.init_state.rot = (1.0, 0.0, 0.0, 0.0)
    right_arm_cfg.init_state.joint_pos = {
        "panda_joint1": 0.0,
        "panda_joint2": -0.569,
        "panda_joint3": 0.0,
        "panda_joint4": -2.5,
        "panda_joint5": 0.0,
        "panda_joint6": 2.5,
        "panda_joint7": 0.741,
        "panda_finger_joint.*": 0.04,
    }
    right_robot = Articulation(cfg=right_arm_cfg)

    # Visualization markers for target and current positions
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/World/VisualizationMarkers",
        markers={
            "target": sim_utils.SphereCfg(
                radius=0.05,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),  # Red
            ),
            "current": sim_utils.SphereCfg(
                radius=0.05,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),  # Green
            ),
        },
    )
    marker = VisualizationMarkers(marker_cfg)

    scene_entities = {
        "left_robot": left_robot,
        "right_robot": right_robot,
        "marker": marker,
    }
    return scene_entities


def run_impedance_control(sim: sim_utils.SimulationContext, entities: dict[str, Articulation]):
    """Runs the simulation loop with GAFRO cooperative impedance control."""
    sim_dt = sim.get_physics_dt()
    count = 0

    left_robot = entities["left_robot"]
    right_robot = entities["right_robot"]
    marker = entities["marker"]

    # Initialize GAFRO controller
    controller = BimanualGAFROController()
    coop_space = controller.coop_space

    print("=" * 60)
    print("CGA-Diffusion: Bimanual GAFRO Cooperative Impedance Control")
    print("=" * 60)
    print("\nUsing Paper 1's cooperative task space with impedance control:")
    print("- Absolute space: centroid of both end-effectors")
    print("- Relative space: relative position/orientation between arms")
    print("- J^T transformation for wrench to joint torques")
    print("\nDemo: Robots hold position with impedance control")
    print("Close window to exit.")

    # Print initial joint positions and EE height
    # Cache the end-effector frame indices
    left_ee_idx = left_robot.find_bodies("panda_hand")[0][0]
    right_ee_idx = right_robot.find_bodies("panda_hand")[0][0]

    # Initial joint positions
    default_q_left = left_robot.data.default_joint_pos[0].cpu().numpy()[:7]
    default_q_right = right_robot.data.default_joint_pos[0].cpu().numpy()[:7]

    # Print initial joint positions and EE height
    print("\n=== 初始状态 ===")
    print(f"左臂关节角度: {default_q_left}")
    print(f"右臂关节角度: {default_q_right}")

    # Target cooperative state (use current state as target - hold position)
    target_abs = None  # Will be set to current state
    target_rel = None

    # Test: change target position every 500 steps to verify impedance behavior
    step_count = 0
    target_y_offset = 0.0  # Will oscillate left-right to test response

    # Joint data history for vibration analysis
    joint_pos_history_l = []
    joint_vel_history_l = []
    joint_effort_history_l = []

    while simulation_app.is_running():
        step_count += 1
        # Get current joint positions
        q_l = left_robot.data.joint_pos[0].cpu().numpy()[:7]
        q_r = right_robot.data.joint_pos[0].cpu().numpy()[:7]

        # Get end-effector poses from Isaac Lab
        pos_left = left_robot.data.body_pose_w[0, left_ee_idx].cpu().numpy()[:3]
        quat_left = left_robot.data.body_pose_w[0, left_ee_idx].cpu().numpy()[3:7]
        pos_right = right_robot.data.body_pose_w[0, right_ee_idx].cpu().numpy()[:3]
        quat_right = right_robot.data.body_pose_w[0, right_ee_idx].cpu().numpy()[3:7]

        # Compute current absolute position in WORLD coordinates (centroid of both EEs)
        current_abs_world = (pos_left + pos_right) / 2.0
        # Compute current absolute orientation (SLERP between both EEs)
        current_quat_world = controller._slerp_quat(quat_left, quat_right, 0.5)

        # Initialize target on first iteration - use fixed absolute height
        if target_abs is None:
            # Fixed target height - adjust this to change desired EE height
            fixed_z_height = 0.60  # Target height in meters
            target_abs_pos = np.array([current_abs_world[0], 0.0, fixed_z_height])  # Y=0 for bimanual center
            target_abs = np.concatenate([target_abs_pos, current_quat_world])
            # target_rel is [pos(3) + quat(4)] = 7 elements
            target_rel_quat = controller._multiply_quat(quat_right, controller._invert_quat(quat_left))
            target_rel = np.concatenate([pos_right - pos_left, target_rel_quat])
            print(f"\n>>> 目标高度设为: {fixed_z_height:.3f}m")

        # Test: move target position every 300 steps to verify impedance behavior
        if step_count > 0 and step_count % 300 == 0:
            # Oscillate target Y position (left-right)
            if target_y_offset == 0.0:
                target_y_offset = 0.08  # Move RIGHT 8cm
            else:
                target_y_offset = 0.0   # Move back to original
            target_abs[1] = 0.0 + target_y_offset
            print(f"\n>>> [Step {step_count}] 目标Y改为 {target_abs[1]:.3f}m")
            print(">>> 机器人平滑移动(弹簧式) = 阻抗控制正常!")
            print(">>> 机器人瞬移 = 位置控制(有bug)")

        # Compute cooperative impedance control
        # Returns wrench_left, wrench_right [6] = [force, torque]
        wrench_left, wrench_right = controller.compute_control(
            q_l, q_r,
            pos_left, quat_left,
            pos_right, quat_right,
            target_abs, target_rel
        )

        # Update visualization markers
        # Target marker (red) at target_abs position, Current marker (green) at current_abs_world position
        marker.visualize(
            translations=np.array([target_abs[:3], current_abs_world]),  # Target (red), Current (green)
            marker_indices=[0, 1],  # 0=target (red sphere), 1=current (green sphere)
        )

        # Convert wrenches to joint torques using pygafro's inverse dynamics
        torques_left, torques_right = coop_space.compute_joint_torques_bimanual(
            q_l, q_r, wrench_left, wrench_right
        )

        # === KEY CHANGE: Use impedance control instead of position control ===
        # Convert numpy to torch tensor for Isaac Lab
        # Isaac Lab has 9 joints (7 arm + 2 fingers), we only control arm joints
        effort_left = torch.zeros(9, device=left_robot.device)
        effort_right = torch.zeros(9, device=right_robot.device)
        effort_left[:7] = torch.from_numpy(torques_left).float()
        effort_right[:7] = torch.from_numpy(torques_right).float()
        left_robot.set_joint_effort_target(effort_left)
        right_robot.set_joint_effort_target(effort_right)

        # Write data to simulation
        left_robot.write_data_to_sim()
        right_robot.write_data_to_sim()

        # Record joint data for vibration analysis
        joint_pos_history_l.append(q_l.copy())
        joint_vel_history_l.append(left_robot.data.joint_vel[0].cpu().numpy()[:7].copy())
        joint_effort_history_l.append(torques_left.copy())

        # Step simulation
        sim.step()

        # Update buffers
        left_robot.update(sim_dt)
        right_robot.update(sim_dt)

        count += 1

        # Analyze joint vibration every 100 steps
        if count % 100 == 0 and len(joint_pos_history_l) >= 100:
            recent_pos = np.array(joint_pos_history_l[-100:])
            recent_vel = np.array(joint_vel_history_l[-100:])
            recent_effort = np.array(joint_effort_history_l[-100:])

            # Compute position change between consecutive steps (jitter indicator)
            pos_diff = np.diff(recent_pos, axis=0)
            vel_diff = np.diff(recent_vel, axis=0)

            # Joint names for display
            joint_names = ['j1', 'j2', 'j3', 'j4', 'j5', 'j6', 'j7']
            max_jitter_idx = np.argmax(np.std(pos_diff, axis=0))
            max_vel_jitter_idx = np.argmax(np.std(vel_diff, axis=0))
            # Use world coordinates for consistency
            current_abs_world = (pos_left + pos_right) / 2.0
            current_rel_world = pos_right - pos_left
            gains = controller.get_adapted_gains()

            # Compute position and velocity std for each joint
            pos_std = np.std(pos_diff, axis=0)
            vel_std = np.std(vel_diff, axis=0)

            print(f"[Step {count}]")
            print(f"  目标(Y): {target_abs[1]:.3f}m, 当前(Y): {current_abs_world[1]:.3f}m, 误差: {abs(target_abs[1] - current_abs_world[1]):.3f}m")
            print(f"  目标(Z): {target_abs[2]:.3f}m, 当前(Z): {current_abs_world[2]:.3f}m")
            print(f"  |左臂力矩|: {np.linalg.norm(torques_left):.2f} Nm, |右臂力矩|: {np.linalg.norm(torques_right):.2f} Nm")
            # Print joint position jitter (most significant joint)
            print(f"  末端关节抖动: {joint_names[max_jitter_idx]}={pos_std[max_jitter_idx]*1000:.3f}mm (std)")
            print(f"  末端关节速度抖动: {joint_names[max_vel_jitter_idx]}={vel_std[max_vel_jitter_idx]:.3f}rad/s (std)")


def main():
    """Main function."""
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)

    # Set camera view
    sim.set_camera_view([2.5, 0.0, 1.5], [0.0, 0.0, 0.5])

    # Design scene
    scene_entities = design_scene()

    # Play the simulator
    sim.reset()

    # Run the simulator
    run_impedance_control(sim, scene_entities)


if __name__ == "__main__":
    main()
    simulation_app.close()
