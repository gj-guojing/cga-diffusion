#!/usr/bin/env python3
"""
Bimanual Official CGA Impedance Control Demo for Isaac Lab.
Implements Paper 1's cooperative impedance control using the official gafro-group algorithm.

Key features:
1. 12x(2*dof) combined Jacobian (6 absolute + 6 relative task space)
2. Damped pseudoinverse: J_pinv = (J^T*J + εI)^-1 * J^T
3. J_dot compensation for velocity-dependent effects
4. Acceleration-based admittance control with inertia, damping, stiffness matrices
"""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Bimanual Official CGA Impedance Control Demo")
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

# Import official CGA controller
import sys
sys.path.insert(0, '/home/jing/GitRepo/cga-diffusion')
from controllers.official_gafro_controller import OfficialCGAAdmittanceController


def design_scene():
    """Designs the scene with two Franka robots for CGA control."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)

    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # Left Franka robot - low stiffness for impedance control
    sim_utils.create_prim("/World/LeftRobotOrigin", "Xform", translation=(0.0, -0.4, 0.0))
    left_arm_cfg = FRANKA_PANDA_CFG.replace(prim_path="/World/LeftRobotOrigin/Robot")
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
        "panda_joint4": -2.810,   # 更弯曲的配置
        "panda_joint5": 0.0,
        "panda_joint6": 2.5,
        "panda_joint7": 0.741,
        "panda_finger_joint.*": 0.04,
    }
    left_robot = Articulation(cfg=left_arm_cfg)

    # Right Franka robot
    sim_utils.create_prim("/World/RightRobotOrigin", "Xform", translation=(0.0, 0.4, 0.0))
    right_arm_cfg = FRANKA_PANDA_CFG.replace(prim_path="/World/RightRobotOrigin/Robot")
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
        "panda_joint4": -2.810,   # 更弯曲的配置
        "panda_joint5": 0.0,
        "panda_joint6": 2.5,
        "panda_joint7": 0.741,
        "panda_finger_joint.*": 0.04,
    }
    right_robot = Articulation(cfg=right_arm_cfg)

    # Visualization markers
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/World/VisualizationMarkers",
        markers={
            "target": sim_utils.SphereCfg(
                radius=0.05,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),  # Red
            ),
            "current": sim_utils.SphereCfg(
                radius=0.04,
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
    """Runs the simulation loop with official CGA cooperative impedance control."""
    sim_dt = sim.get_physics_dt()
    count = 0

    left_robot = entities["left_robot"]
    right_robot = entities["right_robot"]
    marker = entities["marker"]

    # Initialize official CGA controller
    controller = OfficialCGAAdmittanceController()

    print("=" * 60)
    print("CGA-Diffusion: Official CGA Cooperative Impedance Control")
    print("=" * 60)
    print("\nUsing official gafro-group algorithm:")
    print("- 12x14 combined Jacobian (6 absolute + 6 relative)")
    print("- Damped pseudoinverse with J_dot compensation")
    print("- Acceleration-based admittance control")
    print("\nDemo: Robots track target with impedance control")
    print("Close window to exit.")

    # Cache the end-effector frame indices
    left_ee_idx = left_robot.find_bodies("panda_hand")[0][0]
    right_ee_idx = right_robot.find_bodies("panda_hand")[0][0]

    # Initial joint positions
    default_q_left = left_robot.data.default_joint_pos[0].cpu().numpy()[:7]
    default_q_right = right_robot.data.default_joint_pos[0].cpu().numpy()[:7]

    print("\n=== Initial State ===")
    print(f"Left arm joints: {default_q_left}")
    print(f"Right arm joints: {default_q_right}")

    # Target cooperative state
    target_abs = None
    target_rel = None
    step_count = 0
    target_y_offset = 0.0

    # Joint data history
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

        # Compute current absolute position (centroid)
        current_abs_world = (pos_left + pos_right) / 2.0

        # Initialize target on first iteration
        if target_abs is None:
            fixed_z_height = 0.8  # Target height
            # Use actual initial orientation (SLERP between both EEs) as target to maintain natural pose
            target_quat = controller._slerp_quat(quat_left, quat_right, 0.5)
            target_abs_pos = np.array([current_abs_world[0], 0.0, fixed_z_height])
            target_abs = np.concatenate([target_abs_pos, target_quat])
            # Use actual initial relative orientation to maintain arm configuration
            rel_quat = controller._multiply_quat(quat_right, controller._invert_quat(quat_left))
            target_rel = np.concatenate([pos_right - pos_left, rel_quat])
            print(f"\n>>> Target height set to: {fixed_z_height:.3f}m")
            print(f">>> Target orientation: {target_quat}")

        # Test: move target Y every 300 steps (oscillate between -0.2 and +0.2)
        if step_count > 0 and step_count % 300 == 0:
            if target_y_offset == 0.0:
                target_y_offset = 0.2
            elif target_y_offset > 0:
                target_y_offset = -0.2
            else:
                target_y_offset = 0.2
            # Update absolute Y position (centroid)
            target_abs[1] = target_y_offset
            # target_rel[:3] contains the relative position between arms (right - left)
            # So target_rel[:3] should NOT change when we move the centroid
            # The left and right targets are computed as: abs +/- rel/2
            print(f"\n>>> [Step {step_count}] Target Y = {target_abs[1]:.3f}m")

        # Compute control using official CGA algorithm
        torques_left, torques_right = controller.compute_control(
            q_l, q_r,
            pos_left, quat_left,
            pos_right, quat_right,
            target_abs, target_rel
        )

        # Update visualization markers
        marker.visualize(
            translations=np.array([target_abs[:3], current_abs_world]),
            marker_indices=[0, 1],
        )

        # Apply torques
        effort_left = torch.zeros(9, device=left_robot.device)
        effort_right = torch.zeros(9, device=right_robot.device)
        effort_left[:7] = torch.from_numpy(torques_left).float()
        effort_right[:7] = torch.from_numpy(torques_right).float()
        left_robot.set_joint_effort_target(effort_left)
        right_robot.set_joint_effort_target(effort_right)

        # Write data to simulation
        left_robot.write_data_to_sim()
        right_robot.write_data_to_sim()

        # Record data
        joint_pos_history_l.append(q_l.copy())
        joint_vel_history_l.append(left_robot.data.joint_vel[0].cpu().numpy()[:7].copy())
        joint_effort_history_l.append(torques_left.copy())

        # Step simulation
        sim.step()
        left_robot.update(sim_dt)
        right_robot.update(sim_dt)
        count += 1

        # Print status every 100 steps
        if count % 100 == 0 and len(joint_pos_history_l) >= 100:
            recent_pos = np.array(joint_pos_history_l[-100:])
            pos_diff = np.diff(recent_pos, axis=0)

            joint_names = ['j1', 'j2', 'j3', 'j4', 'j5', 'j6', 'j7']
            max_jitter_idx = np.argmax(np.std(pos_diff, axis=0))

            print(f"[Step {count}]")
            print(f"  Target(Y): {target_abs[1]:.3f}m, Current(Y): {current_abs_world[1]:.3f}m")
            print(f"  Target(Z): {target_abs[2]:.3f}m, Current(Z): {current_abs_world[2]:.3f}m")
            print(f"  |Torques L|: {np.linalg.norm(torques_left):.2f} Nm")
            print(f"  |Torques R|: {np.linalg.norm(torques_right):.2f} Nm")
            print(f"  Jitter: {joint_names[max_jitter_idx]}={np.std(pos_diff, axis=0)[max_jitter_idx]*1000:.3f}mm")


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
