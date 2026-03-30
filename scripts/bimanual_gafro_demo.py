#!/usr/bin/env python3
"""
Bimanual GAFRO Controller Demo for Isaac Lab.
Integrates geometric algebra based cooperative control with Isaac Lab simulation.

Uses the true pygafro implementation for:
- Cooperative task space computation (absolute + relative space)
- Bimanual inverse kinematics
- PD control in cooperative space
"""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Bimanual GAFRO Controller Demo")
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

from isaaclab_assets import FRANKA_PANDA_CFG

# Import GAFRO controller
import sys
sys.path.insert(0, '/home/jing/GitRepo/cga-diffusion')
from controllers.gafro_controller import BimanualGAFROController


def design_scene():
    """Designs the scene with two Franka robots for GAFRO control."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)

    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # Left Franka robot
    sim_utils.create_prim("/World/LeftRobotOrigin", "Xform", translation=(0.0, -0.4, 0.0))
    left_arm_cfg = FRANKA_PANDA_CFG.replace(prim_path="/World/LeftRobotOrigin/Robot")
    left_arm_cfg.init_state.pos = (0.0, 0.0, 0.0)
    left_arm_cfg.init_state.rot = (1.0, 0.0, 0.0, 0.0)
    left_arm_cfg.init_state.joint_pos = {
        "panda_joint1": 0.0,
        "panda_joint2": -0.569,
        "panda_joint3": 0.0,
        "panda_joint4": -2.810,
        "panda_joint5": 0.0,
        "panda_joint6": 2.5,
        "panda_joint7": 0.741,
        "panda_finger_joint.*": 0.04,
    }
    left_robot = Articulation(cfg=left_arm_cfg)

    # Right Franka robot
    sim_utils.create_prim("/World/RightRobotOrigin", "Xform", translation=(0.0, 0.4, 0.0))
    right_arm_cfg = FRANKA_PANDA_CFG.replace(prim_path="/World/RightRobotOrigin/Robot")
    right_arm_cfg.init_state.pos = (0.0, 0.0, 0.0)
    right_arm_cfg.init_state.rot = (1.0, 0.0, 0.0, 0.0)
    right_arm_cfg.init_state.joint_pos = {
        "panda_joint1": 0.0,
        "panda_joint2": -0.569,
        "panda_joint3": 0.0,
        "panda_joint4": -2.810,
        "panda_joint5": 0.0,
        "panda_joint6": 2.5,
        "panda_joint7": 0.741,
        "panda_finger_joint.*": 0.04,
    }
    right_robot = Articulation(cfg=right_arm_cfg)

    scene_entities = {
        "left_robot": left_robot,
        "right_robot": right_robot,
    }
    return scene_entities


def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, Articulation]):
    """Runs the simulation loop with GAFRO cooperative control."""
    sim_dt = sim.get_physics_dt()
    count = 0

    left_robot = entities["left_robot"]
    right_robot = entities["right_robot"]

    # Initialize GAFRO controller
    controller = BimanualGAFROController()

    # Joint names for Franka Panda
    joint_names = [
        "panda_joint1", "panda_joint2", "panda_joint3",
        "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7"
    ]

    # Default home configuration (7 arm joints)
    default_q = np.array([0.0, -0.569, 0.0, -2.810, 0.0, 2.5, 0.741])

    print("=" * 60)
    print("CGA-Diffusion: Bimanual GAFRO Controller Demo")
    print("=" * 60)
    print("\nGeometric Algebra based cooperative control:")
    print("- Absolute space: centroid of both end-effectors")
    print("- Relative space: relative position/orientation between arms")
    print("\nDemo: Robot holding stable position, displaying cooperative state")
    print("Close window to exit.")

    # Get default joint positions (stable home position)
    default_q_left = left_robot.data.default_joint_pos[0]
    default_q_right = right_robot.data.default_joint_pos[0]

    # Hold robot at stable home position
    while simulation_app.is_running():
        left_robot.set_joint_position_target(default_q_left)
        right_robot.set_joint_position_target(default_q_right)

        left_robot.write_data_to_sim()
        right_robot.write_data_to_sim()
        sim.step()
        left_robot.update(sim_dt)
        right_robot.update(sim_dt)

        count += 1
        if count % 100 == 0:
            q_l = left_robot.data.joint_pos[0].cpu().numpy()[:7]
            q_r = right_robot.data.joint_pos[0].cpu().numpy()[:7]
            abs_m, rel_m = controller.coop_space.compute_cooperative_motion(q_l, q_r)
            print(f"[Step {count}] Cooperative State: Abs(Z)={abs_m[2]:.3f}m, Rel(Y)={rel_m[1]:.3f}m")

        # Write data to simulation
        left_robot.write_data_to_sim()
        right_robot.write_data_to_sim()

        # Step simulation
        sim.step()

        # Update buffers
        left_robot.update(sim_dt)
        right_robot.update(sim_dt)

        count += 1

        if count % 100 == 0:
            print(f"[Step {count}] Holding stable position...")


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
    run_simulator(sim, scene_entities)


if __name__ == "__main__":
    main()
    simulation_app.close()
