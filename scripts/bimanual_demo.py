#!/usr/bin/env python3
"""
Simple bimanual Franka demo for Isaac Lab.
Displays two Franka Panda robots side by side.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Bimanual Franka Panda demo.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, ISAAC_NUCLEUS_DIR

from isaaclab_assets import FRANKA_PANDA_CFG


def design_scene():
    """Designs the scene with two Franka robots."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)

    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # Left Franka robot (on the left, Y negative)
    sim_utils.create_prim("/World/LeftRobotOrigin", "Xform", translation=(0.0, -0.4, 0.0))
    left_arm_cfg = FRANKA_PANDA_CFG.replace(prim_path="/World/LeftRobotOrigin/Robot")
    left_arm_cfg.init_state.pos = (0.0, 0.0, 0.0)
    left_arm_cfg.init_state.rot = (1.0, 0.0, 0.0, 0.0)  # Identity first, try with no rotation
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

    # Right Franka robot (on the right, Y positive)
    sim_utils.create_prim("/World/RightRobotOrigin", "Xform", translation=(0.0, 0.4, 0.0))
    right_arm_cfg = FRANKA_PANDA_CFG.replace(prim_path="/World/RightRobotOrigin/Robot")
    right_arm_cfg.init_state.pos = (0.0, 0.0, 0.0)
    right_arm_cfg.init_state.rot = (1.0, 0.0, 0.0, 0.0)  # Identity first, try with no rotation
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

    # Return scene entities
    scene_entities = {
        "left_robot": left_robot,
        "right_robot": right_robot,
    }
    return scene_entities


def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, Articulation]):
    """Runs the simulation loop."""
    sim_dt = sim.get_physics_dt()
    count = 0

    left_robot = entities["left_robot"]
    right_robot = entities["right_robot"]

    print("=" * 60)
    print("CGA-Diffusion: Bimanual Franka Viewer")
    print("=" * 60)
    print("\nViewer ready! Close the window to exit.")
    print("Left robot at: (0.0, -0.4, 0.0)")
    print("Right robot at: (0.0, 0.4, 0.0)")

    while simulation_app.is_running():
        # Hold position - no random motions
        for robot in [left_robot, right_robot]:
            robot.set_joint_position_target(robot.data.default_joint_pos)
            robot.write_data_to_sim()

        # Step simulation
        sim.step()

        # Update buffers
        for robot in [left_robot, right_robot]:
            robot.update(sim_dt)

        count += 1


def main():
    """Main function."""
    # Initialize simulation context
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
