#!/usr/bin/env python3
"""
示教脚本 - 找到合适的初始关节角度
修改下方的 JOINT_POS 字典中的值，然后重新运行查看效果
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="示教脚本 - 调整关节角度")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab_assets import FRANKA_PANDA_CFG


# ========== 在这里修改关节角度 ==========
# 调整这些值，然后重新运行脚本查看效果
JOINT_POS = {
    "panda_joint1": 0.0,
    "panda_joint2": -0.569,
    "panda_joint3": 0.0,
    "panda_joint4": -2.810,   # 主要调整这个关节来改变高度
    "panda_joint5": 0.0,
    "panda_joint6": 2.5,
    "panda_joint7": 0.741,
    "panda_finger_joint.*": 0.04,
}
# =======================================


def design_scene():
    """设计场景"""
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)

    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # 左机械臂
    sim_utils.create_prim("/World/LeftRobotOrigin", "Xform", translation=(0.0, -0.4, 0.0))
    left_cfg = FRANKA_PANDA_CFG.replace(prim_path="/World/LeftRobotOrigin/Robot")
    left_cfg.actuators["panda_shoulder"].stiffness = 500.0
    left_cfg.actuators["panda_shoulder"].damping = 200.0
    left_cfg.actuators["panda_forearm"].stiffness = 500.0
    left_cfg.actuators["panda_forearm"].damping = 200.0
    left_cfg.init_state.pos = (0.0, 0.0, 0.0)
    left_cfg.init_state.rot = (1.0, 0.0, 0.0, 0.0)
    left_cfg.init_state.joint_pos = JOINT_POS.copy()
    left_robot = Articulation(cfg=left_cfg)

    # 右机械臂
    sim_utils.create_prim("/World/RightRobotOrigin", "Xform", translation=(0.0, 0.4, 0.0))
    right_cfg = FRANKA_PANDA_CFG.replace(prim_path="/World/RightRobotOrigin/Robot")
    right_cfg.actuators["panda_shoulder"].stiffness = 500.0
    right_cfg.actuators["panda_shoulder"].damping = 200.0
    right_cfg.actuators["panda_forearm"].stiffness = 500.0
    right_cfg.actuators["panda_forearm"].damping = 200.0
    right_cfg.init_state.pos = (0.0, 0.0, 0.0)
    right_cfg.init_state.rot = (1.0, 0.0, 0.0, 0.0)
    right_cfg.init_state.joint_pos = JOINT_POS.copy()
    right_robot = Articulation(cfg=right_cfg)

    return {"left": left_robot, "right": right_robot}


def run(sim, entities):
    """运行示教"""
    left = entities["left"]
    right = entities["right"]
    sim_dt = sim.get_physics_dt()

    left_ee_idx = left.find_bodies("panda_hand")[0][0]
    right_ee_idx = right.find_bodies("panda_hand")[0][0]

    print("\n" + "=" * 60)
    print("示教脚本 - 调整关节角度")
    print("=" * 60)
    print("当前关节角度配置:")
    for k, v in JOINT_POS.items():
        if "joint" in k:
            print(f"  {k}: {v:+.4f}")
    print("=" * 60)

    # 打印初始高度
    pos_l = left.data.body_pose_w[0, left_ee_idx].cpu().numpy()[:3]
    pos_r = right.data.body_pose_w[0, right_ee_idx].cpu().numpy()[:3]
    pos_center = (pos_l + pos_r) / 2.0
    print(f"\n>>> 末端协作空间高度: {pos_center[2]:.3f}m <<<")
    print(f"    左末端: [{pos_l[0]:.3f}, {pos_l[1]:.3f}, {pos_l[2]:.3f}]")
    print(f"    右末端: [{pos_r[0]:.3f}, {pos_r[1]:.3f}, {pos_r[2]:.3f}]")
    print("\n修改 JOINT_POS 字典中的值后重新运行来调整高度")
    print("按 Ctrl+C 关闭")
    print("=" * 60)

    count = 0
    while simulation_app.is_running():
        left.update(sim_dt)
        right.update(sim_dt)
        sim.step()
        count += 1

        if count % 100 == 0:
            pos_l = left.data.body_pose_w[0, left_ee_idx].cpu().numpy()[:3]
            pos_r = right.data.body_pose_w[0, right_ee_idx].cpu().numpy()[:3]
            pos_center = (pos_l + pos_r) / 2.0
            q_l = left.data.joint_pos[0].cpu().numpy()[:7]
            print(f"\r[Step {count}] 高度: {pos_center[2]:.3f}m | j4={q_l[3]:+.3f}     ", end="", flush=True)


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 0.0, 1.5], [0.0, 0.0, 0.5])

    entities = design_scene()
    sim.reset()

    run(sim, entities)


if __name__ == "__main__":
    main()
    simulation_app.close()
