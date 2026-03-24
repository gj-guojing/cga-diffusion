#!/usr/bin/env python3
"""
Isaac Lab bimanual manipulation demo.
Tests the integration of GAFRO and Diffusion controllers.
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List, Dict

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from controllers.gafro_controller import BimanualGAFROController


class SimpleSimEnv:
    """Simple simulation environment placeholder."""

    def __init__(self):
        self.dt = 0.01
        self.current_time = 0.0

        # Initial state
        self.pos_left = np.array([-0.2, 0.0, 0.5])
        self.pos_right = np.array([0.2, 0.0, 0.5])
        self.quat_left = np.array([1.0, 0.0, 0.0, 0.0])
        self.quat_right = np.array([1.0, 0.0, 0.0, 0.0])
        self.q_left = np.zeros(7)
        self.q_right = np.zeros(7)

    def reset(self):
        """Reset environment."""
        self.current_time = 0.0
        self.pos_left = np.array([-0.2, 0.0, 0.5])
        self.pos_right = np.array([0.2, 0.0, 0.5])
        return self._get_obs()

    def step(self, wrench_left, wrench_right):
        """Step simulation."""
        self.current_time += self.dt

        # Simple kinematic update
        k_pos = 0.001
        self.pos_left += k_pos * wrench_left[:3] * self.dt
        self.pos_right += k_pos * wrench_right[:3] * self.dt

        # Simulate contact forces
        self._update_contact_forces()

        return self._get_obs(), 0.0, self.current_time > 10.0, {}

    def _get_obs(self):
        """Get observation."""
        return {
            "joint_pos_left": self.q_left,
            "joint_pos_right": self.q_right,
            "ee_pos_left": self.pos_left.copy(),
            "ee_pos_right": self.pos_right.copy(),
            "ee_quat_left": self.quat_left.copy(),
            "ee_quat_right": self.quat_right.copy(),
            "force_left": np.random.randn(3) * 0.5,
            "force_right": np.random.randn(3) * 0.5,
            "moment_left": np.zeros(3),
            "moment_right": np.zeros(3),
        }

    def _update_contact_forces(self):
        """Simulate contact forces."""
        pass


def main():
    """Main demo function."""
    print("=" * 60)
    print("CGA-Diffusion: Isaac Lab Bimanual Demo")
    print("=" * 60)

    # Create environment and controller
    env = SimpleSimEnv()
    controller = BimanualGAFROController()

    # Reset
    obs = env.reset()

    # History for plotting
    history: Dict[str, List[np.ndarray]] = {
        "time": [],
        "pos_left": [],
        "pos_right": [],
        "force_left": [],
        "force_right": [],
    }

    print("\nStarting simulation...")

    # Run simulation
    num_steps = 500
    for step in range(num_steps):
        # Targets: absolute (center) and relative (0.4m apart)
        t = step * env.dt
        target_abs = np.array([
            0.0 + 0.05 * np.sin(t * 0.5),  # X
            0.0,  # Y
            0.5,  # Z
            1.0, 0.0, 0.0, 0.0  # quat
        ])
        target_rel = np.array([
            0.4, 0.0, 0.0,  # X, Y, Z
            1.0, 0.0, 0.0, 0.0  # quat
        ])

        # Compute control
        wrench_left, wrench_right = controller.compute_control(
            obs["joint_pos_left"],
            obs["joint_pos_right"],
            obs["ee_pos_left"],
            obs["ee_quat_left"],
            obs["ee_pos_right"],
            obs["ee_quat_right"],
            target_abs,
            target_rel
        )

        # Step environment
        obs, reward, done, info = env.step(wrench_left, wrench_right)

        # Record history
        history["time"].append(env.current_time)
        history["pos_left"].append(obs["ee_pos_left"].copy())
        history["pos_right"].append(obs["ee_pos_right"].copy())
        history["force_left"].append(obs["force_left"].copy())
        history["force_right"].append(obs["force_right"].copy())

        # Print progress
        if (step + 1) % 50 == 0:
            pos_l = obs["ee_pos_left"]
            pos_r = obs["ee_pos_right"]
            force_l = np.linalg.norm(obs["force_left"])
            print(f"Step {step+1}/{num_steps} | "
                  f"Left: [{pos_l[0]:.3f}, {pos_l[1]:.3f}, {pos_l[2]:.3f}] | "
                  f"Force: {force_l:.1f}N")

        if done:
            print("Episode done, resetting...")
            obs = env.reset()

    print("\nSimulation complete! Plotting results...")

    # Plot results
    plot_results(history)


def plot_results(history: Dict):
    """Plot simulation results."""
    time = np.array(history["time"])
    pos_left = np.array(history["pos_left"])
    pos_right = np.array(history["pos_right"])
    force_left = np.array(history["force_left"])
    force_right = np.array(history["force_right"])

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))

    # Position - X
    axes[0, 0].plot(time, pos_left[:, 0], label="Left X", color="blue")
    axes[0, 0].plot(time, pos_right[:, 0], label="Right X", color="red")
    axes[0, 0].set_ylabel("Position X [m]")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Position - Y
    axes[1, 0].plot(time, pos_left[:, 1], label="Left Y", color="blue")
    axes[1, 0].plot(time, pos_right[:, 1], label="Right Y", color="red")
    axes[1, 0].set_ylabel("Position Y [m]")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Position - Z
    axes[2, 0].plot(time, pos_left[:, 2], label="Left Z", color="blue")
    axes[2, 0].plot(time, pos_right[:, 2], label="Right Z", color="red")
    axes[2, 0].set_xlabel("Time [s]")
    axes[2, 0].set_ylabel("Position Z [m]")
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)

    # Forces - X
    axes[0, 1].plot(time, force_left[:, 0], label="Left Force X", color="blue", alpha=0.7)
    axes[0, 1].plot(time, force_right[:, 0], label="Right Force X", color="red", alpha=0.7)
    axes[0, 1].set_ylabel("Contact Force X [N]")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Placeholder for stiffness
    axes[1, 1].text(0.5, 0.5, "GAFRO Controller Only\n(Constant stiffness)",
                 ha='center', va='center', transform=axes[1, 1].transAxes,
                 fontsize=12)
    axes[1, 1].set_ylabel("Stiffness Y [N/m]")

    # Distance between arms
    dist = np.linalg.norm(pos_right - pos_left, axis=1)
    axes[2, 1].plot(time, dist, label="Arm Distance", color="green", linewidth=2)
    axes[2, 1].axhline(y=0.4, color='k', linestyle='--', label="Target (0.4m)")
    axes[2, 1].set_xlabel("Time [s]")
    axes[2, 1].set_ylabel("Distance [m]")
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "plots"
    )
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "isaaclab_demo_results.png")
    plt.savefig(output_path, dpi=150)
    print(f"\nPlot saved to: {output_path}")

    plt.close()


if __name__ == "__main__":
    main()
