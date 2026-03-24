#!/usr/bin/env python3
"""
Demo script for bimanual cooperative manipulation.
This is a simplified simulation without Isaac Lab,
to test the controller logic before full integration.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import List, Dict

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from controllers.diffusion_controller import BimanualDiffusionController
from controllers.gafro_controller import BimanualGAFROController


class SimpleBimanualSim:
    """
    Simple kinematic simulation for bimanual manipulation.
    No physics, just kinematic updates for testing controllers.
    """

    def __init__(self):
        # Initial arm positions (left and right)
        self.pos_left = np.array([-0.2, 0.0, 0.5])
        self.pos_right = np.array([0.2, 0.0, 0.5])
        self.quat_left = np.array([1.0, 0.0, 0.0, 0.0])
        self.quat_right = np.array([1.0, 0.0, 0.0, 0.0])

        # Dummy joint positions
        self.q_left = np.zeros(7)
        self.q_right = np.zeros(7)

        # Simulation parameters
        self.dt = 0.01
        self.time = 0.0

        # Contact forces (simulated)
        self.force_left = np.zeros(3)
        self.force_right = np.zeros(3)
        self.moment_left = np.zeros(3)
        self.moment_right = np.zeros(3)

        # History for plotting
        self.history: Dict[str, List[np.ndarray]] = {
            "time": [],
            "pos_left": [],
            "pos_right": [],
            "force_left": [],
            "force_right": [],
            "K_t_left": [],
            "K_t_right": [],
        }

    def step(self, wrench_left: np.ndarray, wrench_right: np.ndarray):
        """
        Simple kinematic step: move towards wrench direction.
        """
        # Convert wrench to position change (simplified)
        k_pos = 0.001
        k_rot = 0.0001

        self.pos_left += k_pos * wrench_left[:3] * self.dt
        self.pos_right += k_pos * wrench_right[:3] * self.dt

        # Simulate some contact forces when moving
        self._update_contact_forces()

        # Update time
        self.time += self.dt

        # Record history
        self.history["time"].append(self.time)
        self.history["pos_left"].append(self.pos_left.copy())
        self.history["pos_right"].append(self.pos_right.copy())
        self.history["force_left"].append(self.force_left.copy())
        self.history["force_right"].append(self.force_right.copy())

    def _update_contact_forces(self):
        """Simulate contact forces based on position."""
        # Simulate a vertical "wall" in the middle
        wall_y = 0.0
        stiffness_wall = 50.0

        # Left arm contact
        penetration_left = wall_y - self.pos_left[1]
        if penetration_left > 0:
            self.force_left[1] = stiffness_wall * penetration_left
        else:
            self.force_left[1] *= 0.9

        # Right arm contact
        penetration_right = self.pos_right[1] - wall_y
        if penetration_right > 0:
            self.force_right[1] = -stiffness_wall * penetration_right
        else:
            self.force_right[1] *= 0.9

        # Add some noise
        self.force_left += np.random.randn(3) * 0.5
        self.force_right += np.random.randn(3) * 0.5


def main():
    """Main demo function."""
    print("=" * 60)
    print("CGA-Diffusion: Bimanual Cooperative Manipulation Demo")
    print("=" * 60)

    # Check if diffusion model is available
    model_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "..", "DiffusionBasedImpedanceLearning",
        "reproduction", "full_paper_results", "best_model.pth"
    )

    has_diffusion = os.path.exists(model_path)
    print(f"\nDiffusion model available: {has_diffusion}")
    if has_diffusion:
        print(f"Model path: {model_path}")
        controller = BimanualDiffusionController(model_path=model_path)
    else:
        print("Using GAFRO cooperative controller only (no diffusion)")
        controller = BimanualGAFROController()

    # Create simulation
    sim = SimpleBimanualSim()

    # Target: move both arms towards center while maintaining distance
    target_abs = np.array([0.0, 0.2, 0.5, 1.0, 0.0, 0.0, 0.0])  # pos + quat
    target_rel = np.array([0.4, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])  # 40cm apart

    print("\nStarting simulation...")

    # Run simulation
    num_steps = 500
    for i in range(num_steps):
        if has_diffusion:
            # Use full diffusion controller
            wrench_left, wrench_right = controller.compute_control(
                sim.q_left, sim.q_right,
                sim.pos_left, sim.quat_left,
                sim.pos_right, sim.quat_right,
                sim.force_left, sim.force_right,
                sim.moment_left, sim.moment_right,
                target_abs, target_rel
            )

            # Record impedance for plotting
            impedance = controller.get_current_impedance()
            if impedance["K_t_left"] is not None:
                sim.history["K_t_left"].append(impedance["K_t_left"].copy())
                sim.history["K_t_right"].append(impedance["K_t_right"].copy())
            else:
                sim.history["K_t_left"].append(np.array([800, 800, 800]))
                sim.history["K_t_right"].append(np.array([800, 800, 800]))

        else:
            # Use only GAFRO controller
            wrench_left, wrench_right = controller.compute_control(
                sim.q_left, sim.q_right,
                sim.pos_left, sim.quat_left,
                sim.pos_right, sim.quat_right,
                target_abs, target_rel
            )

            # Default stiffness
            sim.history["K_t_left"].append(np.array([800, 800, 800]))
            sim.history["K_t_right"].append(np.array([800, 800, 800]))

        # Step simulation
        sim.step(wrench_left, wrench_right)

        # Print progress
        if (i + 1) % 50 == 0:
            print(f"Step {i+1}/{num_steps} | "
                  f"Left: [{sim.pos_left[0]:.3f}, {sim.pos_left[1]:.3f}, {sim.pos_left[2]:.3f}] | "
                  f"Force: {np.linalg.norm(sim.force_left):.1f}N")

    print("\nSimulation complete! Plotting results...")

    # Plot results
    plot_results(sim.history, has_diffusion)


def plot_results(history: Dict, has_diffusion: bool):
    """Plot simulation results."""
    time = np.array(history["time"])
    pos_left = np.array(history["pos_left"])
    pos_right = np.array(history["pos_right"])
    force_left = np.array(history["force_left"])
    force_right = np.array(history["force_right"])
    K_t_left = np.array(history["K_t_left"])
    K_t_right = np.array(history["K_t_right"])

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

    # Forces
    axes[0, 1].plot(time, force_left[:, 1], label="Left Force Y", color="blue", alpha=0.7)
    axes[0, 1].plot(time, force_right[:, 1], label="Right Force Y", color="red", alpha=0.7)
    axes[0, 1].set_ylabel("Contact Force Y [N]")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Stiffness (if diffusion available)
    if has_diffusion:
        axes[1, 1].plot(time, K_t_left[:, 1], label="Left Stiffness Y", color="blue")
        axes[1, 1].plot(time, K_t_right[:, 1], label="Right Stiffness Y", color="red")
        axes[1, 1].set_ylabel("Stiffness Y [N/m]")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, "No diffusion model\n(Constant stiffness)",
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
    output_path = os.path.join(output_dir, "demo_results.png")
    plt.savefig(output_path, dpi=150)
    print(f"\nPlot saved to: {output_path}")

    plt.show()


if __name__ == "__main__":
    main()
