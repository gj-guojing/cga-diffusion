"""
Diffusion model based impedance controller for bimanual manipulation.
Integrates diffusion-based sZFT reconstruction with adaptive impedance control.
"""

import torch
import numpy as np
from typing import Tuple, Optional, List
from collections import deque

from utils.diffusion_utils import DiffusionImpedanceHelper
from controllers.gafro_controller import BimanualGAFROController


class BimanualDiffusionController:
    """
    Integrated controller combining diffusion-based impedance learning
    with cooperative task space control for bimanual manipulation.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        # Base cooperative controller
        self.gafro_controller = BimanualGAFROController()

        # Diffusion impedance helper
        self.diffusion_helper = DiffusionImpedanceHelper(model_path, device)

        # Trajectory buffers for sZFT reconstruction
        self.seq_length = self.diffusion_helper.seq_length
        self.pos_buffer_left = deque(maxlen=self.seq_length)
        self.quat_buffer_left = deque(maxlen=self.seq_length)
        self.force_buffer_left = deque(maxlen=self.seq_length)
        self.moment_buffer_left = deque(maxlen=self.seq_length)

        self.pos_buffer_right = deque(maxlen=self.seq_length)
        self.quat_buffer_right = deque(maxlen=self.seq_length)
        self.force_buffer_right = deque(maxlen=self.seq_length)
        self.moment_buffer_right = deque(maxlen=self.seq_length)

        # Current stiffness values
        self.K_t_left: Optional[np.ndarray] = None
        self.K_r_left: Optional[np.ndarray] = None
        self.K_t_right: Optional[np.ndarray] = None
        self.K_r_right: Optional[np.ndarray] = None

        # Reconstructed sZFT
        self.sZFT_pos_left: Optional[np.ndarray] = None
        self.sZFT_quat_left: Optional[np.ndarray] = None
        self.sZFT_pos_right: Optional[np.ndarray] = None
        self.sZFT_quat_right: Optional[np.ndarray] = None

        # Control mode
        self.use_diffusion = model_path is not None and self.diffusion_helper.model is not None

    def reset(self):
        """Reset controller state for new episodes."""
        self.pos_buffer_left.clear()
        self.quat_buffer_left.clear()
        self.force_buffer_left.clear()
        self.moment_buffer_left.clear()

        self.pos_buffer_right.clear()
        self.quat_buffer_right.clear()
        self.force_buffer_right.clear()
        self.moment_buffer_right.clear()

        self.K_t_left = None
        self.K_r_left = None
        self.K_t_right = None
        self.K_r_right = None

        self.sZFT_pos_left = None
        self.sZFT_quat_left = None
        self.sZFT_pos_right = None
        self.sZFT_quat_right = None

        self.gafro_controller.prev_pos_left = None
        self.gafro_controller.prev_pos_right = None
        self.gafro_controller.prev_quat_left = None
        self.gafro_controller.prev_quat_right = None

    def compute_control(
        self,
        q_left: np.ndarray,
        q_right: np.ndarray,
        pos_left: np.ndarray,
        quat_left: np.ndarray,
        pos_right: np.ndarray,
        quat_right: np.ndarray,
        force_left: Optional[np.ndarray] = None,
        force_right: Optional[np.ndarray] = None,
        moment_left: Optional[np.ndarray] = None,
        moment_right: Optional[np.ndarray] = None,
        target_abs: Optional[np.ndarray] = None,
        target_rel: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute integrated control with diffusion-based impedance adaptation.

        Args:
            q_left: Left arm joint positions [7]
            q_right: Right arm joint positions [7]
            pos_left: Left end-effector position [3]
            quat_left: Left end-effector quaternion [4]
            pos_right: Right end-effector position [3]
            quat_right: Right end-effector quaternion [4]
            force_left: Left arm contact forces [3]
            force_right: Right arm contact forces [3]
            moment_left: Left arm contact moments [3]
            moment_right: Right arm contact moments [3]
            target_abs: Target absolute motion [7]
            target_rel: Target relative motion [7]

        Returns:
            Tuple of (wrench_left, wrench_right) - control wrenches
        """
        # Default force/moment values
        if force_left is None:
            force_left = np.zeros(3)
        if force_right is None:
            force_right = np.zeros(3)
        if moment_left is None:
            moment_left = np.zeros(3)
        if moment_right is None:
            moment_right = np.zeros(3)

        # Store in buffers
        self.pos_buffer_left.append(pos_left.copy())
        self.quat_buffer_left.append(quat_left.copy())
        self.force_buffer_left.append(force_left.copy())
        self.moment_buffer_left.append(moment_left.copy())

        self.pos_buffer_right.append(pos_right.copy())
        self.quat_buffer_right.append(quat_right.copy())
        self.force_buffer_right.append(force_right.copy())
        self.moment_buffer_right.append(moment_right.copy())

        # Check if we have enough data for diffusion
        if len(self.pos_buffer_left) >= self.seq_length and self.use_diffusion:
            # Reconstruct sZFT for both arms
            self._reconstruct_szft()

            # Estimate impedance based on sZFT
            self._estimate_impedance(
                pos_left, quat_left, force_left, moment_left,
                pos_right, quat_right, force_right, moment_right
            )

            # Update cooperative controller gains based on estimated impedance
            self._update_coop_gains()

            # Use sZFT as target if no target provided
            if target_abs is None and self.sZFT_pos_left is not None:
                target_abs, target_rel = self._compute_target_from_szft()

        # Compute cooperative control
        wrench_left, wrench_right = self.gafro_controller.compute_control(
            q_left, q_right, pos_left, quat_left, pos_right, quat_right,
            target_abs, target_rel
        )

        return wrench_left, wrench_right

    def _reconstruct_szft(self):
        """Reconstruct simulated Zero-Force Trajectory for both arms."""
        """Reconstruct simulated Zero-Force Trajectory for both arms."""
        # Get trajectories from buffers
        pos_seq_left = np.array(self.pos_buffer_left)
        quat_seq_left = np.array(self.quat_buffer_left)
        force_seq_left = np.array(self.force_buffer_left)
        moment_seq_left = np.array(self.moment_buffer_left)

        pos_seq_right = np.array(self.pos_buffer_right)
        quat_seq_right = np.array(self.quat_buffer_right)
        force_seq_right = np.array(self.force_buffer_right)
        moment_seq_right = np.array(self.moment_buffer_right)

        # Reconstruct sZFT for left arm
        self.sZFT_pos_left, self.sZFT_quat_left = self.diffusion_helper.reconstruct_szft(
            pos_seq_left, quat_seq_left, force_seq_left, moment_seq_left
        )

        # Reconstruct sZFT for right arm
        self.sZFT_pos_right, self.sZFT_quat_right = self.diffusion_helper.reconstruct_szft(
            pos_seq_right, quat_seq_right, force_seq_right, moment_seq_right
        )

    def _estimate_impedance(
        self,
        pos_left: np.ndarray,
        quat_left: np.ndarray,
        force_left: np.ndarray,
        moment_left: np.ndarray,
        pos_right: np.ndarray,
        quat_right: np.ndarray,
        force_right: np.ndarray,
        moment_right: np.ndarray
    ):
        """Estimate directional impedance for both arms."""
        if self.sZFT_pos_left is None or self.sZFT_pos_right is None:
            return

        # Use the most recent sZFT point
        sZFT_pos_left_curr = self.sZFT_pos_left[-1]
        sZFT_quat_left_curr = self.sZFT_quat_left[-1]
        sZFT_pos_right_curr = self.sZFT_pos_right[-1]
        sZFT_quat_right_curr = self.sZFT_quat_right[-1]

        # Estimate impedance for left arm
        self.K_t_left, self.K_r_left = self.diffusion_helper.estimate_impedance(
            sZFT_pos_left_curr, sZFT_quat_left_curr,
            pos_left, quat_left, force_left, moment_left,
            self.K_t_left, self.K_r_left
        )

        # Estimate impedance for right arm
        self.K_t_right, self.K_r_right = self.diffusion_helper.estimate_impedance(
            sZFT_pos_right_curr, sZFT_quat_right_curr,
            pos_right, quat_right, force_right, moment_right,
            self.K_t_right, self.K_r_right
        )

    def _update_coop_gains(self):
        """Update cooperative controller gains based on estimated impedance."""
        if self.K_t_left is None or self.K_t_right is None:
            return

        # Average impedance for cooperative space
        K_t_avg = (self.K_t_left + self.K_t_right) / 2.0
        K_r_avg = (self.K_r_left + self.K_r_right) / 2.0

        # Update cooperative controller gains
        self.gafro_controller.Kp_abs = K_t_avg
        self.gafro_controller.Kp_rel = K_t_avg * 0.8
        self.gafro_controller.Kp_abs_rot = K_r_avg
        self.gafro_controller.Kp_rel_rot = K_r_avg * 0.8

        # Damping proportional to stiffness
        self.gafro_controller.Kd_abs = np.sqrt(K_t_avg) * 2.0
        self.gafro_controller.Kd_rel = np.sqrt(K_t_avg) * 1.6
        self.gafro_controller.Kd_abs_rot = np.sqrt(K_r_avg) * 2.0
        self.gafro_controller.Kd_rel_rot = np.sqrt(K_r_avg) * 1.6

    def _compute_target_from_szft(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute cooperative space targets from sZFT."""
        # Use most recent sZFT
        pos_left = self.sZFT_pos_left[-1]
        quat_left = self.sZFT_quat_left[-1]
        pos_right = self.sZFT_pos_right[-1]
        quat_right = self.sZFT_quat_right[-1]

        # Compute cooperative motion
        pos_abs = (pos_left + pos_right) / 2.0
        pos_rel = pos_right - pos_left

        # Simple average for quaternions
        dot = np.dot(quat_left, quat_right)
        if dot < 0:
            quat_right_avg = -quat_right
        else:
            quat_right_avg = quat_right
        quat_abs = (quat_left + quat_right_avg) / 2.0
        quat_abs = quat_abs / np.linalg.norm(quat_abs)

        # Relative quaternion
        quat_rel = self._multiply_quat(quat_right, self._invert_quat(quat_left))

        target_abs = np.concatenate([pos_abs, quat_abs])
        target_rel = np.concatenate([pos_rel, quat_rel])

        return target_abs, target_rel

    def get_current_impedance(self) -> dict:
        """Get current impedance values for logging/visualization."""
        return {
            "K_t_left": self.K_t_left.copy() if self.K_t_left is not None else None,
            "K_r_left": self.K_r_left.copy() if self.K_r_left is not None else None,
            "K_t_right": self.K_t_right.copy() if self.K_t_right is not None else None,
            "K_r_right": self.K_r_right.copy() if self.K_r_right is not None else None,
        }

    def get_szft(self) -> dict:
        """Get reconstructed sZFT for logging/visualization."""
        return {
            "pos_left": self.sZFT_pos_left.copy() if self.sZFT_pos_left is not None else None,
            "quat_left": self.sZFT_quat_left.copy() if self.sZFT_quat_left is not None else None,
            "pos_right": self.sZFT_pos_right.copy() if self.sZFT_pos_right is not None else None,
            "quat_right": self.sZFT_quat_right.copy() if self.sZFT_quat_right is not None else None,
        }

    def _multiply_quat(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Quaternion multiplication."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        result = np.array([w, x, y, z])
        return result / np.linalg.norm(result)

    def _invert_quat(self, q: np.ndarray) -> np.ndarray:
        """Quaternion inversion."""
        return np.array([q[0], -q[1], -q[2], -q[3]])
