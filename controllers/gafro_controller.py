"""
Geometric Algebra based cooperative controller for bimanual manipulation.
Implements cooperative task space control using geometric algebra concepts.
"""

import numpy as np
from typing import Tuple, Optional
from utils.gafro_utils import BimanualCooperativeSpace


class BimanualGAFROController:
    """
    Cooperative controller for bimanual manipulation using geometric algebra concepts.
    Computes cooperative task space control signals for both arms.
    """

    def __init__(self):
        self.coop_space = BimanualCooperativeSpace()

        # Control gains
        self.Kp_abs = np.array([400.0, 400.0, 400.0])  # Absolute space stiffness
        self.Kd_abs = np.array([40.0, 40.0, 40.0])       # Absolute space damping
        self.Kp_rel = np.array([300.0, 300.0, 300.0])  # Relative space stiffness
        self.Kd_rel = np.array([30.0, 30.0, 30.0])       # Relative space damping

        # Rotational gains
        self.Kp_abs_rot = np.array([80.0, 80.0, 80.0])
        self.Kd_abs_rot = np.array([8.0, 8.0, 8.0])
        self.Kp_rel_rot = np.array([60.0, 60.0, 60.0])
        self.Kd_rel_rot = np.array([6.0, 6.0, 6.0])

        # State history for velocity estimation
        self.prev_pos_left: Optional[np.ndarray] = None
        self.prev_pos_right: Optional[np.ndarray] = None
        self.prev_quat_left: Optional[np.ndarray] = None
        self.prev_quat_right: Optional[np.ndarray] = None
        self.dt = 0.01  # Time step

    def compute_control(
        self,
        q_left: np.ndarray,
        q_right: np.ndarray,
        pos_left: np.ndarray,
        quat_left: np.ndarray,
        pos_right: np.ndarray,
        quat_right: np.ndarray,
        target_abs: Optional[np.ndarray] = None,
        target_rel: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute cooperative task space control signals.

        Args:
            q_left: Left arm joint positions [7]
            q_right: Right arm joint positions [7]
            pos_left: Left end-effector position [3]
            quat_left: Left end-effector quaternion [4]
            pos_right: Right end-effector position [3]
            quat_right: Right end-effector quaternion [4]
            target_abs: Target absolute motion [7] (pos + quat)
            target_rel: Target relative motion [7] (pos + quat)

        Returns:
            Tuple of ( wrench_left, wrench_right ) - control wrenches for each arm
        """
        # Compute current cooperative motion
        current_abs, current_rel = self.coop_space.compute_cooperative_motion(
            q_left, q_right, pos_left, quat_left, pos_right, quat_right
        )

        # Set default targets if not provided
        if target_abs is None:
            # Default: maintain current absolute position with slight elevation
            target_abs = current_abs.copy()
            target_abs[2] = max(target_abs[2], 0.3)  # Keep at least 30cm height

        if target_rel is None:
            # Default: maintain relative position
            target_rel = current_rel.copy()

        # Estimate velocities
        vel_left = self._estimate_velocity(pos_left, self.prev_pos_left)
        vel_right = self._estimate_velocity(pos_right, self.prev_pos_right)
        omega_left = self._estimate_angular_vel(quat_left, self.prev_quat_left)
        omega_right = self._estimate_angular_vel(quat_right, self.prev_quat_right)

        # Compute cooperative space velocities
        vel_abs = (vel_left + vel_right) / 2.0
        vel_rel = vel_right - vel_left
        omega_abs = (omega_left + omega_right) / 2.0
        omega_rel = omega_right - omega_left

        # Extract current and target states
        pos_abs_current = current_abs[:3]
        quat_abs_current = current_abs[3:]
        pos_rel_current = current_rel[:3]
        quat_rel_current = current_rel[3:]

        pos_abs_target = target_abs[:3]
        quat_abs_target = target_abs[3:]
        pos_rel_target = target_rel[:3]
        quat_rel_target = target_rel[3:]

        # Compute position errors
        e_abs_pos = pos_abs_target - pos_abs_current
        e_rel_pos = pos_rel_target - pos_rel_current

        # Compute rotation errors (simplified)
        e_abs_rot = self._compute_quat_error(quat_abs_current, quat_abs_target)
        e_rel_rot = self._compute_quat_error(quat_rel_current, quat_rel_target)

        # PD control in cooperative space
        F_abs_trans = self.Kp_abs * e_abs_pos - self.Kd_abs * vel_abs
        F_abs_rot = self.Kp_abs_rot * e_abs_rot - self.Kd_abs_rot * omega_abs
        F_rel_trans = self.Kp_rel * e_rel_pos - self.Kd_rel * vel_rel
        F_rel_rot = self.Kp_rel_rot * e_rel_rot - self.Kd_rel_rot * omega_rel

        # Map cooperative space forces to individual arms
        wrench_left, wrench_right = self._coop_to_individual(
            F_abs_trans, F_abs_rot, F_rel_trans, F_rel_rot
        )

        # Update state history
        self.prev_pos_left = pos_left.copy()
        self.prev_pos_right = pos_right.copy()
        self.prev_quat_left = quat_left.copy()
        self.prev_quat_right = quat_right.copy()

        return wrench_left, wrench_right

    def _coop_to_individual(
        self,
        F_abs_trans: np.ndarray,
        F_abs_rot: np.ndarray,
        F_rel_trans: np.ndarray,
        F_rel_rot: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Map cooperative space forces to individual arm wrenches.

        Cooperative to individual mapping:
        F_left = 0.5 * F_abs - 0.5 * F_rel
        F_right = 0.5 * F_abs + 0.5 * F_rel
        """
        # Split absolute and relative forces
        F_left_trans = 0.5 * F_abs_trans - 0.5 * F_rel_trans
        F_right_trans = 0.5 * F_abs_trans + 0.5 * F_rel_trans

        F_left_rot = 0.5 * F_abs_rot - 0.5 * F_rel_rot
        F_right_rot = 0.5 * F_abs_rot + 0.5 * F_rel_rot

        # Combine into wrenches [force, torque]
        wrench_left = np.concatenate([F_left_trans, F_left_rot])
        wrench_right = np.concatenate([F_right_trans, F_right_rot])

        return wrench_left, wrench_right

    def _estimate_velocity(
        self,
        pos: np.ndarray,
        prev_pos: Optional[np.ndarray]
    ) -> np.ndarray:
        """Estimate linear velocity from position history."""
        if prev_pos is None:
            return np.zeros(3)
        return (pos - prev_pos) / self.dt

    def _estimate_angular_vel(
        self,
        quat: np.ndarray,
        prev_quat: Optional[np.ndarray]
    ) -> np.ndarray:
        """Estimate angular velocity from quaternion history."""
        if prev_quat is None:
            return np.zeros(3)

        # Compute quaternion difference
        q_rel = self._multiply_quat(quat, self._invert_quat(prev_quat))
        angle = 2 * np.arccos(np.clip(q_rel[0], -1.0, 1.0))
        axis = q_rel[1:] / (np.sin(angle / 2) + 1e-6) if angle > 1e-6 else np.zeros(3)

        return axis * angle / self.dt

    def _compute_quat_error(self, q_current: np.ndarray, q_target: np.ndarray) -> np.ndarray:
        """Compute rotation error between two quaternions."""
        q_rel = self._multiply_quat(q_target, self._invert_quat(q_current))
        angle = 2 * np.arccos(np.clip(q_rel[0], -1.0, 1.0))
        axis = q_rel[1:] / (np.sin(angle / 2) + 1e-6) if angle > 1e-6 else np.zeros(3)
        return axis * angle

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
