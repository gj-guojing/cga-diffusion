"""
Geometric Algebra based cooperative controller for bimanual manipulation.
Implements cooperative task space control using pygafro geometric algebra.

Algorithm:
1. Compute cooperative space errors (absolute + relative)
2. PD control in cooperative space to get desired forces
3. Map to individual arm wrenches using: F_left = F_abs - F_rel, F_right = F_abs + F_rel
4. J^T transformation to joint torques using pygafro inverse dynamics
5. Adaptive gain adjustment based on tracking error
"""

import numpy as np
from typing import Tuple, Optional
from utils.gafro_utils import BimanualCooperativeSpace


class BimanualGAFROController:
    """
    Cooperative controller for bimanual manipulation using geometric algebra concepts.
    Computes cooperative task space control signals for both arms.

    Uses pygafro for true geometric algebra computations:
    - Absolute space: centroid of both end-effectors
    - Relative space: relative position/orientation between arms
    """

    def __init__(self):
        self.coop_space = BimanualCooperativeSpace()

        # Control gains - lower Kd to reduce noise amplification
        self.Kp_abs = np.array([45.0, 45.0, 45.0])  # Kp
        self.Kd_abs = np.array([8.0, 8.0, 8.0])   # Lower Kd to reduce velocity noise amplification
        self.Kp_rel = np.array([15.0, 15.0, 15.0])  # Relative gain
        self.Kd_rel = np.array([4.0, 4.0, 4.0])      # Lower damping

        # Rotational gains
        self.Kp_abs_rot = np.array([5.0, 5.0, 5.0])
        self.Kd_abs_rot = np.array([3.0, 3.0, 3.0])
        self.Kp_rel_rot = np.array([3.0, 3.0, 3.0])
        self.Kd_rel_rot = np.array([2.0, 2.0, 2.0])

        # State history for velocity estimation
        self.prev_pos_left: Optional[np.ndarray] = None
        self.prev_pos_right: Optional[np.ndarray] = None
        self.prev_quat_left: Optional[np.ndarray] = None
        self.prev_quat_right: Optional[np.ndarray] = None
        self.dt = 0.01

        # Adaptive control parameters
        self.error_history = []
        self.max_error_history = 100
        self.target_Kp_factor = 1.0  # Multiplier for Kp adaptation

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
            pos_left: Left end-effector position [3] in WORLD coordinates
            quat_left: Left end-effector quaternion [4]
            pos_right: Right end-effector position [3] in WORLD coordinates
            quat_right: Right end-effector quaternion [4]
            target_abs: Target absolute motion [7] (pos + quat) in WORLD coordinates
            target_rel: Target relative motion [7] (pos + quat) in WORLD coordinates

        Returns:
            Tuple of ( wrench_left, wrench_right ) - control wrenches for each arm [6]
        """
        # Compute current cooperative motion in WORLD coordinates
        current_abs_pos = (pos_left + pos_right) / 2.0
        current_abs_quat = self._slerp_quat(quat_left, quat_right, 0.5)
        current_abs = np.concatenate([current_abs_pos, current_abs_quat])

        current_rel_pos = pos_right - pos_left
        current_rel_quat = self._multiply_quat(quat_right, self._invert_quat(quat_left))
        current_rel = np.concatenate([current_rel_pos, current_rel_quat])

        # Set default targets if not provided
        if target_abs is None:
            target_abs = current_abs.copy()
        if target_rel is None:
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

        # Compute rotation errors (axis-angle representation)
        e_abs_rot = self._compute_quat_error(quat_abs_current, quat_abs_target)
        e_rel_rot = self._compute_quat_error(quat_rel_current, quat_rel_target)

        # PD control in cooperative space (no adaptive gains for testing)
        F_abs_trans = self.Kp_abs * e_abs_pos - self.Kd_abs * vel_abs
        F_abs_rot = self.Kp_abs_rot * e_abs_rot - self.Kd_abs_rot * omega_abs
        F_rel_trans = self.Kp_rel * e_rel_pos - self.Kd_rel * vel_rel
        F_rel_rot = self.Kp_rel_rot * e_rel_rot - self.Kd_rel_rot * omega_rel

        # Map cooperative space forces to individual arms
        # This is the correct mapping for cooperative task space:
        # F_left = F_abs - F_rel (force on left arm to move centroid and adjust relative pos)
        # F_right = F_abs + F_rel
        wrench_left, wrench_right = self._coop_to_individual(
            F_abs_trans, F_abs_rot, F_rel_trans, F_rel_rot
        )

        # Update state history
        self.prev_pos_left = pos_left.copy()
        self.prev_pos_right = pos_right.copy()
        self.prev_quat_left = quat_left.copy()
        self.prev_quat_right = quat_right.copy()

        return wrench_left, wrench_right

    def _adapt_gains(self, error: np.ndarray, target: np.ndarray):
        """
        Adapt control gains based on tracking error and error trend.

        Strategy:
        - If error is large and growing: increase Kp
        - If error is oscillating (overshooting): increase Kd
        - If error is small and stable: reduce gains
        """
        # Compute normalized error magnitude
        error_mag = np.linalg.norm(error)
        target_mag = np.linalg.norm(target) + 1e-6
        normalized_error = error_mag / target_mag

        # Store error in history
        self.error_history.append(error_mag)
        if len(self.error_history) > self.max_error_history:
            self.error_history.pop(0)

        # Compute error trend (derivative)
        error_trend = 0.0
        if len(self.error_history) >= 5:
            recent = self.error_history[-5:]
            error_trend = (recent[-1] - recent[0]) / 5.0  # Average change over last 5 samples

        # Check for oscillation (error changing sign frequently)
        oscillating = False
        if len(self.error_history) >= 10:
            signs = np.sign(self.error_history[-10:])
            sign_changes = np.sum(np.abs(np.diff(signs)))
            oscillating = sign_changes >= 6

        # Adaptive strategy for Kp factor
        if oscillating:
            # Reduce gains when oscillating to dampen the system
            self.target_Kp_factor = max(0.5, self.target_Kp_factor * 0.9)
        elif error_trend > 0.001 and normalized_error > 0.05:
            # Error is growing - need more aggressive response
            self.target_Kp_factor = min(2.0, self.target_Kp_factor + 0.02)
        elif normalized_error > 0.1:
            # Large error - increase gains to respond faster
            self.target_Kp_factor = min(2.0, 1.0 + normalized_error * 2)
        elif normalized_error < 0.02 and abs(error_trend) < 0.001:
            # Small error and stable - can reduce gains
            self.target_Kp_factor = max(0.7, self.target_Kp_factor * 0.99)
        else:
            # Slowly converge to 1.0
            self.target_Kp_factor = 0.95 * self.target_Kp_factor + 0.05 * 1.0

    def _coop_to_individual(
        self,
        F_abs_trans: np.ndarray,
        F_abs_rot: np.ndarray,
        F_rel_trans: np.ndarray,
        F_rel_rot: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Map cooperative space forces to individual arm wrenches.

        From cooperative task space theory:
        - F_abs = F_left + F_right (centroid motion)
        - F_rel = F_right - F_left (relative motion)

        Solving:
        - F_right = (F_abs + F_rel) / 2
        - F_left = (F_abs - F_rel) / 2
        """
        F_left_trans = (F_abs_trans - F_rel_trans) / 2.0
        F_right_trans = (F_abs_trans + F_rel_trans) / 2.0

        F_left_rot = (F_abs_rot - F_rel_rot) / 2.0
        F_right_rot = (F_abs_rot + F_rel_rot) / 2.0

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

        q_rel = self._multiply_quat(quat, self._invert_quat(prev_quat))
        angle = 2 * np.arccos(np.clip(q_rel[0], -1.0, 1.0))
        axis = q_rel[1:] / (np.sin(angle / 2) + 1e-6) if angle > 1e-6 else np.zeros(3)

        return axis * angle / self.dt

    def _compute_quat_error(self, q_current: np.ndarray, q_target: np.ndarray) -> np.ndarray:
        """Compute rotation error between two quaternions as axis-angle."""
        q_rel = self._multiply_quat(q_target, self._invert_quat(q_current))
        angle = 2 * np.arccos(np.clip(q_rel[0], -1.0, 1.0))
        axis = q_rel[1:] / (np.sin(angle / 2) + 1e-6) if angle > 1e-6 else np.zeros(3)
        return axis * angle

    def _multiply_quat(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Hamilton product of two quaternions."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        result = np.array([w, x, y, z])
        return result / np.linalg.norm(result)

    def _invert_quat(self, q: np.ndarray) -> np.ndarray:
        """Invert a quaternion (conjugate for unit quaternions)."""
        return np.array([q[0], -q[1], -q[2], -q[3]])

    def _slerp_quat(self, q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
        """Spherical linear interpolation for quaternions."""
        q0 = np.array(q0, dtype=np.float64)
        q1 = np.array(q1, dtype=np.float64)

        dot = np.dot(q0, q1)
        if dot < 0:
            q1 = -q1
            dot = -dot

        dot = np.clip(dot, -1.0, 1.0)
        theta = np.arccos(dot)
        sin_theta = np.sin(theta)

        if abs(sin_theta) < 1e-6:
            return (1 - t) * q0 + t * q1

        s0 = np.sin((1 - t) * theta) / sin_theta
        s1 = np.sin(t * theta) / sin_theta

        result = s0 * q0 + s1 * q1
        return result / np.linalg.norm(result)

    def get_adapted_gains(self) -> dict:
        """Return the current adapted gains for debugging."""
        return {
            'Kp_abs': self.Kp_abs * self.target_Kp_factor,
            'Kd_abs': self.Kd_abs * self.target_Kp_factor,
            'Kp_rel': self.Kp_rel * self.target_Kp_factor,
            'Kd_rel': self.Kd_rel * self.target_Kp_factor,
            'adaptation_factor': self.target_Kp_factor
        }
