"""
gafro utilities wrapper for bimanual cooperative manipulation.
Provides geometric algebra based cooperative task space computation.
"""

import numpy as np
from typing import Tuple, Optional

# Try to import pygafro, provide fallback if not available
try:
    import pygafro
    GAFRO_AVAILABLE = True
except ImportError:
    GAFRO_AVAILABLE = False
    print("Warning: pygafro not available. Using fallback implementations.")


class BimanualCooperativeSpace:
    """
    Cooperative task space computation for bimanual manipulation.
    Computes absolute and relative motion components using geometric algebra.
    """

    def __init__(self):
        self.gafro_available = GAFRO_AVAILABLE

        if self.gafro_available:
            self._init_gafro()
        else:
            self._init_fallback()

    def _init_gafro(self):
        """Initialize with pygafro if available."""
        try:
            self.robot_left = pygafro.FrankaEmikaRobot()
            self.robot_right = pygafro.FrankaEmikaRobot()
        except Exception as e:
            print(f"Warning: Failed to initialize pygafro robots: {e}")
            self.gafro_available = False
            self._init_fallback()

    def _init_fallback(self):
        """Fallback initialization without pygafro."""
        pass

    def compute_cooperative_motion(
        self,
        q_left: np.ndarray,
        q_right: np.ndarray,
        pos_left: Optional[np.ndarray] = None,
        quat_left: Optional[np.ndarray] = None,
        pos_right: Optional[np.ndarray] = None,
        quat_right: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute cooperative task space components.

        Args:
            q_left: Left arm joint positions [7]
            q_right: Right arm joint positions [7]
            pos_left: Left end-effector position [3] (optional fallback)
            quat_left: Left end-effector quaternion [4] (optional fallback)
            pos_right: Right end-effector position [3] (optional fallback)
            quat_right: Right end-effector quaternion [4] (optional fallback)

        Returns:
            Tuple of (absolute_motion, relative_motion)
            - absolute_motion: Centroid/average motion [7] (pos + quat)
            - relative_motion: Relative motion between arms [7] (pos + quat)
        """
        if self.gafro_available:
            return self._compute_gafro(q_left, q_right)
        else:
            return self._compute_fallback(
                pos_left, quat_left, pos_right, quat_right
            )

    def _compute_gafro(
        self,
        q_left: np.ndarray,
        q_right: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute using pygafro."""
        try:
            # Get end-effector motors from both arms
            motor_left = self.robot_left.getEEMotor(q_left)
            motor_right = self.robot_right.getEEMotor(q_right)

            # Extract position and quaternion from motors
            # This is a simplified version - actual implementation depends on gafro API
            pos_left = self._motor_to_pos(motor_left)
            quat_left = self._motor_to_quat(motor_left)
            pos_right = self._motor_to_pos(motor_right)
            quat_right = self._motor_to_quat(motor_right)

            return self._compute_fallback(pos_left, quat_left, pos_right, quat_right)
        except Exception as e:
            print(f"Warning: gafro computation failed: {e}")
            # Fallback to basic computation if we don't have positions
            return self._compute_fallback(None, None, None, None)

    def _compute_fallback(
        self,
        pos_left: Optional[np.ndarray],
        quat_left: Optional[np.ndarray],
        pos_right: Optional[np.ndarray],
        quat_right: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fallback cooperative space computation using basic linear algebra.

        Computes:
        - Absolute: Average of both end-effectors
        - Relative: Difference between end-effectors
        """
        # Default values if inputs are None
        if pos_left is None:
            pos_left = np.array([0.0, 0.0, 0.5])
        if quat_left is None:
            quat_left = np.array([1.0, 0.0, 0.0, 0.0])  # Identity
        if pos_right is None:
            pos_right = np.array([0.0, 0.5, 0.5])
        if quat_right is None:
            quat_right = np.array([1.0, 0.0, 0.0, 0.0])  # Identity

        # Absolute motion (average)
        pos_abs = (pos_left + pos_right) / 2.0
        quat_abs = self._slerp_quat(quat_left, quat_right, 0.5)

        # Relative motion (right - left)
        pos_rel = pos_right - pos_left
        quat_rel = self._multiply_quat(quat_right, self._invert_quat(quat_left))

        # Combine into [pos, quat] format
        absolute_motion = np.concatenate([pos_abs, quat_abs])
        relative_motion = np.concatenate([pos_rel, quat_rel])

        return absolute_motion, relative_motion

    def _motor_to_pos(self, motor) -> np.ndarray:
        """Extract position from gafro motor (placeholder)."""
        # Actual implementation depends on gafro's Motor API
        return np.array([0.0, 0.0, 0.5])

    def _motor_to_quat(self, motor) -> np.ndarray:
        """Extract quaternion from gafro motor (placeholder)."""
        # Actual implementation depends on gafro's Motor API
        return np.array([1.0, 0.0, 0.0, 0.0])

    def _slerp_quat(self, q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
        """Spherical linear interpolation for quaternions."""
        q0 = np.array(q0, dtype=np.float64)
        q1 = np.array(q1, dtype=np.float64)

        # Ensure shortest path
        dot = np.dot(q0, q1)
        if dot < 0:
            q1 = -q1
            dot = -dot

        # Clamp for numerical stability
        dot = np.clip(dot, -1.0, 1.0)

        theta = np.arccos(dot)
        sin_theta = np.sin(theta)

        if abs(sin_theta) < 1e-6:
            # Linear interpolation for small angles
            return (1 - t) * q0 + t * q1

        s0 = np.sin((1 - t) * theta) / sin_theta
        s1 = np.sin(t * theta) / sin_theta

        result = s0 * q0 + s1 * q1
        return result / np.linalg.norm(result)

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
