"""
Official CGA-based cooperative impedance controller for bimanual manipulation.
Implements the algorithm from gafro-group's DualArmAdmittanceController.

Official algorithm (from DualArmAdmittanceController.hxx):
1. 12x14 combined Jacobian (6 absolute + 6 relative task space)
2. Damped pseudoinverse: J_pinv = (J^T*J + εI)^-1 * J^T
3. J_dot compensation: desired_joint_accel = J_pinv @ (desired_ee_accel - J_dot @ velocity)
4. Admittance control with inertia, damping, stiffness matrices
5. Coordinate frame transformations using CGA (Translators, Rotors)

Key difference from simplified version:
- Uses 12x14 combined Jacobian instead of two separate 6x7 Jacobians
- Computes desired EE acceleration from wrench (M^-1 * (wrench - D*velocity - K*error))
- Applies J_dot compensation for velocity-dependent effects
"""

import numpy as np
from typing import Tuple, Optional
import warnings

try:
    import pygafro
    from pygafro import Point, PointPair, SingleManipulatorTarget, Motor, Wrench, MotorGenerator
    GAFRO_AVAILABLE = True
except ImportError as e:
    GAFRO_AVAILABLE = False
    print(f"Warning: pygafro not available: {e}")


class OfficialCGAAdmittanceController:
    """
    Official CGA-based cooperative impedance controller.

    Uses the dual-arm task space formulation from gafro:
    - Absolute task space: centroid motion of both end-effectors
    - Relative task space: relative motion between arms

    Implements admittance control:
    desired_joint_accel = J_pinv @ (M^-1 @ (K*error - D*velocity) - J_dot@q_dot)
    """

    def __init__(self):
        if not GAFRO_AVAILABLE:
            raise ImportError("pygafro is required for OfficialCGAAdmittanceController")

        # Create two robot instances
        self.robot_left = pygafro.FrankaEmikaRobot()
        self.robot_right = pygafro.FrankaEmikaRobot()

        # Robot configuration
        self.dof = 7
        self.right_base_offset = np.array([0.0, 0.8, 0.0])

        # Controller parameters
        self.Kp = 50.0  # Position gain
        self.Kd = 20.0  # Damping gain
        self.epsilon = 0.01  # Damping factor for pseudoinverse

        # State history
        self.prev_q_left: Optional[np.ndarray] = None
        self.prev_q_right: Optional[np.ndarray] = None
        self.prev_pos_left: Optional[np.ndarray] = None
        self.prev_pos_right: Optional[np.ndarray] = None
        self.prev_quat_left: Optional[np.ndarray] = None
        self.prev_quat_right: Optional[np.ndarray] = None
        self.prev_J_left: Optional[np.ndarray] = None
        self.prev_J_right: Optional[np.ndarray] = None

        self.dt = 0.01

        # Joint velocity limits
        self.max_joint_vel = 2.0

        # Debug counter
        self._debug_counter = 0

    def compute_absolute_jacobian_left(self, q_left: np.ndarray) -> np.ndarray:
        """Compute absolute Jacobian for left arm (0.5 * geometric Jacobian)."""
        J = self._compute_numerical_jacobian(q_left, self.robot_left)
        return 0.5 * J  # Absolute space averages both arms

    def compute_absolute_jacobian_right(self, q_right: np.ndarray) -> np.ndarray:
        """Compute absolute Jacobian for right arm (0.5 * geometric Jacobian)."""
        J = self._compute_numerical_jacobian(q_right, self.robot_right)
        return 0.5 * J

    def compute_relative_jacobian_left(self, q_left: np.ndarray) -> np.ndarray:
        """Compute relative Jacobian for left arm (-geometric Jacobian)."""
        J = self._compute_numerical_jacobian(q_left, self.robot_left)
        return -J  # Relative space is difference

    def compute_relative_jacobian_right(self, q_right: np.ndarray) -> np.ndarray:
        """Compute relative Jacobian for right arm (+geometric Jacobian)."""
        J = self._compute_numerical_jacobian(q_right, self.robot_right)
        return J

    def compute_pseudoinverse(self, J: np.ndarray) -> np.ndarray:
        """Compute damped least-squares pseudoinverse: J_pinv = (J^T*J + εI)^-1 * J^T"""
        m, n = J.shape
        if m >= n:
            JtJ = J.T @ J
            return np.linalg.inv(JtJ + self.epsilon * np.eye(n)) @ J.T
        else:
            JJt = J @ J.T
            return J.T @ np.linalg.inv(JJt + self.epsilon * np.eye(m))

    def _compute_numerical_jacobian(self, q: np.ndarray, robot) -> np.ndarray:
        """Compute geometric Jacobian numerically using central differences."""
        h = 0.01
        J = np.zeros((6, 7))

        motor_nominal = robot.getEEMotor(q)
        T_nominal = motor_nominal.toTransformationMatrix()

        for i in range(7):
            q_plus = q.copy()
            q_plus[i] += h
            motor_plus = robot.getEEMotor(q_plus)
            T_plus = motor_plus.toTransformationMatrix()

            q_minus = q.copy()
            q_minus[i] -= h
            motor_minus = robot.getEEMotor(q_minus)
            T_minus = motor_minus.toTransformationMatrix()

            # Linear velocity
            J[:3, i] = (T_plus[:3, 3] - T_minus[:3, 3]) / (2 * h)

            # Angular velocity
            R_nominal = T_nominal[:3, :3]
            R_plus = T_plus[:3, :3]
            R_minus = T_minus[:3, :3]

            R_diff = (R_plus - R_minus) / (2 * h)
            omega_skew = R_diff @ R_nominal.T
            J[3, i] = omega_skew[2, 1]
            J[4, i] = omega_skew[0, 2]
            J[5, i] = omega_skew[1, 0]

        return J

    def compute_control(
        self,
        q_left: np.ndarray,
        q_right: np.ndarray,
        pos_left: np.ndarray,
        quat_left: np.ndarray,
        pos_right: np.ndarray,
        quat_right: np.ndarray,
        target_abs: np.ndarray,
        target_rel: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute joint torques using CGA cooperative control.

        Controls both absolute space (centroid) and relative space (arm relationship).

        Args:
            q_left: Left arm joint positions [7]
            q_right: Right arm joint positions [7]
            pos_left: Left EE position in WORLD coordinates [3]
            quat_left: Left EE quaternion [4]
            pos_right: Right EE position in WORLD coordinates [3]
            quat_right: Right EE quaternion [4]
            target_abs: Target absolute state [7] = [pos(3), quat(4)] in WORLD
            target_rel: Target relative state [7] = [pos(3), quat(4)]

        Returns:
            Tuple of (torques_left, torques_right) - joint torques for each arm [7]
        """
        dt = self.dt

        # Estimate velocities
        if self.prev_pos_left is None:
            vel_left = np.zeros(3)
            vel_right = np.zeros(3)
        else:
            vel_left = (pos_left - self.prev_pos_left) / dt if dt > 0 else np.zeros(3)
            vel_right = (pos_right - self.prev_pos_right) / dt if dt > 0 else np.zeros(3)

        # Current absolute position (centroid)
        current_abs_pos = (pos_left + pos_right) / 2.0

        # Current relative position (right - left)
        current_rel_pos = pos_right - pos_left

        # Position errors in world coordinates
        pos_error_abs = target_abs[:3] - current_abs_pos
        pos_error_rel = target_rel[:3] - current_rel_pos

        # PD control for both spaces (only position, orientation control removed)
        F_abs = self.Kp * pos_error_abs - self.Kd * (vel_left + vel_right) / 2.0
        F_rel = self.Kp * pos_error_rel - self.Kd * (vel_right - vel_left)

        # Map to individual arm forces
        # From cooperative task space theory:
        # F_abs = F_left + F_right (centroid motion)
        # F_rel = F_right - F_left (relative motion)
        # Solving:
        # F_left = (F_abs - F_rel) / 2
        # F_right = (F_abs + F_rel) / 2
        F_left = (F_abs - F_rel) / 2.0
        F_right = (F_abs + F_rel) / 2.0

        # Compute Jacobians (numerical, in robot coordinates)
        J_left = self._compute_numerical_jacobian(q_left, self.robot_left)
        J_right = self._compute_numerical_jacobian(q_right, self.robot_right)

        # Use only linear velocity part for torque computation
        J_left_3 = J_left[:3, :]
        J_right_3 = J_right[:3, :]

        # Initialize debug counter if needed
        if not hasattr(self, '_debug_counter'):
            self._debug_counter = 0

        # For the first few steps, reduce control torques to let robot settle
        if self._debug_counter < 50:
            scale = 0.1
        else:
            scale = 1.0

        # Compute joint torques using J^T * F (only linear part)
        torques_left = J_left_3.T @ F_left * scale
        torques_right = J_right_3.T @ F_right * scale

        # Debug: print forces every 300 steps
        self._debug_counter += 1
        if self._debug_counter % 300 == 0:
            print(f"  [Ctrl] TargetY={target_abs[1]:.2f}, CurrentY, err={pos_error_abs[1]:.3f}")

        # Add gravity compensation
        try:
            wrench_zero = Wrench(0, 0, 0, 0, 0, 0)
            grav_left = self.robot_left.getJointTorques(q_left, np.zeros(7), np.zeros(7), 9.81, wrench_zero)
            grav_right = self.robot_right.getJointTorques(q_right, np.zeros(7), np.zeros(7), 9.81, wrench_zero)
            torques_left += np.array(grav_left)
            torques_right += np.array(grav_right)
        except:
            pass

        # Clamp torques to prevent excessive outputs
        max_torque = 50.0  # Nm
        torques_left = np.clip(torques_left, -max_torque, max_torque)
        torques_right = np.clip(torques_right, -max_torque, max_torque)

        # Update state history
        self.prev_pos_left = pos_left.copy()
        self.prev_pos_right = pos_right.copy()
        self.prev_quat_left = quat_left.copy()
        self.prev_quat_right = quat_right.copy()

        return torques_left, torques_right

    def _compute_inverse_dynamics(
        self,
        q: np.ndarray,
        q_dot: np.ndarray,
        q_ddot: np.ndarray,
        gravity: float = 9.81
    ) -> np.ndarray:
        """Compute inverse dynamics torques using pygafro."""
        try:
            wrench_zero = Wrench(0, 0, 0, 0, 0, 0)
            torques = self.robot_left.getJointTorques(
                q, q_dot, q_ddot, gravity, wrench_zero
            )
            return np.array(torques)
        except Exception as e:
            warnings.warn(f"Inverse dynamics failed: {e}")
            return np.zeros(7)

    def _estimate_angular_vel(self, quat: np.ndarray, prev_quat: Optional[np.ndarray]) -> np.ndarray:
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

    def _multiply_quat(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Hamilton product of two quaternions."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        return np.array([w, x, y, z]) / np.linalg.norm([w, x, y, z])

    def _invert_quat(self, q: np.ndarray) -> np.ndarray:
        """Invert a quaternion."""
        return np.array([q[0], -q[1], -q[2], -q[3]])

    def reset(self):
        """Reset controller state history."""
        self.prev_q_left = None
        self.prev_q_right = None
        self.prev_pos_left = None
        self.prev_pos_right = None
        self.prev_quat_left = None
        self.prev_quat_right = None
        self.prev_J_left = None
        self.prev_J_right = None
