"""
GAFRO utilities for bimanual cooperative manipulation.
Uses pygafro for true geometric algebra based computations.
"""

import numpy as np
from typing import Tuple, Optional
import warnings

# Try to import pygafro
try:
    import pygafro
    from pygafro import Point, PointPair, SingleManipulatorTarget, Motor, Wrench
    GAFRO_AVAILABLE = True
except ImportError as e:
    GAFRO_AVAILABLE = False
    print(f"Warning: pygafro not available: {e}")


class BimanualCooperativeSpace:
    """
    Cooperative task space computation for bimanual manipulation.
    Uses pygafro for true geometric algebra based computations.

    Implements the Cooperative Task Space concepts from the paper:
    - Absolute space: centroid of both end-effectors
    - Relative space: relative position/orientation between arms
    """

    def __init__(self):
        if not GAFRO_AVAILABLE:
            raise ImportError("pygafro is required for BimanualCooperativeSpace")

        # Create two robot instances
        self.robot_left = pygafro.FrankaEmikaRobot()
        self.robot_right = pygafro.FrankaEmikaRobot()

        # Robot configuration offsets (right robot is offset by Y)
        self.right_base_offset = np.array([0.0, 0.8, 0.0])  # 80cm offset in Y

    def get_ee_motor(self, joint_positions: np.ndarray, robot: pygafro.FrankaEmikaRobot) -> Motor:
        """
        Get end-effector motor from joint positions.

        Args:
            joint_positions: Joint angles [7]
            robot: The robot instance

        Returns:
            Motor representing the end-effector pose
        """
        return robot.getEEMotor(joint_positions)

    def motor_to_pose(self, motor: Motor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract position and quaternion from a Motor.

        Args:
            motor: The Motor

        Returns:
            Tuple of (position [3], quaternion [4])
        """
        # Convert to 4x4 transformation matrix
        T = motor.toTransformationMatrix()

        # Extract position
        position = T[:3, 3]

        # Extract rotation matrix and convert to quaternion
        R = T[:3, :3]

        # Rotation matrix to quaternion
        q = self._rotation_to_quaternion(R)

        return position, q

    def _rotation_to_quaternion(self, R: np.ndarray) -> np.ndarray:
        """Convert rotation matrix to quaternion [w, x, y, z]."""
        trace = np.trace(R)
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

        q = np.array([w, x, y, z])
        return q / np.linalg.norm(q)

    def compute_cooperative_motion(
        self,
        q_left: np.ndarray,
        q_right: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute cooperative task space components using geometric algebra.

        Args:
            q_left: Left arm joint positions [7]
            q_right: Right arm joint positions [7]

        Returns:
            Tuple of (absolute_motion, relative_motion)
            - absolute_motion: Centroid motion [7] (pos + quat) in WORLD coordinates
            - relative_motion: Relative motion [7] (pos + quat) - EE2 relative to EE1
        """
        # Get end-effector motors
        motor_left = self.get_ee_motor(q_left, self.robot_left)
        motor_right = self.get_ee_motor(q_right, self.robot_right)

        # Extract poses (in各自robot frame)
        pos_left, quat_left = self.motor_to_pose(motor_left)
        pos_right, quat_right = self.motor_to_pose(motor_right)

        # Apply base offset to get world position of right EE
        pos_right_world = pos_right + self.right_base_offset

        # Absolute space: centroid in WORLD coordinates
        pos_abs = (pos_left + pos_right_world) / 2.0
        quat_abs = self._slerp_quat(quat_left, quat_right, 0.5)

        # Relative space: difference between EEs (in robot coordinates, NOT world)
        # This represents how the arms are configured relative to each other
        pos_rel = pos_right - pos_left
        quat_rel = self._multiply_quat(quat_right, self._invert_quat(quat_left))

        # Combine into [pos, quat] format
        absolute_motion = np.concatenate([pos_abs, quat_abs])
        relative_motion = np.concatenate([pos_rel, quat_rel])

        return absolute_motion, relative_motion

    def compute_absolute_jacobian(
        self,
        q_left: np.ndarray,
        q_right: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute absolute space Jacobians for both arms.

        Args:
            q_left: Left arm joint positions [7]
            q_right: Right arm joint positions [7]

        Returns:
            Tuple of (J_left, J_right) - 6x7 Jacobians for absolute space
        """
        # Get geometric Jacobians
        J_left = self.robot_left.getEEGeometricJacobian(q_left)
        J_right = self.robot_right.getEEGeometricJacobian(q_right)

        # Average for absolute space
        J_left_abs = 0.5 * J_left
        J_right_abs = 0.5 * J_right

        return J_left_abs, J_right_abs

    def compute_relative_jacobian(
        self,
        q_left: np.ndarray,
        q_right: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute relative space Jacobians for both arms.

        Args:
            q_left: Left arm joint positions [7]
            q_right: Right arm joint positions [7]

        Returns:
            Tuple of (J_left, J_right) - 6x7 Jacobians for relative space
        """
        # Get geometric Jacobians
        J_left = self.robot_left.getEEGeometricJacobian(q_left)
        J_right = self.robot_right.getEEGeometricJacobian(q_right)

        # Difference for relative space
        J_left_rel = -J_left
        J_right_rel = J_right

        return J_left_rel, J_right_rel

    def inverse_kinematics(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
        robot: pygafro.FrankaEmikaRobot,
        initial_q: Optional[np.ndarray] = None,
        max_iterations: int = 100
    ) -> np.ndarray:
        """
        Solve inverse kinematics using Gauss-Newton optimization with line search.
        Based on the official gafro examples.

        Args:
            target_pos: Target position [3]
            target_quat: Target quaternion [4] (unused, kept for API compatibility)
            robot: The robot instance
            initial_q: Initial joint configuration [7]
            max_iterations: Maximum iterations

        Returns:
            Solved joint configuration [7]
        """
        if initial_q is None:
            initial_q = robot.getRandomConfiguration()

        # Create target point pair
        point1 = Point(target_pos[0], target_pos[1], target_pos[2])
        point2 = Point(target_pos[0] + 0.1, target_pos[1], target_pos[2])
        target_pp = PointPair(point1, point2)

        # Create cost function
        cost_function = SingleManipulatorTarget(robot, Point(), target_pp)

        # Gauss-Newton optimization with line search (from official example)
        x = initial_q.copy()
        cost = np.linalg.norm(cost_function.getError(x))

        for i in range(max_iterations):
            gradient, hessian = cost_function.getGradientAndHessian(x)

            # Compute update using inverse Hessian
            update = -(np.linalg.inv(hessian + 1e-5 * np.eye(7)) @ gradient)

            # Line search
            iter_count = 1
            c = cost
            alpha = 2.0

            while (c >= cost) and (iter_count < 20):
                alpha *= 0.5
                x_new = x + alpha * update
                c = np.linalg.norm(cost_function.getError(x_new))
                iter_count += 1

            update = alpha * update

            if (np.linalg.norm(update) < 1e-10) or np.isnan(update).any():
                break

            x = x + update
            cost = c

            if cost < 1e-10:
                break

        return x

    def solve_bimanual_ik(
        self,
        target_abs_pos: np.ndarray,
        target_abs_quat: np.ndarray,
        target_rel_pos: np.ndarray,
        target_rel_quat: np.ndarray,
        initial_q_left: Optional[np.ndarray] = None,
        initial_q_right: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve bimanual inverse kinematics for cooperative task space targets.

        Args:
            target_abs_pos: Target absolute position [3]
            target_abs_quat: Target absolute quaternion [4]
            target_rel_pos: Target relative position [3]
            target_rel_quat: Target relative quaternion [4]
            initial_q_left: Initial left joint configuration [7]
            initial_q_right: Initial right joint configuration [7]

        Returns:
            Tuple of (q_left, q_right) - solved joint configurations
        """
        # Compute target positions for each arm from absolute and relative
        # abs = (left + right) / 2
        # rel = right - left
        # Solving: left = abs - rel/2, right = abs + rel/2
        target_left_pos = target_abs_pos - target_rel_pos / 2.0
        target_right_pos = target_abs_pos + target_rel_pos / 2.0

        # For orientation, we need more sophisticated handling
        # Simplified: use the same orientation for both arms
        target_left_quat = target_abs_quat
        target_right_quat = target_abs_quat

        # Solve IK for each arm
        q_left = self.inverse_kinematics(
            target_left_pos, target_left_quat,
            self.robot_left, initial_q_left
        )
        q_right = self.inverse_kinematics(
            target_right_pos, target_right_quat,
            self.robot_right, initial_q_right
        )

        return q_left, q_right

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

        result = np.array([w, x, y, z])
        return result / np.linalg.norm(result)

    def _invert_quat(self, q: np.ndarray) -> np.ndarray:
        """Invert a quaternion (conjugate for unit quaternions)."""
        return np.array([q[0], -q[1], -q[2], -q[3]])

    def compute_joint_torques_from_wrench(
        self,
        q: np.ndarray,
        wrench: np.ndarray,
        robot
    ) -> np.ndarray:
        """
        Convert end-effector wrench to joint torques using proper J^T transformation.

        The proper formula is: τ_joints = J^T(q) * F_ee
        Where J is the geometric Jacobian and F_ee is the end-effector wrench.

        Args:
            q: Joint positions [7]
            wrench: End-effector wrench [6] = [force_x, force_y, force_z, torque_x, torque_y, torque_z]
                    in WORLD coordinates (Isaac Lab convention: Z-up)
            robot: pygafro robot (FrankaEmikaRobot)

        Returns:
            Joint torques [7]
        """
        # Get gravity compensation from pygafro
        wrench_gafro_zero = Wrench(0, 0, 0, 0, 0, 0)
        torques_gravity = robot.getJointTorques(
            q, np.zeros(7), np.zeros(7), 9.81, wrench_gafro_zero
        )
        gravity_torques = np.array(torques_gravity)

        # Compute J^T * wrench using numerical Jacobian
        J = self._computeGeometricJacobian(q, robot)
        J_T = J.T
        wrench_torques = J_T @ wrench

        # Combine gravity compensation + torques for desired EE force
        total_torques = gravity_torques + wrench_torques

        return total_torques

    def _computeGeometricJacobian(
        self,
        q: np.ndarray,
        robot
    ) -> np.ndarray:
        """
        Compute geometric Jacobian numerically using central differences.

        The geometric Jacobian J(q) maps joint velocities to end-effector spatial velocity:
        v_ee = J(q) * q_dot

        For J^T * F transformation (wrench to joint torques):
        τ = J^T * F

        Args:
            q: Joint positions [7]
            robot: pygafro robot

        Returns:
            Jacobian matrix [6x7]
        """
        h = 0.01  # Larger step to reduce numerical noise
        J = np.zeros((6, 7))

        # Get nominal EE motor
        motor_nominal = self.get_ee_motor(q, robot)
        T_nominal = motor_nominal.toTransformationMatrix()

        for i in range(7):
            # Positive perturbation
            q_plus = q.copy()
            q_plus[i] += h
            motor_plus = self.get_ee_motor(q_plus, robot)
            T_plus = motor_plus.toTransformationMatrix()

            # Negative perturbation
            q_minus = q.copy()
            q_minus[i] -= h
            motor_minus = self.get_ee_motor(q_minus, robot)
            T_minus = motor_minus.toTransformationMatrix()

            # Linear velocity (central difference)
            J[:3, i] = (T_plus[:3, 3] - T_minus[:3, 3]) / (2 * h)

            # Angular velocity approximation (from rotation matrix difference)
            # Compute rotation matrix difference
            R_nominal = T_nominal[:3, :3]
            R_plus = T_plus[:3, :3]
            R_minus = T_minus[:3, :3]

            # Angular velocity from skew-symmetric matrix
            R_diff = (R_plus - R_minus) / (2 * h)
            # omega = [R_diff * R^T]_{skew}
            omega_skew = R_diff @ R_nominal.T
            J[3, i] = omega_skew[2, 1]  # omega_x
            J[4, i] = omega_skew[0, 2]  # omega_y
            J[5, i] = omega_skew[1, 0]  # omega_z

        return J

    def compute_joint_torques_bimanual(
        self,
        q_left: np.ndarray,
        q_right: np.ndarray,
        wrench_left: np.ndarray,
        wrench_right: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute joint torques for both arms from their respective wrenches.

        Args:
            q_left: Left arm joint positions [7]
            q_right: Right arm joint positions [7]
            wrench_left: Left end-effector wrench [6]
            wrench_right: Right end-effector wrench [6]

        Returns:
            Tuple of (torques_left, torques_right) - joint torques for each arm [7]
        """
        torques_left = self.compute_joint_torques_from_wrench(
            q_left, wrench_left, self.robot_left
        )
        torques_right = self.compute_joint_torques_from_wrench(
            q_right, wrench_right, self.robot_right
        )

        return torques_left, torques_right
