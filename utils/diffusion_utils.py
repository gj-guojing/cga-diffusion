"""
Diffusion model utilities for impedance learning.
Provides noise scheduling, sZFT reconstruction, and impedance estimation.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
import os
import sys

# Add path to your diffusion model repo
DIFFUSION_REPO_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "..", "DiffusionBasedImpedanceLearning"
)
if os.path.exists(DIFFUSION_REPO_PATH):
    sys.path.insert(0, os.path.join(DIFFUSION_REPO_PATH, "ImpedanceLearning"))


class DiffusionImpedanceHelper:
    """
    Helper class for diffusion model based impedance learning.
    Loads pre-trained model and provides sZFT reconstruction.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.model = None
        self.stats = None

        # Default parameters (from your reproduction)
        self.seq_length = 16
        self.hidden_dim = 512
        self.use_forces = True
        self.beta_start = 0.0001
        self.beta_end = 0.04
        self.noiseadding_steps = 5

        # Impedance parameters
        self.K_t_max = 800.0  # Max translational stiffness
        self.K_r_max = 150.0  # Max rotational stiffness
        self.force_thresh = 1.0  # Force threshold for adaptation
        self.moment_thresh = 1.0  # Moment threshold for adaptation

        if model_path is not None:
            self.load_model(model_path)

    def load_model(self, model_path: str):
        """
        Load pre-trained diffusion model.

        Args:
            model_path: Path to model checkpoint (.pth file)
        """
        try:
            # Try to import your model
            from models import NoisePredictorTransformerWithCrossAttentionTime

            print(f"Loading diffusion model from {model_path}")

            self.model = NoisePredictorTransformerWithCrossAttentionTime(
                seq_length=self.seq_length,
                hidden_dim=self.hidden_dim,
                num_timesteps=self.noiseadding_steps,
                use_forces=self.use_forces
            ).to(self.device)

            # Load weights
            checkpoint = torch.load(model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)

            self.model.eval()
            print("Model loaded successfully!")

        except Exception as e:
            print(f"Warning: Failed to load diffusion model: {e}")
            print("Will use placeholder implementation")
            self.model = None

    def reconstruct_szft(
        self,
        noisy_pos: np.ndarray,
        noisy_quat: np.ndarray,
        forces: Optional[np.ndarray] = None,
        moments: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reconstruct simulated Zero-Force Trajectory (sZFT).

        Args:
            noisy_pos: Noisy position trajectory [seq_length, 3]
            noisy_quat: Noisy quaternion trajectory [seq_length, 4]
            forces: External forces [seq_length, 3]
            moments: External moments [seq_length, 3]

        Returns:
            Tuple of (clean_pos, clean_quat) - reconstructed sZFT
        """
        if self.model is not None:
            return self._reconstruct_with_model(
                noisy_pos, noisy_quat, forces, moments
            )
        else:
            return self._reconstruct_placeholder(
                noisy_pos, noisy_quat, forces, moments
            )

    def _reconstruct_with_model(
        self,
        noisy_pos: np.ndarray,
        noisy_quat: np.ndarray,
        forces: Optional[np.ndarray],
        moments: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Reconstruct using the loaded diffusion model."""
        # Convert to torch tensors
        noisy_pos_t = torch.tensor(noisy_pos, dtype=torch.float32, device=self.device).unsqueeze(0)
        noisy_quat_t = torch.tensor(noisy_quat, dtype=torch.float32, device=self.device).unsqueeze(0)

        if forces is not None and moments is not None and self.use_forces:
            forces_t = torch.tensor(forces, dtype=torch.float32, device=self.device).unsqueeze(0)
            moments_t = torch.tensor(moments, dtype=torch.float32, device=self.device).unsqueeze(0)
        else:
            forces_t = torch.zeros(1, self.seq_length, 3, device=self.device)
            moments_t = torch.zeros(1, self.seq_length, 3, device=self.device)

        # Denoising loop
        with torch.no_grad():
            # Start from noisy input
            current_pos = noisy_pos_t.clone()
            current_quat = noisy_quat_t.clone()

            # Iterative denoising
            for t in range(self.noiseadding_steps - 1, -1, -1):
                # Predict noise
                noise_pred = self.model(
                    current_pos, current_quat, t, forces_t, moments_t
                )

                # Split into position and quaternion noise
                noise_pos = noise_pred[..., :3]
                noise_quat = noise_pred[..., 3:]

                # Update current estimate (simplified denoising step)
                alpha_bar = self._compute_alpha_bar(t)
                current_pos = (current_pos - torch.sqrt(1 - alpha_bar) * noise_pos) / torch.sqrt(alpha_bar)
                # Quaternion denoising would use SLERP

        # Return as numpy arrays
        return current_pos[0].cpu().numpy(), current_quat[0].cpu().numpy()

    def _reconstruct_placeholder(
        self,
        noisy_pos: np.ndarray,
        noisy_quat: np.ndarray,
        forces: Optional[np.ndarray],
        moments: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Placeholder reconstruction when model is not available.
        Uses simple smoothing based on force magnitude.
        """
        # Simple low-pass filter as placeholder
        alpha = 0.3

        clean_pos = np.zeros_like(noisy_pos)
        clean_quat = np.zeros_like(noisy_quat)

        clean_pos[0] = noisy_pos[0]
        clean_quat[0] = noisy_quat[0]

        for i in range(1, len(noisy_pos)):
            # Adjust smoothing based on force (higher force = less smoothing)
            if forces is not None:
                force_mag = np.linalg.norm(forces[i])
                adaptive_alpha = alpha * (1.0 - min(force_mag / 20.0, 0.9))
            else:
                adaptive_alpha = alpha

            # Position smoothing
            clean_pos[i] = adaptive_alpha * noisy_pos[i] + (1 - adaptive_alpha) * clean_pos[i-1]

            # Quaternion SLERP smoothing
            clean_quat[i] = self._slerp_quat_np(
                clean_quat[i-1], noisy_quat[i], adaptive_alpha
            )

        return clean_pos, clean_quat

    def estimate_impedance(
        self,
        sZFT_pos: np.ndarray,
        sZFT_quat: np.ndarray,
        current_pos: np.ndarray,
        current_quat: np.ndarray,
        forces: np.ndarray,
        moments: np.ndarray,
        prev_K_t: Optional[np.ndarray] = None,
        prev_K_r: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate directional impedance parameters based on sZFT and force measurements.

        Args:
            sZFT_pos: Reconstructed zero-force position [3]
            sZFT_quat: Reconstructed zero-force quaternion [4]
            current_pos: Current end-effector position [3]
            current_quat: Current end-effector quaternion [4]
            forces: Current external forces [3]
            moments: Current external moments [3]
            prev_K_t: Previous translational stiffness [3]
            prev_K_r: Previous rotational stiffness [3]

        Returns:
            Tuple of (K_t, K_r) - directional stiffness parameters
        """
        if prev_K_t is None:
            prev_K_t = np.array([self.K_t_max, self.K_t_max, self.K_t_max])
        if prev_K_r is None:
            prev_K_r = np.array([self.K_r_max, self.K_r_max, self.K_r_max])

        # Position error
        e_lin = current_pos - sZFT_pos

        # Rotation error (simplified)
        q_rel = self._multiply_quat_np(current_quat, self._invert_quat_np(sZFT_quat))
        e_rot = 2 * q_rel[1:]  # Approximate angular error

        # Compute relative motion importance per axis
        # Axes with more motion keep higher stiffness
        trans_norms = np.abs(e_lin) + 1e-6
        sum_trans = np.sum(trans_norms)
        rel_trans_importance = trans_norms / sum_trans

        rot_norms = np.abs(e_rot) + 1e-6
        sum_rot = np.sum(rot_norms)
        rel_rot_importance = rot_norms / sum_rot

        K_t = np.zeros(3)
        K_r = np.zeros(3)

        # Translational stiffness
        for i in range(3):
            if np.abs(forces[i]) < self.force_thresh:
                K_t[i] = self.K_t_max
            else:
                # Reduce stiffness based on force, but keep task axes stiffer
                scale_factor = 1.0 - rel_trans_importance[i]
                k_drop = np.abs(forces[i] * e_lin[i]) / (e_lin[i]**2 + 1e-6)
                k_drop *= 10.0 * scale_factor  # Aggression factor
                K_t[i] = np.clip(self.K_t_max - k_drop, 0.0, self.K_t_max)

        # Rotational stiffness
        for i in range(3):
            if np.abs(moments[i]) < self.moment_thresh:
                K_r[i] = self.K_r_max
            else:
                scale_factor = 1.0 - rel_rot_importance[i]
                k_drop = np.abs(moments[i] * e_rot[i]) / (e_rot[i]**2 + 1e-6)
                k_drop *= 2.0 * scale_factor
                K_r[i] = np.clip(self.K_r_max - k_drop, 0.0, self.K_r_max)

        # Smooth stiffness changes
        alpha = 0.15
        K_t = alpha * K_t + (1 - alpha) * prev_K_t
        K_r = alpha * K_r + (1 - alpha) * prev_K_r

        return K_t, K_r

    def _compute_alpha_bar(self, t: int) -> float:
        """Compute cumulative alpha bar for diffusion schedule."""
        beta_values = np.linspace(self.beta_start, self.beta_end, self.noiseadding_steps)
        alpha_values = 1.0 - beta_values
        alpha_bar = np.prod(alpha_values[:t+1])
        return float(alpha_bar)

    def _slerp_quat_np(self, q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
        """NumPy version of SLERP."""
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

    def _multiply_quat_np(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """NumPy version of quaternion multiplication."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        result = np.array([w, x, y, z])
        return result / np.linalg.norm(result)

    def _invert_quat_np(self, q: np.ndarray) -> np.ndarray:
        """NumPy version of quaternion inversion."""
        return np.array([q[0], -q[1], -q[2], -q[3]])
