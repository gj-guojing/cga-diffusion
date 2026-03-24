"""
Default configuration for bimanual cooperative manipulation.
"""

import numpy as np

# Simulation parameters
SIM_DT = 0.01
SIM_DURATION = 10.0

# Robot parameters
ROBOT_TYPE = "franka"  # or "kuka"
LEFT_ARM_BASE_POS = np.array([-0.3, 0.0, 0.0])
RIGHT_ARM_BASE_POS = np.array([0.3, 0.0, 0.0])

# Control gains - cooperative space
Kp_abs_trans = np.array([400.0, 400.0, 400.0])
Kd_abs_trans = np.array([40.0, 40.0, 40.0])
Kp_rel_trans = np.array([300.0, 300.0, 300.0])
Kd_rel_trans = np.array([30.0, 30.0, 30.0])

Kp_abs_rot = np.array([80.0, 80.0, 80.0])
Kd_abs_rot = np.array([8.0, 8.0, 8.0])
Kp_rel_rot = np.array([60.0, 60.0, 60.0])
Kd_rel_rot = np.array([6.0, 6.0, 6.0])

# Diffusion model parameters
DIFFUSION_SEQ_LENGTH = 16
DIFFUSION_HIDDEN_DIM = 512
DIFFUSION_NOISE_STEPS = 5
DIFFUSION_BETA_START = 0.0001
DIFFUSION_BETA_END = 0.04

# Impedance parameters
MAX_K_TRANSLATIONAL = 800.0
MAX_K_ROTATIONAL = 150.0
FORCE_THRESHOLD = 1.0
MOMENT_THRESHOLD = 1.0

# Isaac Lab specific
ENV_WIDTH = 2.0
ENV_HEIGHT = 2.0
USE_GPU = True
NUM_ENVS = 1
