# CGA-Diffusion: Bimanual Cooperative Manipulation

Geometric Algebra + Diffusion Model for Bimanual Cooperative Carrying Tasks.

## Setup

### Conda Environment

Use the existing `env_isaaclab` environment (no need to create new):

```bash
# Activate Isaac Lab environment
conda activate env_isaaclab

# (Optional) If you need diffusion model features, switch to:
conda activate impl
```

### Isaac Lab Setup

Ensure Isaac Lab is properly installed and sourced.

## Usage

### Basic Simulation (No Isaac Lab)

```bash
python scripts/demo_simulation.py
```

### Isaac Lab Bimanual Viewer

Display two Franka Panda robots in Isaac Sim:

```bash
cd ~/GitRepo/IsaacLab
conda activate env_isaaclab
./isaaclab.sh -p ~/GitRepo/cga-diffusion/scripts/bimanual_demo.py
```
