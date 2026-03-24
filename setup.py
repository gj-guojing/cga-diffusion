#!/usr/bin/env python3
"""
Setup script for cga-diffusion package.
"""

from setuptools import setup, find_packages

setup(
    name="cga-diffusion",
    version="0.1.0",
    description="Geometric Algebra + Diffusion Model for Bimanual Cooperative Manipulation",
    author="",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20",
        "torch>=2.0",
        "matplotlib>=3.0",
    ],
    extras_require={
        "isaaclab": [
            # Isaac Lab dependencies to be installed separately
        ],
        "gafro": [
            # pygafro to be installed separately
        ],
    },
)
