# Repository Guidelines

- GitHub: `ava-agent/embodied-agent`
- Category: Python robotics simulation and embodied-AI learning demos.

This repository is not an iOS app despite its current workspace path. Keep PyBullet, MuJoCo, PyTorch, and Unitree examples aligned with the matching learning guides.

Run `python3 -m compileall -q demos` for a safe syntax check. Do not launch GUI simulations, long training runs, or real-robot/network interfaces unless the task explicitly requires them.

Do not commit trained models, simulator state, local environments, generated captures, device credentials, or build products.
