# MuJoCo Sawyer Robot Simulator

This repository contains a MuJoCo-based simulation environment for the Rethink Robotics Sawyer robot.

## Prerequisites

Before you begin, ensure you have the following installed:

- **MuJoCo:** Follow the official [MuJoCo documentation](https://github.com/google-deepmind/mujoco/) for installation instructions. The recommended way is to install using PyPI via `pip install mujoco`.
  - Ensure `MJ_KEY_PATH` environment variable is set to your `mjkey.txt` file.
  - Ensure `LD_LIBRARY_PATH` (Linux) or `DYLD_LIBRARY_PATH` (macOS) includes your MuJoCo library path (e.g., `~/.mujoco/mujoco210/bin`).
- **Python 3.x:** Recommended version 3.8 or higher.

## Running a basic simulation

To launch a basic simulation of the Sawyer robot in a default environment:

```bash
python3 simulation_server.py
```

To run the shared autonomy system, run

```bash
roslaunch tabletop_workspace_opt intent_recognition_sim.launch
```

You can go to the launch file to change parameters such as the scene name or whether to use a joystick or keyboard to control the robot.

## Scenes

The default scene is defined in the file `tabletop_workspace_opt/src/assets/simple_scene.xml`. You can create a new scene by copying this file and modifying it. Object meshes are stored in the `tabletop_workspace_opt/src/assets/sawyer/meshes` directory.
