# 🧩 Tabletop Workspace Optimization  
**Perception-Driven Intent Recognition & Workspace Layout Optimization for Human-Robot Collaboration**  
*A ROS1 (Noetic) package for tabletop shared autonomy integrating Sawyer, RealSense, YOLO, and MAP-Elites for Workspace Optimization.*

---

# 📑 Table of Contents
- [Overview](#overview)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Mujoco Simulation](#installation)
- [Working with Rosbags](#working-with-rosbags)
- [Demo Videos](#demo-videos)
- [Intent Recognition & Perception Pipeline](#intent-recognition--perception-pipeline)
- [Workspace Optimization (MAP-Elites)](#workspace-optimization-map-elites)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

# Overview
This repository provides a full-stack ROS1 pipeline for:

- **3D Perception** using RealSense + YOLO  
- **Human Intent Inference** via hand / end-effector tracking  
- **Workspace Layout Optimization** to optimize object placement for fast intent recognition using MAP-Elites  
- **Shared Autonomy** with Sawyer + RelaxedIK  

Designed for collaborative tabletop tasks (e.g., pick-and-place tea/snacks), with easy extensibility.

---

# 📦 System Requirements

<!-- <details> -->
<!-- <summary><strong></strong></summary> -->

### **Operating System**
- Ubuntu **20.04 LTS** (tested)

### **Core Dependencies**
- Python 3.8+  
- ROS Noetic  
- Sawyer SDK (`intera_interface`)  
- RelaxedIK (`relaxed_ik_ros1`)  
- RealSense (`realsense2_camera`)

### **ROS Packages**
`rospy`, `tf`, `tf2_ros`, `cv_bridge`,  
`vision_msgs`, `sensor_msgs`, `geometry_msgs`, `message_filters`, `visualization_msgs`

### **Python Dependencies**
Installed via `requirements.txt`.

### **Hardware**
- Sawyer robot  
- Intel RealSense D435/D435i  
- NVIDIA GPU recommended

<!-- </details> -->

---

# Installation

<!-- <details>
<summary><strong>Click to expand</strong></summary> -->

## 1. Install ROS + Dependencies

```bash
sudo apt update
sudo apt install -y \
  ros-noetic-vision-msgs ros-noetic-cv-bridge ros-noetic-tf \
  ros-noetic-tf2-ros ros-noetic-message-filters ros-noetic-joy \
  ros-noetic-realsense2-camera
```

## 2. Clone & Build

```bash
cd ~/catkin_ws/src
git clone https://github.com/<your-username>/tabletop_workspace_opt.git

# Optional:
# git clone https://github.com/IntelRealSense/realsense-ros
# git clone https://github.com/uwgraphics/relaxed_ik_ros1
# git clone https://github.com/RethinkRobotics/sawyer_robot.git

cd ~/catkin_ws
catkin_make
source devel/setup.bash
```

## 3. Python Environment Setup

```bash
cd ~/catkin_ws
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r src/tabletop_workspace_opt/requirements.txt
```

<!-- </details> -->

## 4. Mujoco Simulation with Sawyer

For detailed instructions on running the simulation only, see the [Simulation README](src/mujoco_sim/README.md).

![](assets/Mujoco_sim.png)

For detailed instructions on using joystick or keyboard [Instructions README](controller.md)



---

#  Working with Rosbags

<!-- <details>
<summary><strong>Click to expand</strong></summary> -->

## 1. Extract

```bash
cd tabletop_workspace_opt/assets
sudo apt install p7zip-full
7z x chai_pick.7z
```

## 2. Play

```bash
rosbag play chai_pick.bag
```

## 3. Launch Pipeline

```bash
roslaunch tabletop_workspace_opt intent_recognizer.launch
```

### Expected Output

* RealSense streams
* YOLO detections
* RViz markers
* Intent inference (`~distribution`, `~top_goal`)
* Optional GUI windows



<!-- </details> -->

---

# 🎮 Demo Videos

<!-- <details>
<summary><strong>Click to expand</strong></summary> -->

* **Baseline Task**
  `assets/baseline_tea_task.mov`

* **Optimized Workspace Layout**
  `assets/workspace_optimized_tea_task.mov`

* **Manual Labeling Tool**
  ![](assets/manual_labelling.gif)

<!-- </details> -->

---

# 🧠 Intent Recognition & Perception Pipeline

<!-- <details>
<summary><strong>Click to expand</strong></summary> -->

Launch the perception → tracking → inference → visualization stack:

```bash
roslaunch tabletop_workspace_opt intent_recognizer.launch
```

Includes:

* RealSense alignment
* YOLO inference
* Hand/end-effector tracking
* Intent distribution
* RViz visualization

<!-- </details> -->

---

# Workspace Optimization (MAP-Elites)

<!-- <details>
<summary><strong>Click to expand</strong></summary> -->

Run MAP-Elites:

```bash
python3 map_elites.py --config config/tea_task.yaml
```

### Outputs

* Optimized object poses `[x, y, theta]`
* `wo_layout_cma-me.png`
* `wo_archive_heatmap_cma-me.png`
* Archive `.pkl`

### Configurable Parameters

* Object sizes
* Colors
* Task graph
* Locked objects

<!-- </details> -->


---

# 📚 Citation

If you use this work in academic publications, please cite:

```
Citation will be added upon publication.
```

---

# 📄 License

This project is released under the **MIT License**.

```
MIT License

Copyright (c) 2025 Yi-Shiuan Tung

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

# 🤝 Acknowledgments

This project builds upon:

* Rethink Robotics Sawyer SDK
* Intel RealSense
* RelaxedIK (UW Graphics Lab)
* Ultralytics YOLO
* Pyribs (MAP-Elites)


