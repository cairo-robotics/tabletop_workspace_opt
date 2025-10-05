## Tabletop Workspace Optimization

A ROS1 package for perception-driven intent recognition and workspace layout optimization to support human-robot collaboration with a Sawyer robot.

---

### System Requirements

- **Operating Systems:** Ubuntu 20.04 LTS (tested)
- **Software Dependencies:**
  - **Programming language:** Python 3.8+
  - **ROS:** ROS Noetic (catkin workspace)
  - **Major ROS packages:** `rospy`, `tf`, `tf2_ros`, `cv_bridge`, `message_filters`, `vision_msgs`, `sensor_msgs`, `geometry_msgs`, `visualization_msgs`
  - **Other ROS stacks used at runtime:**
    - `realsense2_camera` (Intel RealSense depth camera)
    - `relaxed_ik_ros1` (inverse kinematics teleop/planning)
    - `intera_interface` (Sawyer robot SDK)
- **Python libraries (installed via requirements.txt):** numpy, opencv-python, mediapipe (optional), ultralytics (YOLO), matplotlib, pandas, shapely, tqdm, torch, pyribs, rospkg, catkin_pkg
- **Hardware:**
  - Rethink Robotics Sawyer with electric gripper
  - Intel RealSense depth camera (e.g., D435 series)
  - NVIDIA GPU recommended for YOLO inference

---

### Installation Guide

1) Set up ROS Noetic and a catkin workspace

```bash
sudo apt update
# Install ROS Noetic (see ROS docs if not installed)
# sudo apt install ros-noetic-desktop-full

# ROS dependencies used by this package
sudo apt install -y \
  ros-noetic-vision-msgs \
  ros-noetic-cv-bridge \
  ros-noetic-tf \
  ros-noetic-tf2-ros \
  ros-noetic-message-filters \
  ros-noetic-joy

# Optional dependencies from other stacks used in launches
sudo apt install -y ros-noetic-realsense2-camera
```

2) Clone this package into your catkin workspace and build

```bash
cd /catkin_ws
# If not already present, clone dependent stacks (examples):
# git clone https://github.com/IntelRealSense/realsense-ros src/realsense-ros
# git clone https://github.com/uwgraphics/relaxed_ik_ros1 src/relaxed_ik_ros1
# intera SDK should be installed per Sawyer docs

catkin_make
source devel/setup.bash
```

3) Create and activate a Python virtual environment, then install Python deps

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r src/tabletop_workspace_opt/requirements.txt
```

Notes:
- ROS Python packages (e.g., `rospy`, `cv_bridge`, `tf`) are provided by apt/ROS, not pip.
- For GPU acceleration, install the correct CUDA/cuDNN prior to installing `torch`.

---

### Demo

- **Intent recognition and perception pipeline** (camera + YOLO + intent):

```bash
source /catkin_ws/devel/setup.bash
roslaunch tabletop_workspace_opt intent_recognizer.launch
```

- **Transforms/perception sanity check** (camera + visualization):

```bash
source /catkin_ws/devel/setup.bash
roslaunch tabletop_workspace_opt transform_testing.launch
```

- **Expected Output:**
  - RealSense streams are started and aligned; YOLO detections are shown/republished.
  - RViz displays 3D bounding boxes and markers for tracked and detected objects.
  - The `intent_inference` node publishes `~distribution` and `~top_goal` reflecting user intent.
  - If `show_gui:=true`, OpenCV windows display annotated frames and GUI controls.

If you do not have hardware available, record or play back a ROS bag with the following topics: `/right_cam/color/image_raw`, `/right_cam/aligned_depth_to_color/image_raw`, `/right_cam/color/camera_info`, and optional `/robot/limb/right/endpoint_state`.

---

### Instructions for Use

- Configuration parameters are exposed as ROS params in the launch files and nodes:
  - Perception (`stitch_object_detection_vt.py`): model path (`~model`), thresholds, class filters, frame names, GUI toggle.
  - Intent (`intent_inference.py`): `~tracker_type` (hand or end_effector), `~base_frame`, temporal window, beta, and speed threshold.
  - IK/teleop via `relaxed_ik_ros1` and joystick/keyboard teleop are included in the launches.
- To adapt for your robot or camera:
  - Update frame names (`~base_frame`, `~color_optical_frame`) and static transforms in the launch files.
  - Replace YOLO weights (e.g., `yolov8m.pt` or `yolo11m.pt`) via the `~model` param.
  - Adjust detection classes and patch sizes via node params.
  - For end-effector intent, ensure `/robot/limb/right/endpoint_state` exists or remap the topic.
- Optimization components (`src/envopt/map_elites.py`) can be run standalone for workspace layout studies; ensure Python deps from requirements are installed and run it with Python to generate archives and visualizations.

---

### Code Review and Suggestions

- **Hardcoded paths:**
  - `launch/*` uses absolute defaults for `setting_file_path` pointing into `relaxed_ik_ros1`. Consider making this a relative path or a ROS package resource (e.g., via `$(find relaxed_ik_ros1)/...`) and/or a user-facing arg with no absolute default.
  - Camera serial numbers are hardcoded; expose as a launch arg already, but document how to override (`serial_no:=<your_serial>`).
  - YOLO model filename defaults to `yolov8m.pt`/`yolo11m.pt`; document where to download and place the weights, or default to a path in this package’s `src/assets/`.
- **Demo dataset:**
  - No sample bag files are included. For reproducibility, provide a short bag with synchronized color, depth, and camera info topics or link to an external dataset.
- **Python packaging:**
  - `setup.py` lists `packages=["find_packages('src')"]` which is incorrect. Replace with `packages=find_packages('src')` and ensure `package_dir={'': 'src'}`. Optionally add `entry_points` for scripts or rely on ROS `catkin_install_python`.
- **License metadata:**
  - `package.xml` has `<license>TODO</license>`. Update this to `MIT` to match the LICENSE file.
- **README usage tips:**
  - Add instructions for running without GPU (set YOLO device to CPU) or provide a smaller CPU-friendly model.

---

### Citation

If you use this code in academic work, please cite the accompanying paper (add BibTeX entry here when available).
