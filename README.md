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
- [AprilTag User Study](#apriltag-user-study)
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

# Working with Rosbags

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
roslaunch tabletop_workspace_opt apriltag_shared_control.launch
```

### Expected Output

- RealSense streams
- AprilTag / candidate detections
- RViz markers
- Intent inference (`/apriltag_intent_inference/distribution`, `/apriltag_intent_inference/top_goal`)
- shared-autonomy grasp execution topics

<!-- </details> -->

---

# 🎮 Demo Videos

<!-- <details>
<summary><strong>Click to expand</strong></summary> -->

- **Baseline Task**
  `assets/baseline_tea_task.mov`

- **Optimized Workspace Layout**
  `assets/workspace_optimized_tea_task.mov`

- **Manual Labeling Tool**
  ![](assets/manual_labelling.gif)

<!-- </details> -->

---

# User Study

This workflow launches the AprilTag-based shared-autonomy user study interface, task pipeline, and grasp/pour execution stack.

## 1. Switch Back To System Python

If your shell is using a non-system Python, switch back so ROS uses `/usr/bin/python3`.

```bash
cd ~/catkin_ws
export PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/ros/noetic/bin:$PATH
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash

which python
which python3
```

Expected output:

```bash
/usr/bin/python
/usr/bin/python3
```

## 2. Launch The User Study Stack

Always set the run metadata before a real participant session. These fields are the keys used later to join trial logs with questionnaire data.

The default launch configuration is set up for local-network access, so the Web UI can be opened from another computer on the same network.

Recommended shortcut:

```bash
cd ~/catkin_ws/src/tabletop_workspace_opt
./run_user_study.sh P01 optimized B3 pilot_20260701 sandwich
```

This helper script:

- sources the ROS environment
- launches the user study stack
- sets the dashboard host to `0.0.0.0`
- sets the dashboard port to `8766`
- auto-generates `session_id` as `pilot_YYYYMMDD` if omitted
- accepts an optional `frontend_stack` argument

Recommended `frontend_stack` values:

- `classic`: breakfast / legacy AprilTag-only workflow
- `lego`: LEGO-only workflow
- `sandwich`: SAM sandwich pieces + AprilTag destination containers
- `hybrid`: combined development/debug stack

The default camera configuration for `user_study.launch` is `camera_mode:=light`.

- `light`: keeps the RGB image stream on for AprilTag, SAM, and PCA-based LEGO grasping, while disabling `depth`, `align_depth`, and `pointcloud` to reduce RealSense crashes during participant sessions
- `full`: re-enables the heavier depth pipeline for debugging or development only

Equivalent explicit launch:

```bash
roslaunch tabletop_workspace_opt user_study.launch \
  frontend_stack:=sandwich \
  session_id:=pilot_20260701 \
  participant_id:=P01 \
  condition_id:=optimized \
  block_id:=B3
```

Compatibility wrapper:

```bash
roslaunch tabletop_workspace_opt apriltag_user_study.launch \
  frontend_stack:=sandwich \
  session_id:=pilot_20260701 \
  participant_id:=P01 \
  condition_id:=optimized \
  block_id:=B3
```

If you want to override the dashboard host or port manually, use:

```bash
roslaunch tabletop_workspace_opt user_study.launch \
  frontend_stack:=sandwich \
  user_study_dashboard_host:=0.0.0.0 \
  user_study_dashboard_port:=8766 \
  camera_mode:=light \
  session_id:=pilot_20260701 \
  participant_id:=P01 \
  condition_id:=optimized \
  block_id:=B3
```

If you need the heavier RealSense configuration for debugging, use:

```bash
roslaunch tabletop_workspace_opt user_study.launch frontend_stack:=sandwich camera_mode:=full
```

Then find the IP address of the ROS / dashboard computer:

```bash
hostname -I
```

Use one of the reachable local-network IPv4 addresses, for example `192.168.50.194`.

This launch file starts:

- AprilTag workspace scan and recorded destination registry
- real-time SAM sandwich-piece candidates
- sandwich candidate registry with scan-only identity freezing
- task context manager and Web UI
- AprilTag intent inference
- shared-autonomy grasp executor
- optional task-specific pipelines depending on `frontend_stack`

Suggested naming:

- `participant_id`: `P01`, `P02`, ...
- `condition_id`: `baseline`, `unoptimized`, `optimized`, ...
- `block_id`: `B1`, `B2`, ...
- `session_id`: one identifier for the whole run day or session, e.g. `pilot_20260701`

## 2.1 Supported Main Entry Points

These are the maintained launch files for the current system:

- `user_study.launch`: main multi-task user study stack
- `apriltag_user_study.launch`: compatibility wrapper around `user_study.launch`
- `apriltag_shared_control.launch`: AprilTag shared-autonomy grasping stack
- `sam_lego_shared_control.launch`: LEGO shared-autonomy stack
- `apriltag_workspace_scan.launch`: workspace scan and grasp registry recording
- `pca_lego_grasp.launch`: PCA-based LEGO grasp generation
- `sam_lego_grasp.launch`: real-time SAM LEGO candidate generation

Older demo and legacy shared-autonomy launch files were removed during pipeline cleanup and should not be used as setup references.

## 3. Scan The Scene

Before selecting a task, move the end-effector / wrist camera so the system can see and record all relevant AprilTags in the scene.

- Keep each tag visible until detections are stable
- Make sure all target objects for the current experiment are scanned
- The recorded candidate set is what intent inference uses afterward

## 4. Open The Web UI

Open:

```text
Same computer:
http://127.0.0.1:8766/operator
http://127.0.0.1:8766/participant

Another computer on the same network:
http://<dashboard_host_ip>:8766/operator
http://<dashboard_host_ip>:8766/participant
```

Important:

- `127.0.0.1` only works on the same computer that launched the dashboard
- the default launch now listens on a network-accessible host
- for a remote participant or operator device, use the actual host IP from `hostname -I`
- the default dashboard port is `8766`

Choose one of the available tasks:

- `Sorting`
- `Sandwich Assembly`

Task summaries:

- `Sorting`: choose one item from the mixed object set, then guide it toward a destination (`plate`, `bowl`, or a tagged sorting container).
- `Sandwich Assembly`: choose any sandwich piece, then choose a tagged destination container and confirm the release.

Current research framing:

- the main question is whether workspace optimization improves target disambiguation during shared autonomy
- the study is not primarily about generic grasping capability
- `Sorting` serves as the lower-ambiguity task
- `Sandwich Assembly` serves as the higher-ambiguity task because many pieces are visually similar and recipe order matters

Recommended condition names:

- `unoptimized`: crowded or ambiguous scene layout
- `optimized`: layout arranged to improve candidate separability without changing the task goal

## 5. Trigger Intent Inference

Once the task step is active, move the robot end-effector toward the intended target to start a reach.

- The current system uses trajectory-based intent inference
- It infers intent from the observed reach path relative to the scanned candidate objects
- You should make a clear reach toward one target instead of only making tiny local motions

As you move, the probability window will show the changing confidence over the currently allowed candidates.

## 6. Execute The Selected Grasp

When the intended target becomes dominant:

- press `X` to confirm and execute `pregrasp`
- press `X` again to continue to `grasp`
- press `A` to close the gripper when required
- press `Y` to cancel the current execution and return to selection

For `Sorting`, a successful pickup step is followed by destination selection, and after a successful placement the dashboard returns to the next object-selection step so the remaining objects can continue to be sorted. For `Sandwich Assembly`, the task loops between `choose sandwich piece` and `choose destination`, so the participant decides how many layers to build.

## Notes

- The user study dashboard only controls task flow; object scanning still has to happen first
- If the probability output looks degenerate, re-scan the scene so the recorded candidate set contains all intended objects
- If ROS Python packages fail to load, re-check that `python3` resolves to `/usr/bin/python3`
- For sandwich-style assembly, candidate ambiguity matters more than raw grasp count. Keep the interpretation focused on whether the intended piece was easy or hard to disambiguate.

## 7. Validate The Trial Log

After each participant block, validate the latest trial log:

```bash
python3 ~/catkin_ws/src/tabletop_workspace_opt/scripts/validate_user_study_logs.py
```

This checks for common integrity problems such as:

- duplicate `trial_id`
- missing end times
- success rows with incorrect final inference
- missing metadata fields

## 8. Export Trial-Level Analysis CSV

Export a questionnaire-friendly trial table from the latest trial log:

```bash
python3 ~/catkin_ws/src/tabletop_workspace_opt/scripts/trail_analysis_exporter.py
```

Or export a specific log:

```bash
python3 ~/catkin_ws/src/tabletop_workspace_opt/scripts/trail_analysis_exporter.py \
  ~/catkin_ws/src/tabletop_workspace_opt/logs/user_study_trials_20260612_160417.jsonl
```

This produces a `*_analysis.csv` file next to the source log.

The trial exporter:

- keeps one row per trial
- excludes inactive tail trials caused by shutdown, by default
- adds `step_role`
- adds `analysis_focus`
- adds `analysis_outcome`
- preserves teleoperation-effort fields needed for HRI analysis, including:
  - `time_teleop`
  - `active_joystick_time_sec`
  - `teleop_distance_m`
  - `autonomous_distance_m`
  - `teleop_distance_proportion`
  - `avg_teleop_entropy`

`analysis_outcome` distinguishes:

- `success`
- `interrupted_after_commit`
- `interrupted_during_execution`
- `failed_after_commit`
- `failed_without_commit`

Use `--include-all` only when you want raw administrative rows as well.

Practical meaning of the new teleoperation fields:

- `time_teleop` / `teleop_time_sec`
  - total non-autonomous time within the trial
  - this includes waiting, aiming, brief pauses, and other user-controlled phases even when the joystick is not being pushed continuously
- `active_joystick_time_sec`
  - only the time when joystick effort is above the activity threshold
  - use this when you want a closer estimate of actual continuous stick manipulation rather than overall teleoperation-stage duration
- `teleop_distance_m`
  - end-effector path length accumulated outside autonomous execution segments
- `autonomous_distance_m`
  - end-effector path length accumulated during autonomous execution segments
- `teleop_distance_proportion`
  - `teleop_distance_m / ee_path_length_m`
- `avg_teleop_entropy`
  - time-weighted mean entropy of the intent probability distribution during teleoperation only
  - lower values usually mean the system distribution is more decisive; higher values mean the intent estimate stayed more ambiguous

## 9. Export Block-Level Analysis CSV

Export one row per questionnaire unit:

```bash
python3 ~/catkin_ws/src/tabletop_workspace_opt/scripts/block_analysis_exporter.py
```

Or export a specific log:

```bash
python3 ~/catkin_ws/src/tabletop_workspace_opt/scripts/block_analysis_exporter.py \
  ~/catkin_ws/src/tabletop_workspace_opt/logs/user_study_trials_20260612_160417.jsonl
```

This produces a `*_block_analysis.csv` file next to the source log.

The block exporter aggregates trials by:

- `session_id`
- `participant_id`
- `condition_id`
- `task_id`

It reports:

- trial counts
- pickup vs destination counts
- success rates
- destination correct inference rate
- mean timing metrics
- mean autonomous time
- mean teleoperation-stage time
- mean active joystick time
- mean teleoperation and autonomous distance proportions
- mean teleoperation-stage entropy
- mean confirmation / switch / cancel / timeout metrics
- interruption outcome counts

## 10. Join With Questionnaire Data

Use these fields to join exported CSVs with questionnaire data:

- `participant_id`
- `condition_id`
- `task_type`

If your questionnaire is collected per session, also keep:

- `session_id`

For the current paper framing, the recommended primary metrics are:

- target selection correctness
- time to commit to the intended target
- top-goal switching / ambiguity indicators
- cancel count and wrong-target attempts
- sorting success rate
- sandwich completion rate
- task completion time
- autonomous assistance time
- joystick / teleoperation time
- teleoperation distance proportion
- teleoperation-stage average intent entropy

Recommended interpretation by task:

- `Sorting`
  - analyze whether optimized layouts improve object and destination disambiguation
- `Sandwich Assembly`
  - analyze whether optimized layouts improve per-layer target disambiguation
  - treat grasping and placement as downstream execution after target selection becomes sufficiently clear

## 11. User Study SOP

The printable SOP has been moved to:

- [EXPERIMENTER_SOP.md](/home/gyanig/catkin_ws/src/tabletop_workspace_opt/EXPERIMENTER_SOP.md)

Use that file for experiment execution. Keep `README.md` as the system and workflow reference.

## 12. Post-Session Checklist

After each participant block, complete this checklist before starting the next block:

1. Validate the latest trial log.
2. Export the latest trial analysis CSV.
3. Export the latest block analysis CSV.
4. Confirm that `participant_id`, `condition_id`, and `block_id` match the questionnaire spreadsheet.
5. Save the questionnaire responses before moving on.
6. Spot-check the exported CSVs:
   - only real participant trials are present
   - no unexpected inactive `node_shutdown` rows remain
   - `step_role`, `analysis_focus`, and `analysis_outcome` look correct
7. If anything looks wrong, stop and fix it before the next participant.


---

# 🧠 Intent Recognition & Perception Pipeline

<!-- <details>
<summary><strong>Click to expand</strong></summary> -->

Launch the perception → tracking → inference → visualization stack:

```bash
roslaunch tabletop_workspace_opt apriltag_shared_control.launch
```

Includes:

- RealSense image stream
- AprilTag candidate detection / registry
- end-effector tracking
- intent distribution
- RViz visualization

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

- Optimized object poses `[x, y, theta]`
- `wo_layout_cma-me.png`
- `wo_archive_heatmap_cma-me.png`
- Archive `.pkl`

### Configurable Parameters

- Object sizes
- Colors
- Task graph
- Locked objects

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

Copyright (c) 2025 <Your Name>

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

- Rethink Robotics Sawyer SDK
- Intel RealSense
- RelaxedIK (UW Graphics Lab)
- Ultralytics YOLO
- Pyribs (MAP-Elites)
