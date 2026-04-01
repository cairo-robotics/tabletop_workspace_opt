# Grasping (GraspNet-based) — `tabletop_workspace_opt/src/grasping`

This folder provides a ROS1 (Noetic) service node that generates grasp poses from an RGB-D frame using:

- **GraspNet Baseline** (model + inference utilities)
- **graspnetAPI** (for `GraspGroup` data structures)
- YOLO + FastSAM for object mask extraction (see `image_tools.py`)

## External requirements (NOT part of this repo)

This repo (`tabletop_workspace_opt`) does **not** vendor GraspNet code. You must obtain/install:

### 1) `graspnet-baseline`

- Source: `https://github.com/graspnet/graspnet-baseline`
- **Where to put it**: anywhere on disk, but you will point the ROS param `~graspnet_root` to the **`graspnet-baseline` folder itself**.

`~graspnet_root` must be a directory that contains (at minimum) these subfolders from the baseline repo:

- `models/`
- `dataset/`
- `utils/`

The code will add these paths to `sys.path` at runtime (see `graspnet_wrapper.py::_setup_paths()`), so it expects a **source checkout** layout, not a packaged wheel.

### 2) `graspnetAPI`

- **Required**: This package imports `graspnetAPI` (see `graspnet_wrapper.py`). If you followed `graspnet-baseline` installation instructions, you already have it installed.

### 3) Model checkpoints/weights (gitignored)

- **GraspNet checkpoint**: A pretrained GraspNet baseline checkpoint (`.tar` file) is required. Default path: `<tabletop_workspace_opt>/assets/models/checkpoint.tar` (or specify via `~checkpoint_path`).
- **FastSAM weights**: FastSAM model weights (default: `<tabletop_workspace_opt>/assets/models/FastSAM-s.pt`).
- **YOLOv8m weights**: YOLOv8m model weights (default: `<tabletop_workspace_opt>/assets/models/yolov8m.pt`).

## What runs here

### ROS node

- **Node**: `generate_grasps.py` (initializes ROS node name `generate_grasps_service_node`)
- **Service**: `~generate_grasps` of type `tabletop_workspace_opt/GenerateGrasps`
  - In the default namespace, this becomes: `/generate_grasps_service_node/generate_grasps`

### Required ROS parameter

`generate_grasps.py` will raise at startup unless you set:

- **`~graspnet_root`**: absolute path to your **`graspnet-baseline/`** folder

### Optional ROS parameters

Defaults are shown as implemented in `generate_grasps.py`:

- **`~checkpoint_path`**: path to a GraspNet checkpoint tar
  - default: `<tabletop_workspace_opt>/assets/models/checkpoint.tar`
- **`~max_gripper_width`**: clamps predicted grasp width (meters)
  - default: `0.05`
- **`~collision_thresh`**: collision threshold; set `<= 0` to skip collision filtering
  - default: `0.01`
- **`~voxel_size`**: voxel size used by collision detector (meters)
  - default: `0.01`
- **`~default_top_k`**: number of grasps returned when request `top_k <= 0`
  - default: `10`
- **`~yolo_weight_path`**: YOLO weights path
  - default: `<tabletop_workspace_opt>/assets/models/yolov8m.pt`
- **`~fastsam_weight_path`**: FastSAM weights path
  - default: `<tabletop_workspace_opt>/assets/models/FastSAM-s.pt`
- **`~yolo_conf_threshold`**: YOLO confidence threshold
  - default: `0.5`
- **`~fastsam_conf`**, **`~fastsam_iou`**: FastSAM thresholds
  - defaults: `0.5`, `0.7`

## How to run (service + client)

### Start the service node

You need to run `generate_grasps.py` with `~graspnet_root` set to your external baseline checkout.

Example parameters to provide:

- `~graspnet_root:=/abs/path/to/graspnet-baseline`
- `~checkpoint_path:=/abs/path/to/checkpoint.tar` (if you don’t use the default)

### Call the service

There is a small client in `test_files/test_generate_grasps_client.py` which calls:

- `/generate_grasps_service_node/generate_grasps`

The request fields are:

- `color_image` (`bgr8`)
- `depth_image` (`16UC1` or `32FC1` are typical)
- `camera_info` (intrinsics from the color stream)
- `top_k` (how many grasps to return)

The response returns:

- `poses[]` (in the camera optical frame, aligned to the request header)
- `scores[]`
- `widths[]` (meters)


