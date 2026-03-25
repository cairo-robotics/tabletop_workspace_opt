## Experiment Pipeline
1. roscore
2. source devel/setup.bash
3. roslaunch tabletop_workspace_opt intent_recognition_sim.launch
4. python3 ~/catkin_ws/src/tabletop_workspace_opt/src/script/publish_tf_from_jointstate_fk.py \
>   _joint_topic:=/relaxed_ik/joint_angle_solutions \
>   _world_frame:=world \
>   _base_link:=base_link \
>   _tip_link:=right_l6 \
>   _cam_frame:=realsense_color_optical_frame \
>   _tip_to_cam_xyz:="[0.05, 0.0, 0.05]" \
>   _tip_to_cam_quat:="[0.0, 1.0, 0.0, 0.0]"

5. python3 ~/catkin_ws/src/tabletop_workspace_opt/src/script/auto_scan_ring_vel_local.py \ _base_frame:=world \ _tip_frame:=right_l6 \_vel_topic:=/relaxed_ik/ee_vel_goals \ _radius:=0.12 \ _z_offset:=0.12 \ _vmax:=0.03 \ _n_poses:=16 \ _out_dir:=/home/heyang/scans

6. merge and visualize. python3 ~/catkin_ws/src/tabletop_workspace_opt/src/script/ply_visualization.py \
  /home/heyang/merged_icp_clean.ply 0.003
