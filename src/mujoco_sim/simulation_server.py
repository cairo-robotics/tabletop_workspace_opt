#!/usr/bin/env python3
import rospy
import rospkg
import numpy as np
import os
import threading
import time
from mujoco_visualizer import MuJoCoVisualizer
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from intera_core_msgs.msg import EndpointState
from std_msgs.msg import Bool


class SimulationServer:
    def __init__(self):
        rospkg_instance = rospkg.RosPack()
        package_path = rospkg_instance.get_path('tabletop_workspace_opt')

        scene_name = rospy.get_param("~scene_name", "simple_scene")
        scene_xml_path = rospy.get_param("~scene_xml_path", "")
        scene_path = self._resolve_scene_path(package_path, scene_name, scene_xml_path)

        # NEW: headless control
        self.headless = rospy.get_param("~headless", False)
        self.sim_rate_hz = float(rospy.get_param("~sim_rate_hz", 200.0))  # headless step rate
        self.sim_dt = 1.0 / max(self.sim_rate_hz, 1.0)

        # OPTIONAL: force offscreen backend (helps WSL even if something tries to init GL)
        # You can also set this in launch via <env name="MUJOCO_GL" value="osmesa"/>
        if self.headless and "MUJOCO_GL" not in os.environ:
            os.environ["MUJOCO_GL"] = "osmesa"

        rospy.loginfo(f"[simulation_server] Loading MuJoCo scene from: {scene_path}")
        self.visualizer = MuJoCoVisualizer(scene_path, headless=self.headless)

        # Subscribers (fix leading slash)
        rospy.Subscriber("/relaxed_ik/joint_angle_solutions", JointState, self.joint_solution_cb, queue_size=1)
        rospy.Subscriber("/mujoco_sim/joint_trajectory", JointTrajectory, self.joint_trajectory_cb, queue_size=1)
        rospy.Subscriber("/mujoco_sim/gripper_open", Bool, self.gripper_cb, queue_size=1)

        # default joint positions for the Sawyer robot
        starting_config = [0.0, -1.1775, 0.0, 2.1761, 0.0, 0.5663, 3.3124]
        self.visualizer.add_target_to_trajectory(starting_config)

        # Publishers
        self.det_pub = rospy.Publisher("/mujoco_sim/detections", Detection2DArray, queue_size=1)
        self.ee_pub  = rospy.Publisher("/mujoco_sim/endpoint_state", EndpointState, queue_size=1)

        self.object_names = rospy.get_param(
            "~object_names",
            ["block1", "block2", "block3"]
        )
        rospy.loginfo(f"[simulation_server] Tracking objects for detections: {self.object_names}")

        # Thread: publish detections + ee pose
        self.pub_thread = threading.Thread(target=self.publish_loop, daemon=True)
        self.pub_thread.start()

        # NEW: headless simulation stepping thread
        self.sim_thread = None
        if self.headless:
            self.sim_thread = threading.Thread(target=self.sim_loop_headless, daemon=True)
            self.sim_thread.start()
            rospy.logwarn("[simulation_server] Running in HEADLESS mode: no MuJoCo window will be created.")
        else:
            rospy.loginfo("[simulation_server] Running with GUI: MuJoCo window will open.")

        rospy.loginfo("Simulation server ready")

    def _resolve_scene_path(self, package_path, scene_name, scene_xml_path):
        """Resolve scene path from absolute path or scene_name across known asset folders."""
        if scene_xml_path:
            if os.path.isfile(scene_xml_path):
                return scene_xml_path
            raise RuntimeError(
                f"scene_xml_path is set but file does not exist: {scene_xml_path}"
            )

        scene_file = scene_name if scene_name.endswith(".xml") else f"{scene_name}.xml"
        candidates = [
            os.path.join(package_path, "src", "assets", scene_file),
            os.path.join(package_path, "src", "assets", "scenes", scene_file),
            os.path.join(package_path, "src", "assets", "mujoco", scene_file),
        ]

        for candidate in candidates:
            if os.path.isfile(candidate):
                return candidate

        raise RuntimeError(
            "Could not find scene XML. Checked:\n- " + "\n- ".join(candidates)
        )

    def publish_loop(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            try:
                stamp = rospy.Time.now()

                # Detections
                msg = Detection2DArray()
                msg.header.stamp = stamp
                msg.header.frame_id = "world"

                for i, obj_name in enumerate(self.object_names):
                    pos, quat = self.visualizer.get_object_pose(obj_name)
                    if np.isnan(pos).any():
                        continue
                    det = Detection2D()
                    det.header = msg.header

                    hyp = ObjectHypothesisWithPose()
                    hyp.id = i
                    hyp.score = 1.0
                    hyp.pose.pose.position.x = float(pos[0])
                    hyp.pose.pose.position.y = float(pos[1])
                    hyp.pose.pose.position.z = float(pos[2])
                    hyp.pose.pose.orientation.x = float(quat[0])
                    hyp.pose.pose.orientation.y = float(quat[1])
                    hyp.pose.pose.orientation.z = float(quat[2])
                    hyp.pose.pose.orientation.w = float(quat[3])

                    det.results.append(hyp)
                    msg.detections.append(det)

                self.det_pub.publish(msg)

                # EndpointState (do not kill loop if ee site name is unavailable)
                try:
                    pos, quat = self.visualizer.get_pose()
                    ee_msg = EndpointState()
                    ee_msg.header.stamp = stamp
                    ee_msg.header.frame_id = "world"
                    ee_msg.pose.position.x = float(pos[0])
                    ee_msg.pose.position.y = float(pos[1])
                    ee_msg.pose.position.z = float(pos[2])
                    ee_msg.pose.orientation.x = float(quat[0])
                    ee_msg.pose.orientation.y = float(quat[1])
                    ee_msg.pose.orientation.z = float(quat[2])
                    ee_msg.pose.orientation.w = float(quat[3])
                    self.ee_pub.publish(ee_msg)
                except Exception as e:
                    rospy.logwarn_throttle(2.0, f"[simulation_server] endpoint_state publish skipped: {e}")
            except Exception as e:
                rospy.logerr_throttle(2.0, f"[simulation_server] publish_loop error: {e}")
            rate.sleep()

    # NEW: headless stepping loop (no viewer)
    def sim_loop_headless(self):
        """
        Keeps MuJoCo simulation advancing without opening a window.
        This assumes MuJoCoVisualizer provides a way to step simulation without GUI.
        We'll try common method names; if none exist, we log an error.
        """
        # Try to discover a stepping method
        step_fn = None
        for name in ["step", "step_sim", "step_simulation", "step_once", "tick"]:
            if hasattr(self.visualizer, name):
                step_fn = getattr(self.visualizer, name)
                rospy.loginfo(f"[simulation_server] Using visualizer.{name}() for headless stepping.")
                break

        if step_fn is None:
            rospy.logerr(
                "[simulation_server] Headless requested but MuJoCoVisualizer has no step method. "
                "Please add a step() method in MuJoCoVisualizer that advances physics one tick."
            )
            return

        # Run loop
        while not rospy.is_shutdown():
            try:
                step_fn()
            except Exception as e:
                rospy.logerr_throttle(2.0, f"[simulation_server] headless step failed: {e}")
            time.sleep(self.sim_dt)

    def start_simulator(self):
        # GUI mode only
        self.visualizer.simulate()

    def gripper_cb(self, msg):
        self.visualizer.operate_gripper(open=bool(msg.data))
        try:
            idx = self.visualizer.trajectory_index
            self.visualizer.trajectory[idx][7] = self.visualizer.trajectory[0][7]
            self.visualizer.trajectory[idx][8] = self.visualizer.trajectory[0][8]
        except Exception:
            pass

    def joint_solution_cb(self, joint_state: JointState):
        joint_positions = np.array(joint_state.position, dtype=float)
        self.visualizer.add_target_to_trajectory(joint_positions)

    def joint_trajectory_cb(self, msg: JointTrajectory):
        if not msg.points:
            rospy.logwarn("[simulation_server] Received empty joint trajectory; ignoring.")
            return

        current_qpos = np.array(self.visualizer.trajectory[0]).copy()
        gripper_qpos = current_qpos[7:9].copy()
        trajectory = []

        for point in msg.points:
            point_positions = np.array(point.positions, dtype=float)
            if len(point_positions) == 7:
                target_complete = np.concatenate((point_positions, gripper_qpos))
            elif len(point_positions) == self.visualizer.num_joints:
                target_complete = point_positions.copy()
                target_complete[7:9] = gripper_qpos
            else:
                rospy.logwarn(
                    "[simulation_server] Skipping trajectory point with %d positions (expected 7 or %d).",
                    len(point_positions),
                    self.visualizer.num_joints,
                )
                continue

            trajectory.append(target_complete)

        if not trajectory:
            rospy.logwarn("[simulation_server] No valid points in received joint trajectory.")
            return

        self.visualizer.set_trajectory(trajectory)
        rospy.loginfo(
            "[simulation_server] Loaded joint trajectory with %d points from %s.",
            len(trajectory),
            msg._connection_header.get("callerid", "unknown") if hasattr(msg, "_connection_header") else "unknown",
        )


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    rospy.init_node("simulation_server")
    server = SimulationServer()

    # Only open MuJoCo GUI when not headless
    if not server.headless:
        server.start_simulator()  # blocks with GUI viewer

    rospy.spin()
