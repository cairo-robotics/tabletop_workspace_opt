#!/usr/bin/env python3
# Fixed Top-Down Grasp Executor (servo, segmented motion)
import numpy as np
import rospy
from geometry_msgs.msg import Twist
from relaxed_ik_ros1.msg import EEVelGoals
from intera_core_msgs.msg import EndpointState
from std_msgs.msg import Bool
from scipy.spatial.transform import Rotation as R


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


class GraspExecutorServo:
    def __init__(self):
        # pubs
        self.ee_pub = rospy.Publisher("/relaxed_ik/ee_vel_goals", EEVelGoals, queue_size=1)
        self.grip_pub = rospy.Publisher("/mujoco_sim/gripper_open", Bool, queue_size=1)

        # sub
        self.ee_state = None
        rospy.Subscriber("/mujoco_sim/endpoint_state", EndpointState, self._ee_cb)

        # servo gains/limits
        self.kp_pos = rospy.get_param("~kp_pos", 1.2)
        self.kp_rot = rospy.get_param("~kp_rot", 2.0)
        self.vmax = rospy.get_param("~vmax", 0.05)  # m/s (slightly higher than before)
        self.wmax = rospy.get_param("~wmax", 1.2)   # rad/s

        # tolerances
        self.pos_tol = rospy.get_param("~pos_tol", 0.008)  # 8 mm
        self.ang_tol = np.deg2rad(rospy.get_param("~ang_tol_deg", 6.0))

        # near-floor safety tuning (ONLY enable in descend/approach)
        self.near_floor_margin = rospy.get_param("~near_floor_margin", 0.02)      # 2 cm
        self.v_xy_near_floor_max = rospy.get_param("~v_xy_near_floor_max", 0.006) # 6 mm/s

        # stuck detection
        self.stuck_pos_eps = rospy.get_param("~stuck_pos_eps", 5e-4)     # 0.5mm
        self.stuck_cmd_eps = rospy.get_param("~stuck_cmd_eps", 0.004)    # 4mm/s
        self.stuck_count_thresh = rospy.get_param("~stuck_count_thresh", 25)  # ~0.8s @30Hz

    def _ee_cb(self, msg: EndpointState):
        self.ee_state = msg

    def wait_ee(self):
        t0 = rospy.Time.now()
        while self.ee_state is None and not rospy.is_shutdown():
            if (rospy.Time.now() - t0).to_sec() > 5.0:
                raise RuntimeError("No /mujoco_sim/endpoint_state received.")
            rospy.sleep(0.01)

    def _publish_twist(self, v, w):
        msg = EEVelGoals()
        twist = Twist()
        twist.linear.x, twist.linear.y, twist.linear.z = float(v[0]), float(v[1]), float(v[2])
        twist.angular.x, twist.angular.y, twist.angular.z = float(w[0]), float(w[1]), float(w[2])
        msg.ee_vels.append(twist)

        # Some RelaxedIK builds ignore tolerances; keep it present but harmless.
        msg.tolerances.append(Twist())
        self.ee_pub.publish(msg)

    @staticmethod
    def _clip_norm(vec, max_norm):
        n = np.linalg.norm(vec)
        if n < 1e-9:
            return vec
        if n > max_norm:
            return vec / n * max_norm
        return vec

    @staticmethod
    def _quat_xyzw_from_msg(q):
        return np.array([q.x, q.y, q.z, q.w], dtype=float)

    def get_current_pose(self):
        p = np.array([
            self.ee_state.pose.position.x,
            self.ee_state.pose.position.y,
            self.ee_state.pose.position.z
        ], dtype=float)
        q = self._quat_xyzw_from_msg(self.ee_state.pose.orientation)
        return p, q

    def servo_to_pose(
        self,
        p_tgt, q_tgt_xyzw,
        timeout=8.0, rate_hz=30,
        control_orientation=True,
        z_min=None,
        enable_near_floor_safety=False,
        debug_every_sec=1.0
    ):
        """
        z_min: hard floor in EE Z (never command downward at/below z_min)
        enable_near_floor_safety: if True, limit XY speed near floor + stuck detect.
        """
        self.wait_ee()
        r = rospy.Rate(rate_hz)
        t0 = rospy.Time.now()
        t_last_dbg = rospy.Time.now()

        R_tgt = R.from_quat(q_tgt_xyzw) if control_orientation else None

        last_p = None
        stuck = 0

        while not rospy.is_shutdown():
            p_cur, q_cur = self.get_current_pose()
            dp = (p_tgt - p_cur)

            if control_orientation:
                R_cur = R.from_quat(q_cur)
                R_err = R_tgt * R_cur.inv()
                ang = float(R_err.magnitude())
                dw = R_err.as_rotvec()
            else:
                ang = 0.0
                dw = np.zeros(3)

            # stop condition
            if np.linalg.norm(dp) < self.pos_tol and (not control_orientation or ang < self.ang_tol):
                self._publish_twist(np.zeros(3), np.zeros(3))
                return True

            v = self._clip_norm(self.kp_pos * dp, self.vmax)
            w = self._clip_norm(self.kp_rot * dw, self.wmax) if control_orientation else np.zeros(3)

            # ---------------- floor safety ----------------
            if z_min is not None:
                # never command downward below hard floor
                if p_cur[2] <= z_min and v[2] < 0.0:
                    v[2] = 0.0

                if enable_near_floor_safety:
                    near_floor = (p_cur[2] < (z_min + self.near_floor_margin))

                    # near floor: limit XY speed to avoid sweeping
                    if near_floor:
                        v_xy = v[0:2].copy()
                        v_xy = self._clip_norm(v_xy, self.v_xy_near_floor_max)
                        v[0], v[1] = v_xy[0], v_xy[1]

                        # stuck detection near floor only
                        if last_p is not None:
                            moved = np.linalg.norm(p_cur - last_p)
                            commanding = np.linalg.norm(v) > self.stuck_cmd_eps
                            if moved < self.stuck_pos_eps and commanding:
                                stuck += 1
                            else:
                                stuck = 0
                        last_p = p_cur.copy()

                        if stuck > self.stuck_count_thresh:
                            rospy.logwarn("Stuck/contact detected near floor -> stop.")
                            self._publish_twist(np.zeros(3), np.zeros(3))
                            return False
                    else:
                        stuck = 0
                        last_p = p_cur.copy()
            # ---------------------------------------------

            self._publish_twist(v, w)

            # debug print
            if debug_every_sec is not None and debug_every_sec > 0:
                if (rospy.Time.now() - t_last_dbg).to_sec() >= debug_every_sec:
                    rospy.loginfo(
                        "cur=(%.3f %.3f %.3f) tgt=(%.3f %.3f %.3f) |dp|=%.3f v=(%.3f %.3f %.3f) ang=%.1fdeg",
                        p_cur[0], p_cur[1], p_cur[2],
                        p_tgt[0], p_tgt[1], p_tgt[2],
                        float(np.linalg.norm(dp)),
                        v[0], v[1], v[2],
                        float(np.rad2deg(ang))
                    )
                    t_last_dbg = rospy.Time.now()

            if (rospy.Time.now() - t0).to_sec() > timeout:
                self._publish_twist(np.zeros(3), np.zeros(3))
                return False

            r.sleep()

    def move_linear_steps(
        self,
        p_start, p_end, q_xyzw,
        steps=20, per_step_timeout=2.0,
        control_orientation=True,
        z_min=None,
        enable_near_floor_safety=False
    ):
        for i in range(1, steps + 1):
            a = i / float(steps)
            p = (1 - a) * p_start + a * p_end
            ok = self.servo_to_pose(
                p, q_xyzw,
                timeout=per_step_timeout,
                control_orientation=control_orientation,
                z_min=z_min,
                enable_near_floor_safety=enable_near_floor_safety,
                debug_every_sec=None,  # keep quiet per step
            )
            if not ok:
                return False
        return True


def quat_from_topdown_params():
    """
    Define a fixed TopDown orientation.
    Default: gripper tool z-axis points downward in world (depends on your EE convention).
    You might need to tweak roll/pitch signs once.
    """
    roll = np.deg2rad(rospy.get_param("~topdown_roll_deg", 180.0))
    pitch = np.deg2rad(rospy.get_param("~topdown_pitch_deg", 0.0))
    yaw = np.deg2rad(rospy.get_param("~topdown_yaw_deg", 0.0))
    q = R.from_euler("xyz", [roll, pitch, yaw]).as_quat()  # xyzw
    return q


def main():
    rospy.init_node("fixed_grasp_executor_servo")

    exe = GraspExecutorServo()
    exe.wait_ee()

    # ---------------- Fixed grasp params ----------------
    # Target XY in the SAME frame as /mujoco_sim/endpoint_state pose (usually world in your sim)
    x_tgt = rospy.get_param("~target_x", 0.65)
    y_tgt = rospy.get_param("~target_y", -0.05)

    # Table / height params (must match endpoint_state frame)
    table_z = rospy.get_param("~table_z", 0.915)

    # If endpoint_state pose is at wrist (not fingertip), use tool_z_offset to convert "contact height"
    tool_z_offset = rospy.get_param("~tool_z_offset", 0.08)  # meters

    # Top-down grasp depth relative to table
    # Example: object height ~0.08, grasp at mid-ish: 0.05 above table (tune)
    grasp_z_above_table = rospy.get_param("~grasp_z_above_table", 0.05)

    z_safe_above_table = rospy.get_param("~z_safe_above_table", 0.25)   # move in XY at this height
    z_hover_above_table = rospy.get_param("~z_hover_above_table", 0.12) # pregrasp hover

    lift_dist = rospy.get_param("~lift_dist", 0.12)

    # hard floor: prevent going below table+margin in EE pose Z
    z_hard_margin = rospy.get_param("~z_hard_margin", 0.015)

    # ---------------------------------------------------
    # Convert "desired fingertip grasp height" to endpoint Z:
    # If endpoint pose is at wrist and fingertip is lower, you need to command wrist higher.
    # Here we assume tool_z_offset is how much lower the contact point is than endpoint along +Z.
    # So endpoint_z = desired_contact_z + tool_z_offset.
    contact_grasp_z = table_z + grasp_z_above_table
    contact_safe_z  = table_z + z_safe_above_table
    contact_hover_z = table_z + z_hover_above_table

    ee_grasp_z = contact_grasp_z + tool_z_offset
    ee_safe_z  = contact_safe_z  + tool_z_offset
    ee_hover_z = contact_hover_z + tool_z_offset

    z_min_servo = table_z + z_hard_margin  # hard floor in endpoint Z
    # (If endpoint is wrist and tool_z_offset is big, you may need z_min_servo = table_z + z_hard_margin + tool_z_offset)

    # Build poses
    q_top = quat_from_topdown_params()

    p_cur, q_cur = exe.get_current_pose()
    rospy.loginfo("EE init pose: (%.3f %.3f %.3f)", p_cur[0], p_cur[1], p_cur[2])
    rospy.loginfo(
        "Params: table_z=%.3f tool_off=%.3f z_min=%.3f | contact(z_safe=%.3f z_hover=%.3f z_grasp=%.3f) -> ee(z_safe=%.3f z_hover=%.3f z_grasp=%.3f)",
        table_z, tool_z_offset, z_min_servo,
        contact_safe_z, contact_hover_z, contact_grasp_z,
        ee_safe_z, ee_hover_z, ee_grasp_z
    )

    p_lift = np.array([x_tgt, y_tgt, ee_grasp_z + lift_dist], dtype=float)
    p_lift[2] = max(p_lift[2], ee_hover_z)

    # ---------------- Execute segmented plan ----------------
    # 0) open
    rospy.loginfo("Open gripper")
    exe.grip_pub.publish(Bool(True))
    rospy.sleep(0.25)

    # 1) lift straight up to safe Z (keep current x,y)
    rospy.loginfo("1) Lift to safe Z")
    p1 = np.array([p_cur[0], p_cur[1], ee_safe_z], dtype=float)
    ok = exe.servo_to_pose(
        p1, q_top,
        timeout=12.0,
        control_orientation=True,
        z_min=z_min_servo,
        enable_near_floor_safety=False,
        debug_every_sec=1.0
    )
    if not ok:
        rospy.logwarn("Failed: lift to safe Z")
        return

    # 2) move in XY at safe Z
    rospy.loginfo("2) Move XY at safe Z to above target")
    p2 = np.array([x_tgt, y_tgt, ee_safe_z], dtype=float)
    ok = exe.servo_to_pose(
        p2, q_top,
        timeout=18.0,
        control_orientation=True,
        z_min=z_min_servo,
        enable_near_floor_safety=False,
        debug_every_sec=1.0
    )
    if not ok:
        rospy.logwarn("Failed: move XY at safe Z")
        return

    # 3) descend to hover (pure Z)
    rospy.loginfo("3) Descend to hover")
    p3 = np.array([x_tgt, y_tgt, ee_hover_z], dtype=float)
    ok = exe.servo_to_pose(
        p3, q_top,
        timeout=12.0,
        control_orientation=True,
        z_min=z_min_servo,
        enable_near_floor_safety=True,   # only here and below
        debug_every_sec=1.0
    )
    if not ok:
        rospy.logwarn("Failed: descend to hover")
        return

    # 4) descend to grasp (linear steps, Z only)
    rospy.loginfo("4) Approach down to grasp (linear steps)")
    p4 = np.array([x_tgt, y_tgt, ee_grasp_z], dtype=float)
    ok = exe.move_linear_steps(
        p3, p4, q_top,
        steps=18,
        per_step_timeout=2.0,
        control_orientation=True,
        z_min=z_min_servo,
        enable_near_floor_safety=True
    )
    if not ok:
        rospy.logwarn("Failed: approach to grasp")
        return

    # 5) close
    rospy.loginfo("5) Close gripper")
    exe.grip_pub.publish(Bool(False))
    rospy.sleep(0.45)

    # 6) lift
    rospy.loginfo("6) Lift")
    ok = exe.servo_to_pose(
        p_lift, q_top,
        timeout=14.0,
        control_orientation=True,
        z_min=z_min_servo,
        enable_near_floor_safety=False,
        debug_every_sec=1.0
    )
    if not ok:
        rospy.logwarn("Failed: lift after grasp")
        return

    rospy.loginfo("Done. Add grasp success check (e.g. contact/force/joint) if available.")


if __name__ == "__main__":
    main()