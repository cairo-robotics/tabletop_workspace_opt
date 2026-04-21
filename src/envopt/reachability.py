"""IK reachability oracle using MuJoCo's site Jacobian.

Self-contained numerical IK: iterative damped-least-squares starting from
multiple seeds. No ROS / TracIK dependency. Fast enough for use in the
optimizer inner loop with LRU caching on rounded pose keys.

API:
    oracle = ReachabilityOracle(scene_xml_path)
    ok = oracle.is_reachable(pos=[x, y, z], R=rot3x3)
    q  = oracle.solve_ik(pos, R)            # returns joint angles or None

The oracle also exposes scene-aware helpers used by the workspace
optimizer's collision-aware feasibility check:
    oracle.teleport_object(mujoco_name, pos, yaw)
    n = oracle.arm_object_collisions(q_arm, target_body_name)
"""
import functools
from typing import Dict, List, Optional, Tuple

import mujoco
import numpy as np

from envopt.se3_utils import rot_log, yaw_to_quat


# Sawyer arm + gripper body names. Anything whose body is in this set
# is treated as part of the robot.
ARM_BODY_NAMES = frozenset({
    "pedestal", "base", "head",
    "right_l0", "right_l1", "right_l2", "right_l3",
    "right_l4", "right_l5", "right_l6",
    "gripper_body", "gripper_finger_left", "gripper_finger_right",
})
TABLE_BODY_NAMES = frozenset({"tablelink"})


def _rounded_key(pos, R, pos_bin=0.01, rot_bin_deg=10.0):
    """Quantize (pos, R) into a hashable cache key.

    pos bucketed in 1 cm; R bucketed in ~10° of yaw/pitch/roll (approx).
    """
    pos = np.asarray(pos, dtype=float)
    pb = tuple(int(round(x / pos_bin)) for x in pos)
    if R is None:
        return (pb, None)
    R = np.asarray(R, dtype=float)
    # Use first two columns quantized — enough to identify rotation
    cols = R[:, :2].flatten()
    cb = tuple(int(round(x * (180.0 / rot_bin_deg))) for x in cols)
    return (pb, cb)


class ReachabilityOracle:
    """Sawyer arm reachability check via Jacobian-based numerical IK."""

    HOME_JOINTS = np.array(
        [0.0, -1.1775, 0.0, 2.1761, 0.0, 0.5663, 3.3124])

    def __init__(self, scene_xml_path: str,
                 pos_tol: float = 0.01,
                 rot_tol_deg: float = 10.0,
                 max_iters: int = 60,
                 n_seeds: int = 4,
                 damping: float = 0.05,
                 cache_size: int = 8192):
        self.model = mujoco.MjModel.from_xml_path(scene_xml_path)
        self.data = mujoco.MjData(self.model)
        self.ee_site_id = self.model.site("end_effector").id

        self.arm_dof_addrs = []
        self.arm_qpos_addrs = []
        self.joint_lower = []
        self.joint_upper = []
        for i in range(7):
            jid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, f"right_j{i}")
            self.arm_dof_addrs.append(self.model.jnt_dofadr[jid])
            self.arm_qpos_addrs.append(self.model.jnt_qposadr[jid])
            self.joint_lower.append(self.model.jnt_range[jid, 0])
            self.joint_upper.append(self.model.jnt_range[jid, 1])
        self.joint_lower = np.array(self.joint_lower)
        self.joint_upper = np.array(self.joint_upper)

        self.pos_tol = pos_tol
        self.rot_tol = np.deg2rad(rot_tol_deg)
        self.max_iters = max_iters
        self.n_seeds = n_seeds
        self.damping = damping
        self.rng = np.random.default_rng(0)

        # LRU cache for is_reachable
        self._cache = {}
        self._cache_size = cache_size
        self._cache_hits = 0
        self._cache_misses = 0

    def _set_joints(self, q):
        for i, addr in enumerate(self.arm_qpos_addrs):
            self.data.qpos[addr] = q[i]
        mujoco.mj_forward(self.model, self.data)

    def _get_ee_pose(self):
        pos = self.data.site_xpos[self.ee_site_id].copy()
        R = self.data.site_xmat[self.ee_site_id].reshape(3, 3).copy()
        return pos, R

    def _step_ik(self, q, target_pos, target_R):
        """One damped-least-squares step toward the target pose."""
        self._set_joints(q)
        cur_pos, cur_R = self._get_ee_pose()

        pos_err = target_pos - cur_pos
        if target_R is not None:
            R_err = target_R @ cur_R.T
            rot_err = rot_log(R_err)
            err = np.concatenate([pos_err, rot_err])
            n_task = 6
        else:
            err = pos_err
            n_task = 3

        nv = self.model.nv
        jacp = np.zeros((3, nv))
        jacr = np.zeros((3, nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_site_id)
        Jp = jacp[:, self.arm_dof_addrs]
        Jr = jacr[:, self.arm_dof_addrs]
        J = np.vstack([Jp, Jr])[:n_task, :]

        JJT = J @ J.T + self.damping * np.eye(n_task)
        dq = J.T @ np.linalg.solve(JJT, err)
        return dq, err

    def solve_ik(self, target_pos, target_R=None) -> Optional[np.ndarray]:
        """Try to find joint angles that reach the target pose.

        Returns a (7,) array of joint angles on success, or None.
        """
        target_pos = np.asarray(target_pos, dtype=float)
        if target_R is not None:
            target_R = np.asarray(target_R, dtype=float)

        best_q = None
        best_err = np.inf

        seeds = [self.HOME_JOINTS.copy()]
        for _ in range(self.n_seeds - 1):
            # Random seed within joint limits
            seeds.append(self.rng.uniform(
                self.joint_lower + 0.1,
                self.joint_upper - 0.1,
                size=7))

        for q0 in seeds:
            q = q0.copy()
            for _ in range(self.max_iters):
                dq, err = self._step_ik(q, target_pos, target_R)
                # Step-size cap for stability
                dq_norm = np.linalg.norm(dq)
                if dq_norm > 0.5:
                    dq = dq * (0.5 / dq_norm)
                q = q + dq
                q = np.clip(q, self.joint_lower + 1e-3, self.joint_upper - 1e-3)

                pos_err_norm = np.linalg.norm(err[:3])
                if target_R is not None:
                    rot_err_norm = np.linalg.norm(err[3:])
                else:
                    rot_err_norm = 0.0
                if (pos_err_norm < self.pos_tol and
                        rot_err_norm < self.rot_tol):
                    return q.copy()
                total_err = pos_err_norm + 0.1 * rot_err_norm
                if total_err < best_err:
                    best_err = total_err
                    best_q = q.copy()

        return None  # no seed converged within tolerance

    def is_reachable(self, pos, R=None) -> bool:
        """Check whether a target pose is IK-reachable (with caching)."""
        key = _rounded_key(pos, R)
        if key in self._cache:
            self._cache_hits += 1
            return self._cache[key]
        self._cache_misses += 1

        q = self.solve_ik(pos, R)
        ok = q is not None

        if len(self._cache) >= self._cache_size:
            # Simple FIFO eviction
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = ok
        return ok

    def manipulability(self, q) -> float:
        """Yoshikawa manipulability measure at joint config q."""
        self._set_joints(q)
        nv = self.model.nv
        jacp = np.zeros((3, nv))
        jacr = np.zeros((3, nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_site_id)
        J = np.vstack([
            jacp[:, self.arm_dof_addrs],
            jacr[:, self.arm_dof_addrs],
        ])
        return float(np.sqrt(max(np.linalg.det(J @ J.T), 0.0)))

    def cache_stats(self):
        return {
            "size": len(self._cache),
            "hits": self._cache_hits,
            "misses": self._cache_misses,
        }

    # ------------------------------------------------------------------
    # Scene-aware helpers (used by collision-aware feasibility check)
    # ------------------------------------------------------------------

    def _build_geom_category_cache(self):
        """Categorize every geom in the model.

        Returns dict: geom_id -> ('arm' | 'table' | 'object:<body>' | 'other').
        Object geoms are tagged with their root free-joint body name so
        the caller can distinguish target from non-target objects.
        Cached on the instance.
        """
        if hasattr(self, "_geom_cat") and self._geom_cat is not None:
            return self._geom_cat
        cat = {}
        for gid in range(self.model.ngeom):
            bid = self.model.geom_bodyid[gid]
            bname = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
            if bname in ARM_BODY_NAMES:
                cat[gid] = "arm"
            elif bname in TABLE_BODY_NAMES:
                cat[gid] = "table"
            else:
                # Walk up to find a free-jointed ancestor (= object root)
                cur = bid
                root_name = None
                while cur > 0:
                    jadr = self.model.body_jntadr[cur]
                    if (jadr >= 0 and
                            self.model.jnt_type[jadr] == mujoco.mjtJoint.mjJNT_FREE):
                        root_name = mujoco.mj_id2name(
                            self.model, mujoco.mjtObj.mjOBJ_BODY, cur)
                        break
                    cur = self.model.body_parentid[cur]
                cat[gid] = f"object:{root_name}" if root_name else "other"
        self._geom_cat = cat
        return cat

    def teleport_object(self, mujoco_name: str, pos, yaw: float = 0.0) -> bool:
        """Move a free-jointed body in the oracle's internal model.

        Subsequent IK and collision queries will see the new position.
        Returns True if the object was found, False otherwise.
        """
        try:
            bid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, mujoco_name)
        except Exception:
            return False
        if bid < 0:
            return False
        jid = self.model.body_jntadr[bid]
        if jid < 0:
            return False
        if self.model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE:
            return False
        qadr = self.model.jnt_qposadr[jid]
        q = yaw_to_quat(yaw)
        self.data.qpos[qadr:qadr + 3] = pos
        self.data.qpos[qadr + 3:qadr + 7] = q
        # Reset the IK cache because object positions affect what
        # arm-other-object collisions are possible at any given pose.
        self._cache.clear()
        return True

    def arm_object_collisions(self, q_arm,
                              target_body_name: Optional[str] = None
                              ) -> List[Tuple[str, str]]:
        """Set the arm to q_arm, run kinematics + collision, and return
        the list of arm-vs-other-object contact pairs.

        ``target_body_name`` (the object the arm is currently grasping)
        is excluded — touching the target is the whole point of a grasp.
        Self-collisions and arm-vs-table contacts are also excluded
        because the optimizer feasibility check uses this method to
        decide whether *another* object is in the way of the grasp;
        self/table collisions are handled by separate checks.

        Returns a list of (body1, body2) tuples for offending pairs.
        """
        # Apply joint state and run forward kinematics + collision.
        self._set_joints(q_arm)  # already calls mj_forward
        cat = self._build_geom_category_cache()
        offending = []
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            c1 = cat.get(g1, "other")
            c2 = cat.get(g2, "other")
            # We want pairs where one side is arm/gripper and the
            # other is an OBJECT body (NOT the target).
            arm_side = None
            obj_side = None
            if c1 == "arm" and c2.startswith("object:"):
                arm_side, obj_side = c1, c2
            elif c2 == "arm" and c1.startswith("object:"):
                arm_side, obj_side = c2, c1
            else:
                continue
            obj_body = obj_side.split(":", 1)[1]
            if target_body_name and obj_body == target_body_name:
                continue
            arm_body = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_BODY,
                self.model.geom_bodyid[g1 if c1 == "arm" else g2]) or "?"
            offending.append((arm_body, obj_body))
        return offending
