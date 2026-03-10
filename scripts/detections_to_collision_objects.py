#!/usr/bin/env python3
import rospy
from moveit_msgs.msg import CollisionObject
from shape_msgs.msg import SolidPrimitive
from shape_msgs.msg import Mesh, MeshTriangle
from geometry_msgs.msg import Point
from geometry_msgs.msg import Pose
from vision_msgs.msg import Detection2DArray
import os
import re
import rospkg
import math


class DetectionsToCollisionObjects:
    def __init__(self):
        self.world_frame = rospy.get_param("~world_frame", "world")
        self.input_topic = rospy.get_param("~input_topic", "/mujoco_sim/detections")
        self.output_topic = rospy.get_param("~output_topic", "/collision_object")
        self.object_prefix = rospy.get_param("~object_prefix", "mujoco_obj")
        # For sources without stable track IDs, detection index is often more stable than hypothesis id.
        self.use_detection_index_as_id = bool(rospy.get_param("~use_detection_index_as_id", True))

        self.default_size_x = float(rospy.get_param("~default_size_x", 0.05))
        self.default_size_y = float(rospy.get_param("~default_size_y", 0.05))
        self.default_size_z = float(rospy.get_param("~default_size_z", 0.05))

        # Optional per-id override: {"0":[x,y,z], "1":[x,y,z], ...}
        self.id_to_size = rospy.get_param("~id_to_size", {})
        # Optional mesh config: {"0":"/abs/path/to.obj", ...}
        self.id_to_mesh_file = rospy.get_param("~id_to_mesh_file", {})
        # Optional mesh scale: {"0":[sx,sy,sz], ...}
        self.id_to_mesh_scale = rospy.get_param("~id_to_mesh_scale", {})
        # Optional orientation offset in xyzw: {"0":[x,y,z,w], ...}
        self.id_to_orientation_offset = rospy.get_param("~id_to_orientation_offset", {})
        self._mesh_cache = {}
        self._rospack = rospkg.RosPack()
        self._known_objects = set()

        self.pub = rospy.Publisher(self.output_topic, CollisionObject, queue_size=20)
        self.sub = rospy.Subscriber(self.input_topic, Detection2DArray, self._cb, queue_size=1)

        rospy.loginfo(
            "[detections_to_collision_objects] %s -> %s (frame=%s)",
            self.input_topic,
            self.output_topic,
            self.world_frame,
        )

    def _size_for_id(self, det_id):
        val = self.id_to_size.get(str(det_id))
        if isinstance(val, list) and len(val) == 3:
            return float(val[0]), float(val[1]), float(val[2])
        return self.default_size_x, self.default_size_y, self.default_size_z

    def _mesh_scale_for_id(self, det_id):
        val = self.id_to_mesh_scale.get(str(det_id))
        if isinstance(val, list) and len(val) == 3:
            return float(val[0]), float(val[1]), float(val[2])
        return 1.0, 1.0, 1.0

    def _orientation_offset_for_id(self, det_id):
        val = self.id_to_orientation_offset.get(str(det_id))
        if isinstance(val, list) and len(val) == 4:
            return float(val[0]), float(val[1]), float(val[2]), float(val[3])
        return 0.0, 0.0, 0.0, 1.0

    def _quat_mul_xyzw(self, q1, q2):
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        return (
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        )

    def _load_obj_mesh(self, obj_path, scale_xyz):
        vertices = []
        triangles = []

        with open(obj_path, "r", encoding="utf-8", errors="ignore") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue

                if line.startswith("v "):
                    parts = line.split()
                    if len(parts) < 4:
                        continue
                    p = Point()
                    p.x = float(parts[1]) * scale_xyz[0]
                    p.y = float(parts[2]) * scale_xyz[1]
                    p.z = float(parts[3]) * scale_xyz[2]
                    vertices.append(p)
                    continue

                if line.startswith("f "):
                    parts = line.split()[1:]
                    idx = []
                    for token in parts:
                        # OBJ supports forms like "i", "i/j", "i/j/k"
                        vi = token.split("/")[0]
                        if not vi:
                            continue
                        raw_idx = int(vi)
                        # OBJ index rules:
                        #  >0 : absolute, 1-based
                        #  <0 : relative to current vertex list tail
                        if raw_idx > 0:
                            vidx = raw_idx - 1
                        elif raw_idx < 0:
                            vidx = len(vertices) + raw_idx
                        else:
                            continue
                        if vidx < 0 or vidx >= len(vertices):
                            continue
                        idx.append(vidx)

                    # Fan triangulation for polygons with N >= 3
                    for k in range(1, len(idx) - 1):
                        if idx[0] == idx[k] or idx[k] == idx[k + 1] or idx[0] == idx[k + 1]:
                            continue
                        tri = MeshTriangle()
                        tri.vertex_indices = [idx[0], idx[k], idx[k + 1]]
                        triangles.append(tri)

        mesh = Mesh()
        mesh.vertices = vertices
        mesh.triangles = triangles
        return mesh

    def _mesh_for_id(self, det_id):
        sid = str(det_id)
        if sid not in self.id_to_mesh_file:
            return None

        if sid in self._mesh_cache:
            return self._mesh_cache[sid]

        mesh_path = self._resolve_mesh_path(self.id_to_mesh_file[sid])
        if not os.path.isfile(mesh_path):
            rospy.logwarn_throttle(2.0, f"[detections_to_collision_objects] mesh not found for id {sid}: {mesh_path}")
            return None

        scale_xyz = self._mesh_scale_for_id(det_id)
        try:
            mesh = self._load_obj_mesh(mesh_path, scale_xyz)
            self._mesh_cache[sid] = mesh
            rospy.loginfo(f"[detections_to_collision_objects] loaded mesh id={sid}: {mesh_path}")
            return mesh
        except Exception as e:
            rospy.logwarn_throttle(2.0, f"[detections_to_collision_objects] failed loading mesh id {sid}: {e}")
            return None

    def _resolve_mesh_path(self, raw_path):
        # Resolve ROS launch-style token: $(find package_name)/path/to/file
        m = re.match(r"^\$\(find\s+([^)]+)\)(.*)$", raw_path)
        if not m:
            return raw_path
        pkg = m.group(1).strip()
        suffix = m.group(2).strip()
        try:
            pkg_path = self._rospack.get_path(pkg)
            return os.path.normpath(pkg_path + suffix)
        except Exception:
            return raw_path

    def _cb(self, msg):
        for det_idx, det in enumerate(msg.detections):
            if not det.results:
                continue

            hyp = det.results[0]
            det_id = det_idx if self.use_detection_index_as_id else int(hyp.id)
            pose = hyp.pose.pose
            sx, sy, sz = self._size_for_id(det_id)

            # Keep upstream orientation, normalize and fallback to identity if invalid.
            qn = math.sqrt(
                pose.orientation.x * pose.orientation.x +
                pose.orientation.y * pose.orientation.y +
                pose.orientation.z * pose.orientation.z +
                pose.orientation.w * pose.orientation.w
            )
            if qn < 1e-9:
                pose.orientation.x = 0.0
                pose.orientation.y = 0.0
                pose.orientation.z = 0.0
                pose.orientation.w = 1.0
            else:
                pose.orientation.x /= qn
                pose.orientation.y /= qn
                pose.orientation.z /= qn
                pose.orientation.w /= qn

            # Apply optional per-object fixed orientation offset: q_world_mesh = q_world_body * q_body_mesh.
            q_body = (
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            )
            q_off = self._orientation_offset_for_id(det_id)
            q_mesh = self._quat_mul_xyzw(q_body, q_off)
            qn2 = math.sqrt(q_mesh[0] * q_mesh[0] + q_mesh[1] * q_mesh[1] + q_mesh[2] * q_mesh[2] + q_mesh[3] * q_mesh[3])
            if qn2 < 1e-9:
                pose.orientation.x = 0.0
                pose.orientation.y = 0.0
                pose.orientation.z = 0.0
                pose.orientation.w = 1.0
            else:
                pose.orientation.x = q_mesh[0] / qn2
                pose.orientation.y = q_mesh[1] / qn2
                pose.orientation.z = q_mesh[2] / qn2
                pose.orientation.w = q_mesh[3] / qn2

            co = CollisionObject()
            co.header.stamp = rospy.Time.now()
            co.header.frame_id = self.world_frame
            co.id = f"{self.object_prefix}_{det_id}"
            co.pose = pose

            mesh = self._mesh_for_id(det_id)
            known = co.id in self._known_objects

            if known:
                # For MOVE, MoveIt uses object pose and ignores geometry payload.
                co.operation = CollisionObject.MOVE
            else:
                co.operation = CollisionObject.ADD
                identity_pose = Pose()
                identity_pose.orientation.w = 1.0
                if mesh is not None:
                    co.meshes.append(mesh)
                    co.mesh_poses.append(identity_pose)
                else:
                    prim = SolidPrimitive()
                    prim.type = SolidPrimitive.BOX
                    prim.dimensions = [sx, sy, sz]
                    co.primitives.append(prim)
                    co.primitive_poses.append(identity_pose)
                self._known_objects.add(co.id)

            self.pub.publish(co)


if __name__ == "__main__":
    rospy.init_node("detections_to_collision_objects")
    DetectionsToCollisionObjects()
    rospy.spin()
