#!/usr/bin/env python3
import math
import os
import xml.etree.ElementTree as ET

import rospy
from geometry_msgs.msg import Point, Pose
from moveit_msgs.msg import CollisionObject
from moveit_msgs.msg import PlanningScene
from moveit_msgs.msg import PlanningSceneComponents
from moveit_msgs.srv import ApplyPlanningScene, ApplyPlanningSceneRequest
from moveit_msgs.srv import GetPlanningScene, GetPlanningSceneRequest
from shape_msgs.msg import Mesh, MeshTriangle, SolidPrimitive
from visualization_msgs.msg import Marker, MarkerArray


def _parse_floats(text, n=None, default=None):
    if text is None:
        return list(default) if default is not None else None
    vals = [float(x) for x in text.strip().split()]
    if n is not None and len(vals) != n:
        return list(default) if default is not None else None
    return vals


def _quat_wxyz_to_xyzw(qwxyz):
    return [qwxyz[1], qwxyz[2], qwxyz[3], qwxyz[0]]


def _quat_from_euler_xyz(roll, pitch, yaw):
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    # Intrinsic XYZ ~= q = qx * qy * qz
    return [
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ]


def _quat_mul_xyzw(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return [
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ]


def _quat_normalize_xyzw(q):
    n = math.sqrt(sum(v * v for v in q))
    if n < 1e-9:
        return [0.0, 0.0, 0.0, 1.0]
    return [v / n for v in q]


def _rotate_vec_by_quat_xyzw(v, q):
    # v' = q * (v,0) * q^-1
    x, y, z = v
    qx, qy, qz, qw = q
    ix = qw * x + qy * z - qz * y
    iy = qw * y + qz * x - qx * z
    iz = qw * z + qx * y - qy * x
    iw = -qx * x - qy * y - qz * z
    rx = ix * qw + iw * -qx + iy * -qz - iz * -qy
    ry = iy * qw + iw * -qy + iz * -qx - ix * -qz
    rz = iz * qw + iw * -qz + ix * -qy - iy * -qx
    return [rx, ry, rz]


def _load_obj_mesh(obj_path, scale_xyz):
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


def _make_box_primitive(size_xyz):
    prim = SolidPrimitive()
    prim.type = SolidPrimitive.BOX
    prim.dimensions = list(size_xyz)
    return prim


def _mesh_aabb(mesh_msg):
    if not mesh_msg.vertices:
        return None

    xs = [p.x for p in mesh_msg.vertices]
    ys = [p.y for p in mesh_msg.vertices]
    zs = [p.z for p in mesh_msg.vertices]
    return [
        max(1e-4, max(xs) - min(xs)),
        max(1e-4, max(ys) - min(ys)),
        max(1e-4, max(zs) - min(zs)),
    ]


def _points_from_mesh_triangles(mesh_msg):
    points = []
    for tri in mesh_msg.triangles:
        if len(tri.vertex_indices) != 3:
            continue
        try:
            p0 = mesh_msg.vertices[tri.vertex_indices[0]]
            p1 = mesh_msg.vertices[tri.vertex_indices[1]]
            p2 = mesh_msg.vertices[tri.vertex_indices[2]]
        except IndexError:
            continue
        points.append(p0)
        points.append(p1)
        points.append(p2)
    return points


class StaticSceneToCollisionObjects:
    def __init__(self):
        self.scene_xml = rospy.get_param(
            "~scene_xml_path",
            "$(find tabletop_workspace_opt)/src/assets/scenes/scene_swapped.xml",
        )
        self.world_frame = rospy.get_param("~world_frame", "world")
        self.output_topic = rospy.get_param("~output_topic", "/collision_object")
        self.object_prefix = rospy.get_param("~object_prefix", "mujoco_obj")
        self.startup_delay = float(rospy.get_param("~startup_delay", 2.0))
        self.publish_repeats = int(rospy.get_param("~publish_repeats", 3))
        self.repeat_interval = float(rospy.get_param("~repeat_interval", 0.3))
        self.enable_topic_publish = bool(rospy.get_param("~enable_topic_publish", True))
        self.mesh_as_bounding_box = bool(rospy.get_param("~mesh_as_bounding_box", False))
        self.publish_visual_markers = bool(rospy.get_param("~publish_visual_markers", True))
        self.visual_marker_topic = rospy.get_param("~visual_marker_topic", "/static_scene_visual_markers")
        self.keep_visual_markers_alive = bool(rospy.get_param("~keep_visual_markers_alive", True))
        self.visual_marker_republish_interval = float(
            rospy.get_param("~visual_marker_republish_interval", 1.0)
        )
        self.only_free_bodies = bool(rospy.get_param("~only_free_bodies", True))
        self.use_apply_planning_scene = bool(rospy.get_param("~use_apply_planning_scene", True))
        self.apply_planning_scene_service = rospy.get_param(
            "~apply_planning_scene_service", "/apply_planning_scene"
        )
        self.get_planning_scene_service = rospy.get_param(
            "~get_planning_scene_service", "/get_planning_scene"
        )
        self.clear_world_before_apply = bool(rospy.get_param("~clear_world_before_apply", True))
        self.apply_retries = int(rospy.get_param("~apply_retries", 5))
        self.apply_retry_interval = float(rospy.get_param("~apply_retry_interval", 0.5))
        self.post_apply_monitor_duration = float(
            rospy.get_param("~post_apply_monitor_duration", 15.0)
        )
        self.post_apply_monitor_interval = float(
            rospy.get_param("~post_apply_monitor_interval", 1.0)
        )

        self.scene_xml = self._resolve_find_expr(self.scene_xml)
        # Latch so late subscribers (e.g., move_group starts slightly later) can still receive one-shot publishes.
        self.pub = None
        if self.enable_topic_publish:
            self.pub = rospy.Publisher(self.output_topic, CollisionObject, queue_size=50, latch=True)
        self.marker_pub = None
        if self.publish_visual_markers:
            self.marker_pub = rospy.Publisher(
                self.visual_marker_topic, MarkerArray, queue_size=1, latch=True
            )
        self._mesh_cache = {}

        self._mesh_defs = {}  # mesh_name -> {"file": abs_path, "scale":[sx,sy,sz]}
        self._objects = []  # list[CollisionObject]
        self._markers = MarkerArray()

        self._parse_scene()

    def _resolve_find_expr(self, path):
        # Minimal resolver for "$(find pkg)/suffix"
        if not path.startswith("$(find "):
            return path
        close = path.find(")")
        if close < 0:
            return path
        pkg = path[len("$(find ") : close].strip()
        suffix = path[close + 1 :]
        try:
            import rospkg

            pkg_path = rospkg.RosPack().get_path(pkg)
            return os.path.normpath(pkg_path + suffix)
        except Exception:
            return path

    def _parse_scene(self):
        scene_path = os.path.abspath(self.scene_xml)
        scene_dir = os.path.dirname(scene_path)
        roots = self._load_xml_with_includes(scene_path)

        for root, root_dir in roots:
            for mesh in root.findall("./asset/mesh"):
                name = mesh.get("name")
                file_attr = mesh.get("file")
                if not name or not file_attr:
                    continue
                scale = _parse_floats(mesh.get("scale"), 3, [1.0, 1.0, 1.0])
                abs_file = os.path.normpath(os.path.join(root_dir, file_attr))
                self._mesh_defs[name] = {"file": abs_file, "scale": scale}

        scene_root = ET.parse(scene_path).getroot()
        worldbody = scene_root.find("./worldbody")
        if worldbody is None:
            rospy.logwarn("[static_scene_to_collision_objects] no <worldbody> in %s", scene_path)
            return

        obj_idx = 0
        for body in worldbody.findall("./body"):
            if self.only_free_bodies:
                has_free = any(
                    (j.get("type", "").strip() == "free") for j in body.findall("./joint")
                )
                if not has_free:
                    continue

            body_name = body.get("name", f"obj_{obj_idx}")
            body_pos = _parse_floats(body.get("pos"), 3, [0.0, 0.0, 0.0])
            body_quat_wxyz = _parse_floats(body.get("quat"), 4, None)
            body_euler = _parse_floats(body.get("euler"), 3, None)
            if body_quat_wxyz is not None:
                body_q = _quat_normalize_xyzw(_quat_wxyz_to_xyzw(body_quat_wxyz))
            elif body_euler is not None:
                body_q = _quat_normalize_xyzw(
                    _quat_from_euler_xyz(body_euler[0], body_euler[1], body_euler[2])
                )
            else:
                body_q = [0.0, 0.0, 0.0, 1.0]

            geom = body.find("./geom")
            if geom is None:
                continue

            geom_pos_local = _parse_floats(geom.get("pos"), 3, [0.0, 0.0, 0.0])
            geom_quat_wxyz = _parse_floats(geom.get("quat"), 4, None)
            geom_euler = _parse_floats(geom.get("euler"), 3, None)
            if geom_quat_wxyz is not None:
                geom_q = _quat_normalize_xyzw(_quat_wxyz_to_xyzw(geom_quat_wxyz))
            elif geom_euler is not None:
                geom_q = _quat_normalize_xyzw(
                    _quat_from_euler_xyz(geom_euler[0], geom_euler[1], geom_euler[2])
                )
            else:
                geom_q = [0.0, 0.0, 0.0, 1.0]

            world_geom_pos = _rotate_vec_by_quat_xyzw(geom_pos_local, body_q)
            world_geom_pos = [
                body_pos[0] + world_geom_pos[0],
                body_pos[1] + world_geom_pos[1],
                body_pos[2] + world_geom_pos[2],
            ]
            world_geom_q = _quat_normalize_xyzw(_quat_mul_xyzw(body_q, geom_q))

            co = CollisionObject()
            co.header.frame_id = self.world_frame
            co.id = f"{self.object_prefix}_{body_name}"
            co.operation = CollisionObject.ADD
            co.pose.position.x = world_geom_pos[0]
            co.pose.position.y = world_geom_pos[1]
            co.pose.position.z = world_geom_pos[2]
            co.pose.orientation.x = world_geom_q[0]
            co.pose.orientation.y = world_geom_q[1]
            co.pose.orientation.z = world_geom_q[2]
            co.pose.orientation.w = world_geom_q[3]

            geom_type = geom.get("type", "mesh")
            marker = self._make_marker(
                obj_idx=obj_idx,
                body_name=body_name,
                geom=geom,
                geom_type=geom_type,
                pose=co.pose,
            )
            if geom_type == "mesh":
                mesh_name = geom.get("mesh")
                if not mesh_name or mesh_name not in self._mesh_defs:
                    rospy.logwarn(
                        "[static_scene_to_collision_objects] mesh def missing for body=%s mesh=%s",
                        body_name,
                        str(mesh_name),
                    )
                    obj_idx += 1
                    continue
                mesh_msg = self._mesh_from_name(mesh_name)
                if mesh_msg is None:
                    obj_idx += 1
                    continue
                identity = Pose()
                identity.orientation.w = 1.0
                if self.mesh_as_bounding_box:
                    bbox = _mesh_aabb(mesh_msg)
                    if bbox is None:
                        obj_idx += 1
                        continue
                    co.primitives.append(_make_box_primitive(bbox))
                    co.primitive_poses.append(identity)
                else:
                    co.meshes.append(mesh_msg)
                    co.mesh_poses.append(identity)
            elif geom_type == "box":
                size = _parse_floats(geom.get("size"), 3, [0.025, 0.025, 0.025])
                prim = SolidPrimitive()
                prim.type = SolidPrimitive.BOX
                prim.dimensions = [2.0 * size[0], 2.0 * size[1], 2.0 * size[2]]
                identity = Pose()
                identity.orientation.w = 1.0
                co.primitives.append(prim)
                co.primitive_poses.append(identity)
            elif geom_type == "cylinder":
                size = _parse_floats(geom.get("size"), 2, [0.03, 0.05])
                prim = SolidPrimitive()
                prim.type = SolidPrimitive.CYLINDER
                prim.dimensions = [2.0 * size[1], size[0]]  # height, radius
                identity = Pose()
                identity.orientation.w = 1.0
                co.primitives.append(prim)
                co.primitive_poses.append(identity)
            elif geom_type == "sphere":
                size = _parse_floats(geom.get("size"), 1, [0.03])
                prim = SolidPrimitive()
                prim.type = SolidPrimitive.SPHERE
                prim.dimensions = [size[0]]
                identity = Pose()
                identity.orientation.w = 1.0
                co.primitives.append(prim)
                co.primitive_poses.append(identity)
            else:
                rospy.logwarn(
                    "[static_scene_to_collision_objects] unsupported geom type '%s' for %s",
                    geom_type,
                    body_name,
                )
                obj_idx += 1
                continue

            self._objects.append(co)
            if marker is not None:
                self._markers.markers.append(marker)
            obj_idx += 1

        rospy.loginfo(
            "[static_scene_to_collision_objects] parsed %d static objects from %s",
            len(self._objects),
            scene_path,
        )
        rospy.loginfo(
            "[static_scene_to_collision_objects] object ids: %s",
            ", ".join(co.id for co in self._objects),
        )

    def _make_marker(self, obj_idx, body_name, geom, geom_type, pose):
        if not self.publish_visual_markers:
            return None

        marker = Marker()
        marker.header.frame_id = self.world_frame
        marker.ns = "static_scene_visual"
        marker.id = obj_idx
        marker.action = Marker.ADD
        marker.pose = pose
        marker.color.r = 0.8
        marker.color.g = 0.8
        marker.color.b = 0.8
        marker.color.a = 1.0
        marker.lifetime = rospy.Duration(0)

        if geom_type == "mesh":
            mesh_name = geom.get("mesh")
            md = self._mesh_defs.get(mesh_name)
            if md is None:
                return None
            mesh_msg = self._mesh_from_name(mesh_name)
            if mesh_msg is None:
                return None
            marker.type = Marker.TRIANGLE_LIST
            marker.scale.x = 1.0
            marker.scale.y = 1.0
            marker.scale.z = 1.0
            marker.points = _points_from_mesh_triangles(mesh_msg)
            marker.color.r = 0.7
            marker.color.g = 0.7
            marker.color.b = 0.75
            marker.color.a = 1.0
            return marker

        if geom_type == "box":
            size = _parse_floats(geom.get("size"), 3, [0.025, 0.025, 0.025])
            marker.type = Marker.CUBE
            marker.scale.x = 2.0 * size[0]
            marker.scale.y = 2.0 * size[1]
            marker.scale.z = 2.0 * size[2]
            return marker

        if geom_type == "cylinder":
            size = _parse_floats(geom.get("size"), 2, [0.03, 0.05])
            marker.type = Marker.CYLINDER
            marker.scale.x = 2.0 * size[0]
            marker.scale.y = 2.0 * size[0]
            marker.scale.z = 2.0 * size[1]
            return marker

        if geom_type == "sphere":
            size = _parse_floats(geom.get("size"), 1, [0.03])
            marker.type = Marker.SPHERE
            marker.scale.x = 2.0 * size[0]
            marker.scale.y = 2.0 * size[0]
            marker.scale.z = 2.0 * size[0]
            return marker

        rospy.logwarn(
            "[static_scene_to_collision_objects] no visual marker support for geom type '%s' (%s)",
            geom_type,
            body_name,
        )
        return None

    def _publish_visual_markers(self):
        if not self.publish_visual_markers or self.marker_pub is None:
            return

        msg = MarkerArray()
        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        msg.markers.append(delete_all)
        msg.markers.extend(self._markers.markers)
        self.marker_pub.publish(msg)
        rospy.loginfo(
            "[static_scene_to_collision_objects] published %d visual markers to %s",
            len(self._markers.markers),
            self.visual_marker_topic,
        )

    def _keep_visual_markers_alive(self):
        if not self.publish_visual_markers or not self.keep_visual_markers_alive:
            return

        interval = max(0.1, self.visual_marker_republish_interval)
        rospy.loginfo(
            "[static_scene_to_collision_objects] keeping visual markers alive on %s every %.1fs",
            self.visual_marker_topic,
            interval,
        )
        while not rospy.is_shutdown():
            rospy.sleep(interval)
            self._publish_visual_markers()

    def _load_xml_with_includes(self, root_xml):
        out = []
        visited = set()

        def _rec(path):
            apath = os.path.abspath(path)
            if apath in visited:
                return
            visited.add(apath)
            root = ET.parse(apath).getroot()
            root_dir = os.path.dirname(apath)
            out.append((root, root_dir))
            for inc in root.findall("./include"):
                inc_file = inc.get("file")
                if not inc_file:
                    continue
                _rec(os.path.normpath(os.path.join(root_dir, inc_file)))

        _rec(root_xml)
        return out

    def _mesh_from_name(self, mesh_name):
        if mesh_name in self._mesh_cache:
            return self._mesh_cache[mesh_name]
        md = self._mesh_defs[mesh_name]
        mesh_file = md["file"]
        if not os.path.isfile(mesh_file):
            rospy.logwarn(
                "[static_scene_to_collision_objects] mesh file not found: %s", mesh_file
            )
            return None
        try:
            mesh = _load_obj_mesh(mesh_file, md["scale"])
            self._mesh_cache[mesh_name] = mesh
            return mesh
        except Exception as e:
            rospy.logwarn(
                "[static_scene_to_collision_objects] failed loading mesh %s: %s",
                mesh_file,
                str(e),
            )
            return None

    def publish_once(self):
        if self.startup_delay > 0:
            rospy.sleep(self.startup_delay)

        self._publish_visual_markers()

        if self.use_apply_planning_scene:
            try:
                rospy.loginfo(
                    "[static_scene_to_collision_objects] waiting for services: %s, %s",
                    self.apply_planning_scene_service,
                    self.get_planning_scene_service,
                )
                rospy.wait_for_service(self.apply_planning_scene_service)
                rospy.wait_for_service(self.get_planning_scene_service)
                apply_srv = rospy.ServiceProxy(
                    self.apply_planning_scene_service, ApplyPlanningScene
                )
                get_srv = rospy.ServiceProxy(self.get_planning_scene_service, GetPlanningScene)

                if self.clear_world_before_apply:
                    existing_ids = self._get_world_object_ids(get_srv)
                    if existing_ids:
                        clear_req = ApplyPlanningSceneRequest()
                        clear_req.scene = PlanningScene()
                        clear_req.scene.is_diff = True
                        for oid in existing_ids:
                            co = CollisionObject()
                            co.header.frame_id = self.world_frame
                            co.id = oid
                            co.operation = CollisionObject.REMOVE
                            clear_req.scene.world.collision_objects.append(co)
                        clear_resp = apply_srv(clear_req)
                        rospy.loginfo(
                            "[static_scene_to_collision_objects] cleared %d existing world objects (success=%s)",
                            len(existing_ids),
                            str(clear_resp.success),
                        )
                        rospy.sleep(0.2)

                for attempt in range(max(1, self.apply_retries)):
                    req = ApplyPlanningSceneRequest()
                    req.scene = PlanningScene()
                    req.scene.is_diff = True
                    req.scene.world.collision_objects = self._objects
                    resp = apply_srv(req)
                    if not resp.success:
                        rospy.logwarn(
                            "[static_scene_to_collision_objects] apply attempt %d failed",
                            attempt + 1,
                        )
                        rospy.sleep(self.apply_retry_interval)
                        continue

                    world_ids = self._get_world_object_ids(get_srv)
                    wanted = set(co.id for co in self._objects)
                    have = wanted.intersection(set(world_ids))
                    if len(have) == len(wanted):
                        rospy.loginfo(
                            "[static_scene_to_collision_objects] applied %d objects via %s (verified)",
                            len(self._objects),
                            self.apply_planning_scene_service,
                        )
                        self._monitor_world_objects(get_srv, wanted)
                        return

                    rospy.logwarn(
                        "[static_scene_to_collision_objects] apply attempt %d partial (%d/%d present), retrying",
                        attempt + 1,
                        len(have),
                        len(wanted),
                    )
                    rospy.sleep(self.apply_retry_interval)

                if self.enable_topic_publish:
                    rospy.logwarn(
                        "[static_scene_to_collision_objects] apply_planning_scene not fully verified; fallback to topic publish"
                    )
                else:
                    rospy.logwarn(
                        "[static_scene_to_collision_objects] apply_planning_scene not fully verified; topic fallback disabled"
                    )
            except Exception as e:
                if self.enable_topic_publish:
                    rospy.logwarn(
                        "[static_scene_to_collision_objects] apply_planning_scene failed (%s); fallback to topic publish",
                        str(e),
                    )
                else:
                    rospy.logwarn(
                        "[static_scene_to_collision_objects] apply_planning_scene failed (%s); topic fallback disabled",
                        str(e),
                    )

        if not self.enable_topic_publish:
            self._keep_visual_markers_alive()
            return

        for _ in range(max(1, self.publish_repeats)):
            if rospy.is_shutdown():
                return
            stamp = rospy.Time.now()
            for co in self._objects:
                co.header.stamp = stamp
                self.pub.publish(co)
            rospy.sleep(max(0.0, self.repeat_interval))

        self._keep_visual_markers_alive()

    def _get_world_object_ids(self, get_srv):
        # Newer/older MoveIt message variants differ here:
        # - some expose world.collision_object_ids
        # - others expose only world.collision_objects[*].id
        req = GetPlanningSceneRequest()
        req.components = PlanningSceneComponents()
        req.components.components = PlanningSceneComponents.WORLD_OBJECT_NAMES
        resp = get_srv(req)

        world = resp.scene.world
        if hasattr(world, "collision_object_ids"):
            return list(world.collision_object_ids)

        ids = [co.id for co in getattr(world, "collision_objects", []) if getattr(co, "id", "")]
        if ids:
            return ids

        # Fallback: request geometry and read ids from returned objects.
        req2 = GetPlanningSceneRequest()
        req2.components = PlanningSceneComponents()
        req2.components.components = PlanningSceneComponents.WORLD_OBJECT_GEOMETRY
        resp2 = get_srv(req2)
        return [co.id for co in resp2.scene.world.collision_objects if getattr(co, "id", "")]

    def _monitor_world_objects(self, get_srv, wanted_ids):
        duration = max(0.0, self.post_apply_monitor_duration)
        interval = max(0.1, self.post_apply_monitor_interval)
        if duration <= 0.0:
            return

        deadline = rospy.Time.now() + rospy.Duration.from_sec(duration)
        while not rospy.is_shutdown() and rospy.Time.now() < deadline:
            rospy.sleep(interval)
            world_ids = set(self._get_world_object_ids(get_srv))
            missing = sorted(wanted_ids.difference(world_ids))
            if missing:
                rospy.logwarn(
                    "[static_scene_to_collision_objects] world objects disappeared after apply; missing=%s present=%s",
                    ",".join(missing),
                    ",".join(sorted(world_ids)),
                )
                return

        rospy.loginfo(
            "[static_scene_to_collision_objects] world objects still present after %.1fs monitor window",
            duration,
        )


if __name__ == "__main__":
    rospy.init_node("static_scene_to_collision_objects")
    node = StaticSceneToCollisionObjects()
    node.publish_once()
    rospy.loginfo("[static_scene_to_collision_objects] done.")
