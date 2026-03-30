#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auto ring scan at current EE height + posegraph fusion (Open3D) + auto-drop worst frames.

ROS1 (rospy). Publishes RelaxedIK EEVelGoals to move EE in XY ring at fixed Z.
Captures PointCloud2, transforms to base_frame, optional crop, saves scan_*.ply.
Then fuses scans into merged_posegraph_before_drop.ply and merged_posegraph_after_drop.ply.

Author: merged from user's ring scanner + posegraph fusion script
"""

import os, math, time, glob, copy
import numpy as np
import rospy

import tf2_ros
from geometry_msgs.msg import Twist, PointStamped
from sensor_msgs.msg import JointState, PointCloud2
import sensor_msgs.point_cloud2 as pc2
import tf2_geometry_msgs

try:
    from relaxed_ik_ros1.msg import EEVelGoals
    HAS_RELAXED_IK = True
except ImportError:
    HAS_RELAXED_IK = False
    EEVelGoals = None

import open3d as o3d
from collections import defaultdict, deque

# Optional fast transform for PointCloud2
try:
    import tf2_sensor_msgs.tf2_sensor_msgs as tf2sm
    HAS_TF2_SENSOR_MSGS = True
except Exception:
    HAS_TF2_SENSOR_MSGS = False

# -------------------------
# helpers
# -------------------------
def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def normalize_output_dir(path):
    path = os.path.expanduser(str(path).strip())
    legacy_home = "/home/heyang"
    current_home = os.path.expanduser("~")
    if path == legacy_home or path.startswith(legacy_home + os.sep):
        suffix = path[len(legacy_home):].lstrip(os.sep)
        path = os.path.join(current_home, suffix) if suffix else current_home
    return path


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

# =========================
# PoseGraph fusion utilities
# =========================
def remove_plane(pcd, dist):
    if len(pcd.points) < 2000:
        return pcd
    _, inliers = pcd.segment_plane(distance_threshold=dist, ransac_n=3, num_iterations=1000)
    return pcd.select_by_index(inliers, invert=True)

def keep_largest_cluster(pcd, eps, min_points):
    if len(pcd.points) < max(min_points, 200):
        return pcd
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    if labels.size == 0 or labels.max() < 0:
        return pcd
    counts = np.bincount(labels[labels >= 0])
    k = int(counts.argmax())
    idx = np.where(labels == k)[0]
    return pcd.select_by_index(idx)

def estimate_normals(pcd, voxel):
    if len(pcd.points) < 10:
        return pcd
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 3.0, max_nn=30)
    )
    return pcd

def preprocess_for_registration(pcd, voxel, do_remove_plane=True, do_largest_cluster=True):
    p = pcd.voxel_down_sample(voxel)
    if do_remove_plane:
        p = remove_plane(p, dist=voxel * 2.0)
    if do_largest_cluster:
        p = keep_largest_cluster(p, eps=voxel * 6.0, min_points=200)
    estimate_normals(p, voxel)
    return p

def compute_fpfh(pcd, voxel):
    radius_feature = voxel * 5.0
    return o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )

def global_ransac_init(source, target, voxel=0.02):
    s = source.voxel_down_sample(voxel)
    t = target.voxel_down_sample(voxel)
    if len(s.points) < 20 or len(t.points) < 20:
        raise RuntimeError("too few points for global RANSAC init")
    estimate_normals(s, voxel)
    estimate_normals(t, voxel)
    if not s.has_normals() or not t.has_normals():
        raise RuntimeError("normals unavailable for global RANSAC init")
    f_s = compute_fpfh(s, voxel)
    f_t = compute_fpfh(t, voxel)

    dist = voxel * 6.0
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        s, t, f_s, f_t,
        mutual_filter=True,
        max_correspondence_distance=dist,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(dist),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(50000, 0.999)
    )
    return result.transformation

def pairwise_icp_multiscale(source, target, init=np.eye(4)):
    scales = [
        (0.02, 0.15, 40),
        (0.01, 0.06, 50),
        (0.005, 0.02, 70),
    ]
    T = init.copy()
    last_reg = None

    for voxel, max_corr, it in scales:
        s = source.voxel_down_sample(voxel)
        t = target.voxel_down_sample(voxel)
        if len(s.points) < 10 or len(t.points) < 10:
            continue
        estimate_normals(s, voxel)
        estimate_normals(t, voxel)
        if not s.has_normals() or not t.has_normals():
            continue

        reg = o3d.pipelines.registration.registration_icp(
            s, t, max_corr, T,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=it, relative_fitness=1e-6, relative_rmse=1e-6
            )
        )
        T = reg.transformation
        last_reg = reg

    if last_reg is None:
        raise RuntimeError("too few points after downsampling for ICP")

    info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target,
        max_correspondence_distance=scales[-1][1],
        transformation=T
    )
    return T, info, last_reg

def _reg_is_better(candidate, incumbent):
    if incumbent is None:
        return True
    cand_reg = candidate[2]
    inc_reg = incumbent[2]
    if cand_reg.fitness != inc_reg.fitness:
        return cand_reg.fitness > inc_reg.fitness
    return cand_reg.inlier_rmse < inc_reg.inlier_rmse

def robust_register(src, tgt, fitness_gate, rmse_gate):
    best = None

    try:
        direct = (*pairwise_icp_multiscale(src, tgt, init=np.eye(4)), "icp")
        if _reg_is_better(direct, best):
            best = direct
        if (direct[2].fitness >= fitness_gate) and (direct[2].inlier_rmse <= rmse_gate):
            return direct
    except Exception:
        direct = None

    ransac_errors = []
    for voxel in (0.03, 0.02):
        try:
            T0 = global_ransac_init(src, tgt, voxel=voxel)
            candidate = (*pairwise_icp_multiscale(src, tgt, init=T0), f"ransac{voxel:.3f}+icp")
            if _reg_is_better(candidate, best):
                best = candidate
            if (candidate[2].fitness >= fitness_gate) and (candidate[2].inlier_rmse <= rmse_gate):
                return candidate
        except Exception as exc:
            ransac_errors.append(str(exc))

    if best is not None:
        if best[3].startswith("ransac"):
            return best
        if ransac_errors:
            return best[0], best[1], best[2], f"bad({'; '.join(ransac_errors[:2])})"
        return best[0], best[1], best[2], "bad"

    raise RuntimeError("Registration failed: %s" % ("; ".join(ransac_errors[:2]) if ransac_errors else "unknown error"))

def build_pose_graph(
    pcds_reg,
    verbose=True,
    fitness_gate=0.30,
    rmse_gate=0.015,
    loop_k=6,
    loop_fitness_gate=0.40,
    loop_rmse_gate=0.012
):
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odom = np.eye(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(odom)))

    for i in range(1, len(pcds_reg)):
        src = pcds_reg[i - 1]
        tgt = pcds_reg[i]

        T, info, reg, how = robust_register(src, tgt, fitness_gate, rmse_gate)
        if verbose:
            print(f"[edge {i-1}->{i}] {how:10s} fitness={reg.fitness:.3f} rmse={reg.inlier_rmse:.4f}")

        if (reg.fitness < fitness_gate) or (reg.inlier_rmse > rmse_gate):
            if verbose:
                print("  -> reject odom edge (graph may disconnect)")
            pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(odom)))
        else:
            odom = odom @ T
            pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(odom)))
            pose_graph.edges.append(
                o3d.pipelines.registration.PoseGraphEdge(i - 1, i, T, info, uncertain=False)
            )

        # loop closure
        if loop_k is not None and i - loop_k >= 0:
            a = i - loop_k
            src_lc = pcds_reg[a]
            tgt_lc = pcds_reg[i]

            T_lc, info_lc, reg_lc, how_lc = robust_register(src_lc, tgt_lc, loop_fitness_gate, loop_rmse_gate)
            if verbose:
                print(f"[loop {a}->{i}] {how_lc:10s} fitness={reg_lc.fitness:.3f} rmse={reg_lc.inlier_rmse:.4f}")

            if (reg_lc.fitness >= loop_fitness_gate) and (reg_lc.inlier_rmse <= loop_rmse_gate):
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(a, i, T_lc, info_lc, uncertain=True)
                )
            else:
                if verbose:
                    print("  -> reject loop edge")

    return pose_graph

def optimize_pose_graph(pose_graph, max_corr=0.02):
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=max_corr,
        edge_prune_threshold=0.25,
        reference_node=0
    )
    o3d.pipelines.registration.global_optimization(
        pose_graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        option
    )

def largest_connected_component(num_nodes, edges):
    adj = defaultdict(list)
    for e in edges:
        adj[e.source_node_id].append(e.target_node_id)
        adj[e.target_node_id].append(e.source_node_id)

    visited = [False] * num_nodes
    best = []

    for s in range(num_nodes):
        if visited[s]:
            continue
        q = deque([s])
        visited[s] = True
        comp = [s]
        while q:
            u = q.popleft()
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    q.append(v)
                    comp.append(v)
        if len(comp) > len(best):
            best = comp

    return sorted(best)

def merge_with_poses(
    source_pcds,
    pose_graph,
    keep_nodes,
    voxel_final,
    out_path,
    final_remove_outlier=True,
    final_outlier_nb_neighbors=30,
    final_outlier_std_ratio=2.5,
):
    merged = o3d.geometry.PointCloud()
    for i in keep_nodes:
        p2 = copy.deepcopy(source_pcds[i])
        p2.transform(pose_graph.nodes[i].pose)
        merged += p2

    if voxel_final > 0.0:
        merged = merged.voxel_down_sample(voxel_final)

    if final_remove_outlier and len(merged.points) > final_outlier_nb_neighbors:
        _, ind = merged.remove_statistical_outlier(
            nb_neighbors=final_outlier_nb_neighbors,
            std_ratio=final_outlier_std_ratio,
        )
        merged = merged.select_by_index(ind)

    o3d.io.write_point_cloud(out_path, merged, write_ascii=False, compressed=False)
    print(f"Saved: {out_path}  points={len(merged.points)}  used_frames={len(keep_nodes)}/{len(source_pcds)}")
    return merged

def score_frame_against_model(frame_pcd, model_pcd, voxel=0.005):
    a = frame_pcd.voxel_down_sample(voxel)
    b = model_pcd.voxel_down_sample(voxel)
    estimate_normals(a, voxel)
    estimate_normals(b, voxel)
    reg = o3d.pipelines.registration.registration_icp(
        a, b, voxel * 6.0, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=40)
    )
    return reg.fitness, reg.inlier_rmse

def pick_bad_frames(raw, pose_graph, keep_nodes, voxel=0.005, fitness_th=0.35, rmse_th=0.012):
    model = o3d.geometry.PointCloud()
    for i in keep_nodes:
        p = copy.deepcopy(raw[i])
        p.transform(pose_graph.nodes[i].pose)
        model += p
    model = model.voxel_down_sample(voxel)

    bad = []
    for i in keep_nodes:
        p = copy.deepcopy(raw[i])
        p.transform(pose_graph.nodes[i].pose)
        fit, rmse = score_frame_against_model(p, model, voxel=voxel)
        if (fit < fitness_th) or (rmse > rmse_th):
            bad.append((i, float(fit), float(rmse)))
    bad.sort(key=lambda x: (x[1], -x[2]))
    return bad


def refine_in_base_frame(
    raw_pcds,
    voxel_reg=0.005,
    voxel_final=0.003,
    fitness_gate=0.20,
    rmse_gate=0.02,
    final_remove_outlier=True,
    final_outlier_nb_neighbors=30,
    final_outlier_std_ratio=2.5,
    out_path=None,
    verbose=True,
):
    if not raw_pcds:
        raise RuntimeError("No point clouds provided for base-frame refinement.")

    accepted_indices = [0]
    accepted_transforms = [np.eye(4)]
    merged = copy.deepcopy(raw_pcds[0])
    merged_reg = preprocess_for_registration(
        merged,
        voxel_reg,
        do_remove_plane=False,
        do_largest_cluster=False,
    )

    if verbose:
        print(f"[base_icp] seed frame=0 points={len(raw_pcds[0].points)}")

    for i in range(1, len(raw_pcds)):
        src_raw = raw_pcds[i]
        src_reg = preprocess_for_registration(
            src_raw,
            voxel_reg,
            do_remove_plane=False,
            do_largest_cluster=False,
        )
        tgt_reg = merged_reg

        if len(src_reg.points) < 10 or len(tgt_reg.points) < 10:
            if verbose:
                print(f"[base_icp] frame={i} skipped: too few points after preprocessing")
            continue

        try:
            T, _, reg = pairwise_icp_multiscale(src_reg, tgt_reg, init=np.eye(4))
        except Exception as exc:
            if verbose:
                print(f"[base_icp] frame={i} failed: {exc}")
            continue

        accept = (reg.fitness >= fitness_gate) and (reg.inlier_rmse <= rmse_gate)
        if verbose:
            status = "accept" if accept else "reject"
            print(
                f"[base_icp] frame={i} {status} fitness={reg.fitness:.3f} "
                f"rmse={reg.inlier_rmse:.4f} points={len(src_raw.points)}"
            )

        if not accept:
            continue

        src_aligned = copy.deepcopy(src_raw)
        src_aligned.transform(T)
        merged += src_aligned
        if voxel_final > 0.0:
            merged = merged.voxel_down_sample(voxel_final)
        merged_reg = preprocess_for_registration(
            merged,
            voxel_reg,
            do_remove_plane=False,
            do_largest_cluster=False,
        )
        accepted_indices.append(i)
        accepted_transforms.append(T)

    if voxel_final > 0.0:
        merged = merged.voxel_down_sample(voxel_final)

    if final_remove_outlier and len(merged.points) > final_outlier_nb_neighbors:
        _, ind = merged.remove_statistical_outlier(
            nb_neighbors=final_outlier_nb_neighbors,
            std_ratio=final_outlier_std_ratio,
        )
        merged = merged.select_by_index(ind)

    if out_path is not None:
        o3d.io.write_point_cloud(out_path, merged, write_ascii=False, compressed=False)
        print(
            f"Saved: {out_path}  points={len(merged.points)}  "
            f"used_frames={len(accepted_indices)}/{len(raw_pcds)}"
        )

    return merged, accepted_indices, accepted_transforms

def fuse_scans(
    ply_dir,
    pattern="scan_*.ply",
    out_before="merged_posegraph_before_drop.ply",
    out_after="merged_posegraph_after_drop.ply",
    fusion_strategy="posegraph",
    voxel_reg=0.005,
    voxel_final=0.003,
    fitness_gate=0.30,
    rmse_gate=0.015,
    loop_k=6,
    loop_fitness_gate=0.40,
    loop_rmse_gate=0.012,
    opt_max_corr=0.02,
    drop_k=1,
    drop_fitness_th=0.35,
    drop_rmse_th=0.012,
    reg_remove_plane=True,
    reg_keep_largest_cluster=True,
    final_remove_outlier=True,
    final_outlier_nb_neighbors=30,
    final_outlier_std_ratio=2.5,
    raw_crop_box=None,
    final_use_registration_clouds=False,
    verbose=True
):
    files = sorted(glob.glob(os.path.join(ply_dir, pattern)))
    if not files:
        raise RuntimeError(f"No '{pattern}' in {ply_dir}")

    print(f"[fuse] Found {len(files)} frames in {ply_dir}")
    raw = [o3d.io.read_point_cloud(f) for f in files]

    if raw_crop_box is not None and len(raw_crop_box) == 6:
        xmin, xmax, ymin, ymax, zmin, zmax = raw_crop_box
        aabb = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=(xmin, ymin, zmin),
            max_bound=(xmax, ymax, zmax),
        )
        raw = [p.crop(aabb) for p in raw]
        print(f"[fuse] Applied raw crop box: {raw_crop_box}")

    pcds_reg = [
        preprocess_for_registration(
            p,
            voxel_reg,
            do_remove_plane=reg_remove_plane,
            do_largest_cluster=reg_keep_largest_cluster,
        )
        for p in raw
    ]
    valid_pairs = [(r, p) for r, p in zip(raw, pcds_reg) if len(p.points) >= 10]
    if len(valid_pairs) < 2:
        raise RuntimeError("Need at least 2 usable scans after preprocessing for fusion.")
    if len(valid_pairs) < len(raw):
        print(f"[fuse] Dropping {len(raw) - len(valid_pairs)} scans that became too small after preprocessing")
    raw = [pair[0] for pair in valid_pairs]
    pcds_reg = [pair[1] for pair in valid_pairs]
    merge_source = pcds_reg if final_use_registration_clouds else raw

    out_before_path = os.path.join(ply_dir, out_before) if not os.path.isabs(out_before) else out_before
    out_after_path  = os.path.join(ply_dir, out_after)  if not os.path.isabs(out_after)  else out_after

    if fusion_strategy == "base_icp":
        print("[fuse] Using base_icp strategy")
        _, accepted_indices, accepted_transforms = refine_in_base_frame(
            raw,
            voxel_reg=voxel_reg,
            voxel_final=voxel_final,
            fitness_gate=fitness_gate,
            rmse_gate=rmse_gate,
            final_remove_outlier=final_remove_outlier,
            final_outlier_nb_neighbors=final_outlier_nb_neighbors,
            final_outlier_std_ratio=final_outlier_std_ratio,
            out_path=out_before_path,
            verbose=verbose,
        )
        if not accepted_indices:
            raise RuntimeError("base_icp failed to keep any frames.")

        before_model = o3d.io.read_point_cloud(out_before_path)
        bad = []
        for idx, T in zip(accepted_indices, accepted_transforms):
            frame = copy.deepcopy(raw[idx])
            frame.transform(T)
            fit, rmse = score_frame_against_model(frame, before_model, voxel=voxel_reg)
            if (fit < drop_fitness_th) or (rmse > drop_rmse_th):
                bad.append((idx, float(fit), float(rmse)))
        bad.sort(key=lambda x: (x[1], -x[2]))
        print("[fuse] Bad frames (worst->best):")
        for x in bad[:8]:
            print("  frame=%d  fitness=%.3f  rmse=%.4f" % x)

        drop = set(idx for idx, _, _ in bad[:max(0, drop_k)])
        keep2 = [idx for idx in accepted_indices if idx not in drop]
        print(f"[fuse] Dropping frames: {sorted(list(drop))}  keep={len(keep2)}/{len(accepted_indices)}")

        merged_after = o3d.geometry.PointCloud()
        for idx, T in zip(accepted_indices, accepted_transforms):
            if idx not in keep2:
                continue
            p = copy.deepcopy(raw[idx])
            p.transform(T)
            merged_after += p
        if voxel_final > 0.0:
            merged_after = merged_after.voxel_down_sample(voxel_final)
        if final_remove_outlier and len(merged_after.points) > final_outlier_nb_neighbors:
            _, ind = merged_after.remove_statistical_outlier(
                nb_neighbors=final_outlier_nb_neighbors,
                std_ratio=final_outlier_std_ratio,
            )
            merged_after = merged_after.select_by_index(ind)
        o3d.io.write_point_cloud(out_after_path, merged_after, write_ascii=False, compressed=False)
        print(
            f"Saved: {out_after_path}  points={len(merged_after.points)}  "
            f"used_frames={len(keep2)}/{len(raw)}"
        )
        return out_before_path, out_after_path

    loop_k = None if loop_k is not None and loop_k <= 0 else loop_k

    pose_graph = build_pose_graph(
        pcds_reg,
        verbose=verbose,
        fitness_gate=fitness_gate,
        rmse_gate=rmse_gate,
        loop_k=loop_k,
        loop_fitness_gate=loop_fitness_gate,
        loop_rmse_gate=loop_rmse_gate
    )

    keep = largest_connected_component(len(pose_graph.nodes), pose_graph.edges)
    if len(keep) < len(pose_graph.nodes):
        print(f"[WARN] PoseGraph disconnected. Keeping largest component: {len(keep)}/{len(pose_graph.nodes)} nodes")

    optimize_pose_graph(pose_graph, max_corr=opt_max_corr)

    merge_with_poses(
        merge_source,
        pose_graph,
        keep_nodes=keep,
        voxel_final=voxel_final,
        out_path=out_before_path,
        final_remove_outlier=final_remove_outlier,
        final_outlier_nb_neighbors=final_outlier_nb_neighbors,
        final_outlier_std_ratio=final_outlier_std_ratio,
    )

    bad = []
    keep2 = list(keep)
    drop = set()
    if drop_k > 0:
        bad = pick_bad_frames(raw, pose_graph, keep, voxel=voxel_reg, fitness_th=drop_fitness_th, rmse_th=drop_rmse_th)
        print("[fuse] Bad frames (worst->best):")
        for x in bad[:8]:
            print("  frame=%d  fitness=%.3f  rmse=%.4f" % x)

        for k in range(min(drop_k, len(bad))):
            drop.add(bad[k][0])

        keep2 = [i for i in keep if i not in drop]
        print(f"[fuse] Dropping frames: {sorted(list(drop))}  keep={len(keep2)}/{len(keep)}")
    else:
        print("[fuse] drop_k <= 0, keeping all frames in the optimized component")

    merge_with_poses(
        merge_source,
        pose_graph,
        keep_nodes=keep2,
        voxel_final=voxel_final,
        out_path=out_after_path,
        final_remove_outlier=final_remove_outlier,
        final_outlier_nb_neighbors=final_outlier_nb_neighbors,
        final_outlier_std_ratio=final_outlier_std_ratio,
    )

    return out_before_path, out_after_path

# =========================
# Ring scanner + capture
# =========================
class AutoRingScanAndFuse:
    def __init__(self):
        # IO/topics
        self.out_dir     = normalize_output_dir(rospy.get_param("~out_dir", "~/scans_run"))
        self.cloud_topic = rospy.get_param("~cloud_topic", "/realsense/depth/points")
        self.js_topic    = rospy.get_param("~js_topic", "/relaxed_ik/joint_angle_solutions")
        self.vel_topic   = rospy.get_param("~vel_topic", "/relaxed_ik/ee_vel_goals")

        # Frames (IMPORTANT: base_link/right_hand is what you publish)
        self.base_frame  = rospy.get_param("~base_frame", "base_link")
        self.tip_frame   = rospy.get_param("~tip_frame", "right_hand")

        # ring params (scan at current EE height + z_offset)
        self.radius  = float(rospy.get_param("~radius", 0.08))
        self.n_poses = int(rospy.get_param("~n_poses", 16))
        self.z_offset = float(rospy.get_param("~z_offset", 0.0))

        # controller
        self.rate_hz = float(rospy.get_param("~rate", 60.0))
        self.kp_pos  = float(rospy.get_param("~kp_pos", 1.5))
        self.vmax    = float(rospy.get_param("~vmax", 0.03))
        self.pos_tol = float(rospy.get_param("~pos_tol", 0.006))
        self.settle_time = float(rospy.get_param("~settle_time", 0.25))
        self.timeout = float(rospy.get_param("~timeout", 10.0))

        # capture params
        self.take_multiple_frames = int(rospy.get_param("~take_multiple_frames", 1))
        self.voxel = float(rospy.get_param("~voxel", 0.003))

        # crop_box: [xmin,xmax,ymin,ymax,zmin,zmax] in base_frame; empty => no crop
        self.crop_box = rospy.get_param("~crop_box", [])
        if isinstance(self.crop_box, str):
            # sometimes passed as "[]"
            self.crop_box = eval(self.crop_box)

        self.use_latest_tf_for_cloud = bool(rospy.get_param("~use_latest_tf_for_cloud", True))

        ensure_dir(self.out_dir)
        self.pub_vel = rospy.Publisher(self.vel_topic, EEVelGoals, queue_size=1)

        self.last_cloud = None
        self.last_js = None
        rospy.Subscriber(self.cloud_topic, PointCloud2, self._cloud_cb, queue_size=1)
        rospy.Subscriber(self.js_topic, JointState, self._js_cb, queue_size=1)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        rospy.loginfo("[ring_scan] waiting for first cloud and jointstate...")
        while not rospy.is_shutdown() and (self.last_cloud is None or self.last_js is None):
            rospy.sleep(0.05)

        # sanity TF
        self._wait_tf(self.base_frame, self.tip_frame)

        # center at current EE
        self.center_xyz = self._get_tip_xyz()
        self.center_xyz[2] += self.z_offset
        rospy.loginfo("[ring_scan] base_frame=%s tip_frame=%s center_xyz=%s",
                      self.base_frame, self.tip_frame, np.round(self.center_xyz, 3))

        if not HAS_TF2_SENSOR_MSGS:
            rospy.logwarn("[ring_scan] tf2_sensor_msgs not found -> cloud transform may be slow (point-by-point).")

    def _cloud_cb(self, msg): self.last_cloud = msg
    def _js_cb(self, msg): self.last_js = msg

    def _wait_tf(self, target, source, timeout=3.0):
        t0 = time.time()
        while not rospy.is_shutdown() and time.time() - t0 < timeout:
            try:
                self.tf_buffer.lookup_transform(target, source, rospy.Time(0), rospy.Duration(0.2))
                return
            except Exception:
                rospy.sleep(0.05)
        raise RuntimeError(f"TF not available: {target} <- {source}")

    def _get_tip_xyz(self):
        tf = self.tf_buffer.lookup_transform(self.base_frame, self.tip_frame, rospy.Time(0), rospy.Duration(1.0))
        t = tf.transform.translation
        return np.array([t.x, t.y, t.z], dtype=np.float64)

    def _publish_twist(self, vx, vy, vz):
        msg = EEVelGoals()
        tw = Twist()
        tw.linear.x, tw.linear.y, tw.linear.z = float(vx), float(vy), float(vz)
        tw.angular.x, tw.angular.y, tw.angular.z = 0.0, 0.0, 0.0
        msg.ee_vels.append(tw)
        msg.tolerances.append(Twist())
        self.pub_vel.publish(msg)

    def _stop(self, duration=0.2):
        t_end = time.time() + duration
        r = rospy.Rate(self.rate_hz)
        while not rospy.is_shutdown() and time.time() < t_end:
            self._publish_twist(0,0,0)
            r.sleep()

    def _drive_to_xyz(self, target_xyz):
        t0 = time.time()
        r = rospy.Rate(self.rate_hz)
        reached_since = None

        while not rospy.is_shutdown() and (time.time() - t0) < self.timeout:
            cur = self._get_tip_xyz()
            err = target_xyz - cur

            vx = clamp(self.kp_pos * err[0], -self.vmax, self.vmax)
            vy = clamp(self.kp_pos * err[1], -self.vmax, self.vmax)
            vz = clamp(self.kp_pos * err[2], -self.vmax, self.vmax)

            self._publish_twist(vx, vy, vz)

            if np.linalg.norm(err) < self.pos_tol:
                if reached_since is None:
                    reached_since = time.time()
                if time.time() - reached_since >= self.settle_time:
                    self._stop(0.15)
                    return True
            else:
                reached_since = None
            r.sleep()

        self._stop(0.2)
        return False

    def _transform_cloud_to_base(self, cloud_msg):
        # transform PointCloud2 into base_frame
        try:
            if self.use_latest_tf_for_cloud:
                tf = self.tf_buffer.lookup_transform(self.base_frame, cloud_msg.header.frame_id,
                                                     rospy.Time(0), rospy.Duration(1.0))
            else:
                tf = self.tf_buffer.lookup_transform(self.base_frame, cloud_msg.header.frame_id,
                                                     cloud_msg.header.stamp, rospy.Duration(1.0))
        except Exception as e:
            rospy.logwarn("[ring_scan] TF lookup failed: %s", str(e))
            return None

        if HAS_TF2_SENSOR_MSGS:
            try:
                out = tf2sm.do_transform_cloud(cloud_msg, tf)
                return out
            except Exception as e:
                rospy.logwarn("[ring_scan] tf2_sensor_msgs transform failed, fallback point-by-point: %s", str(e))

        # fallback point-by-point (slow)
        pts = []
        for p in pc2.read_points(cloud_msg, skip_nans=True):
            pt = PointStamped()
            pt.header = cloud_msg.header
            pt.point.x, pt.point.y, pt.point.z = p[0], p[1], p[2]
            pw = tf2_geometry_msgs.do_transform_point(pt, tf)
            pts.append([pw.point.x, pw.point.y, pw.point.z])

        if len(pts) < 200:
            return None

        # build "fake" cloud-like list
        return np.asarray(pts, dtype=np.float64)

    def _cloud_to_o3d(self, cloud_msg_or_pts):
        # If we got a numpy Nx3 from fallback, use directly
        if isinstance(cloud_msg_or_pts, np.ndarray):
            pts = cloud_msg_or_pts
        else:
            pts = []
            for p in pc2.read_points(cloud_msg_or_pts, skip_nans=True, field_names=("x","y","z")):
                pts.append([p[0], p[1], p[2]])
            pts = np.asarray(pts, dtype=np.float64)

        if pts.shape[0] < 200:
            return None

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)

        # crop
        if isinstance(self.crop_box, (list, tuple)) and len(self.crop_box) == 6:
            xmin,xmax,ymin,ymax,zmin,zmax = map(float, self.crop_box)
            aabb = o3d.geometry.AxisAlignedBoundingBox(
                min_bound=(xmin, ymin, zmin),
                max_bound=(xmax, ymax, zmax)
            )
            pcd = pcd.crop(aabb)

        # downsample
        if self.voxel > 0:
            pcd = pcd.voxel_down_sample(self.voxel)

        if len(pcd.points) < 200:
            return None
        return pcd

    def _capture(self, idx, z_tag):
        cloud = self.last_cloud
        if cloud is None:
            rospy.logwarn("[ring_scan] No cloud")
            return False

        # transform to base frame first
        transformed = self._transform_cloud_to_base(cloud)
        if transformed is None:
            rospy.logwarn("[ring_scan] transform produced None/too few points")
            return False

        pcd = self._cloud_to_o3d(transformed)
        if pcd is None:
            rospy.logwarn("[ring_scan] After crop/downsample too few points: 0")
            return False

        out = os.path.join(self.out_dir, f"scan_{idx:04d}_z{z_tag}.ply")
        o3d.io.write_point_cloud(out, pcd)
        rospy.loginfo("[ring_scan] Saved %s (points=%d)", out, len(pcd.points))
        return True

    def run_scan(self):
        cx, cy, cz = self.center_xyz
        r = rospy.Rate(self.rate_hz)

        for i in range(self.n_poses):
            theta = 2.0 * math.pi * float(i) / float(self.n_poses)

            x = cx + self.radius * math.cos(theta)
            y = cy + self.radius * math.sin(theta)
            z = cz

            target = np.array([x,y,z], dtype=np.float64)
            rospy.loginfo("[ring_scan] target %d/%d xyz=%s",
                          i+1, self.n_poses, np.round(target, 3))

            ok = self._drive_to_xyz(target)
            if not ok:
                rospy.logwarn("[ring_scan] failed to reach target %d, capturing anyway", i)

            # capture multiple frames (optional)
            got = 0
            for k in range(max(1, self.take_multiple_frames)):
                if self._capture(i*self.take_multiple_frames + k, z_tag="+0"):
                    got += 1
                rospy.sleep(0.05)

            if got == 0:
                rospy.logwarn("[ring_scan] idx=%d captured 0 frames (all too few after crop)", i)

            r.sleep()

        self._stop(0.5)
        rospy.loginfo("[ring_scan] done. scans in %s", self.out_dir)

# =========================
# main
# =========================
def main():
    scanner = AutoRingScanAndFuse()
    scanner.run_scan()

    do_fuse = bool(rospy.get_param("~do_fuse", True))
    if not do_fuse:
        rospy.loginfo("[main] do_fuse=false, done.")
        return

    # Fusion params (ROS params)
    ply_dir = scanner.out_dir
    pattern = str(rospy.get_param("~fuse_pattern", "scan_*.ply"))
    out_before = str(rospy.get_param("~out_before", "merged_posegraph_before_drop.ply"))
    out_after  = str(rospy.get_param("~out_after",  "merged_posegraph_after_drop.ply"))
    fusion_strategy = str(rospy.get_param("~fusion_strategy", "posegraph"))

    voxel_reg   = float(rospy.get_param("~voxel_reg", 0.005))
    voxel_final = float(rospy.get_param("~voxel_final", 0.003))

    fitness_gate = float(rospy.get_param("~fitness_gate", 0.30))
    rmse_gate    = float(rospy.get_param("~rmse_gate", 0.015))

    loop_k = int(rospy.get_param("~loop_k", 6))
    loop_fitness_gate = float(rospy.get_param("~loop_fitness_gate", 0.40))
    loop_rmse_gate    = float(rospy.get_param("~loop_rmse_gate", 0.012))

    opt_max_corr = float(rospy.get_param("~opt_max_corr", 0.02))

    drop_k = int(rospy.get_param("~drop_k", 1))
    drop_fitness_th = float(rospy.get_param("~drop_fitness_th", 0.35))
    drop_rmse_th    = float(rospy.get_param("~drop_rmse_th", 0.012))

    verbose = bool(rospy.get_param("~fuse_verbose", True))

    rospy.loginfo("[fuse] start posegraph fusion...")
    try:
        b, a = fuse_scans(
            ply_dir=ply_dir,
            pattern=pattern,
            out_before=out_before,
            out_after=out_after,
            fusion_strategy=fusion_strategy,
            voxel_reg=voxel_reg,
            voxel_final=voxel_final,
            fitness_gate=fitness_gate,
            rmse_gate=rmse_gate,
            loop_k=loop_k,
            loop_fitness_gate=loop_fitness_gate,
            loop_rmse_gate=loop_rmse_gate,
            opt_max_corr=opt_max_corr,
            drop_k=drop_k,
            drop_fitness_th=drop_fitness_th,
            drop_rmse_th=drop_rmse_th,
            verbose=verbose
        )
        rospy.loginfo("[fuse] done. before=%s after=%s", b, a)
    except Exception as e:
        import traceback
        rospy.logerr("[fuse] FAILED: %s", str(e))
        traceback.print_exc()
        raise

if __name__ == "__main__":
    rospy.init_node("auto_scan_ring_and_fuse")
    main()
