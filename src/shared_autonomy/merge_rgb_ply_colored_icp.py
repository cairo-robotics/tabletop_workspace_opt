#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge RGB PLY scans using pose-graph optimization with colored ICP refinement.

Input:
- a directory containing scan_rgb_XXXX.ply files

Output:
- merged_colored_posegraph_before_drop.ply
- merged_colored_posegraph_after_drop.ply

This script follows the same broad idea as the manual capture/fuse pipeline:
- preprocess each scan
- register adjacent scans
- add loop closures
- optimize a pose graph
- optionally drop a few bad frames
- merge all aligned clouds
"""

import argparse
import copy
import glob
import os
from collections import defaultdict, deque

import numpy as np
import open3d as o3d

try:
    from auto_scan_ring_and_fuse import ensure_dir, remove_plane, keep_largest_cluster
except ImportError:
    def ensure_dir(path):
        os.makedirs(path, exist_ok=True)

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


def crop_foreground_by_depth(
    pcd,
    max_depth_quantile=0.55,
    pad=0.03,
    min_keep_points=800,
    cluster_eps=0.02,
    cluster_min_points=60,
):
    if len(pcd.points) < min_keep_points:
        return pcd

    pts = np.asarray(pcd.points)
    z = pts[:, 2]
    valid = np.isfinite(z) & (z > 0.0)
    if int(valid.sum()) < min_keep_points:
        return pcd

    z_valid = z[valid]
    z_hi = float(np.quantile(z_valid, max_depth_quantile)) + pad
    z_lo = max(float(z_valid.min()) - 0.01, 0.0)
    keep_idx = np.where(valid & (z >= z_lo) & (z <= z_hi))[0]
    if keep_idx.size < min_keep_points:
        return pcd

    cropped = pcd.select_by_index(keep_idx.tolist())
    if len(cropped.points) < min_keep_points:
        return pcd

    labels = np.array(
        cropped.cluster_dbscan(
            eps=cluster_eps,
            min_points=cluster_min_points,
            print_progress=False,
        )
    )
    if labels.size == 0 or labels.max() < 0:
        return cropped

    pts_crop = np.asarray(cropped.points)
    best_label = None
    best_score = None
    for label in np.unique(labels):
        if label < 0:
            continue
        idx = np.where(labels == label)[0]
        if idx.size < cluster_min_points:
            continue
        mean_z = float(np.mean(pts_crop[idx, 2]))
        # Prefer larger and nearer clusters, which is usually the foreground object.
        score = (mean_z, -idx.size)
        if best_score is None or score < best_score:
            best_score = score
            best_label = int(label)

    if best_label is None:
        return cropped

    best_idx = np.where(labels == best_label)[0]
    if best_idx.size < cluster_min_points:
        return cropped
    return cropped.select_by_index(best_idx.tolist())


def preprocess_for_registration(
    pcd,
    voxel,
    remove_plane_flag=True,
    keep_cluster_flag=True,
    foreground_crop=True,
):
    p = copy.deepcopy(pcd)
    if foreground_crop:
        p = crop_foreground_by_depth(
            p,
            max_depth_quantile=0.60,
            pad=max(voxel * 6.0, 0.02),
            min_keep_points=800,
            cluster_eps=max(voxel * 10.0, 0.02),
            cluster_min_points=60,
        )
    if voxel > 0.0:
        p = p.voxel_down_sample(voxel)
    if remove_plane_flag:
        p = remove_plane(p, dist=max(voxel * 2.0, 0.003))
    if keep_cluster_flag:
        p = keep_largest_cluster(p, eps=max(voxel * 6.0, 0.015), min_points=150)
    if len(p.points) > 50:
        _, ind = p.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        p = p.select_by_index(ind)
    estimate_normals(p, voxel if voxel > 0.0 else 0.005)
    return p


def preprocess_for_merge(
    pcd,
    voxel_hint,
    remove_plane_flag=True,
    keep_cluster_flag=True,
    foreground_crop=True,
):
    p = copy.deepcopy(pcd)
    if foreground_crop:
        p = crop_foreground_by_depth(
            p,
            max_depth_quantile=0.65,
            pad=max(voxel_hint * 8.0, 0.03),
            min_keep_points=800,
            cluster_eps=max(voxel_hint * 12.0, 0.02),
            cluster_min_points=60,
        )
    if remove_plane_flag:
        p = remove_plane(p, dist=max(voxel_hint * 2.0, 0.003))
    if keep_cluster_flag:
        p = keep_largest_cluster(p, eps=max(voxel_hint * 6.0, 0.015), min_points=150)
    if len(p.points) > 50:
        _, ind = p.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        p = p.select_by_index(ind)
    return p


def compute_fpfh(pcd, voxel):
    radius_feature = voxel * 5.0
    return o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100),
    )


def global_ransac_init(source, target, voxel):
    s = source.voxel_down_sample(voxel)
    t = target.voxel_down_sample(voxel)
    if len(s.points) < 20 or len(t.points) < 20:
        raise RuntimeError("too few points for global init")
    estimate_normals(s, voxel)
    estimate_normals(t, voxel)
    if not s.has_normals() or not t.has_normals():
        raise RuntimeError("normals unavailable for global init")
    f_s = compute_fpfh(s, voxel)
    f_t = compute_fpfh(t, voxel)
    dist = voxel * 6.0
    reg = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        s,
        t,
        f_s,
        f_t,
        mutual_filter=True,
        max_correspondence_distance=dist,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(dist),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(50000, 0.999),
    )
    return reg.transformation


def global_fgr_init(source, target, voxel):
    s = source.voxel_down_sample(voxel)
    t = target.voxel_down_sample(voxel)
    if len(s.points) < 20 or len(t.points) < 20:
        raise RuntimeError("too few points for FGR init")
    estimate_normals(s, voxel)
    estimate_normals(t, voxel)
    f_s = compute_fpfh(s, voxel)
    f_t = compute_fpfh(t, voxel)
    reg = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        s,
        t,
        f_s,
        f_t,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=max(voxel * 4.0, 0.02),
        ),
    )
    return reg.transformation


def pairwise_icp_multiscale(source, target, init=np.eye(4)):
    scales = [
        (0.02, 0.15, 40),
        (0.01, 0.06, 50),
        (0.005, 0.02, 70),
    ]
    T = init.copy()
    last_reg = None
    for voxel, max_corr, iters in scales:
        s = source.voxel_down_sample(voxel)
        t = target.voxel_down_sample(voxel)
        if len(s.points) < 10 or len(t.points) < 10:
            continue
        estimate_normals(s, voxel)
        estimate_normals(t, voxel)
        if not s.has_normals() or not t.has_normals():
            continue
        reg = o3d.pipelines.registration.registration_icp(
            s,
            t,
            max_corr,
            T,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=iters,
                relative_fitness=1e-6,
                relative_rmse=1e-6,
            ),
        )
        T = reg.transformation
        last_reg = reg
    if last_reg is None:
        raise RuntimeError("too few points after downsampling for ICP")
    info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, max_correspondence_distance=scales[-1][1], transformation=T
    )
    return T, info, last_reg


def pairwise_gicp_multiscale(source, target, init=np.eye(4)):
    scales = [
        (0.03, 0.12, 40),
        (0.015, 0.06, 50),
        (0.0075, 0.03, 60),
    ]
    T = init.copy()
    last_reg = None
    for voxel, max_corr, iters in scales:
        s = source.voxel_down_sample(voxel)
        t = target.voxel_down_sample(voxel)
        if len(s.points) < 10 or len(t.points) < 10:
            continue
        estimate_normals(s, voxel)
        estimate_normals(t, voxel)
        if not s.has_normals() or not t.has_normals():
            continue
        reg = o3d.pipelines.registration.registration_generalized_icp(
            s,
            t,
            max_corr,
            T,
            o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=iters,
                relative_fitness=1e-6,
                relative_rmse=1e-6,
            ),
        )
        T = reg.transformation
        last_reg = reg
    if last_reg is None:
        raise RuntimeError("too few points after downsampling for GICP")
    info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, max_correspondence_distance=scales[-1][1], transformation=T
    )
    return T, info, last_reg


def pairwise_colored_icp_multiscale(source, target, init):
    if not source.has_colors() or not target.has_colors():
        raise RuntimeError("colored ICP requires point colors")
    scales = [
        (0.01, 0.04, 30),
        (0.005, 0.02, 50),
    ]
    T = init.copy()
    last_reg = None
    for voxel, max_corr, iterations in scales:
        s = source.voxel_down_sample(voxel)
        t = target.voxel_down_sample(voxel)
        if len(s.points) < 20 or len(t.points) < 20:
            continue
        if not s.has_colors() or not t.has_colors():
            raise RuntimeError("downsampled clouds lost colors")
        estimate_normals(s, voxel)
        estimate_normals(t, voxel)
        if not s.has_normals() or not t.has_normals():
            raise RuntimeError("normals unavailable for colored ICP")
        reg = o3d.pipelines.registration.registration_colored_icp(
            s,
            t,
            max_corr,
            T,
            o3d.pipelines.registration.TransformationEstimationForColoredICP(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=iterations,
                relative_fitness=1e-6,
                relative_rmse=1e-6,
            ),
        )
        T = reg.transformation
        last_reg = reg
    if last_reg is None:
        raise RuntimeError("too few points for colored ICP")
    info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, max_correspondence_distance=scales[-1][1], transformation=T
    )
    return T, info, last_reg


def transformation_magnitude(T):
    delta_t = float(np.linalg.norm(T[:3, 3]))
    rot = T[:3, :3]
    trace_val = float(np.clip((np.trace(rot) - 1.0) * 0.5, -1.0, 1.0))
    delta_r = float(np.degrees(np.arccos(trace_val)))
    return delta_t, delta_r


def transform_is_plausible(T, max_translation=0.25, max_rotation_deg=70.0):
    delta_t, delta_r = transformation_magnitude(T)
    return delta_t <= max_translation and delta_r <= max_rotation_deg


def better(candidate, incumbent):
    if incumbent is None:
        return True
    c = candidate[2]
    i = incumbent[2]
    if c.fitness != i.fitness:
        return c.fitness > i.fitness
    return c.inlier_rmse < i.inlier_rmse


def robust_register(src, tgt, fitness_gate, rmse_gate):
    best = None

    try:
        direct = (*pairwise_icp_multiscale(src, tgt, init=np.eye(4)), "icp")
        direct_plausible = transform_is_plausible(direct[0])
        if direct_plausible and better(direct, best):
            best = direct
        if direct_plausible and direct[2].fitness >= fitness_gate and direct[2].inlier_rmse <= rmse_gate:
            if src.has_colors() and tgt.has_colors():
                try:
                    colored = (*pairwise_colored_icp_multiscale(src, tgt, direct[0]), "icp+colored")
                    if transform_is_plausible(colored[0]) and better(colored, direct):
                        return colored
                except Exception:
                    pass
            return direct
    except Exception:
        pass

    errors = []
    init_attempts = [
        ("fgr0.030", lambda: global_fgr_init(src, tgt, 0.03)),
        ("ransac0.030", lambda: global_ransac_init(src, tgt, 0.03)),
        ("fgr0.020", lambda: global_fgr_init(src, tgt, 0.02)),
        ("ransac0.020", lambda: global_ransac_init(src, tgt, 0.02)),
    ]
    for label, init_fn in init_attempts:
        try:
            T0 = init_fn()
            candidate = (*pairwise_gicp_multiscale(src, tgt, init=T0), f"{label}+gicp")
            if not transform_is_plausible(candidate[0]):
                raise RuntimeError("implausible transform")
            if src.has_colors() and tgt.has_colors():
                try:
                    colored = (*pairwise_colored_icp_multiscale(src, tgt, candidate[0]), f"{label}+gicp+colored")
                    if transform_is_plausible(colored[0]) and better(colored, candidate):
                        candidate = colored
                except Exception:
                    pass
            if better(candidate, best):
                best = candidate
            if candidate[2].fitness >= fitness_gate and candidate[2].inlier_rmse <= rmse_gate:
                return candidate
        except Exception as exc:
            errors.append(f"{label}: {str(exc)}")

    if best is not None:
        return best
    raise RuntimeError("Registration failed: %s" % ("; ".join(errors[:2]) if errors else "unknown error"))


def build_pose_graph(
    pcds_reg,
    fitness_gate,
    rmse_gate,
    loop_k,
    loop_fitness_gate,
    loop_rmse_gate,
    max_translation,
    max_rotation_deg,
    verbose,
):
    graph = o3d.pipelines.registration.PoseGraph()
    odom = np.eye(4)
    graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(odom)))

    for i in range(1, len(pcds_reg)):
        src = pcds_reg[i - 1]
        tgt = pcds_reg[i]
        T, info, reg, how = robust_register(src, tgt, fitness_gate, rmse_gate)
        if verbose:
            print(f"[edge {i-1}->{i}] {how:16s} fitness={reg.fitness:.3f} rmse={reg.inlier_rmse:.4f}")
        plausible = transform_is_plausible(T, max_translation=max_translation, max_rotation_deg=max_rotation_deg)
        if reg.fitness < fitness_gate or reg.inlier_rmse > rmse_gate or not plausible:
            if verbose:
                why = "implausible transform" if not plausible else "score gate"
                print(f"  -> reject odom edge ({why})")
            graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(odom)))
        else:
            odom = odom @ T
            graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(odom)))
            graph.edges.append(
                o3d.pipelines.registration.PoseGraphEdge(i - 1, i, T, info, uncertain=False)
            )

        if loop_k is not None and i - 2 >= 0:
            best_loop = None
            start = max(0, i - loop_k)
            for a in range(start, i - 1):
                try:
                    T_lc, info_lc, reg_lc, how_lc = robust_register(
                        pcds_reg[a], pcds_reg[i], loop_fitness_gate, loop_rmse_gate
                    )
                except Exception:
                    continue
                plausible_lc = transform_is_plausible(
                    T_lc, max_translation=max_translation * 1.5, max_rotation_deg=max_rotation_deg * 1.5
                )
                if not plausible_lc:
                    continue
                candidate = (a, T_lc, info_lc, reg_lc, how_lc)
                if best_loop is None or better(candidate[1:], best_loop[1:]):
                    best_loop = candidate
            if best_loop is not None:
                a, T_lc, info_lc, reg_lc, how_lc = best_loop
                if verbose:
                    print(f"[loop {a}->{i}] {how_lc:16s} fitness={reg_lc.fitness:.3f} rmse={reg_lc.inlier_rmse:.4f}")
                if reg_lc.fitness >= loop_fitness_gate and reg_lc.inlier_rmse <= loop_rmse_gate:
                    graph.edges.append(
                        o3d.pipelines.registration.PoseGraphEdge(a, i, T_lc, info_lc, uncertain=True)
                    )
                elif verbose:
                    print("  -> reject loop edge")

    return graph


def optimize_pose_graph(graph, max_corr):
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=max_corr,
        edge_prune_threshold=0.25,
        reference_node=0,
    )
    o3d.pipelines.registration.global_optimization(
        graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        option,
    )


def largest_connected_component(num_nodes, edges):
    adj = defaultdict(list)
    for e in edges:
        adj[e.source_node_id].append(e.target_node_id)
        adj[e.target_node_id].append(e.source_node_id)

    visited = [False] * num_nodes
    best = []
    for start in range(num_nodes):
        if visited[start]:
            continue
        q = deque([start])
        visited[start] = True
        comp = [start]
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
    graph,
    keep_nodes,
    voxel_final,
    out_path,
    final_remove_outlier,
    nb_neighbors,
    std_ratio,
    final_keep_largest_cluster=False,
    final_cluster_eps=0.015,
    final_cluster_min_points=200,
):
    merged = o3d.geometry.PointCloud()
    for i in keep_nodes:
        p = copy.deepcopy(source_pcds[i])
        p.transform(graph.nodes[i].pose)
        merged += p

    if voxel_final > 0.0:
        merged = merged.voxel_down_sample(voxel_final)

    if final_remove_outlier and len(merged.points) > nb_neighbors:
        _, ind = merged.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        merged = merged.select_by_index(ind)

    if final_keep_largest_cluster:
        merged = keep_largest_cluster(
            merged,
            eps=final_cluster_eps,
            min_points=final_cluster_min_points,
        )

    o3d.io.write_point_cloud(out_path, merged, write_ascii=False, compressed=False)
    print(f"Saved: {out_path}  points={len(merged.points)}  used_frames={len(keep_nodes)}/{len(source_pcds)}")
    return merged


def score_frame_against_model(frame_pcd, model_pcd, voxel=0.005):
    a = frame_pcd.voxel_down_sample(voxel)
    b = model_pcd.voxel_down_sample(voxel)
    if len(a.points) < 10 or len(b.points) < 10:
        return 0.0, 1e9
    estimate_normals(a, voxel)
    estimate_normals(b, voxel)
    reg = o3d.pipelines.registration.registration_icp(
        a,
        b,
        voxel * 6.0,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=40),
    )
    return reg.fitness, reg.inlier_rmse


def pick_bad_frames(raw, graph, keep_nodes, voxel=0.005, fitness_th=0.35, rmse_th=0.012):
    model = o3d.geometry.PointCloud()
    for i in keep_nodes:
        p = copy.deepcopy(raw[i])
        p.transform(graph.nodes[i].pose)
        model += p
    model = model.voxel_down_sample(voxel)

    bad = []
    for i in keep_nodes:
        p = copy.deepcopy(raw[i])
        p.transform(graph.nodes[i].pose)
        fit, rmse = score_frame_against_model(p, model, voxel=voxel)
        if fit < fitness_th or rmse > rmse_th:
            bad.append((i, float(fit), float(rmse)))
    bad.sort(key=lambda x: (x[1], -x[2]))
    return bad


def merge_with_transforms(
    source_pcds,
    transforms,
    voxel_final,
    out_path,
    final_remove_outlier,
    nb_neighbors,
    std_ratio,
    final_keep_largest_cluster=False,
    final_cluster_eps=0.015,
    final_cluster_min_points=200,
):
    merged = o3d.geometry.PointCloud()
    for pcd, T in zip(source_pcds, transforms):
        q = copy.deepcopy(pcd)
        q.transform(T)
        merged += q

    if voxel_final > 0.0:
        merged = merged.voxel_down_sample(voxel_final)

    if final_remove_outlier and len(merged.points) > nb_neighbors:
        _, ind = merged.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        merged = merged.select_by_index(ind)

    if final_keep_largest_cluster:
        merged = keep_largest_cluster(
            merged,
            eps=final_cluster_eps,
            min_points=final_cluster_min_points,
        )

    o3d.io.write_point_cloud(out_path, merged, write_ascii=False, compressed=False)
    print(f"Saved: {out_path}  points={len(merged.points)}  used_frames={len(source_pcds)}")
    return merged


def refine_world_aligned_clouds(
    raw,
    merge_source,
    voxel_reg,
    voxel_final,
    fitness_gate,
    rmse_gate,
    drop_k,
    drop_fitness_th,
    drop_rmse_th,
    max_pair_translation,
    max_pair_rotation_deg,
    final_remove_outlier,
    final_outlier_nb_neighbors,
    final_outlier_std_ratio,
    final_keep_largest_cluster,
    final_cluster_eps,
    final_cluster_min_points,
    out_before_path,
    out_after_path,
    verbose,
):
    accepted_source = [merge_source[0]]
    accepted_transforms = [np.eye(4)]
    model = preprocess_for_registration(
        merge_source[0],
        voxel_reg,
        remove_plane_flag=False,
        keep_cluster_flag=False,
        foreground_crop=False,
    )

    if verbose:
        print(f"[world_refine] seed frame=0 points={len(raw[0].points)}")

    for i in range(1, len(merge_source)):
        src = preprocess_for_registration(
            merge_source[i],
            voxel_reg,
            remove_plane_flag=False,
            keep_cluster_flag=False,
            foreground_crop=False,
        )
        if len(src.points) < 20 or len(model.points) < 20:
            if verbose:
                print(f"[world_refine] frame={i} skipped: too few points after preprocessing")
            continue
        try:
            T, _, reg, how = robust_register(src, model, fitness_gate, rmse_gate)
        except Exception as exc:
            if verbose:
                print(f"[world_refine] frame={i} rejected: {exc}")
            continue

        plausible = transform_is_plausible(
            T,
            max_translation=max_pair_translation,
            max_rotation_deg=max_pair_rotation_deg,
        )
        accept = plausible and reg.fitness >= fitness_gate and reg.inlier_rmse <= rmse_gate
        dt, dr = transformation_magnitude(T)
        if verbose:
            print(
                f"[world_refine] frame={i} {'accept' if accept else 'reject'} method={how} "
                f"fitness={reg.fitness:.3f} rmse={reg.inlier_rmse:.4f} dtrans={dt:.4f} drot={dr:.2f}"
            )
        if not accept:
            continue

        accepted_source.append(merge_source[i])
        accepted_transforms.append(T)

        model_accum = o3d.geometry.PointCloud()
        for pcd, T_acc in zip(accepted_source, accepted_transforms):
            q = copy.deepcopy(pcd)
            q.transform(T_acc)
            model_accum += q
        if voxel_final > 0.0:
            model_accum = model_accum.voxel_down_sample(voxel_final)
        model = preprocess_for_registration(
            model_accum,
            voxel_reg,
            remove_plane_flag=False,
            keep_cluster_flag=False,
            foreground_crop=False,
        )

    before_model = merge_with_transforms(
        accepted_source,
        accepted_transforms,
        voxel_final,
        out_before_path,
        final_remove_outlier,
        final_outlier_nb_neighbors,
        final_outlier_std_ratio,
        final_keep_largest_cluster,
        final_cluster_eps,
        final_cluster_min_points,
    )

    bad = []
    for i, (pcd, T) in enumerate(zip(accepted_source, accepted_transforms)):
        q = copy.deepcopy(pcd)
        q.transform(T)
        fit, rmse = score_frame_against_model(q, before_model, voxel=voxel_reg)
        if fit < drop_fitness_th or rmse > drop_rmse_th:
            bad.append((i, float(fit), float(rmse)))
    bad.sort(key=lambda x: (x[1], -x[2]))
    if verbose:
        print("[world_refine] Bad frames (worst->best):")
        for item in bad[:8]:
            print("  frame=%d  fitness=%.3f  rmse=%.4f" % item)

    keep = list(range(len(accepted_source)))
    if drop_k > 0:
        drop = {bad[i][0] for i in range(min(drop_k, len(bad)))}
        keep = [i for i in keep if i not in drop]
        if verbose:
            print(f"[world_refine] Dropping frames: {sorted(list(drop))} keep={len(keep)}/{len(accepted_source)}")

    merge_with_transforms(
        [accepted_source[i] for i in keep],
        [accepted_transforms[i] for i in keep],
        voxel_final,
        out_after_path,
        final_remove_outlier,
        final_outlier_nb_neighbors,
        final_outlier_std_ratio,
        final_keep_largest_cluster,
        final_cluster_eps,
        final_cluster_min_points,
    )
    return out_before_path, out_after_path


def merge_rgb_ply(
    ply_dir,
    pattern="scan_rgb_*.ply",
    out_before="merged_colored_posegraph_before_drop.ply",
    out_after="merged_colored_posegraph_after_drop.ply",
    voxel_reg=0.005,
    voxel_final=0.003,
    fitness_gate=0.45,
    rmse_gate=0.010,
    loop_k=2,
    loop_fitness_gate=0.50,
    loop_rmse_gate=0.010,
    opt_max_corr=0.02,
    drop_k=3,
    drop_fitness_th=0.45,
    drop_rmse_th=0.010,
    reg_remove_plane=True,
    reg_keep_largest_cluster=True,
    merge_remove_plane=True,
    merge_keep_largest_cluster=True,
    final_remove_outlier=True,
    final_outlier_nb_neighbors=30,
    final_outlier_std_ratio=2.5,
    final_keep_largest_cluster=True,
    final_cluster_eps=0.015,
    final_cluster_min_points=200,
    foreground_crop=True,
    max_pair_translation=0.25,
    max_pair_rotation_deg=70.0,
    initial_alignment_mode="pairwise_posegraph",
    verbose=True,
):
    files = sorted(glob.glob(os.path.join(ply_dir, pattern)))
    if not files:
        raise RuntimeError(f"No '{pattern}' found in {ply_dir}")

    print(f"[merge_rgb] Found {len(files)} files")
    raw = [o3d.io.read_point_cloud(f) for f in files]
    valid = [p for p in raw if len(p.points) >= 20]
    if len(valid) < 2:
        raise RuntimeError("Need at least 2 valid RGB PLY scans")
    raw = valid

    pcds_reg = [
        preprocess_for_registration(
            p,
            voxel_reg,
            remove_plane_flag=reg_remove_plane,
            keep_cluster_flag=reg_keep_largest_cluster,
            foreground_crop=foreground_crop,
        )
        for p in raw
    ]
    valid_pairs = [(r, p) for r, p in zip(raw, pcds_reg) if len(p.points) >= 20]
    if len(valid_pairs) < 2:
        raise RuntimeError("Need at least 2 usable scans after preprocessing")
    raw = [x[0] for x in valid_pairs]
    pcds_reg = [x[1] for x in valid_pairs]
    merge_source = [
        preprocess_for_merge(
            p,
            voxel_reg,
            remove_plane_flag=merge_remove_plane,
            keep_cluster_flag=merge_keep_largest_cluster,
            foreground_crop=foreground_crop,
        )
        for p in raw
    ]
    merge_pairs = [(r, p) for r, p in zip(raw, merge_source) if len(p.points) >= 20]
    if len(merge_pairs) == len(raw):
        merge_source = [x[1] for x in merge_pairs]
    else:
        merge_source = raw

    out_before_path = os.path.join(ply_dir, out_before)
    out_after_path = os.path.join(ply_dir, out_after)

    if initial_alignment_mode == "world_pose_refine":
        return refine_world_aligned_clouds(
            raw=raw,
            merge_source=merge_source,
            voxel_reg=voxel_reg,
            voxel_final=voxel_final,
            fitness_gate=fitness_gate,
            rmse_gate=rmse_gate,
            drop_k=drop_k,
            drop_fitness_th=drop_fitness_th,
            drop_rmse_th=drop_rmse_th,
            max_pair_translation=max_pair_translation,
            max_pair_rotation_deg=max_pair_rotation_deg,
            final_remove_outlier=final_remove_outlier,
            final_outlier_nb_neighbors=final_outlier_nb_neighbors,
            final_outlier_std_ratio=final_outlier_std_ratio,
            final_keep_largest_cluster=final_keep_largest_cluster,
            final_cluster_eps=final_cluster_eps,
            final_cluster_min_points=final_cluster_min_points,
            out_before_path=out_before_path,
            out_after_path=out_after_path,
            verbose=verbose,
        )

    graph = build_pose_graph(
        pcds_reg,
        fitness_gate=fitness_gate,
        rmse_gate=rmse_gate,
        loop_k=loop_k if loop_k > 0 else None,
        loop_fitness_gate=loop_fitness_gate,
        loop_rmse_gate=loop_rmse_gate,
        max_translation=max_pair_translation,
        max_rotation_deg=max_pair_rotation_deg,
        verbose=verbose,
    )

    keep = largest_connected_component(len(graph.nodes), graph.edges)
    if len(keep) < len(graph.nodes):
        print(f"[WARN] PoseGraph disconnected. Keeping largest component: {len(keep)}/{len(graph.nodes)} nodes")

    optimize_pose_graph(graph, opt_max_corr)

    merge_with_poses(
        merge_source,
        graph,
        keep,
        voxel_final,
        out_before_path,
        final_remove_outlier,
        final_outlier_nb_neighbors,
        final_outlier_std_ratio,
        final_keep_largest_cluster,
        final_cluster_eps,
        final_cluster_min_points,
    )

    keep2 = list(keep)
    if drop_k > 0:
        bad = pick_bad_frames(merge_source, graph, keep, voxel=voxel_reg, fitness_th=drop_fitness_th, rmse_th=drop_rmse_th)
        print("[merge_rgb] Bad frames (worst->best):")
        for item in bad[:8]:
            print("  frame=%d  fitness=%.3f  rmse=%.4f" % item)
        drop = {bad[i][0] for i in range(min(drop_k, len(bad)))}
        keep2 = [i for i in keep if i not in drop]
        print(f"[merge_rgb] Dropping frames: {sorted(list(drop))} keep={len(keep2)}/{len(keep)}")
    else:
        print("[merge_rgb] drop_k <= 0, keeping all frames")

    merge_with_poses(
        merge_source,
        graph,
        keep2,
        voxel_final,
        out_after_path,
        final_remove_outlier,
        final_outlier_nb_neighbors,
        final_outlier_std_ratio,
        final_keep_largest_cluster,
        final_cluster_eps,
        final_cluster_min_points,
    )
    return out_before_path, out_after_path


def main():
    parser = argparse.ArgumentParser(description="Merge RGB PLY scans using colored ICP pose-graph fusion")
    parser.add_argument("--ply_dir", default=os.path.join(os.path.dirname(__file__), "..", "captured_rgb_ply"))
    parser.add_argument("--pattern", default="scan_rgb_*.ply")
    parser.add_argument("--voxel_reg", type=float, default=0.005)
    parser.add_argument("--voxel_final", type=float, default=0.003)
    parser.add_argument("--loop_k", type=int, default=2)
    parser.add_argument("--drop_k", type=int, default=3)
    parser.add_argument("--fitness_gate", type=float, default=0.45)
    parser.add_argument("--rmse_gate", type=float, default=0.010)
    parser.add_argument("--loop_fitness_gate", type=float, default=0.50)
    parser.add_argument("--loop_rmse_gate", type=float, default=0.010)
    parser.add_argument("--opt_max_corr", type=float, default=0.02)
    parser.add_argument("--drop_fitness_th", type=float, default=0.45)
    parser.add_argument("--drop_rmse_th", type=float, default=0.010)
    parser.add_argument("--no_reg_remove_plane", action="store_true")
    parser.add_argument("--no_reg_keep_largest_cluster", action="store_true")
    parser.add_argument("--no_merge_remove_plane", action="store_true")
    parser.add_argument("--no_merge_keep_largest_cluster", action="store_true")
    parser.add_argument("--no_final_outlier", action="store_true")
    parser.add_argument("--final_outlier_nb_neighbors", type=int, default=30)
    parser.add_argument("--final_outlier_std_ratio", type=float, default=2.5)
    parser.add_argument("--no_final_keep_largest_cluster", action="store_true")
    parser.add_argument("--final_cluster_eps", type=float, default=0.015)
    parser.add_argument("--final_cluster_min_points", type=int, default=200)
    parser.add_argument("--no_foreground_crop", action="store_true")
    parser.add_argument("--max_pair_translation", type=float, default=0.25)
    parser.add_argument("--max_pair_rotation_deg", type=float, default=70.0)
    parser.add_argument("--initial_alignment_mode", choices=["pairwise_posegraph", "world_pose_refine"], default="pairwise_posegraph")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    ply_dir = os.path.abspath(os.path.expanduser(args.ply_dir))
    ensure_dir(ply_dir)
    before, after = merge_rgb_ply(
        ply_dir=ply_dir,
        pattern=args.pattern,
        voxel_reg=args.voxel_reg,
        voxel_final=args.voxel_final,
        fitness_gate=args.fitness_gate,
        rmse_gate=args.rmse_gate,
        loop_k=args.loop_k,
        loop_fitness_gate=args.loop_fitness_gate,
        loop_rmse_gate=args.loop_rmse_gate,
        opt_max_corr=args.opt_max_corr,
        drop_k=args.drop_k,
        drop_fitness_th=args.drop_fitness_th,
        drop_rmse_th=args.drop_rmse_th,
        reg_remove_plane=not args.no_reg_remove_plane,
        reg_keep_largest_cluster=not args.no_reg_keep_largest_cluster,
        merge_remove_plane=not args.no_merge_remove_plane,
        merge_keep_largest_cluster=not args.no_merge_keep_largest_cluster,
        final_remove_outlier=not args.no_final_outlier,
        final_outlier_nb_neighbors=args.final_outlier_nb_neighbors,
        final_outlier_std_ratio=args.final_outlier_std_ratio,
        final_keep_largest_cluster=not args.no_final_keep_largest_cluster,
        final_cluster_eps=args.final_cluster_eps,
        final_cluster_min_points=args.final_cluster_min_points,
        foreground_crop=not args.no_foreground_crop,
        max_pair_translation=args.max_pair_translation,
        max_pair_rotation_deg=args.max_pair_rotation_deg,
        initial_alignment_mode=args.initial_alignment_mode,
        verbose=not args.quiet,
    )
    print("\nDone.")
    print(f"Before-drop result: {before}")
    print(f"After-drop result:  {after}")


if __name__ == "__main__":
    main()
