#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sequential RGB PLY merging using chain-style registration.

This script is intentionally more conservative than the pose-graph version:
- only registers each scan to the previous accepted scan
- no loop closure
- bad registrations are skipped instead of globally contaminating the result

Designed for objects like boxes/cartons where global pose-graph optimization can
easily overfit wrong planar correspondences and produce "exploded" shapes.
"""

import argparse
import copy
import glob
import os

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
        o3d.geometry.KDTreeSearchParamHybrid(radius=max(voxel * 3.0, 0.01), max_nn=30)
    )
    return pcd


def remove_small_outliers(pcd, nb_neighbors=20, std_ratio=2.0):
    if len(pcd.points) <= nb_neighbors:
        return pcd
    _, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return pcd.select_by_index(ind)


def preprocess_cloud(pcd, voxel, remove_plane_flag=True, keep_cluster_flag=True, outlier_flag=True):
    p = copy.deepcopy(pcd)
    if voxel > 0.0:
        p = p.voxel_down_sample(voxel)
    if remove_plane_flag:
        p = remove_plane(p, dist=max(voxel * 2.0, 0.003))
    if keep_cluster_flag:
        p = keep_largest_cluster(p, eps=max(voxel * 6.0, 0.015), min_points=150)
    if outlier_flag:
        p = remove_small_outliers(p, nb_neighbors=20, std_ratio=2.0)
    estimate_normals(p, voxel if voxel > 0.0 else 0.005)
    return p


def compute_fpfh(pcd, voxel):
    return o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=max(voxel * 5.0, 0.02), max_nn=100),
    )


def global_ransac_init(source, target, voxel=0.02):
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
    dist = max(voxel * 6.0, 0.03)
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


def global_fgr_init(source, target, voxel=0.02):
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


def pairwise_icp_multiscale(source, target, init):
    scales = [
        (0.02, 0.08, 40),
        (0.01, 0.04, 50),
        (0.005, 0.02, 70),
    ]
    T = init.copy()
    last_reg = None
    for voxel, max_corr, iterations in scales:
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
                max_iteration=iterations,
                relative_fitness=1e-6,
                relative_rmse=1e-6,
            ),
        )
        T = reg.transformation
        last_reg = reg
    if last_reg is None:
        raise RuntimeError("too few points for ICP")
    return T, last_reg


def pairwise_gicp_multiscale(source, target, init):
    scales = [
        (0.03, 0.12, 40),
        (0.015, 0.06, 50),
        (0.0075, 0.03, 60),
    ]
    T = init.copy()
    last_reg = None
    for voxel, max_corr, iterations in scales:
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
                max_iteration=iterations,
                relative_fitness=1e-6,
                relative_rmse=1e-6,
            ),
        )
        T = reg.transformation
        last_reg = reg
    if last_reg is None:
        raise RuntimeError("too few points for GICP")
    return T, last_reg


def pairwise_colored_icp_multiscale(source, target, init):
    if not source.has_colors() or not target.has_colors():
        raise RuntimeError("colored ICP requires colors")
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
    return T, last_reg


def transformation_magnitude(T):
    delta_t = float(np.linalg.norm(T[:3, 3]))
    rot = T[:3, :3]
    trace_val = float(np.clip((np.trace(rot) - 1.0) * 0.5, -1.0, 1.0))
    delta_r = float(np.degrees(np.arccos(trace_val)))
    return delta_t, delta_r


def transform_is_plausible(T, max_translation=0.25, max_rotation_deg=70.0):
    delta_t, delta_r = transformation_magnitude(T)
    return delta_t <= max_translation and delta_r <= max_rotation_deg


def register_pair(source, target, fitness_gate, rmse_gate, max_translation=0.25, max_rotation_deg=70.0):
    best = None
    errors = []

    try:
        T_id, reg_id = pairwise_icp_multiscale(source, target, np.eye(4))
        if transform_is_plausible(T_id, max_translation=max_translation, max_rotation_deg=max_rotation_deg):
            best = ("icp", T_id, reg_id)
    except Exception as exc:
        errors.append(str(exc))

    init_attempts = [
        ("fgr", lambda: global_fgr_init(source, target, voxel=0.02)),
        ("ransac", lambda: global_ransac_init(source, target, voxel=0.02)),
    ]
    for label, init_fn in init_attempts:
        try:
            T0 = init_fn()
            T1, reg1 = pairwise_gicp_multiscale(source, target, T0)
            if not transform_is_plausible(T1, max_translation=max_translation, max_rotation_deg=max_rotation_deg):
                raise RuntimeError("implausible transform")
            cand = (f"{label}+gicp", T1, reg1)
            if best is None or reg1.fitness > best[2].fitness or (
                reg1.fitness == best[2].fitness and reg1.inlier_rmse < best[2].inlier_rmse
            ):
                best = cand
        except Exception as exc:
            errors.append(f"{label}: {str(exc)}")

    if best is not None and source.has_colors() and target.has_colors():
        try:
            T2, reg2 = pairwise_colored_icp_multiscale(source, target, best[1])
            if transform_is_plausible(T2, max_translation=max_translation, max_rotation_deg=max_rotation_deg) and (
                reg2.fitness > best[2].fitness or (
                reg2.fitness == best[2].fitness and reg2.inlier_rmse < best[2].inlier_rmse
                )
            ):
                best = (best[0] + "+colored", T2, reg2)
        except Exception as exc:
            errors.append(str(exc))

    if best is None:
        raise RuntimeError("registration failed: %s" % ("; ".join(errors[:2]) if errors else "unknown error"))

    if best[2].fitness < fitness_gate or best[2].inlier_rmse > rmse_gate:
        raise RuntimeError(
            "registration rejected: method=%s fitness=%.3f rmse=%.4f"
            % (best[0], best[2].fitness, best[2].inlier_rmse)
        )
    return best


def merge_clouds(clouds, poses, voxel_final, final_remove_outlier, nb_neighbors, std_ratio, final_keep_largest_cluster):
    merged = o3d.geometry.PointCloud()
    for pcd, pose in zip(clouds, poses):
        q = copy.deepcopy(pcd)
        q.transform(pose)
        merged += q

    if voxel_final > 0.0:
        merged = merged.voxel_down_sample(voxel_final)
    if final_remove_outlier and len(merged.points) > nb_neighbors:
        _, ind = merged.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        merged = merged.select_by_index(ind)
    if final_keep_largest_cluster:
        merged = keep_largest_cluster(merged, eps=max(voxel_final * 6.0, 0.015), min_points=200)
    return merged


def merge_rgb_ply_chain(
    ply_dir,
    pattern="scan_rgb_*.ply",
    out_name="merged_chain_colored_icp.ply",
    voxel_reg=0.005,
    voxel_final=0.003,
    fitness_gate=0.20,
    rmse_gate=0.020,
    reg_remove_plane=False,
    reg_keep_largest_cluster=False,
    reg_remove_outlier=True,
    final_remove_outlier=True,
    final_outlier_nb_neighbors=30,
    final_outlier_std_ratio=2.5,
    final_keep_largest_cluster=False,
    anchor_lookback=3,
    max_pair_translation=0.25,
    max_pair_rotation_deg=70.0,
    verbose=True,
):
    files = sorted(glob.glob(os.path.join(ply_dir, pattern)))
    if not files:
        raise RuntimeError(f"No '{pattern}' found in {ply_dir}")

    raw = [o3d.io.read_point_cloud(f) for f in files]
    valid = [(f, p) for f, p in zip(files, raw) if len(p.points) >= 20]
    if len(valid) < 2:
        raise RuntimeError("Need at least 2 valid RGB PLY scans")
    files = [x[0] for x in valid]
    raw = [x[1] for x in valid]

    reg_clouds = [
        preprocess_cloud(
            p,
            voxel_reg,
            remove_plane_flag=reg_remove_plane,
            keep_cluster_flag=reg_keep_largest_cluster,
            outlier_flag=reg_remove_outlier,
        )
        for p in raw
    ]

    valid = [(f, r, p) for f, r, p in zip(files, raw, reg_clouds) if len(p.points) >= 20]
    if len(valid) < 2:
        raise RuntimeError("Need at least 2 usable scans after preprocessing")
    files = [x[0] for x in valid]
    raw = [x[1] for x in valid]
    reg_clouds = [x[2] for x in valid]

    accepted_raw = [raw[0]]
    accepted_reg = [reg_clouds[0]]
    accepted_files = [files[0]]
    poses = [np.eye(4)]

    if verbose:
        print(f"[chain] seed frame=0 file={os.path.basename(files[0])} points={len(raw[0].points)}")

    for i in range(1, len(reg_clouds)):
        src = reg_clouds[i]
        best_candidate = None
        best_error = None
        start_idx = max(0, len(accepted_reg) - anchor_lookback)
        for anchor_idx in range(len(accepted_reg) - 1, start_idx - 1, -1):
            if verbose:
                print(
                    "[chain] try frame=%d file=%s against accepted[%d]=%s"
                    % (i, os.path.basename(files[i]), anchor_idx, os.path.basename(accepted_files[anchor_idx]))
                )
            try:
                method, T_rel, reg = register_pair(
                    src,
                    accepted_reg[anchor_idx],
                    fitness_gate,
                    rmse_gate,
                    max_translation=max_pair_translation,
                    max_rotation_deg=max_pair_rotation_deg,
                )
                candidate = (anchor_idx, method, T_rel, reg)
                if best_candidate is None or reg.fitness > best_candidate[3].fitness or (
                    reg.fitness == best_candidate[3].fitness and reg.inlier_rmse < best_candidate[3].inlier_rmse
                ):
                    best_candidate = candidate
            except Exception as exc:
                best_error = str(exc)

        if best_candidate is None:
            if verbose:
                print(
                    "[chain] skip frame=%d file=%s reason=%s"
                    % (i, os.path.basename(files[i]), best_error if best_error else "no valid anchor")
                )
            continue

        anchor_idx, method, T_rel, reg = best_candidate
        pose_i = poses[anchor_idx] @ T_rel
        poses.append(pose_i)
        accepted_raw.append(raw[i])
        accepted_reg.append(reg_clouds[i])
        accepted_files.append(files[i])
        if verbose:
            print(
                "[chain] accept frame=%d file=%s anchor=%d method=%s fitness=%.3f rmse=%.4f kept=%d"
                % (i, os.path.basename(files[i]), anchor_idx, method, reg.fitness, reg.inlier_rmse, len(accepted_raw))
            )

    if len(accepted_raw) < 2:
        raise RuntimeError("Too few accepted scans after sequential registration")

    merged = merge_clouds(
        accepted_raw,
        poses,
        voxel_final=voxel_final,
        final_remove_outlier=final_remove_outlier,
        nb_neighbors=final_outlier_nb_neighbors,
        std_ratio=final_outlier_std_ratio,
        final_keep_largest_cluster=final_keep_largest_cluster,
    )

    out_path = os.path.join(ply_dir, out_name)
    ok = o3d.io.write_point_cloud(out_path, merged, write_ascii=False, compressed=False)
    if not ok:
        raise RuntimeError(f"Failed to write {out_path}")

    print("\nDone.")
    print(f"Accepted scans: {len(accepted_raw)}/{len(raw)}")
    print(f"Saved merged cloud: {out_path}")
    print(f"Merged points: {len(merged.points)}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Sequential colored ICP merge for RGB PLY scans")
    parser.add_argument("--ply_dir", default=os.path.join(os.path.dirname(__file__), "..", "captured_rgb_ply"))
    parser.add_argument("--pattern", default="scan_rgb_*.ply")
    parser.add_argument("--out_name", default="merged_chain_colored_icp.ply")
    parser.add_argument("--voxel_reg", type=float, default=0.005)
    parser.add_argument("--voxel_final", type=float, default=0.003)
    parser.add_argument("--fitness_gate", type=float, default=0.20)
    parser.add_argument("--rmse_gate", type=float, default=0.020)
    parser.add_argument("--no_reg_remove_plane", action="store_true")
    parser.add_argument("--no_reg_keep_largest_cluster", action="store_true")
    parser.add_argument("--no_reg_remove_outlier", action="store_true")
    parser.add_argument("--no_final_outlier", action="store_true")
    parser.add_argument("--final_outlier_nb_neighbors", type=int, default=30)
    parser.add_argument("--final_outlier_std_ratio", type=float, default=2.5)
    parser.add_argument("--final_keep_largest_cluster", action="store_true")
    parser.add_argument("--anchor_lookback", type=int, default=3)
    parser.add_argument("--max_pair_translation", type=float, default=0.25)
    parser.add_argument("--max_pair_rotation_deg", type=float, default=70.0)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    ply_dir = os.path.abspath(os.path.expanduser(args.ply_dir))
    ensure_dir(ply_dir)
    merge_rgb_ply_chain(
        ply_dir=ply_dir,
        pattern=args.pattern,
        out_name=args.out_name,
        voxel_reg=args.voxel_reg,
        voxel_final=args.voxel_final,
        fitness_gate=args.fitness_gate,
        rmse_gate=args.rmse_gate,
        reg_remove_plane=not args.no_reg_remove_plane,
        reg_keep_largest_cluster=not args.no_reg_keep_largest_cluster,
        reg_remove_outlier=not args.no_reg_remove_outlier,
        final_remove_outlier=not args.no_final_outlier,
        final_outlier_nb_neighbors=args.final_outlier_nb_neighbors,
        final_outlier_std_ratio=args.final_outlier_std_ratio,
        final_keep_largest_cluster=args.final_keep_largest_cluster,
        anchor_lookback=args.anchor_lookback,
        max_pair_translation=args.max_pair_translation,
        max_pair_rotation_deg=args.max_pair_rotation_deg,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
