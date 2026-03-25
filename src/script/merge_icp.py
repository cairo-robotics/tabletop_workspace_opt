#!/usr/bin/env python3
import os, glob, copy
import numpy as np
import open3d as o3d
from collections import defaultdict, deque

# -------------------------
# Preprocess
# -------------------------
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

# -------------------------
# FPFH + RANSAC fallback
# -------------------------
def compute_fpfh(pcd, voxel):
    radius_feature = voxel * 5.0
    return o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )

def global_ransac_init(source, target, voxel=0.02):
    s = source.voxel_down_sample(voxel)
    t = target.voxel_down_sample(voxel)
    estimate_normals(s, voxel)
    estimate_normals(t, voxel)
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

# -------------------------
# Multi-scale ICP (source -> target)
# -------------------------
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
        estimate_normals(s, voxel)
        estimate_normals(t, voxel)

        reg = o3d.pipelines.registration.registration_icp(
            s, t, max_corr, T,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=it, relative_fitness=1e-6, relative_rmse=1e-6
            )
        )
        T = reg.transformation
        last_reg = reg

    info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target,
        max_correspondence_distance=scales[-1][1],
        transformation=T
    )
    return T, info, last_reg

def robust_register(src, tgt, fitness_gate, rmse_gate):
    # First try ICP from identity
    T, info, reg = pairwise_icp_multiscale(src, tgt, init=np.eye(4))
    if (reg.fitness >= fitness_gate) and (reg.inlier_rmse <= rmse_gate):
        return T, info, reg, "icp"

    # If totally failed, do global init then ICP
    if reg.fitness == 0.0:
        try:
            T0 = global_ransac_init(src, tgt, voxel=0.02)
            T2, info2, reg2 = pairwise_icp_multiscale(src, tgt, init=T0)
            return T2, info2, reg2, "ransac+icp"
        except Exception as e:
            return T, info, reg, f"fail({e})"

    return T, info, reg, "bad"

# -------------------------
# Pose graph with loop closure
# -------------------------
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

        # loop closure (sparse)
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

# -------------------------
# Connectivity: keep largest component
# -------------------------
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

# -------------------------
# Merge
# -------------------------
def merge_with_poses(raw_pcds, pose_graph, keep_nodes, voxel_final, out_path):
    merged = o3d.geometry.PointCloud()
    for i in keep_nodes:
        p2 = copy.deepcopy(raw_pcds[i])
        p2.transform(pose_graph.nodes[i].pose)
        merged += p2

    merged = merged.voxel_down_sample(voxel_final)
    cl, ind = merged.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.5)
    merged = merged.select_by_index(ind)

    o3d.io.write_point_cloud(out_path, merged, write_ascii=False, compressed=False)
    print(f"Saved: {out_path}  points={len(merged.points)}  used_frames={len(keep_nodes)}/{len(raw_pcds)}")
    return merged

# -------------------------
# Auto bad-frame detection & drop
# -------------------------
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
    # Build coarse global model
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

    # worst first: low fitness, high rmse
    bad.sort(key=lambda x: (x[1], -x[2]))
    return bad

# -------------------------
# Main
# -------------------------
def main():
    ply_dir = "/home/heyang/scans" # modify this to your actual save directory
    out_before = "/home/heyang/scans/merged_posegraph_before_drop.ply"
    out_after  = "/home/heyang/scans/merged_posegraph_after_drop.ply"

    files = sorted(glob.glob(os.path.join(ply_dir, "scan_*.ply")))
    if not files:
        raise RuntimeError(f"No scan_*.ply in {ply_dir}")

    print(f"Found {len(files)} frames")
    raw = [o3d.io.read_point_cloud(f) for f in files]

    # Registration clouds
    voxel_reg = 0.005
    pcds_reg = [preprocess_for_registration(p, voxel_reg, do_remove_plane=True, do_largest_cluster=True)
                for p in raw]

    # Build pose graph
    pose_graph = build_pose_graph(
        pcds_reg,
        verbose=True,
        fitness_gate=0.30,
        rmse_gate=0.015,
        loop_k=6,
        loop_fitness_gate=0.40,
        loop_rmse_gate=0.012
    )

    # Keep largest connected component
    keep = largest_connected_component(len(pose_graph.nodes), pose_graph.edges)
    if len(keep) < len(pose_graph.nodes):
        print(f"[WARN] PoseGraph disconnected. Keeping largest component: {len(keep)}/{len(pose_graph.nodes)} nodes")

    # Optimize
    optimize_pose_graph(pose_graph, max_corr=0.02)

    # Merge before drop
    merge_with_poses(raw, pose_graph, keep_nodes=keep, voxel_final=0.003, out_path=out_before)

    # Auto-pick bad frames (usually 1 frame for your remaining tiny ghost cup)
    bad = pick_bad_frames(raw, pose_graph, keep, voxel=0.005, fitness_th=0.35, rmse_th=0.012)
    print("Bad frames (sorted worst->best):")
    for x in bad[:8]:
        print("  frame=%d  fitness=%.3f  rmse=%.4f" % x)

    # Drop worst 1 frame if any
    drop = set()
    if bad:
        drop.add(bad[0][0])
        # if len(bad) > 1: drop.add(bad[1][0])

    keep2 = [i for i in keep if i not in drop]
    print(f"Dropping frames: {sorted(list(drop))}  keep={len(keep2)}/{len(keep)}")

    # Merge after drop
    merge_with_poses(raw, pose_graph, keep_nodes=keep2, voxel_final=0.003, out_path=out_after)

if __name__ == "__main__":
    main()
