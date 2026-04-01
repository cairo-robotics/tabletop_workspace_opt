#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os

import numpy as np
import open3d as o3d


def load_cloud(path):
    pcd = o3d.io.read_point_cloud(path)
    if pcd.is_empty():
        raise RuntimeError(f"Point cloud is empty: {path}")
    return pcd


def remove_dominant_plane(pcd, plane_dist, min_plane_points):
    if len(pcd.points) < min_plane_points:
        return pcd, None
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=plane_dist,
        ransac_n=3,
        num_iterations=1000,
    )
    if len(inliers) < min_plane_points:
        return pcd, None
    foreground = pcd.select_by_index(inliers, invert=True)
    return foreground, plane_model


def cluster_objects(pcd, eps, min_points, min_cluster_points):
    if len(pcd.points) == 0:
        return []
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    if labels.size == 0 or labels.max() < 0:
        return []

    clusters = []
    for label in sorted(np.unique(labels)):
        if label < 0:
            continue
        idx = np.where(labels == label)[0]
        if idx.size < min_cluster_points:
            continue
        cluster = pcd.select_by_index(idx.tolist())
        clusters.append((int(label), cluster))
    clusters.sort(key=lambda item: len(item[1].points), reverse=True)
    return clusters


def colorize_clusters(clusters):
    palette = np.array(
        [
            [0.95, 0.35, 0.35],
            [0.20, 0.60, 0.95],
            [0.25, 0.75, 0.45],
            [0.95, 0.75, 0.20],
            [0.80, 0.40, 0.95],
            [0.20, 0.85, 0.85],
        ],
        dtype=np.float64,
    )
    geoms = []
    for idx, (_, cluster) in enumerate(clusters):
        color = palette[idx % len(palette)]
        cluster_vis = o3d.geometry.PointCloud(cluster)
        cluster_vis.paint_uniform_color(color)
        geoms.append(cluster_vis)

        obb = cluster.get_oriented_bounding_box()
        obb.color = color
        geoms.append(obb)
    return geoms


def save_clusters(clusters, out_dir, prefix):
    os.makedirs(out_dir, exist_ok=True)
    saved = []
    for idx, (_, cluster) in enumerate(clusters):
        path = os.path.join(out_dir, f"{prefix}_cluster_{idx:02d}.ply")
        ok = o3d.io.write_point_cloud(path, cluster, write_ascii=False, compressed=False)
        if ok:
            saved.append(path)
    return saved


def print_cluster_summary(clusters):
    for idx, (_, cluster) in enumerate(clusters):
        pts = np.asarray(cluster.points)
        mins = pts.min(axis=0)
        maxs = pts.max(axis=0)
        center = pts.mean(axis=0)
        print(
            "[segment] cluster=%02d points=%d center=[%.3f, %.3f, %.3f] "
            "min=[%.3f, %.3f, %.3f] max=[%.3f, %.3f, %.3f]"
            % (
                idx,
                len(pts),
                center[0], center[1], center[2],
                mins[0], mins[1], mins[2],
                maxs[0], maxs[1], maxs[2],
            )
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Segment tabletop objects from a merged PLY")
    parser.add_argument("ply_path", help="Input merged PLY")
    parser.add_argument(
        "--out_dir",
        default="",
        help="Directory to save segmented clusters; defaults next to the input PLY",
    )
    parser.add_argument("--plane_dist", type=float, default=0.004, help="RANSAC plane distance threshold")
    parser.add_argument("--min_plane_points", type=int, default=3000, help="Minimum inliers to accept a plane")
    parser.add_argument("--cluster_eps", type=float, default=0.012, help="DBSCAN epsilon in meters")
    parser.add_argument("--cluster_min_points", type=int, default=120, help="DBSCAN min points")
    parser.add_argument("--min_cluster_points", type=int, default=1000, help="Minimum points kept per object cluster")
    parser.add_argument("--top_k", type=int, default=5, help="Keep at most this many largest clusters")
    parser.add_argument("--no_plane_removal", action="store_true", help="Skip dominant plane removal")
    parser.add_argument("--no_vis", action="store_true", help="Skip Open3D visualization")
    return parser.parse_args()


def main():
    args = parse_args()
    ply_path = os.path.abspath(os.path.expanduser(args.ply_path))
    pcd = load_cloud(ply_path)
    print(f"[segment] loaded {ply_path} points={len(pcd.points)}")

    work_cloud = pcd
    plane_model = None
    if not args.no_plane_removal:
        work_cloud, plane_model = remove_dominant_plane(
            pcd,
            plane_dist=args.plane_dist,
            min_plane_points=args.min_plane_points,
        )
        print(
            "[segment] after plane removal points=%d plane=%s"
            % (len(work_cloud.points), "none" if plane_model is None else np.round(plane_model, 5).tolist())
        )

    clusters = cluster_objects(
        work_cloud,
        eps=args.cluster_eps,
        min_points=args.cluster_min_points,
        min_cluster_points=args.min_cluster_points,
    )
    clusters = clusters[: max(0, args.top_k)]
    print(f"[segment] kept_clusters={len(clusters)}")
    print_cluster_summary(clusters)

    out_dir = args.out_dir.strip()
    if not out_dir:
        out_dir = os.path.join(os.path.dirname(ply_path), "segments")
    saved = save_clusters(clusters, out_dir, os.path.splitext(os.path.basename(ply_path))[0])
    for path in saved:
        print(f"[segment] saved {path}")

    if not args.no_vis:
        geoms = colorize_clusters(clusters)
        if plane_model is not None:
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
            geoms.append(frame)
        if geoms:
            o3d.visualization.draw_geometries(
                geoms,
                window_name="Segmented Objects",
                width=1600,
                height=900,
            )


if __name__ == "__main__":
    main()
