#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os

import open3d as o3d


def load_cloud(path):
    pcd = o3d.io.read_point_cloud(path)
    if pcd.is_empty():
        raise RuntimeError(f"Point cloud is empty: {path}")
    return pcd


def main():
    parser = argparse.ArgumentParser(description="View one or more PLY point clouds with Open3D")
    parser.add_argument("ply", nargs="+", help="PLY file(s) to visualize")
    args = parser.parse_args()

    geometries = []
    for path in args.ply:
        abs_path = os.path.abspath(os.path.expanduser(path))
        pcd = load_cloud(abs_path)
        print(f"[view_ply] loaded {abs_path} points={len(pcd.points)}")
        geometries.append(pcd)

    o3d.visualization.draw_geometries(
        geometries,
        window_name="PLY Viewer",
        width=1600,
        height=900,
    )


if __name__ == "__main__":
    main()
