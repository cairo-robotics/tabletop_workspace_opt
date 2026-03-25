# ply_visualization.py
import sys
import os
import numpy as np

try:
    import open3d as o3d
except ImportError:
    print("Open3D not found. Install with: pip install open3d")
    sys.exit(1)

def main(path: str, voxel: float = 0.0):
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    geom = o3d.io.read_point_cloud(path)
    if geom.is_empty():
        mesh = o3d.io.read_triangle_mesh(path)
        if mesh.is_empty():
            raise RuntimeError("Cannot read as point cloud or mesh. Is this a valid .ply?")
        mesh.compute_vertex_normals()
        print(f"Loaded mesh: vertices={len(mesh.vertices)}, triangles={len(mesh.triangles)}")
        o3d.visualization.draw_geometries([mesh], window_name="PLY Viewer (mesh)")
        return

    print(f"Loaded point cloud: points={len(geom.points)}")
    if len(geom.colors) > 0:
        print("Point cloud has RGB colors.")
    else:
        print("Point cloud has no colors.")

    if voxel and voxel > 0:
        before = len(geom.points)
        geom = geom.voxel_down_sample(voxel_size=voxel)
        after = len(geom.points)
        print(f"Downsampled with voxel={voxel}: {before} -> {after}")

    if len(geom.normals) == 0:
        geom.estimate_normals()

    o3d.visualization.draw_geometries(
        [geom],
        window_name="PLY Viewer (point cloud)",
        point_show_normal=False
    )

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ply_visualization.py /path/to/file.ply [voxel_size]")
        sys.exit(0)
    ply_path = sys.argv[1]
    voxel_size = float(sys.argv[2]) if len(sys.argv) >= 3 else 0.0
    main(ply_path, voxel_size)
