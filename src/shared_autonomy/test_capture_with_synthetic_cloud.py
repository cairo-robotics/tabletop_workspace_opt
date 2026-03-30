#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test version of manual_capture_and_fuse_real.py that generates synthetic point clouds
for testing when the RealSense camera is not available or has driver issues.

This allows testing the capture pipeline without hardware.
"""

import os
import glob
import numpy as np
import open3d as o3d
import rospy

from auto_scan_ring_and_fuse import ensure_dir, fuse_scans


def generate_synthetic_cloud(center, radius=0.3, num_points=50000):
    """Generate a synthetic point cloud (sphere with noise)."""
    # Sphere
    u = np.random.uniform(0, 2 * np.pi, num_points)
    v = np.random.uniform(0, np.pi, num_points)
    x = radius * np.cos(u) * np.sin(v) + center[0]
    y = radius * np.sin(u) * np.sin(v) + center[1]
    z = radius * np.cos(v) + center[2]
    
    # Add noise
    noise = np.random.normal(0, 0.01, (num_points, 3))
    points = np.column_stack([x, y, z]) + noise
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


class SyntheticCapture:
    def __init__(self):
        self.out_dir = os.path.expanduser("~/scans_manual_synthetic")
        ensure_dir(self.out_dir)
        self.capture_count = self._find_next_index()
        rospy.loginfo("[synthetic_capture] output dir: %s, next_index: %d", self.out_dir, self.capture_count)
    
    def _find_next_index(self):
        files = sorted(glob.glob(os.path.join(self.out_dir, "scan_*.ply")))
        if not files:
            return 0
        last_name = os.path.basename(files[-1])
        try:
            prefix = last_name.split("_")[1]
            return int(prefix) + 1
        except Exception:
            return len(files)
    
    def capture_once(self, center_offset=None):
        """Capture one synthetic cloud and save it."""
        if center_offset is None:
            center_offset = [0, 0, 0]
        
        center = np.array([0.5, 0, 0.3]) + np.array(center_offset)
        pcd = generate_synthetic_cloud(center)
        
        out_path = os.path.join(self.out_dir, f"scan_{self.capture_count:04d}.ply")
        o3d.io.write_point_cloud(out_path, pcd)
        rospy.loginfo("[synthetic_capture] saved %s (points=%d)", out_path, len(pcd.points))
        self.capture_count += 1
        return out_path
    
    def fuse(self):
        """Fuse all saved scans."""
        files = sorted(glob.glob(os.path.join(self.out_dir, "scan_*.ply")))
        if len(files) < 2:
            rospy.logwarn("[synthetic_capture] need at least 2 scans, have %d", len(files))
            return None, None
        
        rospy.loginfo("[synthetic_capture] fusing %d scans...", len(files))
        before_path, after_path = fuse_scans(
            ply_dir=self.out_dir,
            pattern="scan_*.ply",
            out_before="merged_posegraph_before_drop.ply",
            out_after="merged_posegraph_after_drop.ply",
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
            verbose=True,
        )
        rospy.loginfo("[synthetic_capture] fusion done. before=%s after=%s", before_path, after_path)
        return before_path, after_path
    
    @staticmethod
    def print_help():
        print(
            "\nSynthetic Capture Controls (no RealSense required)\n"
            "  s : save one synthetic point-cloud frame\n"
            "  f : fuse saved frames\n"
            "  p : print saved-scan count\n"
            "  h : print this help\n"
            "  q : quit\n"
            "\nNote: Press 's' multiple times with small offsets to generate\n"
            "multi-view scans from different viewpoints.\n"
        )
    
    def run(self):
        self.print_help()
        print(f"Saving into {self.out_dir}")
        
        view_angle = 0
        while not rospy.is_shutdown():
            key = input("Command> ").lower().strip()
            
            if key == "s":
                try:
                    # Rotate around the object
                    offset = [0.2 * np.cos(np.radians(view_angle)), 
                              0.2 * np.sin(np.radians(view_angle)), 
                              0]
                    self.capture_once(offset)
                    view_angle += 45  # Rotate by 45 degrees each capture
                except Exception as exc:
                    rospy.logwarn("[synthetic_capture] capture failed: %s", str(exc))
            elif key == "f":
                try:
                    self.fuse()
                except Exception as exc:
                    rospy.logwarn("[synthetic_capture] fusion failed: %s", str(exc))
            elif key == "p":
                print(f"Saved scans: {self.capture_count}")
            elif key == "h":
                self.print_help()
            elif key == "q":
                rospy.loginfo("[synthetic_capture] quitting.")
                return
            else:
                print("Unknown command. Press 'h' for help.")


def main():
    node = SyntheticCapture()
    node.run()


if __name__ == "__main__":
    rospy.init_node("synthetic_capture_and_fuse")
    main()
