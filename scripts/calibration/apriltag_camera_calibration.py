#!/usr/bin/env python3
"""
AprilTag-based Camera Calibration Script for hand-eye calibration
"""

import numpy as np
import cv2
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize
import argparse
import os
import json


PERM_CANDS = [
    [0,1,2,3],
    [1,2,3,0],
    [2,3,0,1],
    [3,0,1,2],
    [0,3,2,1],
    [3,2,1,0],
    [2,1,0,3],
    [1,0,3,2],
]

def order_corners_tl_tr_br_bl(corners_4x2: np.ndarray) -> np.ndarray:
    """
    Returns corners reordered to [TL, TR, BR, BL] in IMAGE coordinates.
    corners_4x2: (4,2) float
    """
    c = corners_4x2
    s = c[:, 0] + c[:, 1]      # x + y
    d = c[:, 0] - c[:, 1]      # x - y

    tl = np.argmin(s)
    br = np.argmax(s)
    tr = np.argmax(d)
    bl = np.argmin(d)

    return c[[tl, tr, br, bl]]


def reproj_rms_for_perm(obj_points_groups, img_points_groups, perm, K, dist):
    """Solve PnP for a given corner permutation using ITERATIVE solver.

    Returns (rms, max_err, rvec, tvec).
    """
    OP = np.concatenate(obj_points_groups, axis=0).astype(np.float32)
    IP = np.concatenate([c[perm] for c in img_points_groups], axis=0).astype(np.float32)

    ok, rvec, tvec = cv2.solvePnP(OP, IP, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return np.inf, np.inf, None, None

    proj, _ = cv2.projectPoints(OP, rvec, tvec, K, dist)
    proj = proj.reshape(-1, 2)
    err = np.linalg.norm(proj - IP, axis=1)
    rms = float(np.sqrt(np.mean(err ** 2)))
    mx = float(err.max())
    return rms, mx, rvec, tvec


class AprilTagCameraCalibration:
    def __init__(self, tag_size, tag_family='tag36h11'):
        """
        Initialize the calibration object.
        
        Args:
            tag_size: Size of the AprilTag in meters
            tag_family: AprilTag family (default: tag36h11)
        """
        self.tag_size = tag_size
        self.tag_family = tag_family
        
        # Try to import apriltag library
        try:
            import apriltag
            self.detector = apriltag.Detector(apriltag.DetectorOptions(families=tag_family))
            print(f"Using apriltag library (python apriltag)")
        except Exception as e:
            print(f"Warning: apriltag library failed ({type(e).__name__}: {e}). Using cv2.aruco instead.")
            self.detector = None
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
            self.aruco_params = cv2.aruco.DetectorParameters()
        
        # Storage for observations
        self.observations = []

    def detect_apriltag(self, image, camera_matrix, dist_coeffs, return_corners=False):
        """
        Returns:
            T_tag_cam: 4x4 transform tag->camera (camera frame consistent with OpenCV: x right, y down, z forward)
            tag_id
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # --- detect corners ---
        if self.detector is not None:
            detections = self.detector.detect(gray)
            if len(detections) == 0:
                return None, None
            detection = detections[0]
            tag_id = detection.tag_id
            corners = detection.corners.reshape((4, 2)).astype(np.float32)
        else:
            detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
            corners_list, ids, _ = detector.detectMarkers(gray)
            if ids is None or len(ids) == 0:
                return None, None
            tag_id = int(ids[0][0])
            corners = corners_list[0].reshape((4, 2)).astype(np.float32)

        # --- object points for a square ---
        half = self.tag_size / 2.0
        obj = np.array([
            [-half, -half, 0.0],
            [ half, -half, 0.0],
            [ half,  half, 0.0],
            [-half,  half, 0.0],
        ], dtype=np.float32)

        # --- get BOTH IPPE solutions ---
        try:
            ok, rvecs, tvecs, reprojErrs = cv2.solvePnPGeneric(
                obj, corners, camera_matrix, dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )
        except TypeError:
            # Older OpenCV sometimes returns different tuple shapes
            out = cv2.solvePnPGeneric(
                obj, corners, camera_matrix, dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )
            ok = out[0]; rvecs = out[1]; tvecs = out[2]
            reprojErrs = out[3] if len(out) > 3 else None

        if not ok or rvecs is None or len(rvecs) == 0:
            return None, None

        # Build candidate transforms
        cands = []
        for i in range(len(rvecs)):
            rvec = rvecs[i]
            tvec = tvecs[i]
            R, _ = cv2.Rodrigues(rvec)
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = np.array(tvec).reshape(3)

            err = float(reprojErrs[i]) if reprojErrs is not None else 0.0
            cands.append((T, err))

        # --- cheirality / plausibility filter: tag must be in front of camera ---
        # (OpenCV camera convention: +Z forward)
        cands_front = [(T, err) for (T, err) in cands if T[2, 3] > 0.05]  # 5 cm
        if len(cands_front) == 0:
            # If everything fails cheirality, fall back to smallest reproj error
            cands_front = cands

        # --- choose stable solution ---
        if getattr(self, "last_T_tag_cam", None) is None:
            # First time: choose smallest reprojection error
            T_best = min(cands_front, key=lambda x: x[1])[0]
        else:
            T_prev = self.last_T_tag_cam

            def cost(T):
                # translation continuity
                dtrans = np.linalg.norm(T[:3, 3] - T_prev[:3, 3])
                # rotation continuity
                R_rel = Rotation.from_matrix(T[:3, :3]) * Rotation.from_matrix(T_prev[:3, :3]).inv()
                dang = np.linalg.norm(R_rel.as_rotvec())
                return 1.0 * dtrans + 0.3 * dang  # tune weights if needed

            # Also reject extreme jumps when camera is not moving (optional but helpful)
            T_best = min(cands_front, key=lambda x: cost(x[0]))[0]

            # Hard gate to prevent flip-flop if you are stationary
            R_rel = Rotation.from_matrix(T_best[:3, :3]) * Rotation.from_matrix(T_prev[:3, :3]).inv()
            dang = np.linalg.norm(R_rel.as_rotvec())
            dtrans = np.linalg.norm(T_best[:3, 3] - T_prev[:3, 3])
            if dang > np.deg2rad(90) and dtrans > 0.05:
                # likely planar flip or bogus detection; keep previous
                T_best = T_prev

        self.last_T_tag_cam = T_best.copy()
        if return_corners:
            return T_best, tag_id, corners
        return T_best, tag_id

    def detect_apriltag_gridboard(self, image, camera_matrix, dist_coeffs,
                                rows, cols, tag_size, tag_spacing,
                                id_to_rc=None,
                                min_tags=2):
        """
        Estimate board pose from multiple AprilTags (python apriltag detector) by stacking all corners.
        Returns:
            T_board_cam (board->camera), used_tag_ids, corners_by_id (optional)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        if self.detector is not None:
            detections = self.detector.detect(gray)
            if len(detections) == 0:
                return None, None
            det_list = [(int(d.tag_id), d.corners.reshape((4, 2)).astype(np.float32))
                        for d in detections]
        else:
            detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
            corners_list, ids, _ = detector.detectMarkers(gray)
            if ids is None or len(ids) == 0:
                return None, None
            det_list = [(int(ids[i][0]), corners_list[i].reshape((4, 2)).astype(np.float32))
                        for i in range(len(ids))]

        pitch = tag_size + tag_spacing
        half = tag_size / 2.0

        obj_points = []
        img_points = []
        used_ids = []
        corners_by_id = {}

        for tid, corners in det_list:

            # map tag_id to (row, col)
            if id_to_rc is None:
                r = tid // cols
                c = tid % cols
            else:
                if tid not in id_to_rc:
                    continue
                r, c = id_to_rc[tid]

            # ignore tags outside board index range
            if not (0 <= r < rows and 0 <= c < cols):
                continue
            corners_by_id[tid] = corners

            cx = c * pitch
            cy = r * pitch

            obj = np.array([
                [cx - half, cy - half, 0.0],  # TL
                [cx + half, cy - half, 0.0],  # TR
                [cx + half, cy + half, 0.0],  # BR
                [cx - half, cy + half, 0.0],  # BL
            ], dtype=np.float32)

            obj_points.append(obj)
            img_points.append(corners)
            used_ids.append(tid)

        if len(obj_points) < min_tags:
            return None, None

        # choose best perm
        best = (np.inf, None, None, None, None)  # rms, perm, mx, rvec, tvec
        for perm in PERM_CANDS:
            rms, mx, rvec, tvec = reproj_rms_for_perm(obj_points, img_points, perm, camera_matrix, dist_coeffs)
            if rms < best[0]:
                best = (rms, perm, mx, rvec, tvec)

        best_rms, best_perm, best_mx, rvec, tvec = best

        if rvec is None:
            return None, None

        R, _ = cv2.Rodrigues(rvec)
        T_board_cam = np.eye(4)
        T_board_cam[:3, :3] = R
        T_board_cam[:3, 3] = tvec.reshape(3)

        z_dist = float(tvec.reshape(-1)[2])

        if best_mx > 5:
            print(f"[board reproj] REJECTED  max reproj {best_mx:.2f}px > 5px")
            return None, None

        # Cheirality check: board must be in front of camera
        if z_dist <= 0.05:
            print(f"[board pose] REJECTED  z={z_dist:.4f}m (board not in front of camera)")
            return None, None

        return T_board_cam, {
            "tag_ids": used_ids,
            "corners_by_id": corners_by_id,
            "reproj_rms": best_rms,
            "reproj_max": best_mx,
            "num_tags": len(used_ids),
            "best_perm": list(best_perm),
        }

def load_camera_params(intrinsics_file):
    """
    Load camera intrinsics from file.
    
    Args:
        intrinsics_file: Path to camera intrinsics file (.npz or .json)
        
    Returns:
        camera_matrix: 3x3 camera matrix
        dist_coeffs: Distortion coefficients
    """
    ext = os.path.splitext(intrinsics_file)[1]
    
    if ext == '.npz':
        data = np.load(intrinsics_file)
        camera_matrix = data['camera_matrix']
        dist_coeffs = data['dist_coeffs']
    elif ext == '.json':
        with open(intrinsics_file, 'r') as f:
            data = json.load(f)
        camera_matrix = np.array(data['camera_matrix'])
        dist_coeffs = np.array(data['dist_coeffs'])
    else:
        raise ValueError(f"Unsupported file extension: {ext}")
    
    return camera_matrix, dist_coeffs


def main():
    parser = argparse.ArgumentParser(description='AprilTag Camera Calibration')
    parser.add_argument('--tag-size', type=float, required=True,
                        help='Size of the AprilTag in meters')
    parser.add_argument('--tag-family', type=str, default='tag36h11',
                        help='AprilTag family (default: tag36h11)')
    parser.add_argument('--cam1-intrinsics', type=str, required=True,
                        help='Path to camera1 intrinsics file')
    parser.add_argument('--cam2-intrinsics', type=str, required=True,
                        help='Path to camera2 intrinsics file')
    parser.add_argument('--output', type=str, default='camera_transform.npz',
                        help='Output file for calibration result')
    parser.add_argument('--method', type=str, default='mean',
                        choices=['mean', 'optimize'],
                        help='Method to combine observations')
    
    args = parser.parse_args()
    
    # Load camera intrinsics
    print("Loading camera intrinsics...")
    cam1_matrix, cam1_dist = load_camera_params(args.cam1_intrinsics)
    cam2_matrix, cam2_dist = load_camera_params(args.cam2_intrinsics)
    
    # Initialize calibration
    calib = AprilTagCameraCalibration(args.tag_size, args.tag_family)
    
    print("\n=== AprilTag Camera Calibration ===")
    print(f"Tag size: {args.tag_size} m")
    print(f"Tag family: {args.tag_family}")
    print("\nInstructions:")
    print("1. Place the AprilTag in view of both cameras")
    print("2. Press SPACE to capture an observation")
    print("3. Move the tag to a different position and repeat")
    print("4. Press 'q' to finish and compute the transform")
    print("5. Aim for at least 10-20 observations for good results\n")
    
    # Open cameras
    cap1 = cv2.VideoCapture(0)  # Adjust camera indices as needed
    cap2 = cv2.VideoCapture(1)
    
    if not cap1.isOpened() or not cap2.isOpened():
        print("Error: Could not open cameras")
        print("Note: You may need to adjust camera indices in the script")
        print("      or use a different method to load images")
        return
    
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if not ret1 or not ret2:
            print("Error reading from cameras")
            break
        
        # Display frames
        cv2.imshow('Camera 1', frame1)
        cv2.imshow('Camera 2', frame2)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):
            # Capture observation
            print("\nCapturing observation...")
            success = calib.add_observation(
                frame1, frame2,
                cam1_matrix, cam1_dist,
                cam2_matrix, cam2_dist
            )
            if success:
                print(f"Total observations: {len(calib.observations)}")
        
        elif key == ord('q'):
            # Finish calibration
            break
    
    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()
    
    # Compute final transform
    if len(calib.observations) == 0:
        print("\nNo observations captured. Exiting.")
        return
    
    print(f"\n\nComputing transform from {len(calib.observations)} observations...")
    T_cam1_cam2 = calib.compute_transform(method=args.method)
    
    # Display result
    print("\n=== Calibration Result ===")
    print("Transform from camera2 to camera1:")
    print(T_cam1_cam2)
    print("\nRotation matrix:")
    print(T_cam1_cam2[:3, :3])
    print("\nTranslation vector (meters):")
    print(T_cam1_cam2[:3, 3])
    
    # Display statistics
    stats = calib.get_statistics()
    if stats:
        print("\n=== Statistics ===")
        print(f"Number of observations: {stats['num_observations']}")
        print(f"Translation std dev: {stats['translation_std']:.2f} mm")
        print(f"Translation max error: {stats['translation_max_error']:.2f} mm")
        print(f"Rotation std dev: {stats['rotation_std']:.2f} degrees")
        print(f"Rotation max error: {stats['rotation_max_error']:.2f} degrees")
    
    # Save result
    calib.save_result(args.output, T_cam1_cam2)
    print(f"\nCalibration complete! Result saved to {args.output}")


if __name__ == '__main__':
    main()
