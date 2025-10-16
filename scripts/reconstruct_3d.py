import numpy as np
import cv2

def reconstruct_3d(keypoints_all, descriptors_all, matches_all, K):
    points3D = []
    camera_poses = [np.eye(4)]  # First camera at the origin

    for i, matches in enumerate(matches_all):
        # Get matched keypoints
        src_pts = np.array([keypoints_all[i][m.queryIdx].pt for m in matches], dtype=np.float32)
        dst_pts = np.array([keypoints_all[i+1][m.trainIdx].pt for m in matches], dtype=np.float32)

        # Find Essential matrix
        E, mask = cv2.findEssentialMat(src_pts, dst_pts, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

        # Recover relative pose
        _, R, t, _ = cv2.recoverPose(E, src_pts, dst_pts, K)

        # Build current camera pose in world coordinates
        rel_pose = np.eye(4)
        rel_pose[:3, :3] = R
        rel_pose[:3, 3] = t.ravel()

        # Compose with previous pose to get absolute pose
        curr_pose = camera_poses[i] @ rel_pose
        camera_poses.append(curr_pose)

        # Projection matrices (use intrinsics)
        P1 = K @ camera_poses[i][:3, :]
        P2 = K @ curr_pose[:3, :]

        # Triangulate
        points_3d_hom = cv2.triangulatePoints(P1, P2, src_pts.T, dst_pts.T)
        points_3d = (points_3d_hom[:3] / points_3d_hom[3]).T  # Nx3

        points3D.append(points_3d)

    return np.vstack(points3D), camera_poses