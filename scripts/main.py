import cv2
import numpy as np
import extract_features
import reconstruct_3d
import mask_extraction
import voxel_carving
import utils

def main(image_paths, K, grid_size=50, voxel_length=0.01):
    """
    Main pipeline for 3D reconstruction and voxel carving from input images.
    
    Args:
        image_paths (list[str]): List of image file paths.
        K (np.ndarray): Camera intrinsics matrix.
        grid_size (int): Number of voxels per axis in the grid.
        voxel_length (float): Physical size of each voxel.

    Returns:
        points3D (np.ndarray): Reconstructed 3D points (point cloud).
        camera_poses (list): Camera extrinsic parameters for each image.
        carved_voxels (np.ndarray): Final carved voxel grid (binary).
    """

    # Step 0: Background removal (segmentation)
    images = [cv2.imread(p) for p in image_paths]
    masks = [mask_extraction.deeplabv3_segmentation(p) for p in image_paths]

    # Step 1: Feature extraction and matching
    keypoints_all, descriptors_all, matches_all = extract_features.extract_features_and_match(image_paths)

    # Step 2: 3D reconstruction from features
    points3D, camera_poses = reconstruct_3d.reconstruct_3d(keypoints_all, descriptors_all, matches_all, K)
    
    utils.save_camera_data(camera_poses, points3D, "output/reconstruction_data.json")
    utils.visualize_3d(points3D)

    # Step 3: Voxel grid creation and carving
    voxels, coords = voxel_carving.create_voxel_grid(grid_size=grid_size, voxel_length=voxel_length)
    carved_voxels = voxel_carving.carve_voxels(voxels, coords, images, masks, camera_poses, K)

    voxel_carving.visualize_voxels(carved_voxels, voxel_length=voxel_length)

    return points3D, camera_poses, carved_voxels



if __name__ == "__main__":
    image_paths = [
        "placeholder.png", 
        "placeholder.png"
    ]

    # Camera intrinsic matrix K (for now)
    K = np.array([[800, 0, 320],
                  [0, 800, 240],
                  [0, 0, 1]], dtype=np.float64)

    points3D, camera_poses, carved_voxels = main(image_paths, K)
