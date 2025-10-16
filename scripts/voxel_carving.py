import numpy as np
import cv2
import open3d as o3d

def create_voxel_grid(grid_size=100, voxel_length=0.01):
    """Initialize a cubic voxel grid representing a 3D volume.
    Args:
        grid_size: Number of voxels along each axis (int)
        voxel_length: Physical size of each voxel (float)
    Returns:
        voxels: numpy array (grid_size x grid_size x grid_size) initialized to 1 (solid)
        coords: numpy array (N x 3) with coordinates of each voxel center
    """
    voxels = np.ones((grid_size, grid_size, grid_size), dtype=np.uint8)

    # Create a grid of voxel centers in 3D space, centered at origin
    axis = (np.arange(grid_size) - grid_size / 2) * voxel_length
    X, Y, Z = np.meshgrid(axis, axis, axis, indexing='ij')
    coords = np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T  # Shape (N, 3)
    return voxels, coords

def project_voxels_to_image(voxel_coords, K, R, t):
    """Project voxel centers (3D points) into a 2D image plane using camera intrinsics and extrinsics
    Args:
        voxel_coords: (N,3) array of voxel positions
        K: (3,3) camera intrinsic matrix
        R: (3,3) rotation matrix of camera
        t: (3,1) translation vector of camera
    Returns:
        projected_pixels: (N,2) array of 2D pixel coordinates (float)
    """
    # Transform voxels to camera coordinates
    cam_coords = (R @ voxel_coords.T) + t  # shape (3, N)
    
    # Keep points with positive z (in front of camera)
    valid = cam_coords[2, :] > 0
    cam_coords = cam_coords[:, valid]

    # Project points onto image plane: p = K * (X / Z)
    pts_2d = K @ (cam_coords / cam_coords[2, :])
    pts_2d = pts_2d[:2, :].T  # (M, 2)

    return pts_2d, valid

def carve_voxels(voxels, voxel_coords, images, masks, camera_poses, K):
    """Perform voxel carving from multiple views
    Args:
        voxels: 3D voxel grid (numpy array)
        voxel_coords: (N,3) array of voxel centers
        images: list of color images (optional, for visualization)
        masks: list of binary masks (foreground=1, background=0)
        camera_poses: list of (R, t) camera pose tuples
        K: camera intrinsic matrix
    Returns:
        carved_voxels: updated voxel grid after carving
    """
    grid_size = voxels.shape[0]
    for i, (img, mask, (R, t)) in enumerate(zip(images, masks, camera_poses)):
        print(f"Processing view {i+1}/{len(images)}")
        pts_2d, valid = project_voxels_to_image(voxel_coords, K, R, t)
        valid_voxel_indices = np.where(valid)[0]

        # Round projected pixel coords to nearest integer
        pix_x = np.round(pts_2d[:, 0]).astype(int)
        pix_y = np.round(pts_2d[:, 1]).astype(int)

        height, width = mask.shape
        # Check which pixels project inside image bounds
        inside = (pix_x >= 0) & (pix_x < width) & (pix_y >= 0) & (pix_y < height)

        # For voxels that project outside the image, carve it 
        outside_voxels = valid_voxel_indices[~inside]
        for idx in outside_voxels:
            # Convert flat voxel index to 3D index
            z = idx % grid_size
            y = (idx // grid_size) % grid_size
            x = (idx // (grid_size*grid_size)) % grid_size
            voxels[x, y, z] = 0

        # For voxels inside image, check mask pixel value
        inside_voxels = valid_voxel_indices[inside]
        inside_pix_x = pix_x[inside]
        inside_pix_y = pix_y[inside]

        for v_idx, px, py in zip(inside_voxels, inside_pix_x, inside_pix_y):
            if mask[py, px] == 0:  # background pixel
                z = v_idx % grid_size
                y = (v_idx // grid_size) % grid_size
                x = (v_idx // (grid_size*grid_size)) % grid_size
                voxels[x, y, z] = 0

    return voxels

def visualize_voxels(voxels, voxel_length=0.01):
    """Visualize carved voxel grid using Open3D"""
    grid_size = voxels.shape[0]
    # Get voxel indices where value is 1
    vox_indices = np.array(np.where(voxels == 1)).T  # (M,3)

    # Convert voxel indices to coordinates centered at origin
    coords = (vox_indices - grid_size / 2) * voxel_length

    # Create Open3D voxel grid
    voxel_grid = o3d.geometry.VoxelGrid()

    for coord in coords:
        voxel = o3d.geometry.Voxel(coord, voxel_length)
        voxel_grid.voxels.append(voxel)

    # visualize as point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    o3d.visualization.draw_geometries([pcd])  # type: ignore

# # Example usage
# if __name__ == "__main__":
#     # Step 0: Prepare images and masks (binary segmented images)
#     # images = [cv2.imread(...), ...]
#     # masks = [binary masks of same size as images]

#     grid_size = 50  # smaller for speed; increase if needed
#     voxel_length = 0.01  # adjust scale to your scene units

#     voxels, coords = create_voxel_grid(grid_size=grid_size, voxel_length=voxel_length)

#     # camera_poses = [(R1, t1), (R2, t2), ...]  
#     carved_voxels = carve_voxels(voxels, coords, images, masks, camera_poses, K)

#     visualize_voxels(carved_voxels, voxel_length=voxel_length)
