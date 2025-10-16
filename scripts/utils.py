import json
import numpy as np

def save_camera_data(camera_poses, points3D, output_file):
    data = {
        'camera_poses': [pose.tolist() for pose in camera_poses],
        'points3D': points3D.tolist()
    }

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)

def load_camera_data(input_file):
    with open(input_file, 'r') as f:
        data = json.load(f)
    camera_poses = [np.array(pose) for pose in data['camera_poses']]
    points3D = np.array(data['points3D'])
    return camera_poses, points3D

def visualize_3d(points3D):
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points3D)
    o3d.visualization.draw_geometries([pcd])  # type: ignore