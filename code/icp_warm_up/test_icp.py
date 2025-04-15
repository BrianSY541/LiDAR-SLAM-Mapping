import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from utils import read_canonical_model, load_pc, visualize_icp_result

def voxel_downsample(pc, voxel_size=0.02):
    """
    Downsample the point cloud using Open3D voxel downsampling.
    pc: (N, 3) numpy array representing the point cloud.
    voxel_size: size of the voxel (in meters).
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
    return np.asarray(pcd_down.points)

def compute_rigid_transform_3d(src, dst):
    """
    Given corresponding 3D point pairs (src[i], dst[i]), solve for the optimal rigid transform R and t using SVD.
    Returns R (3x3) and t (3,), such that dst ≈ R @ src + t.
    """
    centroid_src = np.mean(src, axis=0)
    centroid_dst = np.mean(dst, axis=0)
    src_centered = src - centroid_src
    dst_centered = dst - centroid_dst
    H = src_centered.T @ dst_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = centroid_dst - R @ centroid_src
    return R, t

def apply_transform(pc, R, t):
    """Apply the rigid transform R, t to the point cloud pc (N, 3)."""
    return (R @ pc.T).T + t

def icp_3d(source, target, max_iterations=50, distance_threshold=0.1, tolerance=1e-6, min_inliers=3):
    """
    3D ICP algorithm:
      - source, target: (N, 3) and (M, 3) point clouds.
      - distance_threshold: initial outlier threshold (in meters).
      - If the number of inliers is insufficient, the threshold is dynamically increased.
    Returns: R_total, t_total, mean_error.
    """
    kdtree = cKDTree(target)
    R_total = np.eye(3)
    t_total = np.zeros(3)
    prev_error = float('inf')
    src_transformed = source.copy()
    curr_threshold = distance_threshold
    for i in range(max_iterations):
        distances, indices = kdtree.query(src_transformed, k=1)
        mask = distances < curr_threshold
        inlier_src = src_transformed[mask]
        inlier_dst = target[indices[mask]]
        if len(inlier_src) < min_inliers:
            curr_threshold *= 1.5
            print(f"Warning: Insufficient inliers, increasing threshold to {curr_threshold:.3f}")
            # If even after expanding the threshold there are still insufficient inliers,
            # skip this iteration and wait for the next one.
            continue
        R, t = compute_rigid_transform_3d(inlier_src, inlier_dst)
        R_total = R @ R_total
        t_total = R @ t_total + t
        src_transformed = apply_transform(source, R_total, t_total)
        mean_error = np.mean(distances[mask])
        if abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error
    return R_total, t_total, prev_error

def multi_scale_icp(source, target):
    """
    Multi-scale ICP:
      1. Coarse alignment: Use voxel downsampling with a larger voxel size (voxel_size=0.05)
         and a looser distance threshold (0.2 m).
      2. Fine alignment: Apply the coarse alignment result to the original source point cloud,
         then use a smaller voxel size (voxel_size=0.01) and a tighter threshold (0.05 m) for refinement.
    Also print bounding box information to verify the scale and units.
    """
    print("Source bounding box:")
    print("min:", np.min(source, axis=0))
    print("max:", np.max(source, axis=0))
    print("Target bounding box:")
    print("min:", np.min(target, axis=0))
    print("max:", np.max(target, axis=0))
    
    # Coarse alignment stage: perform voxel downsampling for a rough estimation.
    src_coarse = voxel_downsample(source, voxel_size=0.05)
    tgt_coarse = voxel_downsample(target, voxel_size=0.05)
    print(f"Coarse ICP: source points: {src_coarse.shape[0]}, target points: {tgt_coarse.shape[0]}")
    R_coarse, t_coarse, err_coarse = icp_3d(src_coarse, tgt_coarse,
                                              max_iterations=30,
                                              distance_threshold=0.2,
                                              tolerance=1e-6,
                                              min_inliers=3)
    # Apply the coarse alignment result to the full original source point cloud.
    source_aligned = apply_transform(source, R_coarse, t_coarse)
    
    # Fine alignment stage: downsample using a smaller voxel size to achieve a more precise alignment.
    src_fine = voxel_downsample(source_aligned, voxel_size=0.01)
    tgt_fine = voxel_downsample(target, voxel_size=0.01)
    print(f"Fine ICP: source points: {src_fine.shape[0]}, target points: {tgt_fine.shape[0]}")
    R_fine, t_fine, err_fine = icp_3d(src_fine, tgt_fine,
                                      max_iterations=30,
                                      distance_threshold=0.05,
                                      tolerance=1e-6,
                                      min_inliers=3)
    R_final = R_fine @ R_coarse
    t_final = R_fine @ t_coarse + t_fine
    return R_final, t_final, err_fine

if __name__ == "__main__":
    obj_name = 'drill'  # Options: 'drill' or 'liq_container'
    num_pc = 4
    # Load the canonical model (used as source) and target point clouds (full data, no random downsampling)
    source_pc = read_canonical_model(obj_name)
    for i in range(num_pc):
        target_pc = load_pc(obj_name, i)
        R_final, t_final, final_error = multi_scale_icp(source_pc, target_pc)
        print(f"Point cloud {i} final error: {final_error:.6f}")
        # Form a 4x4 homogeneous transformation matrix for visualization.
        T = np.eye(4)
        T[:3, :3] = R_final
        T[:3, 3] = t_final
        visualize_icp_result(source_pc, target_pc, T)
