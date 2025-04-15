"""
scan_matching.py

This script demonstrates how to perform ICP (Iterative Closest Point) scan matching
between consecutive LiDAR scans, using an initial guess from odometry data.
We intentionally swap the 'source' and 'target' inputs to the ICP function
to observe how it affects the resulting trajectory (i.e., to see if it flips).

Steps:
1. Load LiDAR data (Hokuyo) and convert polar coordinates to Cartesian points.
2. Load and compute odometry data (encoder + IMU) for an initial guess.
3. For each pair of consecutive scans, compute an initial relative transform from odometry,
   then run ICP to refine the transform.
4. Accumulate the refined transform to build the global trajectory.
5. Plot both the raw odometry trajectory and the ICP-refined trajectory for comparison.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from tqdm import tqdm

# Import odometry utility functions (loading sensor data, computing velocity, integrating)
from odometry import load_sensor_data, compute_velocity_and_yawrate, integrate_odometry

def load_hokuyo_data(dataset_id=20):
    """
    Load Hokuyo (LiDAR) data from a .npz file.
    Returns angle parameters, range parameters, all scans, and their timestamps.
    """
    hokuyo_file = f"../data/Hokuyo{dataset_id}.npz"
    data = np.load(hokuyo_file)
    angle_min = data["angle_min"].item()          # Minimum angle of the scan
    angle_increment = data["angle_increment"].item()  # Angular increment between each measurement
    range_min = data["range_min"].item()          # Minimum valid range
    range_max = float(data["range_max"].item())   # Maximum valid range (converted to float)
    ranges_all = data["ranges"]                   # Shape: (1081, number_of_scans)
    hokuyo_time_stamps = data["time_stamps"]      # Shape: (number_of_scans,)
    print(f"[INFO] Loaded {ranges_all.shape[1]} LiDAR scans.")
    return angle_min, angle_increment, range_min, range_max, ranges_all, hokuyo_time_stamps

def polar_to_cartesian(ranges, angle_min, angle_increment, range_min, range_max):
    """
    Convert a single LiDAR scan from polar to Cartesian coordinates,
    discarding values outside the valid range.
    """
    angles = angle_min + np.arange(len(ranges)) * angle_increment
    mask = (ranges > range_min) & (ranges < range_max)
    valid_ranges = ranges[mask]
    valid_angles = angles[mask]
    x = valid_ranges * np.cos(valid_angles)
    y = valid_ranges * np.sin(valid_angles)
    return np.column_stack((x, y))

def compute_rigid_transform_2d(src, dst):
    """
    Given corresponding 2D point sets (src and dst),
    compute the best rigid transform (R, t) such that dst ≈ R*src + t.
    Uses SVD to solve for the transformation.
    """
    centroid_src = np.mean(src, axis=0)
    centroid_dst = np.mean(dst, axis=0)
    src_centered = src - centroid_src
    dst_centered = dst - centroid_dst
    H = src_centered.T @ dst_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    # If det(R) < 0, flip the last row of Vt to correct for a reflection
    if np.linalg.det(R) < 0:
        Vt[1, :] *= -1
        R = Vt.T @ U.T
    t = centroid_dst - R @ centroid_src
    return R, t

def icp_2d(source, target, init_R=np.eye(2), init_t=np.zeros(2),
           max_iterations=50, tolerance=1e-6, distance_threshold=0.1):
    """
    Perform ICP (Iterative Closest Point) in 2D:
    - source: Nx2 array of source points
    - target: Mx2 array of target points
    - init_R, init_t: initial guess for the rigid transform
    - max_iterations, tolerance, distance_threshold: ICP parameters

    Returns:
      - R_total: final rotation (2x2)
      - t_total: final translation (2,)
      - final_error: mean error of inlier correspondences
    """
    R_total = init_R.copy()
    t_total = init_t.copy()
    # Transform source points by the initial guess
    source_transformed = (R_total @ source.T).T + t_total

    prev_error = float('inf')
    tree = cKDTree(target)  # Build a KD-tree for the target point cloud

    for i in range(max_iterations):
        # Find nearest neighbors from the target for each point in source_transformed
        distances, indices = tree.query(source_transformed)
        # Filter out correspondences that exceed the distance threshold
        mask = distances < distance_threshold
        if np.sum(mask) < 3:
            # Not enough inliers, break out
            break
        src_inliers = source_transformed[mask]
        dst_inliers = target[indices[mask]]

        # Compute the incremental rigid transform between these inlier sets
        R_delta, t_delta = compute_rigid_transform_2d(src_inliers, dst_inliers)

        # Update the total rotation and translation
        R_total = R_delta @ R_total
        t_total = R_delta @ t_total + t_delta

        # Apply the updated transform to the original source
        source_transformed = (R_total @ source.T).T + t_total

        # Compute mean error
        error = np.mean(distances[mask])
        if abs(prev_error - error) < tolerance:
            # Converged
            break
        prev_error = error

    return R_total, t_total, error

def pose_to_transform(x, y, theta):
    """
    Convert (x, y, theta) to a 3x3 homogeneous transform matrix in 2D.
    """
    return np.array([[np.cos(theta), -np.sin(theta), x],
                     [np.sin(theta),  np.cos(theta), y],
                     [0, 0, 1]])

def transform_to_pose(T):
    """
    Convert a 3x3 homogeneous transform matrix to (x, y, theta).
    """
    x = T[0, 2]
    y = T[1, 2]
    theta = np.arctan2(T[1, 0], T[0, 0])
    return x, y, theta

def relative_transform_from_poses(pose1, pose2):
    """
    Given two absolute poses (x, y, theta),
    compute the relative transform that takes pose1 to pose2.
    """
    T1 = pose_to_transform(*pose1)
    T2 = pose_to_transform(*pose2)
    T_rel = np.linalg.inv(T1) @ T2
    return T_rel[:2, :2], T_rel[:2, 2]

def main():
    # Choose the dataset ID (e.g., 20 or 21)
    dataset_id = 21

    # 1) Load LiDAR data
    angle_min, angle_increment, range_min, range_max, ranges_all, hokuyo_time_stamps = load_hokuyo_data(dataset_id)
    num_scans = ranges_all.shape[1]

    # 2) Load and compute odometry data
    encoder_counts, encoder_stamps, imu_angular_velocity, imu_stamps = load_sensor_data(dataset_id)
    t_all, v_all, w_all, dt_all = compute_velocity_and_yawrate(
        encoder_counts, encoder_stamps, imu_angular_velocity, imu_stamps,
        ticks_to_meter=0.0022
    )
    X, Y, THETA = integrate_odometry(dt_all, v_all, w_all, x0=0.0, y0=0.0, theta0=0.0)
    encoder_times = encoder_stamps

    # Helper function to get the odometry pose at a given LiDAR timestamp
    def get_odometry_pose(time_stamp):
        idx = np.argmin(np.abs(encoder_times - time_stamp))
        return (X[idx], Y[idx], THETA[idx])

    # Build the odometry trajectory by querying each LiDAR timestamp
    odom_trajectory = np.array([get_odometry_pose(t) for t in hokuyo_time_stamps])

    # 3) Use ICP to refine consecutive LiDAR scans, accumulating the global poses
    icp_global_poses = [get_odometry_pose(hokuyo_time_stamps[0])]  # Initialize with the first pose from odometry

    for i in tqdm(range(num_scans - 1)):
        # Extract consecutive scans
        scan1 = ranges_all[:, i]
        scan2 = ranges_all[:, i+1]
        # Convert polar to Cartesian
        points1 = polar_to_cartesian(scan1, angle_min, angle_increment, range_min, range_max)
        points2 = polar_to_cartesian(scan2, angle_min, angle_increment, range_min, range_max)

        # Get the initial guess from odometry (pose1 -> pose2)
        pose1 = get_odometry_pose(hokuyo_time_stamps[i])
        pose2 = get_odometry_pose(hokuyo_time_stamps[i+1])
        init_R, init_t = relative_transform_from_poses(pose1, pose2)

        # *** Modified here: swap source and target for ICP to see if it flips the trajectory ***
        # Original: R_icp, t_icp, error = icp_2d(points1, points2, init_R, init_t)
        # Now we do: R_icp, t_icp, error = icp_2d(points2, points1, init_R, init_t)
        R_icp, t_icp, error = icp_2d(points2, points1,
                                     init_R=init_R, init_t=init_t,
                                     max_iterations=50,
                                     tolerance=1e-6,
                                     distance_threshold=0.1)

        # Convert the resulting R, t into a 3x3 homogeneous matrix
        T_icp = np.array([[R_icp[0, 0], R_icp[0, 1], t_icp[0]],
                          [R_icp[1, 0], R_icp[1, 1], t_icp[1]],
                          [0,          0,          1 ]])

        # Accumulate with the previous global pose
        T_prev = pose_to_transform(*icp_global_poses[i])
        T_new = T_prev @ T_icp
        refined_pose = transform_to_pose(T_new)
        icp_global_poses.append(refined_pose)

    icp_global_poses = np.array(icp_global_poses)

    # 4) Plot the odometry trajectory vs. the ICP-refined trajectory
    plt.figure()
    plt.plot(odom_trajectory[:, 0], odom_trajectory[:, 1],
             'r--', marker='o', markersize=2, label='Odometry Trajectory')
    plt.plot(icp_global_poses[:, 0], icp_global_poses[:, 1],
             'b-', marker='.', markersize=2, label='ICP Refined Trajectory')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title(f'Odometry & ICP Refined Trajectories for dataset {dataset_id}')
    plt.axis('equal')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
