#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
occupancy_map_icp.py

This program implements occupancy grid mapping of the entire environment using the robot's LiDAR scans and the ICP-corrected robot trajectory.

Main steps:
1. Read sensor data from Hokuyo (LiDAR) and odometry, and perform scan matching using odometry plus ICP to obtain the globally refined poses.
2. For each LiDAR scan, first convert the polar coordinate data into Cartesian points in the robot's coordinate frame, and then transform these points into world coordinates using the ICP pose at that scan.
3. For the points in world coordinates, use the Bresenham algorithm to compute the free cells between the robot's position and each scan point, and update the cell corresponding to the scan point as occupied.
4. Use the log-odds model to accumulate updates in the occupancy map, and finally apply a sigmoid mapping to convert log-odds into occupancy probabilities for visualization.

Adjust the map resolution, range, and log-odds update parameters as needed.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import time
from tqdm import tqdm

# Load existing modules
from odometry import load_sensor_data, compute_velocity_and_yawrate, integrate_odometry
from pr2_utils import bresenham2D, plot_map  # Functions provided in pr2_utils.py

# ------------------------------
# Helper Functions
# ------------------------------

def load_hokuyo_data(dataset_id=20):
    """
    Load Hokuyo LiDAR data
    """
    hokuyo_file = f"../data/Hokuyo{dataset_id}.npz"
    data = np.load(hokuyo_file)
    angle_min = data["angle_min"].item()
    angle_increment = data["angle_increment"].item()  # Note: this is a scalar
    range_min = data["range_min"].item()
    range_max = float(data["range_max"].item())
    ranges_all = data["ranges"]       # shape: (1081, num_scans)
    hokuyo_time_stamps = data["time_stamps"]  # shape: (num_scans,)
    print(f"[INFO] Loaded {ranges_all.shape[1]} LiDAR scans.")
    return angle_min, angle_increment, range_min, range_max, ranges_all, hokuyo_time_stamps

def polar_to_cartesian(ranges, angle_min, angle_increment, range_min, range_max):
    """
    Convert a single LiDAR scan from polar to Cartesian coordinates (only keeping points within valid range)
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
    Given 2D point correspondences, compute the optimal rigid transformation R and t such that dst ≈ R * src + t
    """
    centroid_src = np.mean(src, axis=0)
    centroid_dst = np.mean(dst, axis=0)
    src_centered = src - centroid_src
    dst_centered = dst - centroid_dst
    H = src_centered.T @ dst_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    # Correct for reflection if necessary
    if np.linalg.det(R) < 0:
        Vt[1, :] *= -1
        R = Vt.T @ U.T
    t = centroid_dst - R @ centroid_src
    return R, t

def icp_2d(source, target, init_R=np.eye(2), init_t=np.zeros(2),
           max_iterations=50, tolerance=1e-6, distance_threshold=0.1):
    """
    2D Iterative Closest Point (ICP)
    """
    R_total = init_R.copy()
    t_total = init_t.copy()
    source_transformed = (R_total @ source.T).T + t_total

    prev_error = float('inf')
    tree = cKDTree(target)
    for i in range(max_iterations):
        distances, indices = tree.query(source_transformed)
        mask = distances < distance_threshold
        if np.sum(mask) < 3:
            break
        src_inliers = source_transformed[mask]
        dst_inliers = target[indices[mask]]
        R_delta, t_delta = compute_rigid_transform_2d(src_inliers, dst_inliers)
        R_total = R_delta @ R_total
        t_total = R_delta @ t_total + t_delta
        source_transformed = (R_total @ source.T).T + t_total
        error = np.mean(distances[mask])
        if abs(prev_error - error) < tolerance:
            break
        prev_error = error
    return R_total, t_total, error

def pose_to_transform(x, y, theta):
    """
    Convert (x, y, theta) to a 3x3 homogeneous transformation matrix
    """
    return np.array([[np.cos(theta), -np.sin(theta), x],
                     [np.sin(theta),  np.cos(theta), y],
                     [0, 0, 1]])

def transform_to_pose(T):
    """
    Convert a 3x3 homogeneous transformation matrix to (x, y, theta)
    """
    x = T[0, 2]
    y = T[1, 2]
    theta = np.arctan2(T[1, 0], T[0, 0])
    return x, y, theta

def relative_transform_from_poses(pose1, pose2):
    """
    Compute the relative transformation from pose1 to pose2
    """
    T1 = pose_to_transform(*pose1)
    T2 = pose_to_transform(*pose2)
    T_rel = np.linalg.inv(T1) @ T2
    return T_rel[:2, :2], T_rel[:2, 2]

# ------------------------------
# Occupancy Map Update Functions
# ------------------------------

def init_map():
    """
    Initialize the occupancy grid map and related parameters
    """
    MAP = {}
    MAP['res'] = np.array([0.05, 0.05])    # Size of each cell (meters)
    MAP['min'] = np.array([-20.0, -20.0])  # Lower bound of map range
    MAP['max'] = np.array([20.0, 20.0])    # Upper bound of map range
    MAP['size'] = np.ceil((MAP['max'] - MAP['min']) / MAP['res']).astype(int)
    isEven = MAP['size'] % 2 == 0
    MAP['size'][isEven] = MAP['size'][isEven] + 1  # Ensure a central cell exists
    # Initialize using log-odds; 0 indicates unknown
    MAP['log_odds'] = np.zeros(MAP['size'])
    return MAP

def world_to_map(point, MAP):
    """
    Convert world coordinates to map grid indices
    """
    idx = np.floor((point - MAP['min']) / MAP['res']).astype(int)
    return idx

def update_map(MAP, robot_pose, scan_points, log_free=-0.4, log_occ=0.85):
    """
    Update the occupancy map based on a single scan
      - robot_pose: (x, y, theta), the current ICP refined pose
      - scan_points: (N,2) scan points in Cartesian coordinates in the robot's frame
    1. Transform scan points to world coordinates based on the robot pose.
    2. Use the Bresenham algorithm to compute the grid cells (free cells) along the ray from the robot position to each scan point.
    3. Update the log-odds for free cells and occupied cells.
    """
    # Robot pose transformation matrix
    T = pose_to_transform(*robot_pose)
    
    # Robot position in world coordinates
    robot_world = np.array([robot_pose[0], robot_pose[1]])
    robot_idx = world_to_map(robot_world, MAP)
    
    # Transform all scan points from robot coordinates to world coordinates
    scan_points_world = (T[:2, :2] @ scan_points.T).T + robot_world

    for pt in scan_points_world:
        pt_idx = world_to_map(pt, MAP)
        # Compute all grid cells along the line from robot_idx to pt_idx using Bresenham's algorithm
        ray = bresenham2D(robot_idx[0], robot_idx[1], pt_idx[0], pt_idx[1]).astype(int)
        # Check if the cells are within the map boundaries
        valid = (ray[0, :] >= 0) & (ray[0, :] < MAP['log_odds'].shape[0]) & \
                (ray[1, :] >= 0) & (ray[1, :] < MAP['log_odds'].shape[1])
        ray = ray[:, valid]
        if ray.shape[1] == 0:
            continue

        # Update the cells along the ray (except the last cell) as free
        for i in range(ray.shape[1]-1):
            MAP['log_odds'][ray[0, i], ray[1, i]] += log_free

        # Update the last cell as occupied
        MAP['log_odds'][ray[0, -1], ray[1, -1]] += log_occ

    return MAP

# ------------------------------
# Main Process
# ------------------------------

def main():
    # Choose the dataset ID (e.g., 20 or 21)
    dataset_id = 21

    # 1. Load sensor data
    angle_min, angle_increment, range_min, range_max, ranges_all, hokuyo_time_stamps = load_hokuyo_data(dataset_id)
    encoder_counts, encoder_stamps, imu_angular_velocity, imu_stamps = load_sensor_data(dataset_id)
    
    # 2. Compute the odometry trajectory (initial estimate)
    t_all, v_all, w_all, dt_all = compute_velocity_and_yawrate(
        encoder_counts, encoder_stamps, imu_angular_velocity, imu_stamps, ticks_to_meter=0.0022)
    X, Y, THETA = integrate_odometry(dt_all, v_all, w_all, x0=0.0, y0=0.0, theta0=0.0)
    
    # Helper function: get the nearest odometry pose based on the LiDAR timestamp
    def get_odometry_pose(time_stamp):
        idx = np.argmin(np.abs(encoder_stamps - time_stamp))
        return (X[idx], Y[idx], THETA[idx])
    
    # Build the initial odometry trajectory (used for ICP initialization)
    odom_trajectory = np.array([get_odometry_pose(t) for t in hokuyo_time_stamps])
    
    # 3. Refine odometry using ICP
    # Initialize ICP trajectory; the first pose is taken from odometry
    icp_global_poses = [get_odometry_pose(hokuyo_time_stamps[0])]
    
    num_scans = ranges_all.shape[1]
    print("[INFO] Starting ICP scan matching on {} scans...".format(num_scans))
    
    for i in tqdm(range(num_scans - 1)):
        # Get two consecutive LiDAR scans
        scan1 = ranges_all[:, i]
        scan2 = ranges_all[:, i+1]
        # Convert to Cartesian coordinates (in robot frame)
        points1 = polar_to_cartesian(scan1, angle_min, angle_increment, range_min, range_max)
        points2 = polar_to_cartesian(scan2, angle_min, angle_increment, range_min, range_max)
        
        # Initial estimate: use odometry pose difference to obtain an initial relative transformation
        pose1 = get_odometry_pose(hokuyo_time_stamps[i])
        pose2 = get_odometry_pose(hokuyo_time_stamps[i+1])
        init_R, init_t = relative_transform_from_poses(pose1, pose2)
        
        # Use ICP directly here: use scan1 as source and scan2 as target
        R_icp, t_icp, error = icp_2d(points2, points1, init_R, init_t,
                                       max_iterations=50, tolerance=1e-6, distance_threshold=0.1)
        
        # Form the 3x3 homogeneous transformation matrix
        T_icp = np.array([[R_icp[0,0], R_icp[0,1], t_icp[0]],
                          [R_icp[1,0], R_icp[1,1], t_icp[1]],
                          [0, 0, 1]])
        
        # Accumulate the ICP transformation
        T_prev = pose_to_transform(*icp_global_poses[i])
        T_new = T_prev @ T_icp
        refined_pose = transform_to_pose(T_new)
        icp_global_poses.append(refined_pose)
        
    icp_global_poses = np.array(icp_global_poses)
    print("[INFO] ICP scan matching completed.")
    
    # 4. Build the occupancy grid map using the ICP-corrected trajectory
    MAP = init_map()
    
    # Process each scan sequentially to update the map
    for i in tqdm(range(num_scans)):
        # ICP pose for the current scan
        pose = icp_global_poses[i]
        # Get the current scan and convert to Cartesian coordinates in the robot frame
        scan = ranges_all[:, i]
        scan_points = polar_to_cartesian(scan, angle_min, angle_increment, range_min, range_max)
        MAP = update_map(MAP, pose, scan_points, log_free=-0.4, log_occ=0.85)
    
    # 5. Convert log-odds to occupancy probabilities and display the final map
    prob_map = 1 - 1/(1 + np.exp(MAP['log_odds']))
    
    plt.figure()
    plt.title(f"Occupancy Grid Map from ICP trajectory for dataset {dataset_id}")
    plot_map(prob_map, cmap="binary")
    plt.xlabel("X cells")
    plt.ylabel("Y cells")
    plt.show(block=True)

if __name__ == '__main__':
    start = time.time()
    main()
    print("Total processing time: {:.2f} sec.".format(time.time()-start))
