#!/usr/bin/env python
"""
texture_mapping.py

This script demonstrates the following steps:
1) Compute an ICP refined trajectory (icp_poses) from Hokuyo + odometry data, while also returning the corresponding LiDAR timestamps (icp_timestamps).
2) Load disparity_time_stamps and rgb_time_stamps from a Kinect npz file, and load disparity/RGB images from a folder.
3) For each Kinect image, find the closest ICP pose (matched via timestamps) and project the points onto a map.
4) Force the world coordinate z=0 (treated as the floor) and update the texture_map.

Finally, the texture_map (H, W, 3) is output.
"""

import os
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- Import necessary functions from scan_matching.py and odometry.py (please adjust the import paths as needed) ---
from scan_matching import load_hokuyo_data, polar_to_cartesian, icp_2d, pose_to_transform, transform_to_pose, relative_transform_from_poses
from odometry import load_sensor_data, compute_velocity_and_yawrate, integrate_odometry


##########################
# (A) Kinect Data Loading
##########################
def load_kinect_timestamps(dataset_id=20, base_path="../data"):
    """
    Load disparity_time_stamps and rgb_time_stamps from the Kinect{dataset_id}.npz file.
    Returns: (disp_ts, rgb_ts)
    """
    npz_file = os.path.join(base_path, f"Kinect{dataset_id}.npz")
    data = np.load(npz_file, allow_pickle=True)
    disp_ts = data['disparity_time_stamps']   # shape: (N_disp,)
    rgb_ts  = data['rgb_time_stamps']           # shape: (N_rgb,)
    print(f"[INFO] Kinect{dataset_id}.npz loaded. disparity_ts={disp_ts.shape}, rgb_ts={rgb_ts.shape}")
    return disp_ts, rgb_ts

def load_kinect_images(dataset_id=20, base_path="../data/dataRGBD"):
    """
    Read disparity{dataset_id}_{index}.png and rgb{dataset_id}_{index}.png from the folder.
    Returns:
      disp_list = [(disp_idx, disp_img), ...] sorted by file index
      rgb_list  = [(rgb_idx,  rgb_img ), ...]
    """
    disp_folder = os.path.join(base_path, f"Disparity{dataset_id}")
    rgb_folder  = os.path.join(base_path, f"RGB{dataset_id}")

    disp_pattern = re.compile(rf"disparity{dataset_id}_(\d+)\.png$")
    rgb_pattern  = re.compile(rf"rgb{dataset_id}_(\d+)\.png$")

    disp_files = []
    rgb_files  = []

    # Scan the folder
    for fname in os.listdir(disp_folder):
        m = disp_pattern.match(fname)
        if m:
            idx = int(m.group(1))
            disp_files.append((idx, fname))
    disp_files.sort(key=lambda x: x[0])

    for fname in os.listdir(rgb_folder):
        m = rgb_pattern.match(fname)
        if m:
            idx = int(m.group(1))
            rgb_files.append((idx, fname))
    rgb_files.sort(key=lambda x: x[0])

    disp_list = []
    for (idx, fname) in disp_files:
        path = os.path.join(disp_folder, fname)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"[WARNING] Fail to load {path}")
            continue
        disp_list.append((idx, img.astype(np.float32)))

    rgb_list = []
    for (idx, fname) in rgb_files:
        path = os.path.join(rgb_folder, fname)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARNING] Fail to load {path}")
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb_list.append((idx, img_rgb))

    print(f"[INFO] load_kinect_images: disparity_count={len(disp_list)}, rgb_count={len(rgb_list)}")
    return disp_list, rgb_list


##########################
# (B) Compute ICP Pose + LiDAR Timestamps
##########################
def compute_icp_path_and_timestamps(dataset_id=20):
    """
    1) Load Hokuyo data => (ranges_all, hokuyo_time_stamps)
    2) Load odometry data => compute_velocity_and_yawrate => integrate_odometry
    3) ICP: register consecutive scans => icp_poses (num_scans, 3)
    4) Return (icp_poses, icp_timestamps), where icp_timestamps[i] = hokuyo_time_stamps[i],
       so that we know which LiDAR timestamp corresponds to each pose.
    """
    angle_min, angle_increment, range_min, range_max, ranges_all, hokuyo_time_stamps = load_hokuyo_data(dataset_id)
    num_scans = ranges_all.shape[1]

    encoder_counts, encoder_stamps, imu_angular_velocity, imu_stamps = load_sensor_data(dataset_id)
    t_all, v_all, w_all, dt_all = compute_velocity_and_yawrate(
        encoder_counts, encoder_stamps,
        imu_angular_velocity, imu_stamps,
        ticks_to_meter=0.0022
    )
    X, Y, THETA = integrate_odometry(dt_all, v_all, w_all)

    def get_odometry_pose(t):
        idx = np.argmin(np.abs(encoder_stamps - t))
        return (X[idx], Y[idx], THETA[idx])

    icp_poses = []
    icp_poses.append(get_odometry_pose(hokuyo_time_stamps[0]))  # initialization

    for i in range(num_scans - 1):
        scan1 = ranges_all[:, i]
        scan2 = ranges_all[:, i+1]
        points1 = polar_to_cartesian(scan1, angle_min, angle_increment, range_min, range_max)
        points2 = polar_to_cartesian(scan2, angle_min, angle_increment, range_min, range_max)

        pose1 = get_odometry_pose(hokuyo_time_stamps[i])
        pose2 = get_odometry_pose(hokuyo_time_stamps[i+1])
        init_R, init_t = relative_transform_from_poses(pose1, pose2)

        R_icp, t_icp, error = icp_2d(points2, points1,
                                      init_R=init_R, init_t=init_t,
                                      max_iterations=50,
                                      tolerance=1e-6,
                                      distance_threshold=0.1)
        # Accumulate
        T_prev = pose_to_transform(*icp_poses[-1])
        T_icp = np.array([
            [R_icp[0,0], R_icp[0,1], t_icp[0]],
            [R_icp[1,0], R_icp[1,1], t_icp[1]],
            [0,0,1]
        ])
        T_new = T_prev @ T_icp
        new_pose = transform_to_pose(T_new)
        icp_poses.append(new_pose)

    icp_poses = np.array(icp_poses)          # shape=(num_scans, 3)
    icp_timestamps = hokuyo_time_stamps[:]   # shape=(num_scans,)

    return icp_poses, icp_timestamps


##########################
# (C) Main Texture Mapping Process
##########################
def disparity_to_depth(disparity):
    """
    Converts disparity to depth.
    Formula:
      dd = -0.00304 * disparity + 3.31
      depth = 1.03 / dd
    """
    dd = -0.00304 * disparity + 3.31
    depth = 1.03 / dd
    return depth, dd

def project_depth_to_3d(depth_img, camera_intrinsics):
    """
    Project the depth image into the camera coordinate system.
    """
    fx, fy = camera_intrinsics['fx'], camera_intrinsics['fy']
    cx, cy = camera_intrinsics['cx'], camera_intrinsics['cy']
    H, W = depth_img.shape
    u = np.arange(W)
    v = np.arange(H)
    uu, vv = np.meshgrid(u, v)
    X = (uu - cx) * depth_img / fx
    Y = (vv - cy) * depth_img / fy
    Z = depth_img
    points = np.stack((X, Y, Z), axis=2)  # shape (H, W, 3)
    return points, uu, vv

def get_rgb_from_depth_indices(uu, vv, dd):
    rgbi = (526.37 * vv + 19276 - 7877.07 * dd) / 585.051
    rgbj = (526.37 * uu + 16662) / 585.051
    return rgbi, rgbj

def transform_points(points, R, t):
    """
    Transform points using a rotation R and translation t.
    points: (N, 3) or (H, W, 3)
    R: (3, 3), t: (3,)
    """
    if points.ndim == 3:
        H, W, _ = points.shape
        pts_flat = points.reshape(-1, 3).T
        transformed = R @ pts_flat + t.reshape(3,1)
        return transformed.T.reshape(H, W, 3)
    else:
        return (R @ points.T).T + t

def get_camera_to_robot_transform():
    """
    Extrinsic parameters provided: (0.18, 0.005, 0.36) with roll=0, pitch=0.36, yaw=0.021.
    Rotation order: Rz, then Ry, then Rx.
    """
    t = np.array([0.18, 0.005, 0.36])
    roll = 0.0
    pitch = 0.36
    yaw = 0.021

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll),  np.cos(roll)]
    ])
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0, 0, 1]
    ])
    R = Rz @ Ry @ Rx
    return R, t

def robot_to_world(points, robot_pose):
    """
    Convert points from the robot coordinate system to the world coordinate system.
    2D homogeneous transformation given by (x, y, theta); z remains unchanged.
    """
    x, y, theta = robot_pose
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0, 0, 1]
    ])
    t = np.array([x, y, 0])
    pts_world = transform_points(points, R, t)
    # Force z=0 (treated as floor)
    pts_world[..., 2] = 0
    return pts_world

def world_to_map(pt, MAP):
    """
    Convert a point (x, y) in world coordinates to map indices.
    """
    return np.floor((pt - MAP['min']) / MAP['res']).astype(int)

def update_texture_map_from_points(texture_map, pts_world, rgb_values, MAP):
    count_updated = 0
    for pt, color in zip(pts_world, rgb_values):
        idx = world_to_map(pt[:2], MAP)
        if 0 <= idx[0] < texture_map.shape[0] and 0 <= idx[1] < texture_map.shape[1]:
            texture_map[idx[0], idx[1], :] = color
            count_updated += 1
    return texture_map, count_updated

def get_nearest_pose(icp_poses, icp_timestamps, t_img):
    """
    Given an image timestamp t_img, find the closest pose in icp_timestamps.
    """
    idx = np.argmin(np.abs(icp_timestamps - t_img))
    return icp_poses[idx]


##########################
# main()
##########################
def main_texture_mapping(dataset_id=20):
    # 1) Load Kinect timestamps
    disp_ts, rgb_ts = load_kinect_timestamps(dataset_id, base_path="../data")
    # 2) Load Kinect images
    disp_list, rgb_list = load_kinect_images(dataset_id, base_path="../data/dataRGBD")
    if len(disp_list) == 0 or len(rgb_list) == 0:
        print("[ERROR] Kinect images not loaded. Check folder.")
        return

    # 3) Compute ICP pose + timestamps
    icp_poses, icp_timestamps = compute_icp_path_and_timestamps(dataset_id)
    print(f"[INFO] icp_poses: {icp_poses.shape}, icp_timestamps: {icp_timestamps.shape}")

    # 4) Prepare map
    MAP = {}
    MAP['res'] = 0.05
    MAP['min'] = np.array([-30.0, -30.0])
    MAP['max'] = np.array([30.0, 30.0])
    MAP['size'] = np.ceil((MAP['max'] - MAP['min']) / MAP['res']).astype(int)
    even_mask = (MAP['size'] % 2 == 0)
    MAP['size'][even_mask] += 1
    texture_map = np.zeros((MAP['size'][0], MAP['size'][1], 3), dtype=np.uint8)
    print(f"[INFO] MAP size={MAP['size']}, shape={texture_map.shape}")

    # Camera intrinsics
    camera_intrinsics = {
        'fx': 585.05, 'fy': 585.05,
        'cx': 242.94, 'cy': 315.84
    }
    # Extrinsics
    R_cam2rob, t_cam2rob = get_camera_to_robot_transform()

    # 5) Process Kinect images one by one
    min_len = min(len(disp_list), len(rgb_list))
    for i in range(min_len):
        disp_idx, disp_img = disp_list[i]
        rgb_idx, rgb_img   = rgb_list[i]

        # (a) Get timestamp (using disparity or rgb as reference)
        if disp_idx - 1 < len(disp_ts):
            t_disp = disp_ts[disp_idx - 1]
        else:
            t_disp = disp_ts[-1]
        if rgb_idx - 1 < len(rgb_ts):
            t_rgb = rgb_ts[rgb_idx - 1]
        else:
            t_rgb = rgb_ts[-1]
        # Here you can use t_img = (t_disp + t_rgb) / 2 for average, or just use t_disp
        t_img = 0.5 * (t_disp + t_rgb)

        # (b) Find the nearest ICP pose
        robot_pose = get_nearest_pose(icp_poses, icp_timestamps, t_img)

        # (c) Convert disparity to depth
        depth_img, dd = disparity_to_depth(disp_img)
        valid_mask = (depth_img > 0)

        # (d) Project to camera coordinates
        points_cam, uu, vv = project_depth_to_3d(depth_img, camera_intrinsics)
        rgbi, rgbj = get_rgb_from_depth_indices(uu, vv, dd)

        pts_cam_valid = points_cam[valid_mask]
        rgbi_valid = np.rint(rgbi[valid_mask]).astype(int)
        rgbj_valid = np.rint(rgbj[valid_mask]).astype(int)

        H_rgb, W_rgb, _ = rgb_img.shape
        inside_mask = (rgbi_valid >= 0) & (rgbi_valid < H_rgb) & (rgbj_valid >= 0) & (rgbj_valid < W_rgb)
        pts_cam_valid = pts_cam_valid[inside_mask]
        rgbi_valid    = rgbi_valid[inside_mask]
        rgbj_valid    = rgbj_valid[inside_mask]
        rgb_values    = rgb_img[rgbi_valid, rgbj_valid, :]

        # (e) Transform from camera to robot to world
        pts_robot = transform_points(pts_cam_valid, R_cam2rob, t_cam2rob)
        pts_world = robot_to_world(pts_robot, robot_pose)  # z is set to 0 inside

        # (f) Update map
        texture_map, updated_cnt = update_texture_map_from_points(texture_map, pts_world, rgb_values, MAP)
        if i % 200 == 0:
            print(f"[DEBUG] i={i}, disp_idx={disp_idx}, rgb_idx={rgb_idx}, updated={updated_cnt} cells")

    # 6) Display the result
    plt.figure()
    plt.imshow(texture_map)
    plt.title(f"Texture Map for dataset {dataset_id}")
    plt.axis('equal')
    plt.show(block=True)

    return texture_map

if __name__=="__main__":
    texture_map = main_texture_mapping(dataset_id=20)
