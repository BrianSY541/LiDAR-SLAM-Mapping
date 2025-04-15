#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pr2_utils import bresenham2D
from scan_matching import polar_to_cartesian, load_hokuyo_data
from scipy.sparse import csr_matrix
from pathlib import Path

# --------------------------------------------
# Configuration and Constants
# --------------------------------------------
class MapConfig:
    RESOLUTION = 0.05  # meters per cell
    MAP_MIN = np.array([-20.0, -20.0])
    MAP_MAX = np.array([20.0, 20.0])
    OCC_UPDATE = 0.85
    FREE_UPDATE = -0.4
    KINECT_PATH = Path("data")
    K = np.array([[585.05, 0, 242.94],
                 [0, 585.05, 315.84],
                 [0, 0, 1]])
    T_DEPTH_TO_ROBOT = np.eye(4)  # Placeholder, update with actual calibration

# --------------------------------------------
# Helper Functions
# --------------------------------------------
def pose_to_transform(x, y, theta):
    """Convert pose to 3x3 homogeneous transformation matrix."""
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    return np.array([[cos_theta, -sin_theta, x],
                    [sin_theta, cos_theta, y],
                    [0, 0, 1]])

def world_to_map(points, map_min, resolution):
    """Vectorized conversion of world coordinates to map indices."""
    return np.floor((points - map_min) / resolution).astype(int)

# --------------------------------------------
# Efficient Map Creation
# --------------------------------------------
def initialize_maps(config=MapConfig):
    """Initialize sparse occupancy and texture maps."""
    map_size = np.ceil((config.MAP_MAX - config.MAP_MIN) / config.RESOLUTION).astype(int)
    map_size[map_size % 2 == 0] += 1  # Ensure odd size for center alignment
    
    # Use sparse matrices for occupancy to save memory
    log_odds = csr_matrix(map_size, dtype=np.float32)
    occupancy_map = np.zeros(map_size, dtype=np.uint8)
    texture_map = np.zeros((*map_size, 3), dtype=np.uint8)
    
    return occupancy_map, log_odds, texture_map, map_size

# --------------------------------------------
# Optimized Occupancy Update
# --------------------------------------------
def update_occupancy(log_odds, occupancy_map, sensor_origin, points_world, 
                    map_min, resolution, config=MapConfig):
    """Vectorized occupancy grid update."""
    origins = np.tile(sensor_origin, (points_world.shape[0], 1))
    origins_idx = world_to_map(origins, map_min, resolution)
    points_idx = world_to_map(points_world, map_min, resolution)
    
    # Filter valid indices
    valid_mask = ((points_idx >= 0) & (points_idx < occupancy_map.shape[0])).all(axis=1)
    origins_idx, points_idx = origins_idx[valid_mask], points_idx[valid_mask]
    
    # Batch process rays using Bresenham
    for orig_idx, pt_idx in zip(origins_idx, points_idx):
        ray = bresenham2D(orig_idx[0], orig_idx[1], pt_idx[0], pt_idx[1]).astype(int)
        
        # Update free cells (excluding endpoint)
        for x, y in ray[:-1].T:
            if 0 <= x < log_odds.shape[0] and 0 <= y < log_odds.shape[1]:
                log_odds[x, y] += config.FREE_UPDATE
        
        # Update occupied cell
        x, y = ray[-1]
        if 0 <= x < log_odds.shape[0] and 0 <= y < log_odds.shape[1]:
            log_odds[x, y] += config.OCC_UPDATE
    
    # Update occupancy map from log-odds
    occupancy_map[log_odds.toarray() > 0] = 1
    occupancy_map[log_odds.toarray() <= 0] = 0
    return log_odds, occupancy_map

# --------------------------------------------
# Optimized Texture Mapping
# --------------------------------------------
def update_texture_map(texture_map, optimized_trajectory, num_kinect_frames, 
                      map_min, resolution, config=MapConfig):
    """Efficient texture mapping with batch processing."""
    for i in range(min(num_kinect_frames, len(optimized_trajectory))):
        rgb_path = config.KINECT_PATH / f"kinect_rgb_{i}.png"
        depth_path = config.KINECT_PATH / f"kinect_depth_{i}.png"
        
        rgb_img = cv2.imread(str(rgb_path))
        depth_img = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        
        if rgb_img is None or depth_img is None:
            print(f"Warning: Missing Kinect data for frame {i}")
            continue
        
        # Vectorized depth to 3D points
        H, W = depth_img.shape
        u, v = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
        depth = depth_img.astype(np.float32) / 1000.0
        
        x_cam = (u - config.K[0, 2]) * depth / config.K[0, 0]
        y_cam = (v - config.K[1, 2]) * depth / config.K[1, 1]
        z_cam = depth
        points_cam = np.stack((x_cam, y_cam, z_cam), axis=-1).reshape(-1, 3)
        
        # Transform to robot frame
        points_cam_hom = np.hstack((points_cam, np.ones((points_cam.shape[0], 1))))
        points_robot = (config.T_DEPTH_TO_ROBOT @ points_cam_hom.T).T[:, :3]
        
        # Transform to world frame
        x_r, y_r, theta_r = optimized_trajectory[i]
        T_robot = pose_to_transform(x_r, y_r, theta_r)
        points_world = (T_robot @ np.hstack((points_robot[:, :2], 
                                            np.ones((points_robot.shape[0], 1)))).T).T
        
        # Filter floor points efficiently
        floor_mask = (np.abs(points_robot[:, 2]) < 0.15) & (depth.flatten() > 0)
        floor_points = points_world[floor_mask]
        floor_colors = rgb_img.reshape(-1, 3)[floor_mask]
        
        # Vectorized mapping to texture map
        idx = world_to_map(floor_points[:, :2], map_min, resolution)
        valid_idx = ((idx >= 0) & (idx < texture_map.shape[:2])).all(axis=1)
        idx, colors = idx[valid_idx], floor_colors[valid_idx]
        
        # Update texture map (average colors for overlapping points)
        for (x, y), color in zip(idx, colors):
            if np.all(texture_map[x, y] == 0):
                texture_map[x, y] = color
            else:
                texture_map[x, y] = (texture_map[x, y] + color) / 2

    return texture_map

# --------------------------------------------
# Main Function
# --------------------------------------------
def main(config=MapConfig):
    # Load data
    try:
        optimized_trajectory = np.load("optimized_trajectory.npy")
        angle_min, angle_increment, range_min, range_max, ranges_all, _ = load_hokuyo_data(20)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    num_poses, num_scans = optimized_trajectory.shape[0], ranges_all.shape[1]
    
    # Initialize maps
    occupancy_map, log_odds, texture_map, map_size = initialize_maps(config)
    
    # Update occupancy map
    for i in range(min(num_poses, num_scans)):
        scan = ranges_all[:, i]
        points_sensor = polar_to_cartesian(scan, angle_min, angle_increment, range_min, range_max)
        
        x, y, theta = optimized_trajectory[i]
        T = pose_to_transform(x, y, theta)
        points_world = (T @ np.hstack((points_sensor, np.ones((points_sensor.shape[0], 1)))).T).T[:, :2]
        
        log_odds, occupancy_map = update_occupancy(
            log_odds, occupancy_map, [x, y], points_world, 
            config.MAP_MIN, config.RESOLUTION, config
        )
    
    # Update texture map
    texture_map = update_texture_map(
        texture_map, optimized_trajectory, num_kinect_frames=10, 
        map_min=config.MAP_MIN, resolution=config.RESOLUTION, config=config
    )
    
    # Visualization
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(occupancy_map.T, origin='lower', cmap='gray')
    plt.title("Occupancy Grid Map")
    plt.xlabel("X (cells)")
    plt.ylabel("Y (cells)")
    
    plt.subplot(122)
    plt.imshow(cv2.cvtColor(texture_map, cv2.COLOR_BGR2RGB))
    plt.title("Texture Map")
    plt.xlabel("X (cells)")
    plt.ylabel("Y (cells)")
    
    plt.tight_layout()
    plt.savefig("maps.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()