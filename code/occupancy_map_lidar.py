#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from pr2_utils import bresenham2D  # Use the bresenham2D function provided in pr2_utils.py

def load_hokuyo_data(npz_file):
    data = np.load(npz_file)
    angle_min = data["angle_min"].item()
    # Note: angle_increment is stored with shape (1,1) in the file
    angle_increment = data["angle_increment"].item()
    range_min = data["range_min"].item()
    range_max = float(data["range_max"].item())
    ranges = data["ranges"]   # Shape: (1081, 4962)
    time_stamps = data["time_stamps"]
    return angle_min, angle_increment, range_min, range_max, ranges, time_stamps

def polar_to_cartesian(ranges, angles, range_min, range_max):
    # Filter out points that are too close or too far
    valid = (ranges > range_min) & (ranges < range_max)
    valid_ranges = ranges[valid]
    valid_angles = angles[valid]
    x = valid_ranges * np.cos(valid_angles)
    y = valid_ranges * np.sin(valid_angles)
    return np.column_stack((x, y))

def create_occupancy_map(points, res=0.05, map_min=np.array([-20, -20]), map_max=np.array([20, 20]),
                         log_odds_occ=2.0, log_odds_free=-1.0):
    # Calculate the map size (in cells) and ensure the size is odd so that the origin is at the center cell
    map_size = np.ceil((map_max - map_min) / res).astype(int)
    isEven = map_size % 2 == 0
    map_size[isEven] += 1

    # Initialize the log-odds map with 0 (representing uncertainty, i.e., 0.5 probability)
    log_odds_map = np.zeros(map_size)

    # Assume the robot is at world coordinate (0,0), compute its corresponding map cell
    robot_cell = np.floor((np.array([0, 0]) - map_min) / res).astype(int)

    # For each valid point from the LiDAR, compute the ray and update the map using Bresenham's algorithm
    for pt in points:
        # Compute the map cell for the endpoint
        end_cell = np.floor((pt - map_min) / res).astype(int)
        # Use Bresenham's algorithm to compute all cells along the ray from the robot to the point
        cells = bresenham2D(robot_cell[0], robot_cell[1], end_cell[0], end_cell[1])
        cells = np.round(cells).astype(int)  # Convert to integers

        # If the ray contains more than one cell (at least free cells and the endpoint)
        if cells.shape[1] >= 1:
            # Update all cells along the ray (except the last one) as free space
            for i in range(cells.shape[1]-1):
                x_cell, y_cell = cells[0, i], cells[1, i]
                # Check if the cell is within the map bounds
                if 0 <= x_cell < log_odds_map.shape[0] and 0 <= y_cell < log_odds_map.shape[1]:
                    log_odds_map[x_cell, y_cell] += log_odds_free
            # Update the endpoint cell as occupied
            x_cell, y_cell = cells[0, -1], cells[1, -1]
            if 0 <= x_cell < log_odds_map.shape[0] and 0 <= y_cell < log_odds_map.shape[1]:
                log_odds_map[x_cell, y_cell] += log_odds_occ

    return log_odds_map, map_size

def log_odds_to_probability(log_odds_map):
    # Convert log-odds to probability using the formula: p = 1 - 1/(1 + exp(L))
    prob_map = 1 - 1 / (1 + np.exp(log_odds_map))
    return prob_map

def main():
    # Choose the dataset ID (e.g., 20 or 21)
    dataset_id = 20
    # Load Hokuyo data file from dataset (e.g., 20 or 21)
    npz_file = f"../data/Hokuyo{dataset_id}.npz"
    angle_min, angle_increment, range_min, range_max, ranges, time_stamps = load_hokuyo_data(npz_file)
    
    # Retrieve the first LiDAR scan (column 0)
    scan0 = ranges[:, 0]
    num_beams = scan0.shape[0]
    
    # Compute the angle for each beam
    angles = angle_min + np.arange(num_beams) * angle_increment
    
    # Convert to Cartesian coordinates, keeping only valid points based on range_min and range_max
    points = polar_to_cartesian(scan0, angles, range_min, range_max)
    
    # Create the occupancy grid map using log-odds updates
    log_odds_map, map_size = create_occupancy_map(points)
    prob_map = log_odds_to_probability(log_odds_map)
    
    # Display the occupancy grid map
    plt.figure(figsize=(8,8))
    plt.imshow(prob_map.T, origin="lower", cmap="gray", extent=[-20, 20, -20, 20])
    plt.title(f"Occupancy Grid Map from 1st LiDAR Scan for dataset {dataset_id}")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.colorbar(label="Occupancy Probability")
    plt.grid(False)
    plt.show(block=True)

if __name__ == '__main__':
    main()
