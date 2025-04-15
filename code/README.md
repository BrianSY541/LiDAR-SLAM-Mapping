# LiDAR-Based SLAM (ECE276A Project 2)

## Overview
This project implements **Simultaneous Localization and Mapping (SLAM)** using **LiDAR, IMU, and encoder** data from a differential-drive robot. The system performs **odometry estimation**, **ICP scan matching**, **occupancy grid mapping**, and **pose graph optimization** with loop closure constraints. Additionally, it utilizes **RGB-D images** to generate a textured floor map.

### Features:
- **Odometry Estimation:** Uses encoder and IMU data to compute robot motion.
- **ICP Scan Matching:** Improves pose estimation by aligning consecutive LiDAR scans.
- **Occupancy Grid Mapping:** Builds a 2D map of the environment using LiDAR data.
- **Texture Mapping:** Projects RGB-D images onto the floor for visual representation.
- **Pose Graph Optimization:** Refines the trajectory using **GTSAM** for loop closure.

## File Structure
```
.
├── data/                  # Contains sensor datasets (Encoders, LiDAR, IMU, Kinect)
├── pr2_utils.py           # Utility functions (ray tracing, visualization, etc.)
├── load_data.py           # Loads and inspects sensor data
├── odometry.py            # Computes odometry from encoder and IMU data
├── scan_matching.py       # Implements ICP-based LiDAR scan matching
├── test_icp.py            # Tests ICP implementation with 3D point cloud alignment
├── utils.py               # Helper functions for point cloud handling and visualization
├── occupancy_map_lidar.py # Generates an occupancy grid map
├── occupancy_map_icp.py   # Refines the map using ICP-corrected poses
├── texture_mapping.py     # Creates a color texture map using RGB-D images
├── optimize.py            # Implements pose graph optimization using GTSAM
├── apply_optimized_trajectory.py # Reconstructs the map using optimized trajectory
└── README.md              # This file
```

## Dependencies
Before running the code, install the required Python packages:

```bash
pip install numpy matplotlib open3d gtsam scipy
```

For **Windows users**, install **Windows Subsystem for Linux (WSL)** or use a **Linux virtual machine** before proceeding.

## How to Run

### 1. Inspect Data
To visualize the sensor data:

```bash
python load_data.py
```

### 2. Compute Odometry
Run the script to compute odometry using encoders and IMU:

```bash
python odometry.py
```

### 3. Perform Scan Matching with ICP
To align consecutive LiDAR scans using **Iterative Closest Point (ICP)**:

```bash
python scan_matching.py
```

### 4. Test ICP Implementation
To evaluate ICP on 3D point clouds:

```bash
python test_icp.py
```

### 5. Generate Occupancy Grid Map
Using **LiDAR data** to construct an occupancy map:

```bash
python occupancy_map_lidar.py
```

Refining the map with **ICP-corrected poses**:

```bash
python occupancy_map_icp.py
```

### 6. Apply Texture Mapping
To overlay RGB-D images onto the 2D map:

```bash
python texture_mapping.py
```

### 7. Optimize the Trajectory
To perform **pose graph optimization** and apply loop closure corrections:

```bash
python optimize.py
```

### 8. Apply Optimized Trajectory
Recompute the map with the optimized trajectory:

```bash
python apply_optimized_trajectory.py
```

## Results
- **Odometry Trajectory vs. ICP-refined Trajectory**
- **Occupancy Grid Map**
- **Textured Floor Map**
- **Optimized Pose Graph with Loop Closure**

For more details, refer to the **project report**.

---

### Contributors
This project was developed as part of **ECE276A: Sensing & Estimation in Robotics**.

