# LiDAR-Based SLAM and Environment Mapping

## 🔍 Project Overview
This project presents a LiDAR-based Simultaneous Localization and Mapping (SLAM) system for a differential-drive robot. The system combines encoder and IMU data for initial pose estimation, refines poses using LiDAR scan matching via ICP, constructs 2D occupancy grid maps, and enriches these maps with RGB-D texture information. Pose graph optimization with loop closure detection further improves map consistency and accuracy.

## 🛠️ Technical Components

### 1️⃣ Encoder & IMU Odometry
- Differential-drive kinematic model for pose estimation.
- Integration of encoder-derived linear velocities and IMU yaw rates.

### 2️⃣ Iterative Closest Point (ICP) Algorithm
- ICP implementation for accurate point cloud registration.
- Multi-scale ICP strategy (coarse to fine alignment).

### 3️⃣ Occupancy Grid Mapping
- Conversion of LiDAR scans into global occupancy grids.
- Incremental map updating using log-odds occupancy values.

### 4️⃣ Texture Mapping
- RGB-D data used to overlay visual textures onto occupancy maps.
- Depth-to-color point cloud transformations.

### 5️⃣ Pose Graph Optimization and Loop Closure
- Factor graph formulation integrating odometry and ICP constraints.
- Loop closure detection using fixed-interval and proximity-based methods.
- Optimization performed with GTSAM to refine global trajectory estimates.

## 📂 Project Structure
```
.
├── code/
│   ├── icp_warm_up/
│   │   ├── test_icp.py
│   │   └── utils.py
│   ├── load_data.py
│   ├── occupancy_map_icp.py
│   ├── occupancy_map_lidar.py
│   ├── odometry.py
│   ├── optimize.py
│   ├── scan_matching.py
│   ├── test_gtsam.py
│   └── texture_mapping.py
├── docs/
│   └── RobotConfiguration.pdf
├── plot/
│   └── (trajectory and map plots)
└── report/
    └── ECE276A_Project2_Report.pdf
```

## 📈 Results & Performance
- Effective reduction of odometry drift through ICP scan matching.
- Accurate occupancy maps and coherent texture maps.
- Enhanced trajectory accuracy via pose graph optimization and loop closure.

## 🛠️ Technologies
- **Python** for implementation.
- **GTSAM** for factor graph optimization.
- **OpenCV, NumPy, Open3D** for point cloud processing and visualization.

## 🎯 Future Improvements
- Improved loop closure detection and robustness.
- Enhanced synchronization and calibration of RGB-D and LiDAR sensors.
- Real-time processing optimizations.
- Exploration of alternative registration techniques (e.g., NDT).

## 📚 Documentation
Detailed implementation and results discussion available in [`report/ECE276A_Project2_Report.pdf`](report/ECE276A_Project2_Report.pdf).

---

## 📧 Contact
- **Brian (Shou-Yu) Wang**  
  - Email: briansywang541@gmail.com  
  - LinkedIn: [linkedin.com/in/sywang541](https://linkedin.com/in/sywang541)
  - GitHub: [BrianSY541](https://github.com/BrianSY541)

---

**Project developed as part of ECE 276A: Sensing & Estimation in Robotics at UC San Diego.**

