#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import gtsam
from gtsam import Pose2, noiseModel, NonlinearFactorGraph, Values, PriorFactorPose2, BetweenFactorPose2
from scan_matching import polar_to_cartesian, perform_icp

# Import required functions from odometry module
from odometry import load_sensor_data, compute_velocity_and_yawrate, integrate_odometry

# ------------------------------
# Compute the actual trajectory using odometry and IMU data
# ------------------------------
def load_odometry(dataset_id=20, ticks_to_meter=0.0022):
    """
    Reads odometry and IMU sensor data, computes the robot trajectory, and converts it into a list of gtsam.Pose2 objects.
    """
    # (1) Load sensor data
    encoder_counts, encoder_stamps, imu_angular_velocity, imu_stamps = load_sensor_data(dataset_id)
    # (2) Compute linear velocity, angular velocity, and dt for each interval
    t_all, v_all, w_all, dt_all = compute_velocity_and_yawrate(
        encoder_counts, encoder_stamps, imu_angular_velocity, imu_stamps,
        ticks_to_meter=ticks_to_meter
    )
    # (3) Integrate the trajectory using the differential drive model 
    # Note: The arrays X, Y, THETA have length len(dt_all)+1
    X, Y, THETA = integrate_odometry(dt_all, v_all, w_all, x0=0.0, y0=0.0, theta0=0.0)
    
    # (4) Convert the trajectory to a list of gtsam.Pose2 objects
    odom_poses = []
    for i in range(len(X)):
        odom_poses.append(Pose2(X[i], Y[i], THETA[i]))
    return odom_poses

# ------------------------------
# Helper function: Compute the relative transformation between two Pose2 objects
# ------------------------------
def compute_relative_pose(pose1, pose2):
    """
    Given two absolute poses (gtsam.Pose2), compute the relative transformation from pose1 to pose2.
    """
    return pose1.between(pose2)

# ------------------------------
# Load and prepare LiDAR data
# ------------------------------
def load_lidar_data(dataset_id=20):
    """
    Load LiDAR data for scan matching and loop closure detection.
    """
    from scan_matching import load_hokuyo_data
    
    # Load LiDAR data
    angle_min, angle_increment, range_min, range_max, ranges_all, hokuyo_time_stamps = load_hokuyo_data(dataset_id)
    
    return angle_min, angle_increment, range_min, range_max, ranges_all, hokuyo_time_stamps

# ------------------------------
# Detect loop closures using scan matching
# ------------------------------
def detect_loop_closures(odom_poses, angle_min, angle_increment, range_min, range_max, ranges_all, 
                        min_node_separation=20, proximity_threshold=1.0, icp_fitness_threshold=0.7):
    """
    Detect loop closures using both proximity and scan matching.
    
    Args:
        odom_poses: List of robot poses (gtsam.Pose2)
        ranges_all: LiDAR scan data
        min_node_separation: Minimum number of nodes between potential loop closures
        proximity_threshold: Maximum Euclidean distance for potential loop closure
        icp_fitness_threshold: Minimum ICP fitness score to accept a loop closure
    
    Returns:
        list of tuples (i, j, rel_pose, fitness_score)
    """
    loop_closure_pairs = []
    num_poses = len(odom_poses)
    num_scans = ranges_all.shape[1]
    
    # Ensure we don't exceed the number of available scans
    num_poses = min(num_poses, num_scans)
    
    print(f"Detecting loop closures among {num_poses} poses...")
    
    # Look for potential loop closures
    for i in range(num_poses):
        # Only check every few poses to speed up the process
        if i % 5 != 0:
            continue
            
        for j in range(i + min_node_separation, num_poses):
            # Only check every few poses
            if j % 5 != 0:
                continue
                
            # Check if poses are close in Euclidean space
            pos_i = np.array([odom_poses[i].x(), odom_poses[i].y()])
            pos_j = np.array([odom_poses[j].x(), odom_poses[j].y()])
            distance = np.linalg.norm(pos_i - pos_j)
            
            if distance < proximity_threshold:
                # Extract and prepare the LiDAR scans
                scan_i = ranges_all[:, i]
                scan_j = ranges_all[:, j]
                
                # Convert to Cartesian coordinates
                points_i = polar_to_cartesian(scan_i, angle_min, angle_increment, range_min, range_max)
                points_j = polar_to_cartesian(scan_j, angle_min, angle_increment, range_min, range_max)
                
                # Use ICP to check if scans match well
                initial_guess = np.eye(3)  # Initial identity transformation
                T_final, fitness_score = perform_icp(points_i, points_j, initial_guess, max_iterations=20)
                
                # Extract the 2D transformation (x, y, theta)
                x = T_final[0, 2]
                y = T_final[1, 2]
                theta = np.arctan2(T_final[1, 0], T_final[0, 0])
                
                # Create a Pose2 object for the relative transformation
                rel_pose = Pose2(x, y, theta)
                
                # If the ICP match is good, add as a loop closure
                if fitness_score > icp_fitness_threshold:
                    loop_closure_pairs.append((i, j, rel_pose, fitness_score))
                    print(f"Loop closure found: {i} -> {j}, fitness: {fitness_score:.3f}")
    
    return loop_closure_pairs

# ------------------------------
# Remove outlier loop closures
# ------------------------------
def remove_outlier_loop_closures(loop_closure_pairs, max_loop_closures=50, outlier_threshold=2.0):
    """
    Filter out potentially incorrect loop closures.
    
    Args:
        loop_closure_pairs: List of (i, j, rel_pose, fitness_score) tuples
        max_loop_closures: Maximum number of loop closures to keep
        outlier_threshold: Threshold for Mahalanobis distance to detect outliers
    
    Returns:
        Filtered list of loop closures
    """
    if not loop_closure_pairs:
        return []
    
    # Sort by fitness score (highest first)
    sorted_pairs = sorted(loop_closure_pairs, key=lambda x: x[3], reverse=True)
    
    # Limit the number of loop closures
    sorted_pairs = sorted_pairs[:max_loop_closures]
    
    # Extract transformations for statistical analysis
    transforms = []
    for i, j, pose, _ in sorted_pairs:
        transforms.append([pose.x(), pose.y(), pose.theta()])
    
    transforms = np.array(transforms)
    
    # Calculate mean and standard deviation
    mean_transform = np.mean(transforms, axis=0)
    std_transform = np.std(transforms, axis=0)
    
    # Avoid division by zero
    std_transform[std_transform < 1e-6] = 1e-6
    
    # Filter out outliers using Mahalanobis distance
    filtered_pairs = []
    for k, (i, j, pose, fitness) in enumerate(sorted_pairs):
        transform = transforms[k]
        mahalanobis_dist = np.sqrt(np.sum(((transform - mean_transform) / std_transform) ** 2))
        
        if mahalanobis_dist < outlier_threshold:
            filtered_pairs.append((i, j, pose, fitness))
            
    print(f"Kept {len(filtered_pairs)} loop closures out of {len(sorted_pairs)} candidates")
    return filtered_pairs

# ------------------------------
# Build the factor graph with adaptive noise models
# ------------------------------
def build_pose_graph(odom_poses, loop_closures=None):
    """
    Build the factor graph using odometry and loop closure constraints with adaptive noise models.
    
    Args:
        odom_poses: List of robot poses (gtsam.Pose2)
        loop_closures: List of tuples (i, j, rel_pose, fitness_score) for loop closures
    
    Returns:
        graph: NonlinearFactorGraph object
        initial_estimate: Values object with initial pose estimates
    """
    graph = NonlinearFactorGraph()
    initial_estimate = Values()
    
    # Set a prior factor on the first node to remove ambiguity
    prior_noise = noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.1]))
    graph.add(PriorFactorPose2(0, odom_poses[0], prior_noise))
    initial_estimate.insert(0, odom_poses[0])
    
    # Add odometry factors between consecutive nodes
    num_poses = len(odom_poses)
    for i in range(1, num_poses):
        rel_pose = compute_relative_pose(odom_poses[i-1], odom_poses[i])
        
        # Adaptive noise model - higher confidence for smaller transformations
        trans_dist = np.sqrt(rel_pose.x()**2 + rel_pose.y()**2)
        rot_dist = abs(rel_pose.theta())
        
        # Scale noise based on distance and rotation
        trans_noise = 0.1 + 0.05 * trans_dist
        rot_noise = 0.1 + 0.1 * rot_dist
        
        odom_noise = noiseModel.Diagonal.Sigmas(np.array([trans_noise, trans_noise, rot_noise]))
        graph.add(BetweenFactorPose2(i-1, i, rel_pose, odom_noise))
        initial_estimate.insert(i, odom_poses[i])
    
    # Add loop closure factors with adaptive noise models
    if loop_closures:
        for i, j, rel_pose, fitness_score in loop_closures:
            # Adaptive noise model based on ICP fitness score
            # Higher fitness score = lower noise (more confidence)
            noise_scale = 1.0 / (fitness_score + 0.3)
            loop_noise = noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.1]) * noise_scale)
            graph.add(BetweenFactorPose2(i, j, rel_pose, loop_noise))
    
    return graph, initial_estimate

# ------------------------------
# Optimize the pose graph using GTSAM
# ------------------------------
def optimize_pose_graph(graph, initial_estimate, verbose=True, max_iterations=100):
    """
    Optimize the pose graph using Levenberg-Marquardt optimization.
    
    Args:
        graph: NonlinearFactorGraph object
        initial_estimate: Values object with initial pose estimates
        verbose: Whether to print optimization status
        max_iterations: Maximum number of iterations for optimization
    
    Returns:
        result: Optimized poses
    """
    # Create parameters for optimization
    params = gtsam.LevenbergMarquardtParams()
    params.setVerbosity("ERROR" if not verbose else "TERMINATION")
    params.setMaxIterations(max_iterations)
    params.setRelativeErrorTol(1e-5)
    
    # Create optimizer and optimize
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
    result = optimizer.optimize()
    
    # Calculate and print error
    initial_error = graph.error(initial_estimate)
    final_error = graph.error(result)
    improvement = (initial_error - final_error) / initial_error * 100
    
    print(f"Initial error: {initial_error:.3f}")
    print(f"Final error: {final_error:.3f}")
    print(f"Improvement: {improvement:.2f}%")
    
    return result

# ------------------------------
# Plot the initial and optimized trajectories for comparison
# ------------------------------
def plot_trajectories(initial_poses, optimized_poses, loop_closures=None):
    """
    Plot the initial and optimized trajectories for comparison.
    
    Args:
        initial_poses: List of initial robot poses (gtsam.Pose2)
        optimized_poses: Values object with optimized pose estimates
        loop_closures: Optional list of loop closure constraints to visualize
    """
    init_x = [pose.x() for pose in initial_poses]
    init_y = [pose.y() for pose in initial_poses]
    
    opt_x = [optimized_poses.atPose2(i).x() for i in range(optimized_poses.size())]
    opt_y = [optimized_poses.atPose2(i).y() for i in range(optimized_poses.size())]
    
    plt.figure(figsize=(12, 10))
    
    # Plot trajectories
    plt.plot(init_x, init_y, 'r--', linewidth=1.5, alpha=0.7, label='Initial Trajectory')
    plt.plot(opt_x, opt_y, 'b-', linewidth=2, label='Optimized Trajectory')
    
    # Plot loop closures if provided
    if loop_closures:
        for i, j, _, _ in loop_closures:
            plt.plot([init_x[i], init_x[j]], [init_y[i], init_y[j]], 'g-', alpha=0.3, linewidth=0.5)
            plt.plot([opt_x[i], opt_x[j]], [opt_y[i], opt_y[j]], 'g-', alpha=0.5, linewidth=1)
    
    plt.xlabel('X (m)', fontsize=14)
    plt.ylabel('Y (m)', fontsize=14)
    plt.title('Pose Graph Optimization', fontsize=16)
    plt.legend(fontsize=12)
    plt.axis('equal')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('optimized_trajectory.png', dpi=300)
    plt.show()

# ------------------------------
# Main function
# ------------------------------
def main():
    # Set dataset ID
    dataset_id = 20
    
    # Compute the actual trajectory using odometry and IMU data
    odom_poses = load_odometry(dataset_id=dataset_id)
    
    # Load LiDAR data for loop closure detection
    angle_min, angle_increment, range_min, range_max, ranges_all, hokuyo_time_stamps = load_lidar_data(dataset_id)
    
    # Detect loop closures with scan matching
    loop_closures = detect_loop_closures(
        odom_poses, 
        angle_min, 
        angle_increment, 
        range_min, 
        range_max, 
        ranges_all,
        min_node_separation=20,
        proximity_threshold=1.0,
        icp_fitness_threshold=0.7
    )
    
    # Filter out outlier loop closures
    filtered_loop_closures = remove_outlier_loop_closures(
        loop_closures,
        max_loop_closures=30,
        outlier_threshold=2.0
    )
    
    # Build the factor graph with adaptive noise models
    graph, initial_estimate = build_pose_graph(odom_poses, filtered_loop_closures)
    
    # Optimize the graph using GTSAM
    result = optimize_pose_graph(graph, initial_estimate)
    
    # Plot the initial vs. optimized trajectories
    plot_trajectories(odom_poses, result, filtered_loop_closures)
    
    # Save the optimized trajectory for mapping
    optimized_array = np.array([
        [result.atPose2(i).x(), result.atPose2(i).y(), result.atPose2(i).theta()] 
        for i in range(result.size())
    ])
    np.save("optimized_trajectory.npy", optimized_array)
    print("Optimized trajectory saved to optimized_trajectory.npy.")

if __name__ == '__main__':
    main()