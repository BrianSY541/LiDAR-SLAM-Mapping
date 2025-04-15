import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------
# 1) Load Data (using np.load to read Encoders and IMU npz files)
# -----------------------------------------------------------
def load_sensor_data(dataset_id=20):
    """
    Loads Encoders and IMU data for the given dataset_id.
    
    Returns:
        encoder_counts: (4, N) Encoder counts for the four wheels
        encoder_stamps: (N,) Timestamps for encoder readings
        imu_angular_velocity: (3, M) IMU angular velocities (x, y, z)
        imu_stamps: (M,) Timestamps for IMU readings
    """
    with np.load(f"../data/Encoders{dataset_id}.npz") as data:
        encoder_counts = data["counts"]      # (4, N)
        encoder_stamps = data["time_stamps"]   # (N,)
    with np.load(f"../data/Imu{dataset_id}.npz") as data:
        imu_angular_velocity = data["angular_velocity"]  # (3, M)
        imu_stamps = data["time_stamps"]                 # (M,)
    return encoder_counts, encoder_stamps, imu_angular_velocity, imu_stamps


# -----------------------------------------------------------
# 2) Time Alignment and Interpolation (Compute v(t), w(t), and dt)
# -----------------------------------------------------------
def compute_velocity_and_yawrate(encoder_counts, encoder_stamps, imu_angular_velocity, imu_stamps, ticks_to_meter=0.0022):
    """
    Computes the linear velocity v(t) from encoder_counts and the angular velocity w(t) (yaw rate)
    from the IMU data. For each encoder interval, it calculates dt and selects the IMU yaw rate corresponding
    to the midpoint of the interval.
    
    Returns:
        t_all: Start times for each encoder interval (encoder_stamps[:-1])
        v_all: Linear velocity for each interval (length N-1)
        w_all: Yaw rate for each interval (length N-1)
        dt_all: Time differences for each interval (length N-1)
    """
    dt_all = []
    distances = []
    N = len(encoder_stamps)
    # Compute dt using np.diff
    dt_all = np.diff(encoder_stamps)
    
    # Compute the distance traveled in each interval
    for i in range(N - 1):
        # If dt is 0, set distance to 0 (to avoid division by zero)
        dt = dt_all[i]
        # Get the encoder increment for this interval; encoder_counts[0, i] is the increment at this time step
        delta_fr = encoder_counts[0, i]
        delta_fl = encoder_counts[1, i]
        delta_rr = encoder_counts[2, i]
        delta_rl = encoder_counts[3, i]
        # Calculate the distance traveled by the right and left wheels
        dist_right = 0.5 * (delta_fr + delta_rr) * ticks_to_meter
        dist_left  = 0.5 * (delta_fl + delta_rl) * ticks_to_meter
        # Calculate the average distance traveled
        dist_avg = 0.5 * (dist_right + dist_left)
        distances.append(dist_avg)
    
    # Compute linear velocity v = distance / dt
    v_all = np.array([distances[i] / dt_all[i] if dt_all[i] != 0 else 0 for i in range(len(dt_all))])
    
    # Extract the yaw rate (z-axis) from the IMU data
    imu_yaw = imu_angular_velocity[2, :]
    w_all = []
    t_all = encoder_stamps[:-1]  # Start time for each interval
    for i in range(len(t_all)):
        # Use the midpoint of the interval as the corresponding time
        t_mid = 0.5 * (encoder_stamps[i] + encoder_stamps[i+1])
        # Find the index in the IMU timestamps that is closest to t_mid
        idx = np.argmin(np.abs(imu_stamps - t_mid))
        w_all.append(imu_yaw[idx])
    
    return t_all, v_all, np.array(w_all), dt_all


# -----------------------------------------------------------
# 3) Differential Drive Odometry Integration
# -----------------------------------------------------------
def integrate_odometry(dt_all, v_all, w_all, x0=0.0, y0=0.0, theta0=0.0):
    """
    Integrates the odometry using a differential drive model:
        x_{t+1} = x_t + v_t * dt * cos(theta_t)
        y_{t+1} = y_t + v_t * dt * sin(theta_t)
        theta_{t+1} = theta_t + w_t * dt
    
    Returns:
        X, Y, THETA: Arrays representing the trajectory (length = len(dt_all)+1)
    """
    N = len(dt_all)
    X = np.zeros(N + 1)
    Y = np.zeros(N + 1)
    THETA = np.zeros(N + 1)
    
    # Set initial pose
    X[0], Y[0], THETA[0] = x0, y0, theta0
    
    # Integrate the pose for each interval
    for i in range(N):
        X[i+1] = X[i] + v_all[i] * dt_all[i] * np.cos(THETA[i])
        Y[i+1] = Y[i] + v_all[i] * dt_all[i] * np.sin(THETA[i])
        THETA[i+1] = THETA[i] + w_all[i] * dt_all[i]
    
    return X, Y, THETA


# -----------------------------------------------------------
# 4) Main Function
# -----------------------------------------------------------
def main():
    # List of dataset IDs to process
    dataset_ids = [20, 21]
    for id in dataset_ids:
        # (a) Load sensor data
        encoder_counts, encoder_stamps, imu_angular_velocity, imu_stamps = load_sensor_data(id)
        
        # (b) Compute linear velocity, angular velocity, and dt for each interval
        t_all, v_all, w_all, dt_all = compute_velocity_and_yawrate(
            encoder_counts, encoder_stamps, 
            imu_angular_velocity, imu_stamps,
            ticks_to_meter=0.0022
        )
        
        # (c) Integrate odometry using the differential drive model
        X, Y, THETA = integrate_odometry(dt_all, v_all, w_all, x0=0.0, y0=0.0, theta0=0.0)
        
        # (d) Plot the robot trajectory with grid and equal axis scaling
        plt.figure()
        plt.plot(X, Y, label='Odometry Trajectory')
        plt.grid(True)
        plt.axis('equal')
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.title(f'Robot Trajectory from Encoder+IMU Odometry for Dataset {id}')
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main()
