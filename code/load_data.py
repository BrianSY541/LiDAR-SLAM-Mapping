import numpy as np

'''
if __name__ == '__main__':
  dataset = 20
  
  with np.load("../data/Encoders%d.npz"%dataset) as data:
    encoder_counts = data["counts"] # 4 x n encoder counts
    encoder_stamps = data["time_stamps"] # encoder time stamps

  with np.load("../data/Hokuyo%d.npz"%dataset) as data:
    lidar_angle_min = data["angle_min"] # start angle of the scan [rad]
    lidar_angle_max = data["angle_max"] # end angle of the scan [rad]
    lidar_angle_increment = data["angle_increment"] # angular distance between measurements [rad]
    lidar_range_min = data["range_min"] # minimum range value [m]
    lidar_range_max = data["range_max"] # maximum range value [m]
    lidar_ranges = data["ranges"]       # range data [m] (Note: values < range_min or > range_max should be discarded)
    lidar_stamps = data["time_stamps"]  # acquisition times of the lidar scans
    
  with np.load("../data/Imu%d.npz"%dataset) as data:
    imu_angular_velocity = data["angular_velocity"] # angular velocity in rad/sec
    imu_linear_acceleration = data["linear_acceleration"] # accelerations in gs (gravity acceleration scaling)
    imu_stamps = data["time_stamps"]  # acquisition times of the imu measurements
  
  with np.load("../data/Kinect%d.npz"%dataset) as data:
    disp_stamps = data["disparity_time_stamps"] # acquisition times of the disparity images
    rgb_stamps = data["rgb_time_stamps"] # acquisition times of the rgb images
'''

def inspect_npz(file_path):
    with np.load(file_path) as data:
        print("File:", file_path)
        print("Keys:", data.files)
        for key in data.files:
            arr = data[key]
            print("  - Key: '{}'\t Shape: {} \t Type: {}".format(key, arr.shape, arr.dtype))
        print("\n" + "="*50 + "\n")

def main():
    dataset = 20  # 可根據需要修改 dataset 編號
    sensor_files = {
        "Encoders": f"../data/Encoders{dataset}.npz",
        "Hokuyo": f"../data/Hokuyo{dataset}.npz",
        "Imu": f"../data/Imu{dataset}.npz",
        "Kinect": f"../data/Kinect{dataset}.npz"
    }
    
    for sensor, file_path in sensor_files.items():
        print("Inspecting {} data:".format(sensor))
        inspect_npz(file_path)

if __name__ == '__main__':
    main()
