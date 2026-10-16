'''
This script runs our InEKF with Partial Orientation Measurements, the MEKF, and the InEKF with full orientation measurements
on the Holoocean simulation data, and generates the results for Figures 3 and 4 in the paper.

As this script is running 3 different filters 100 times across the entire trajectory, it can take a while to run.
Outputs are saved to an npz file for later plotting if needed without re-running the entire simulation. The generated plots are also saved to the figures folder.
'''
from pathlib import Path
from rosbags.image import message_to_cvimage
from rosbags.highlevel import AnyReader
import numpy as np

import sys
import os

from tqdm import tqdm
# Add the project root to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from helpers.function_helpers import convert_filter_mus_to_rpy_xyz, compute_position_mae, imu_add_noise, gps_add_noise, heading_add_noise, add_initial_state_noise, plot_figure_3, plot_figure_4
from src.core.jitter_buffer import JitterBuffer
from src.core.image_handler import ImageProcessor

from src.estimators.invariant_ekf import InEKF
from src.estimators.multiplicative_ekf import MEKF

from helpers.params import HolooceanConfigParams

from scipy.spatial.transform import Rotation

'''
Topic: /holoocean/asv0/FrontCamera        | Type: sensor_msgs/msg/Image             | Used for horizon measurement
Topic: /holoocean/asv0/GPSSensor          | Type: nav_msgs/msg/Odometry             | Used for position measurement
Topic: /holoocean/asv0/IMUSensor          | Type: sensor_msgs/msg/Imu               | Used for state propagation
Topic: /holoocean/asv0/RotationSensor     | Type: geometry_msgs/msg/Vector3Stamped  | Used for heading measurement
Topic: /holoocean/asv0/PoseSensor         | Type: nav_msgs/msg/Odometry             | Used for Ground Truth
'''

plot_images = False

path_to_bag = "data/holoocean_sim_data"
num_msgs = 0
num_images = 0


# Parameters
JITTER_THRESHOLD = 1_000_000_000  # ms lookahead
config_params = HolooceanConfigParams()
camera_processor = ImageProcessor(config_params.mtx, config_params.dist, config_params.img_width, config_params.img_height, camera_height=config_params.camera_height)

from src.core.measurements import * # Import measurement types


# It's infeasible to process every image for every monte carlo simulation, so we store the first result here
dict_horizon_measurements = {} # indexed by timestamp

print("Running the full Monte Carlo simulations can take awhile, if you just want the data you can find\
      how to download it in the README")

print("You can also decrease the number of simulations or the length of the trajectory used.")

num_simulations = 100
convergence_plot_length_seconds = 10
simulation_length_s = 172

inekf_full_traj_maes = [] # Tracking MAE for each simulation across entire trajectory
mekf_full_traj_maes = []
inEKF_full_pose_full_traj_maes = []
inekf_5hz_conv_trajectories = [] # Tracking convergence over first N seconds
inekf_no_horizon_trajectories = [] # Tracking convergence over first N seconds
mekf_5hz_conv_trajectories = []
mekf_no_horizon_trajectories = []


for i in tqdm(range(num_simulations), desc="Simulations"):
    buffer = JitterBuffer(delay_ns=JITTER_THRESHOLD)
    first_timestamp = None
    simulation_enabled = False
    waiting_threshold = 5 # Simulation is a little noisy at the start as the vehicle drops into the water
    exit_flag = False
    ground_truth_poses = []

    # Initial random walk for IMU noise
    current_random_walk = np.zeros(6) + np.random.normal(0, config_params.init_cov[9:15, 9:15].diagonal()**0.5, size=6)

    last_horizon_observation = None

    with AnyReader([Path(path_to_bag)]) as reader:
        connections = [x for x in reader.connections]

        for connection, timestamp, rawdata in reader.messages(connections=connections):
            if not simulation_enabled:
                if connection.topic == "/holoocean/asv0/PoseSensor":
                    msg = reader.deserialize(rawdata, connection.msgtype)
                    timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

                    if first_timestamp is None:
                        first_timestamp = timestamp
            
                    if timestamp - first_timestamp > waiting_threshold: # Waiting threshold used for vehicle to stabilize in simulation
                        simulation_enabled = True  # Skip until after waiting period
                        first_sim_timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9 # Reset first timestamp for simulation time, not bag time
                        orientation_quat=np.array([msg.pose.pose.orientation.x,
                                                msg.pose.pose.orientation.y,
                                                msg.pose.pose.orientation.z,
                                                msg.pose.pose.orientation.w]),
                        rpy = Rotation.from_quat(orientation_quat).as_euler('xyz', degrees=False)[0]
                        init_state = np.array([np.deg2rad(rpy[0]), np.deg2rad(rpy[1]), np.deg2rad(rpy[2]), 0, 0, 0, msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z])  # [roll, pitch, yaw, vx, vy, vz, x, y, z]
                        ground_truth_poses.append((first_sim_timestamp, rpy[0], rpy[1], rpy[2], msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z))
                        
                        init_state_noisy = init_state

                        inEKF_full_traj = InEKF(config_params.init_cov, config_params.imu_noise_params, init_state=init_state_noisy, first_timestamp=first_sim_timestamp)
                        MEKF_full_traj = MEKF(config_params.init_cov, config_params.imu_noise_params, init_state=init_state_noisy, first_timestamp=first_sim_timestamp)
                        inEKF_full_pose_full_traj = InEKF(config_params.init_cov, config_params.imu_noise_params, init_state=init_state_noisy, first_timestamp=first_sim_timestamp)
                        
                        init_state_high_noise = add_initial_state_noise(init_state, config_params.init_cov_high)
                        inEKF_5hz_conv = InEKF(config_params.init_cov_high, config_params.imu_noise_params, init_state=init_state_high_noise, first_timestamp=first_sim_timestamp)
                        inEKF_no_horizon = InEKF(config_params.init_cov_high, config_params.imu_noise_params, init_state=init_state_high_noise, first_timestamp=first_sim_timestamp)

                        MEKF_5hz_conv = MEKF(config_params.init_cov_high, config_params.imu_noise_params, init_state=init_state_high_noise, first_timestamp=first_sim_timestamp)
                        MEKF_no_horizon = MEKF(config_params.init_cov_high, config_params.imu_noise_params, init_state=init_state_high_noise, first_timestamp=first_sim_timestamp)

                continue  # Skip processing until simulation is enabled

            buffer.push(timestamp, connection, rawdata) # Messages pushed to jitter buffer
                                                        # Not needed for simulation, but keeping for real world data use.

            for ready_msg in buffer.pop_ready():
                msg = reader.deserialize(ready_msg.data, ready_msg.connection.msgtype)
                connection = ready_msg.connection
                if connection.topic == "/holoocean/asv0/IMUSensor":
                    timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                    msg.angular_velocity
                    msg.linear_acceleration
                    imu_measurement = ImuMeasurement(timestamp_sec=msg.header.stamp.sec,
                                                timestamp_nsec=msg.header.stamp.nanosec,
                                                gyro=np.array([msg.angular_velocity.x,
                                                                msg.angular_velocity.y,
                                                                msg.angular_velocity.z]),
                                                accel=np.array([msg.linear_acceleration.x,
                                                                msg.linear_acceleration.y,
                                                                msg.linear_acceleration.z]))
                    
                    imu_measurement, current_random_walk = imu_add_noise(imu_measurement, config_params.imu_noise_params, current_random_walk)

                    inEKF_full_traj.imu_callback(imu_measurement)
                    inEKF_full_pose_full_traj.imu_callback(imu_measurement)
                    MEKF_full_traj.imu_callback(imu_measurement)
                    
                    if timestamp - first_sim_timestamp < convergence_plot_length_seconds:
                        inEKF_5hz_conv.imu_callback(imu_measurement)
                        inEKF_no_horizon.imu_callback(imu_measurement)
                        MEKF_5hz_conv.imu_callback(imu_measurement)
                        MEKF_no_horizon.imu_callback(imu_measurement)

                    if timestamp - first_sim_timestamp > simulation_length_s:
                        exit_flag = True
                        break

                elif connection.topic == "/holoocean/asv0/GPSSensor":
                    timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

                    gps_measurement = GPSMeasurement(timestamp_sec=msg.header.stamp.sec,
                                                    timestamp_nsec=msg.header.stamp.nanosec,
                                                x=msg.pose.pose.position.x,
                                                y=msg.pose.pose.position.y,
                                                z=msg.pose.pose.position.z,
                                                position_cov=config_params.gps_position_cov)
                    
                    gps_measurement = gps_add_noise(gps_measurement)

                    inEKF_full_traj.gps_callback(gps_measurement)
                    MEKF_full_traj.gps_callback(gps_measurement)
                    inEKF_full_pose_full_traj.gps_callback(gps_measurement)
                    if timestamp - first_sim_timestamp < convergence_plot_length_seconds:
                        inEKF_5hz_conv.gps_callback(gps_measurement)
                        inEKF_no_horizon.gps_callback(gps_measurement)
                        MEKF_5hz_conv.gps_callback(gps_measurement)
                        MEKF_no_horizon.gps_callback(gps_measurement)
                
                elif connection.topic == "/holoocean/asv0/RotationSensor": # Used for heading measurement
                    timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                    yaw = msg.vector.z  
                    heading_meas = HeadingMeasurement(timestamp_sec=msg.header.stamp.sec,
                                                            timestamp_nsec=msg.header.stamp.nanosec,
                                                            heading=np.deg2rad(yaw),
                                                            heading_cov=config_params.heading_noise_var)
                    
                    heading_meas = heading_add_noise(heading_meas)

                    inEKF_full_traj.heading_callback(heading_meas)
                    MEKF_full_traj.heading_callback(heading_meas)
                    if last_horizon_observation is not None:
                        inEKF_full_pose_full_traj.orientation_callback_combined(heading_meas, last_horizon_observation)

                    if timestamp - first_sim_timestamp < convergence_plot_length_seconds:
                        inEKF_5hz_conv.heading_callback(heading_meas)   
                        inEKF_no_horizon.heading_callback(heading_meas)
                        MEKF_5hz_conv.heading_callback(heading_meas)
                        MEKF_no_horizon.heading_callback(heading_meas)

                elif connection.topic == "/holoocean/asv0/FrontCamera":
                    num_images += 1
                    timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

                    # Avoid re-processing images for monte carlo simulations
                    if timestamp not in dict_horizon_measurements:
                        cv_image = message_to_cvimage(msg, "bgr8")
                        horizon_observation = camera_processor.lsd_image_callback(cv_image, debug=plot_images)
                        dict_horizon_measurements[timestamp] = horizon_observation  # Store first result only
                    else:
                        horizon_observation = dict_horizon_measurements[timestamp]

                    if horizon_observation.valid:
                        last_horizon_observation = horizon_observation
                        inEKF_full_traj.image_callback(horizon_observation)
                        MEKF_full_traj.image_callback(horizon_observation)
                        if num_images % 4 == 0: # Grab every 4th image for convergence plots
                            if timestamp - first_sim_timestamp < convergence_plot_length_seconds:
                                inEKF_5hz_conv.image_callback(horizon_observation)
                                MEKF_5hz_conv.image_callback(horizon_observation)
                                        
                elif connection.topic == "/holoocean/asv0/PoseSensor": 
                    timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                    pose_meas = PoseMeasurement(timestamp_sec=msg.header.stamp.sec,
                                            timestamp_nsec=msg.header.stamp.nanosec,
                                            position=np.array([msg.pose.pose.position.x,
                                                                msg.pose.pose.position.y,
                                                                msg.pose.pose.position.z]),
                                            orientation=np.array([msg.pose.pose.orientation.x,
                                                                msg.pose.pose.orientation.y,
                                                                msg.pose.pose.orientation.z,
                                                                msg.pose.pose.orientation.w]),
                                            pose_cov=np.eye(6) * 0.0)  # GT meas is exact
                    
                    rpy = Rotation.from_quat([msg.pose.pose.orientation.x,
                                            msg.pose.pose.orientation.y,
                                            msg.pose.pose.orientation.z,
                                            msg.pose.pose.orientation.w]).as_euler('xyz', degrees=True)
                    pos = pose_meas.position
                    ground_truth_poses.append((timestamp, rpy[0], rpy[1], rpy[2], pos[0], pos[1], pos[2]))

            if exit_flag:
                break

        for final_msg in buffer.flush(): # Clear the buffer for now
        # Process the last few bits of data if any
            pass

    inekf_5hz_conv_trajectories.append(convert_filter_mus_to_rpy_xyz(inEKF_5hz_conv))
    inekf_no_horizon_trajectories.append(convert_filter_mus_to_rpy_xyz(inEKF_no_horizon))
    mekf_5hz_conv_trajectories.append(convert_filter_mus_to_rpy_xyz(MEKF_5hz_conv))
    mekf_no_horizon_trajectories.append(convert_filter_mus_to_rpy_xyz(MEKF_no_horizon))

    error_xy, error_z, error_rpy = compute_position_mae(convert_filter_mus_to_rpy_xyz(inEKF_full_traj), np.array(ground_truth_poses))
    error_xy_full_pose, error_z_full_pose, error_rpy_full_pose = compute_position_mae(convert_filter_mus_to_rpy_xyz(inEKF_full_pose_full_traj), np.array(ground_truth_poses))
    error_xy_mekf, error_z_mekf, error_rpy_mekf = compute_position_mae(convert_filter_mus_to_rpy_xyz(MEKF_full_traj), np.array(ground_truth_poses))
    
    inekf_full_traj_maes.append((error_xy, error_z, error_rpy[0], error_rpy[1], error_rpy[2]))
    inEKF_full_pose_full_traj_maes.append((error_xy_full_pose, error_z_full_pose, error_rpy_full_pose[0], error_rpy_full_pose[1], error_rpy_full_pose[2]))
    mekf_full_traj_maes.append((error_xy_mekf, error_z_mekf, error_rpy_mekf[0], error_rpy_mekf[1], error_rpy_mekf[2]))



# Save to npz for later plotting
np.savez("figure_data/inekf_holoocean_simulation_results.npz",
        inekf_full_traj_maes=np.array(inekf_full_traj_maes),
        inEKF_full_pose_full_traj_maes=np.array(inEKF_full_pose_full_traj_maes),
        inekf_5hz_conv_trajectories=np.array(inekf_5hz_conv_trajectories, dtype=object),
        inekf_no_horizon_trajectories=np.array(inekf_no_horizon_trajectories, dtype=object),
        mekf_full_traj_maes=np.array(mekf_full_traj_maes),
        mekf_5hz_conv_trajectories=np.array(mekf_5hz_conv_trajectories, dtype=object),
        mekf_no_horizon_trajectories=np.array(mekf_no_horizon_trajectories, dtype=object),
        gt_data=np.array(ground_truth_poses)
        )

print("InEKF Full Trajectory MAEs (XY, Z, RPY) over simulations:", np.mean(inekf_full_traj_maes, axis=0))
plot_figure_3(np.array(inekf_full_traj_maes), np.array(mekf_full_traj_maes), np.array(inEKF_full_pose_full_traj_maes), show_plot=False)
plot_figure_4(np.array(inekf_5hz_conv_trajectories), np.array(inekf_no_horizon_trajectories), np.array(mekf_5hz_conv_trajectories), np.array(mekf_no_horizon_trajectories), gt_data=np.array(ground_truth_poses), show_plot=True)

