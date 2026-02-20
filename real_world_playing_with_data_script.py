from pathlib import Path
from matplotlib.lines import Line2D
from rosbags.image import message_to_cvimage
from rosbags.highlevel import AnyReader
from matplotlib import pyplot as plt
import numpy as np

import sys
import os

from tqdm import tqdm
# Add the project root to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from helpers.function_helpers import compare_two_filters, convert_filter_mus_to_rpy_xyz, compute_position_mae, fuse_sbg_ekf_msgs, get_filter_error_metrics, imu_add_noise, gps_add_noise, heading_add_noise, add_initial_state_noise, plot_biases, plot_cov, plot_figure_3, plot_figure_4, plot_figure_5_bar_graph, plot_results, plot_results_comparison_gt, plot_results_cov, convert_gps_to_xyz, sbg_ekf_to_filter, plot_results_comparison_gt_cov
from src.core.jitter_buffer import JitterBuffer
from src.core.image_handler import ImageProcessor

from src.estimators.invariant_ekf import InEKF
from src.estimators.multiplicative_ekf import MEKF
from helpers.load_sbg_types import typestore as sbg_typestore

from helpers.params import RealWorldOAKConfigParams, RealWorldSBGConfigParams

import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation

plot_images = False

# path_to_bag = "data/nav_and_forward_cam_seg"
path_to_bag = "/home/laser14/Documents/inekf_hawaii/data/nav_and_forward_cam_seg_fixed_rollover"
# path_to_bag = "/home/laser14/Documents/inekf_hawaii/data/reordered_nav_and_forward_cam_seg_fixed_rollover"
num_msgs = 0
num_images = 0

rp_measurement_history = [] # TODO: Remove this: For debugging horizon measurement validity and values over time, which could be useful for diagnosing convergence issues
measurement_order_history = [] # TODO: Remove this: For debugging message ordering issues
raw_arrival_history = []  # (timestamp, topic)
num_consecutive_image_topics = 0


gps_measurements = [] # timestamp, x, y, alt
ekf_nav_measurements = [] # timestamp, x, y, alt

# # Parameters
config_params_sbg = RealWorldSBGConfigParams()
config_params_oak = RealWorldOAKConfigParams()
camera_processor = ImageProcessor(config_params_sbg.mtx, config_params_sbg.dist, config_params_sbg.img_width, config_params_sbg.img_height, camera_height=config_params_sbg.camera_height)

from src.core.measurements import * # Import measurement types

convergence_plot_length_seconds = 10
simulation_length_s = 80

inekf_full_traj_maes = [] # Tracking MAE for each simulation across entire trajectory
mekf_full_traj_maes = []
inEKF_full_pose_full_traj_maes = []
inekf_5hz_conv_trajectories = [] # Tracking convergence over first N seconds
inekf_no_horizon_trajectories = [] # Tracking convergence over first N seconds
mekf_5hz_conv_trajectories = []
mekf_no_horizon_trajectories = []

JITTER_THRESHOLD = 700_000_000 # 1_000_000_000  # 1000ms lookahead
buffer = JitterBuffer(delay_ns=JITTER_THRESHOLD)
first_timestamp = None
filter_started = False
exit_flag = False
ground_truth_poses = []
sbg_nav = []
sbg_rpy = []

num_heading_msgs = 0
num_gps_msgs = 0

last_horizon_observation = None

last_popped_topic = None
with AnyReader([Path(path_to_bag)]) as reader:
    connections = [x for x in reader.connections]

    for connection, timestamp, rawdata in reader.messages(connections=connections):
        if connection.topic == "/imu/nav_sat_fix" or connection.topic == '/sbg/gps_hdt' or connection.topic == '/oak_d_lr_1/rgb/image_raw' or connection.topic == '/oak_d_lr_1/imu/data' or connection.topic == '/imu/data':
            msg = sbg_typestore.deserialize_cdr(rawdata, connection.msgtype)
            timestamp_ns = timestamp_ns = (msg.header.stamp.sec * 1_000_000_000) + msg.header.stamp.nanosec
            raw_arrival_history.append((timestamp_ns, connection.topic)) # Messages pushed to jitter buffer.sec * 1_000_000_000 + timestamp.nanosec
        
        if not filter_started:
            # if connection.topic == "/holoocean/asv0/PoseSensor":


            if connection.topic == '/sbg/ekf_nav':
                msg = sbg_typestore.deserialize_cdr(rawdata, connection.msgtype)
                timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                lat, long, alt = msg.latitude, msg.longitude, msg.altitude
                x, y = convert_gps_to_xyz(lat, long)
                vx = msg.velocity.x
                vy = msg.velocity.y
                vz = msg.velocity.z

            
                # msg = reader.deserialize(rawdata, connection.msgtype)
                # timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

                if first_timestamp is None:
                    first_timestamp = timestamp
        
                # if timestamp - first_timestamp < 45:
                #     continue

                first_timestamp = timestamp
                sbg_nav.append((timestamp, x, y, alt, vx, vy, vz))
                filter_started = True 
                # orientation_quat=np.array([msg.pose.pose.orientation.x,
                                        # msg.pose.pose.orientation.y,
                                        # msg.pose.pose.orientation.z,
                                        # msg.pose.pose.orientation.w]),
                # rpy = Rotation.from_quat(orientation_quat).as_euler('xyz', degrees=False)[0]
                # rpy = [5.65, -5.28, 159.6] 
                rpy = [0, 0, 159.6] 
                # rpy = [0, 0, 0] # Starting at the identity
                init_state = np.array([np.deg2rad(rpy[0]), np.deg2rad(rpy[1]), np.deg2rad(rpy[2]), 0, 0, 0, x, y, alt])  # [roll, pitch, yaw, vx, vy, vz, x, y, z]
                # ground_truth_poses.append((first_timestamp, rpy[0], rpy[1], rpy[2], msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z))
                
                # init_state_noisy = add_initial_state_noise(init_state,  config_params_sbg.init_cov)
                
                inEKF_sbg_full_traj = InEKF(config_params_sbg.init_cov, config_params_sbg.imu_noise_params, init_state=init_state, first_timestamp=first_timestamp, est_bias=True)
                MEKF_sbg_full_traj = MEKF(config_params_sbg.init_cov, config_params_sbg.imu_noise_params, init_state=init_state, first_timestamp=first_timestamp, est_bias=True)
                inEKF_sbg_full_pose_full_traj = InEKF(config_params_sbg.init_cov, config_params_sbg.imu_noise_params, init_state=init_state, first_timestamp=first_timestamp, est_bias=True)
                
                inEKF_oak_full_traj = InEKF(config_params_oak.init_cov, config_params_oak.imu_noise_params, init_state=init_state, first_timestamp=first_timestamp, update_rate=178)
                MEKF_oak_full_traj = MEKF(config_params_oak.init_cov, config_params_oak.imu_noise_params, init_state=init_state, first_timestamp=first_timestamp, update_rate=178)
                inEKF_oak_full_pose_full_traj = InEKF(config_params_oak.init_cov, config_params_oak.imu_noise_params, init_state=init_state, first_timestamp=first_timestamp, update_rate=178)
                    
                    # init_state_high_noise = add_initial_state_noise(init_state, config_params.init_cov_high)
                    # inEKF_5hz_conv = InEKF(config_params.init_cov_high, config_params.imu_noise_params, init_state=init_state_high_noise, first_timestamp=first_timestamp)
                    # inEKF_no_horizon = InEKF(config_params.init_cov_high, config_params.imu_noise_params, init_state=init_state_high_noise, first_timestamp=first_timestamp)

                    # MEKF_5hz_conv = MEKF(config_params.init_cov_high, config_params.imu_noise_params, init_state=init_state_high_noise, first_timestamp=first_timestamp)
                    # MEKF_no_horizon = MEKF(config_params.init_cov_high, config_params.imu_noise_params, init_state=init_state_high_noise, first_timestamp=first_timestamp)

            continue  # Skip processing until simulation is enabled

        try:
            msg = reader.deserialize(rawdata, connection.msgtype)
            # msg = sbg_typestore.deserialize_cdr(rawdata, connection.msgtype)
            # timestamp = (msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)*1e9
            timestamp_ns = (msg.header.stamp.sec * 1_000_000_000) + msg.header.stamp.nanosec
            buffer.push(timestamp_ns, connection, rawdata) # Messages pushed to jitter buffer
        except:
            
            # print(f"Failed to deserialize message on topic {connection.topic} at timestamp {timestamp}. Skipping.")
            continue
                                                    # Not needed for simulation, but keeping for real world data use.

        for ready_msg in buffer.pop_ready():
            popped_topic = ready_msg.connection.topic
            if popped_topic == last_popped_topic and popped_topic == '/oak_d_lr_1/rgb/image_raw':
                # Known issue with Camera driver where multiple consecutive messages can have the same timestamp
                # Our quick fix is to only take the first one and then ignore subsequent messages with the same timestamp until we see a different timestamp. 
                print(f"Warning: Consecutive messages popped from buffer have the same topic {popped_topic}. This could indicate a large gap in timestamps or a burst of messages from this topic. Check for potential timestamp issues or bursts in the data.")
                num_consecutive_image_topics += 1
                continue

            last_popped_topic = popped_topic
            try:
                msg = reader.deserialize(ready_msg.data, ready_msg.connection.msgtype)
            except:
                # TODO: Remove messages in bag that aren't needed in this experiment
                continue
            connection = ready_msg.connection
            if connection.topic == '/imu/data':
                timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                measurement_order_history.append((timestamp, connection.topic)) # TODO: DELETEME
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
                
                # inEKF_sbg_full_traj.imu_callback(imu_measurement)
                # inEKF_sbg_full_pose_full_traj.imu_callback(imu_measurement)
                # MEKF_sbg_full_traj.imu_callback(imu_measurement)
                
                # if timestamp - first_timestamp < convergence_plot_length_seconds:
                #     inEKF_5hz_conv.imu_callback(imu_measurement)
                #     inEKF_no_horizon.imu_callback(imu_measurement)
                #     MEKF_5hz_conv.imu_callback(imu_measurement)
                #     MEKF_no_horizon.imu_callback(imu_measurement)

                if msg.header.stamp.sec - first_timestamp > simulation_length_s:
                    exit_flag = True

            elif connection.topic == '/oak_d_lr_1/imu/data': # OAK IMU topic
                
                timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                measurement_order_history.append((timestamp, connection.topic)) # TODO: DELETEME
                #
                angular_vel = [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]
                linear_accel = [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]
                rot = config_params_oak.rot_sbg_to_oak_c_imu # Need to rotate IMU measurements into robot frame
                angular_vel = rot @ np.array(angular_vel)
                linear_accel = rot @ np.array(linear_accel)
                
                imu_measurement = ImuMeasurement(timestamp_sec=msg.header.stamp.sec,
                                            timestamp_nsec=msg.header.stamp.nanosec,
                                            gyro= angular_vel,
                                            accel = linear_accel)
                
                # inEKF_oak_full_traj.imu_callback(imu_measurement)
                # inEKF_oak_full_pose_full_traj.imu_callback(imu_measurement)
                # MEKF_oak_full_traj.imu_callback(imu_measurement)
                
                # if timestamp - first_timestamp < convergence_plot_length_seconds:
                #     inEKF_5hz_conv.imu_callback(imu_measurement)
                #     inEKF_no_horizon.imu_callback(imu_measurement)
                #     MEKF_5hz_conv.imu_callback(imu_measurement)
                #     MEKF_no_horizon.imu_callback(imu_measurement)

                # if timestamp - first_timestamp > simulation_length_s:
                #     exit_flag = True


            # elif connection.topic == "/imu/nav_sat_fix":
            elif connection.topic == "/sbg/gps_pos": # SBG GPS topic
                # continue
                # msg = reader.deserialize(rawdata, connection.msgtype)
                num_gps_msgs += 1
                # if num_gps_msgs % 5 != 0:
                #     continue
                timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                measurement_order_history.append((timestamp, connection.topic)) # TODO: DELETEME
            
                lat, lon, alt = msg.latitude, msg.longitude, msg.altitude # IMU_nav_sat_fix and sbg_ekf_nav use different altitude references, so we need to align them for fair comparison
                x, y = convert_gps_to_xyz(lat, lon)
                gps_measurements.append((timestamp, x, y, alt))

                # position_covariance = msg.position_covariance
                # position_covariance = msg.position_accuracy
            
                gps_measurement = GPSMeasurement(timestamp_sec=msg.header.stamp.sec,
                                                timestamp_nsec=msg.header.stamp.nanosec,
                                                x=x,
                                                y=y,
                                                z=alt,
                                                position_cov=config_params_sbg.gps_position_cov)
                
                gps_measurement = gps_add_noise(gps_measurement) # Adding noise to corrupt RTK GPS measurement for testing

                # inEKF_sbg_full_traj.gps_callback(gps_measurement)
                # MEKF_sbg_full_traj.gps_callback(gps_measurement)
                # inEKF_sbg_full_pose_full_traj.gps_callback(gps_measurement)

                # inEKF_oak_full_traj.gps_callback(gps_measurement)
                # MEKF_oak_full_traj.gps_callback(gps_measurement)
                # inEKF_oak_full_pose_full_traj.gps_callback(gps_measurement)
                                
                # if timestamp - first_timestamp < convergence_plot_length_seconds:
                #     inEKF_5hz_conv.gps_callback(gps_measurement)
                #     inEKF_no_horizon.gps_callback(gps_measurement)
                #     MEKF_5hz_conv.gps_callback(gps_measurement)
                #     MEKF_no_horizon.gps_callback(gps_measurement)
            
            elif connection.topic == '/sbg/gps_hdt':
                # continue
                num_heading_msgs += 1
                if num_heading_msgs % 5 != 0:
                    continue
                msg = sbg_typestore.deserialize_cdr(ready_msg.data, ready_msg.connection.msgtype)
                timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                measurement_order_history.append((timestamp, connection.topic)) # TODO: DELETEME

                # True heading angle (0 to 360 deg)
                # ENU convention: Rotation about the up axis. Zero when the X axis is pointing East. (opposite sign compared to NED)
                heading = msg.true_heading - 180  # Degree
                heading_acc = msg.true_heading_acc # # 1 sigma True heading estimated accuracy

                # yaw = msg.vector.z  # Assuming Z is yaw
                heading_meas = HeadingMeasurement(timestamp_sec=msg.header.stamp.sec,
                                                        timestamp_nsec=msg.header.stamp.nanosec,
                                                        heading=np.deg2rad(heading),
                                                        heading_cov=config_params_sbg.heading_noise_var)
                
                heading_meas = heading_add_noise(heading_meas)

                # inEKF_sbg_full_traj.heading_callback(heading_meas)
                # MEKF_sbg_full_traj.heading_callback(heading_meas)
                # inEKF_oak_full_traj.heading_callback(heading_meas)
                # MEKF_oak_full_traj.heading_callback(heading_meas)

                # if last_horizon_observation is not None:
                    # inEKF_sbg_full_pose_full_traj.orientation_callback_combined(heading_meas, last_horizon_observation)
                    # inEKF_oak_full_pose_full_traj.orientation_callback_combined(heading_meas, last_horizon_observation)

                # if timestamp - first_timestamp < convergence_plot_length_seconds:
                    # inEKF_5hz_conv.heading_callback(heading_meas)   
                    # inEKF_no_horizon.heading_callback(heading_meas)
                    # MEKF_5hz_conv.heading_callback(heading_meas)
                    # MEKF_no_horizon.heading_callback(heading_meas)

            elif connection.topic == '/oak_d_lr_1/rgb/image_raw':
                # continue
                num_images += 1
                # continue # TODO: REMOVE ME!
                # if not num_images % 4 == 0: # TODO: Fix this!
                    # continue
                # msg = reader.deserialize(rawdata, connection.msgtype)
                timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                measurement_order_history.append((timestamp, connection.topic)) # TODO: DELETEME

                # cv_image = message_to_cvimage(msg, "bgr8")
                # horizon_observation = camera_processor.lsd_image_callback(cv_image, debug=plot_images)
                # rp_measurement_history.append((timestamp, horizon_observation.roll_rad, horizon_observation.pitch_rad, horizon_observation.valid))
                # if horizon_observation.valid:
                #     last_horizon_observation = horizon_observation
                #     # inEKF_sbg_full_traj.image_callback(horizon_observation)
                #     # MEKF_sbg_full_traj.image_callback(horizon_observation)
                #     # inEKF_oak_full_traj.image_callback(horizon_observation)
                #     # MEKF_oak_full_traj.image_callback(horizon_observation)
                # else:
                #     last_horizon_observation = None
                    # if num_images % 4 == 0:
                        # if timestamp - first_timestamp < convergence_plot_length_seconds:
                            # inEKF_5hz_conv.image_callback(horizon_observation)
                            # MEKF_5hz_conv.image_callback(horizon_observation)
                                    
            elif connection.topic == '/sbg/ekf_quat':
                msg = sbg_typestore.deserialize_cdr(ready_msg.data, ready_msg.connection.msgtype)
                timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                quat = [msg.quaternion.x, msg.quaternion.y, msg.quaternion.z, msg.quaternion.w]
                rpy = Rotation.from_quat(quat).as_euler('xyz', degrees=True)
                sbg_rpy.append((timestamp, rpy[0], rpy[1], rpy[2]))

            elif connection.topic == '/sbg/ekf_nav':
                msg = sbg_typestore.deserialize_cdr(ready_msg.data, ready_msg.connection.msgtype)
                timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                lat, long, alt = msg.latitude, msg.longitude, msg.altitude
                x, y = convert_gps_to_xyz(lat, long)
                ekf_nav_measurements.append((timestamp, x, y, alt))
                vx = msg.velocity.x
                vy = msg.velocity.y
                vz = msg.velocity.z

                sbg_nav.append((timestamp, x, y, alt, vx, vy, vz))


        if exit_flag:
            break

        
    
    # manager.plot_inekf_results()


    for final_msg in buffer.flush(): # Clear the buffer for now
    # Process the last few bits of data if any
        pass

gps_measurements = np.array(gps_measurements) # timestamp, x, y, alt
ekf_nav_measurements = np.array(ekf_nav_measurements) # timestamp, x, y, alt

def compare_gps_and_ekf_nav(gps_measurements, ekf_nav_measurements):
    # Create a 2x2 grid of subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # --- Subplot 1: XY Position (Top Left) ---
    axs[0, 0].plot(gps_measurements[:, 1], gps_measurements[:, 2], 'ro', alpha=0.4, label='GPS', markersize=3)
    axs[0, 0].plot(ekf_nav_measurements[:, 1], ekf_nav_measurements[:, 2], 'b-', label='EKF Nav')
    axs[0, 0].set_xlabel('$x$ Position $(m)$')
    axs[0, 0].set_ylabel('$y$ Position $(m)$')
    axs[0, 0].set_title('XY Trajectory')
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # --- Subplot 2: X vs Timestamp (Top Right) ---
    axs[0, 1].plot(gps_measurements[:, 0], gps_measurements[:, 1], 'ro', alpha=0.4, label='GPS', markersize=3)
    axs[0, 1].plot(ekf_nav_measurements[:, 0], ekf_nav_measurements[:, 1], 'b-', label='EKF Nav')
    axs[0, 1].set_xlabel('Timestamp $(s)$')
    axs[0, 1].set_ylabel('$x$ Position $(m)$')
    axs[0, 1].set_title('$x$ vs Time')
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # --- Subplot 3: Y vs Timestamp (Bottom Left) ---
    axs[1, 0].plot(gps_measurements[:, 0], gps_measurements[:, 2], 'ro', alpha=0.4, label='GPS', markersize=3)
    axs[1, 0].plot(ekf_nav_measurements[:, 0], ekf_nav_measurements[:, 2], 'b-', label='EKF Nav')
    axs[1, 0].set_xlabel('Timestamp $(s)$')
    axs[1, 0].set_ylabel('$y$ Position $(m)$')
    axs[1, 0].set_title('$y$ vs Time')
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # --- Subplot 4: Altitude vs Timestamp (Bottom Right) ---
    axs[1, 1].plot(gps_measurements[:, 0], gps_measurements[:, 3], 'ro', alpha=0.4, label='GPS', markersize=3)
    axs[1, 1].plot(ekf_nav_measurements[:, 0], ekf_nav_measurements[:, 3], 'b-', label='EKF Nav')
    axs[1, 1].set_xlabel('Timestamp $(s)$')
    axs[1, 1].set_ylabel('Altitude $(m)$')
    axs[1, 1].set_title('Altitude vs Time')
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    # Adjust layout to prevent label overlap
    plt.tight_layout()
    plt.show()

compare_gps_and_ekf_nav(gps_measurements, ekf_nav_measurements)

def plot_horizon_consistency(history):
    # history = [(timestamp, roll, pitch, valid), ...]
    timestamps = [x[0] for x in history]
    rolls = [x[1] for x in history]
    valid_mask = [x[3] for x in history]

    plt.figure(figsize=(14, 7))
    
    # 1. Plot the points
    for i in range(len(history)):
        if valid_mask[i]:
            plt.scatter(timestamps[i], rolls[i], color='blue', marker='o', s=40, alpha=0.6)
        else:
            plt.scatter(timestamps[i], rolls[i], color='red', marker='x', s=120, linewidths=3)

    # 2. Draw arrows to show sequence of arrival (the order in the list)
    for i in range(len(history) - 1):
        plt.annotate('', xy=(timestamps[i+1], rolls[i+1]), 
                     xytext=(timestamps[i], rolls[i]),
                     arrowprops=dict(arrowstyle="->", color='gray', alpha=0.3, lw=1))

    # 3. Highlight "Time Travel" (Out of order timestamps)
    for i in range(len(history) - 1):
        if timestamps[i+1] < timestamps[i]:
            plt.axvspan(timestamps[i+1], timestamps[i], color='red', alpha=0.1)
            plt.text((timestamps[i+1] + timestamps[i])/2, plt.ylim()[1], 'BACKWARDS', 
                     color='red', fontsize=8, ha='center', va='top')

    plt.title("Horizon Measurement Sequence & Outlier Rejection")
    plt.xlabel("Message Timestamp (seconds)")
    plt.ylabel("Roll (radians)")
    plt.grid(True, linestyle='--', alpha=0.5)

    # Custom Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Valid Measurement', markerfacecolor='blue', markersize=8),
        Line2D([0], [0], marker='x', color='w', label='Rejected (Outlier)', markeredgecolor='red', markersize=12, markeredgewidth=3),
        Line2D([0], [0], color='gray', lw=1, label='Arrival Order (Sequence)')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.show()

def plot_measurement_flow(history, title="Measurement Flow Over Time"):
    """
    history: List of tuples (timestamp_ns, topic_string)
    """
    plt.figure(figsize=(14, 8))
    if not history:
        print("History is empty.")
        return

    # 1. Extract and Normalize Data
    raw_timestamps = [int(x[0]) if isinstance(x[0], (int, float, str)) else x[0] for x in history]
    topics = [x[1] for x in history]
    
    start_time = raw_timestamps[0]
    normalized_times = [(t - start_time) / 1e9 for t in raw_timestamps]
    indices = np.arange(len(history))

    # 2. Setup Accessible Markers and High-Contrast Colors
    unique_topics = sorted(list(set(topics)))
    # Defining specific markers to ensure they are distinct
    markers = ['o', 's', '^', 'D', 'P', '*', 'v', 'X'] 
    # High contrast colors (Tableau 10)
    colors = plt.cm.get_cmap('tab10', len(unique_topics))

    # 3. Plot each topic with a unique marker + color combo
    for i, topic in enumerate(unique_topics):
        idx_mask = [j for j, t in enumerate(topics) if t == topic]
        t_vals = [normalized_times[j] for j in idx_mask]
        i_vals = [indices[j] for j in idx_mask]
        
        plt.scatter(i_vals, t_vals, 
                    label=topic, 
                    color=colors(i), 
                    marker=markers[i % len(markers)], 
                    s=60, 
                    edgecolor='black', 
                    linewidth=0.5,
                    alpha=0.9)

    # 4. Sequence Line (Lowered opacity to not hide markers)
    plt.plot(indices, normalized_times, color='black', alpha=0.15, linewidth=1, label='Arrival Sequence')

    # 5. Highlight Out-of-Order Dips
    for i in range(1, len(normalized_times)):
        if normalized_times[i] < normalized_times[i-1]:
            # Use a thick hollow circle to highlight backwards steps
            plt.scatter(indices[i], normalized_times[i], 
                        facecolors='none', edgecolors='red', 
                        s=150, lw=2, marker='o')

    plt.title(title)
    plt.xlabel("Arrival Index (Order received by filter)")
    plt.ylabel("Time since start (seconds)")
    plt.grid(True, linestyle=':', alpha=0.4)
    
    # Legend with shapes
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), title="Topics (By Shape)")
    
    plt.tight_layout()

print(f"Num total consecutive image topics: {num_consecutive_image_topics}")
print(f"Total image messages received: {num_images}")
plot_measurement_flow(raw_arrival_history, title="Raw Message Arrival Sequence (Colored by Topic, Shape Coded, with Out-of-Order Highlighting)")
plot_measurement_flow(measurement_order_history, title="Message Processing Sequence (Colored by Topic, Shape Coded, with Out-of-Order Highlighting)")
plt.show()
exit()
# plot_horizon_consistency(rp_measurement_history)

gt_poses = sbg_ekf_to_filter(fuse_sbg_ekf_msgs(np.array(sbg_rpy), np.array(sbg_nav)))

plot_cov(inEKF_sbg_full_traj, title="SBG InEKF Full Trajectory Covariance", show=False)
plot_cov(MEKF_sbg_full_traj, title="SBG MEKF Full Trajectory Covariance", show=False)
plot_cov(inEKF_sbg_full_pose_full_traj, title="SBG InEKF Full Pose Full Trajectory Covariance", show=False)
plot_cov(inEKF_oak_full_traj, title="OAK InEKF Full Trajectory Covariance", show=False)
plot_cov(MEKF_oak_full_traj, title="OAK MEKF Full Trajectory Covariance", show=False)
plot_cov(inEKF_oak_full_pose_full_traj, title="OAK InEKF Full Pose Full Trajectory Covariance", show=False)


plot_results_comparison_gt_cov(inEKF_sbg_full_traj, inEKF_oak_full_traj, gt_poses, filter1_name="SBG InEKF", filter2_name="OAK InEKF", title="InEKF Full Trajectory Comparison with GT", show=False)
plot_results_comparison_gt_cov(inEKF_sbg_full_pose_full_traj, inEKF_sbg_full_traj, gt_poses, filter1_name="SBG InEKF Full Pose", filter2_name="SBG InEKF", title="InEKF Full Trajectory Comparison with GT", show=False)
plot_results_comparison_gt_cov(inEKF_oak_full_pose_full_traj, inEKF_oak_full_traj, gt_poses, filter1_name="OAK InEKF Full Pose", filter2_name="OAK InEKF", title="InEKF Full Trajectory Comparison with GT", show=False)
plot_results_comparison_gt_cov(MEKF_sbg_full_traj, inEKF_sbg_full_traj, gt_poses, filter1_name="MEKF SBG", filter2_name="InEKF SBG", title="InEKF Full Trajectory Comparison with GT", show=False)
plot_results_comparison_gt_cov(MEKF_oak_full_traj, inEKF_oak_full_traj, gt_poses, filter1_name="MEKF OAK", filter2_name="InEKF OAK", title="InEKF Full Trajectory Comparison with GT", show=False)

convergence_period_seconds = 20
compare_two_filters(inEKF_sbg_full_traj, gt_poses, filter1_name="InEKF SBG Full Trajectory", filter2_name="GT", timeoffset=convergence_period_seconds)
compare_two_filters(inEKF_oak_full_traj, gt_poses, filter1_name="InEKF OAK Full Trajectory", filter2_name="GT", timeoffset=convergence_period_seconds)

compare_two_filters(inEKF_sbg_full_pose_full_traj, gt_poses, filter1_name="InEKF SBG Full Pose", filter2_name="GT", timeoffset=convergence_period_seconds)
compare_two_filters(inEKF_oak_full_pose_full_traj, gt_poses, filter1_name="InEKF OAK Full Pose", filter2_name="GT", timeoffset=convergence_period_seconds)
compare_two_filters(MEKF_sbg_full_traj, gt_poses, filter1_name="MEKF SBG", filter2_name="GT", timeoffset=convergence_period_seconds)
compare_two_filters(MEKF_oak_full_traj, gt_poses, filter1_name="MEKF OAK", filter2_name="GT", timeoffset=convergence_period_seconds)

# Metrics are mean, std, and max of xy position error, z position error, roll error, pitch error, and yaw error
sbg_inekf_metrics = get_filter_error_metrics(inEKF_sbg_full_traj, gt_poses, timeoffset=convergence_period_seconds)
sbg_mekf_metrics = get_filter_error_metrics(MEKF_sbg_full_traj, gt_poses, timeoffset=convergence_period_seconds)
sbg_inekf_full_pose_metrics = get_filter_error_metrics(inEKF_sbg_full_pose_full_traj, gt_poses, timeoffset=convergence_period_seconds)
oak_inekf_metrics = get_filter_error_metrics(inEKF_oak_full_traj, gt_poses, timeoffset=convergence_period_seconds)
oak_mekf_metrics = get_filter_error_metrics(MEKF_oak_full_traj, gt_poses, timeoffset=convergence_period_seconds)
oak_inekf_full_pose_metrics = get_filter_error_metrics(inEKF_oak_full_pose_full_traj, gt_poses, timeoffset=convergence_period_seconds) 


# TODO: Delete me after finalizing plotting
import pandas as pd

# Define the columns in the exact order of your return statement
columns = [
    'pos_mean', 'pos_std', 'pos_max',
    'z_mean', 'z_std', 'z_max',
    'roll_mean', 'roll_std', 'roll_max',
    'pitch_mean', 'pitch_std', 'pitch_max',
    'yaw_mean', 'yaw_std', 'yaw_max'
]

# Create a list of the data objects
raw_data = [
    sbg_inekf_metrics,
    sbg_mekf_metrics,
    sbg_inekf_full_pose_metrics,
    oak_inekf_metrics,
    oak_mekf_metrics,
    oak_inekf_full_pose_metrics
]

# Optional: Add names to identify the rows
filter_names = [
    "sbg_inekf", "sbg_mekf", "sbg_inekf_full_pose",
    "oak_inekf", "oak_mekf", "oak_inekf_full_pose"
]

# Create the DataFrame
df = pd.DataFrame(raw_data, columns=columns)

# Insert the names as the first column
df.insert(0, "filter_name", filter_names)

# Save to CSV
df.to_csv('filter_metrics_comparison.csv', index=False)
print("Saved comparison to filter_metrics_comparison.csv")

plot_biases(inEKF_sbg_full_traj, "SBG InEKF Bias", show=False)
plot_biases(MEKF_sbg_full_traj, "SBG MEKF Bias", show=False)
plot_biases(inEKF_sbg_full_pose_full_traj, "SBG InEKF Full Pose Bias", show=False)
plot_biases(inEKF_oak_full_traj, "OAK InEKF Bias", show=False)
plot_biases(MEKF_oak_full_traj, "OAK MEKF Bias", show=False)
plot_biases(inEKF_oak_full_pose_full_traj, "OAK InEKF Full Pose Bias", show=False)

plot_figure_5_bar_graph(df)

#SBG InEKF
#SBG MEKF
#SBG InEKF Full Pose
#OAK InEKF
#OAK MEKF
#OAK InEKF Full Pose
# plot_figure_5(inEKF_sbg_full_traj, MEKF_sbg_full_traj, inEKF_sbg_full_pose_full_traj, inEKF_oak_full_traj, MEKF_oak_full_traj, inEKF_oak_full_pose_full_traj, gt_poses, show_plot=True)

# plt.show()

# evaluate the simulation
# inekf_full_traj_maes = [] # Tracking MAE for each simulation across entire trajectory
# inekf_5hz_conv_trajectories = [] # Tracking convergence over first N seconds
# inekf_no_horizon_trajectories = [] # Tracking convergence over first N seconds

# inekf_5hz_conv_trajectories.append(convert_filter_mus_to_rpy_xyz(inEKF_5hz_conv))
# inekf_no_horizon_trajectories.append(convert_filter_mus_to_rpy_xyz(inEKF_no_horizon))
# mekf_5hz_conv_trajectories.append(convert_filter_mus_to_rpy_xyz(MEKF_5hz_conv))
# mekf_no_horizon_trajectories.append(convert_filter_mus_to_rpy_xyz(MEKF_no_horizon))

# sbg_error_xy, sbg_error_z, sbg_error_rpy = compute_position_mae(convert_filter_mus_to_rpy_xyz(inEKF_sbg_full_traj), np.array(ground_truth_poses))
# sbg_error_xy_full_pose, sbg_error_z_full_pose, sbg_error_rpy_full_pose = compute_position_mae(convert_filter_mus_to_rpy_xyz(inEKF_sbg_full_pose_full_traj), np.array(ground_truth_poses))
# sbg_error_xy_mekf, sbg_error_z_mekf, sbg_error_rpy_mekf = compute_position_mae(convert_filter_mus_to_rpy_xyz(MEKF_sbg_full_traj), np.array(ground_truth_poses))

# oak_error_xy, oak_error_z, oak_error_rpy = compute_position_mae(convert_filter_mus_to_rpy_xyz(inEKF_oak_full_traj), np.array(ground_truth_poses))
# oak_error_xy_full_pose, oak_error_z_full_pose, oak_error_rpy_full_pose = compute_position_mae(convert_filter_mus_to_rpy_xyz(inEKF_oak_full_pose_full_traj), np.array(ground_truth_poses))
# oak_error_xy_mekf, oak_error_z_mekf, oak_error_rpy_mekf = compute_position_mae(convert_filter_mus_to_rpy_xyz(MEKF_oak_full_traj), np.array(ground_truth_poses))





# inekf_full_traj_maes.append((error_xy, error_z, error_rpy[0], error_rpy[1], error_rpy[2]))
# inEKF_full_pose_full_traj_maes.append((error_xy_full_pose, error_z_full_pose, error_rpy_full_pose[0], error_rpy_full_pose[1], error_rpy_full_pose[2]))
# mekf_full_traj_maes.append((error_xy_mekf, error_z_mekf, error_rpy_mekf[0], error_rpy_mekf[1], error_rpy_mekf[2]))

# Save to npz for later plotting
# np.savez("inekf_real_world_results.npz",
#         inekf_full_traj_maes=np.array(inekf_full_traj_maes),
#         inEKF_full_pose_full_traj_maes=np.array(inEKF_full_pose_full_traj_maes),
#         inekf_5hz_conv_trajectories=np.array(inekf_5hz_conv_trajectories, dtype=object),
#         inekf_no_horizon_trajectories=np.array(inekf_no_horizon_trajectories, dtype=object),
#         mekf_full_traj_maes=np.array(mekf_full_traj_maes),
#         mekf_5hz_conv_trajectories=np.array(mekf_5hz_conv_trajectories, dtype=object),
#         mekf_no_horizon_trajectories=np.array(mekf_no_horizon_trajectories, dtype=object),
#         gt_data=np.array(ground_truth_poses)
#         )

# TODO: Uncomment all of these:
# plot_results(MEKF_full_traj, gt_data=np.array(ground_truth_poses), show=False, title="MEKF Full Trajectory", horizon_dict=dict_horizon_measurements)
# plot_results(inEKF_full_traj, gt_data=np.array(ground_truth_poses), show=False, title="InEKF Full Trajectory", horizon_dict=dict_horizon_measurements)
# plot_cov(inEKF_full_traj, title="InEKF Full Trajectory Covariance", show=False)
# plot_cov(MEKF_full_traj, title="MEKF Full Trajectory Covariance", show=False)

# print("InEKF Full Trajectory MAEs (XY, Z, RPY) over simulations:", np.mean(inekf_full_traj_maes, axis=0))
# plot_figure_3(np.array(inekf_full_traj_maes), np.array(mekf_full_traj_maes), np.array(inEKF_full_pose_full_traj_maes), show_plot=False)
# plot_figure_4(np.array(inekf_5hz_conv_trajectories), np.array(inekf_no_horizon_trajectories), np.array(mekf_5hz_conv_trajectories), np.array(mekf_no_horizon_trajectories), gt_data=np.array(ground_truth_poses), show_plot=True)


