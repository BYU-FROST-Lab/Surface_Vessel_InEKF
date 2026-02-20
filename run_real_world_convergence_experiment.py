from pathlib import Path
from rosbags.image import message_to_cvimage
from rosbags.highlevel import AnyReader
from matplotlib import pyplot as plt
import numpy as np

import sys
import os

from tqdm import tqdm
# Add the project root to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from helpers.function_helpers import compare_two_filters, convert_filter_mus_to_rpy_xyz, compute_position_mae, fuse_sbg_ekf_msgs, get_filter_error_metrics, imu_add_noise, gps_add_noise, heading_add_noise, add_initial_state_noise, plot_biases, plot_cov, plot_figure_3, plot_figure_4, plot_figure_5_bar_graph, plot_figure_6_rw_convergence, plot_results, plot_results_comparison_gt, plot_results_cov, convert_gps_to_xyz, sbg_ekf_to_filter, plot_results_comparison_gt_cov
from src.core.jitter_buffer import JitterBuffer
from src.core.image_handler import ImageProcessor

from src.estimators.invariant_ekf import InEKF
from src.estimators.multiplicative_ekf import MEKF
from helpers.load_sbg_types import typestore as sbg_typestore

from helpers.params import RealWorldOAKConfigParams, RealWorldSBGConfigParams

import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation
import pickle

plot_images = False

# path_to_bag = "data/nav_and_forward_cam_seg"
path_to_bag = "/home/laser14/Documents/inekf_hawaii/data/nav_and_forward_cam_seg_fixed_rollover"
# /home/laser14/Documents/holoocean_ros_ws/src/holoocean-ros/holoocean_open_water_loop_waves_gps_offset

# path_to_bag = "/home/laser14/Documents/inekf_hawaii/data/reordered_nav_and_forward_cam_seg_fixed_rollover"
num_msgs = 0
num_images = 0


# # Parameters
JITTER_THRESHOLD = 700_000_000  # 0.7 second lookahead
config_params_sbg = RealWorldSBGConfigParams()
config_params_oak = RealWorldOAKConfigParams()
camera_processor = ImageProcessor(config_params_sbg.mtx, config_params_sbg.dist, config_params_sbg.img_width, config_params_sbg.img_height, camera_height=config_params_sbg.camera_height)

from src.core.measurements import * # Import measurement types

convergence_plot_length_seconds = 2
convergence_plot_gap_seconds = 0.1

end_time = 0

# inekf_full_traj_maes = [] # Tracking MAE for each simulation across entire trajectory
# mekf_full_traj_maes = []
# inEKF_full_pose_full_traj_maes = []
# inekf_5hz_conv_trajectories = [] # Tracking convergence over first N seconds
# inekf_no_horizon_trajectories = [] # Tracking convergence over first N seconds
# mekf_no_horizon_trajectories = []

sbg_inekf_mus = []
oak_inekf_mus = []

sbg_mekf_mus = []
oak_mekf_mus = []

sbg_timestamps = []
oak_timestamps = []
gt_pose_timestamps = []

gt_poses = []

buffer = JitterBuffer(delay_ns=JITTER_THRESHOLD)
first_timestamp = None
filter_started = False
first_lock = False
exit_flag = False

ground_truth_poses = []
sbg_nav = []
sbg_rpy = []

num_heading_msgs = 0
num_gps_msgs = 0

num_simulations = 0

last_horizon_observation = None

with AnyReader([Path(path_to_bag)]) as reader:
    connections = [x for x in reader.connections]

    for connection, timestamp, rawdata in reader.messages(connections=connections):
        try:
            msg = reader.deserialize(rawdata, connection.msgtype)
            timestamp_ns = (msg.header.stamp.sec * 1_000_000_000) + msg.header.stamp.nanosec
            buffer.push(timestamp_ns, connection, rawdata) # Messages pushed to jitter buffer
        except:
            continue

        # Process ready messages from the jitter buffer
        for ready_msg in buffer.pop_ready():
            try:
                msg = reader.deserialize(ready_msg.data, ready_msg.connection.msgtype)
            except:
                # TODO: Remove messages in bag that aren't needed in this experiment
                continue


            # if not first_lock and ready_msg.connection.topic == '/sbg/gps_hdt':
            #     first_lock = True

            if not filter_started:
                if ready_msg.connection.topic == '/sbg/ekf_nav':
                    timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

                    if timestamp - end_time < convergence_plot_gap_seconds and end_time != 0: # TODO: Check values here. ns vs seconds
                        continue  # Skip until we have a gap of convergence_plot_gap_seconds between simulations

                    # filter_started = True 
                    first_lock = True

                    lat, long, alt = msg.latitude, msg.longitude, msg.altitude
                    x, y = convert_gps_to_xyz(lat, long)
                    vx = msg.velocity.x
                    vy = msg.velocity.y
                    vz = msg.velocity.z
            
                    sbg_nav.append((timestamp, x, y, alt, vx, vy, vz))
                    rpy = [0, 0, 0] # Starting at the identity
                    
                    init_state = np.array([np.deg2rad(rpy[0]), np.deg2rad(rpy[1]), np.deg2rad(rpy[2]), 0, 0, 0, x, y, alt])  # [roll, pitch, yaw, vx, vy, vz, x, y, z]                    
                    
                if ready_msg.connection.topic == '/sbg/gps_hdt' and first_lock:
                    first_lock = False
                    filter_started = True
                    timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                    first_timestamp = timestamp
                    
                    inEKF_sbg_full_traj = InEKF(config_params_sbg.init_cov_high, config_params_sbg.imu_noise_params, init_state=init_state, first_timestamp=first_timestamp)
                    MEKF_sbg_full_traj = MEKF(config_params_sbg.init_cov_high, config_params_sbg.imu_noise_params, init_state=init_state, first_timestamp=first_timestamp)
                    inEKF_sbg_full_pose_full_traj = InEKF(config_params_sbg.init_cov_high, config_params_sbg.imu_noise_params, init_state=init_state, first_timestamp=first_timestamp)
                    
                    inEKF_oak_full_traj = InEKF(config_params_oak.init_cov_high, config_params_oak.imu_noise_params, init_state=init_state, first_timestamp=first_timestamp, update_rate=178)
                    MEKF_oak_full_traj = MEKF(config_params_oak.init_cov_high, config_params_oak.imu_noise_params, init_state=init_state, first_timestamp=first_timestamp, update_rate=178)
                    inEKF_oak_full_pose_full_traj = InEKF(config_params_oak.init_cov_high, config_params_oak.imu_noise_params, init_state=init_state, first_timestamp=first_timestamp, update_rate=178)
                    
                
                continue


            connection = ready_msg.connection
            if connection.topic == '/imu/data':
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
                
                inEKF_sbg_full_traj.imu_callback(imu_measurement)
                inEKF_sbg_full_pose_full_traj.imu_callback(imu_measurement)
                MEKF_sbg_full_traj.imu_callback(imu_measurement)

                if msg.header.stamp.sec - first_timestamp > convergence_plot_length_seconds:
                    exit_flag = True

            elif connection.topic == '/oak_d_lr_1/imu/data': # OAK IMU topic
                
                timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

                angular_vel = [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]
                linear_accel = [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]
                rot = config_params_oak.rot_sbg_to_oak_c_imu # Need to rotate IMU measurements into robot frame
                angular_vel = rot @ np.array(angular_vel)
                linear_accel = rot @ np.array(linear_accel)
                
                imu_measurement = ImuMeasurement(timestamp_sec=msg.header.stamp.sec,
                                            timestamp_nsec=msg.header.stamp.nanosec,
                                            gyro= angular_vel,
                                            accel = linear_accel)
                
                inEKF_oak_full_traj.imu_callback(imu_measurement)
                inEKF_oak_full_pose_full_traj.imu_callback(imu_measurement)
                MEKF_oak_full_traj.imu_callback(imu_measurement)
                
            elif connection.topic == "/sbg/gps_pos":
                # continue

                num_gps_msgs += 1
                if num_gps_msgs % 5 != 0:
                    continue
                timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            
                lat, lon, alt = msg.latitude, msg.longitude, msg.altitude # IMU_nav_sat_fix and sbg_ekf_nav use different altitude references, so we need to align them for fair comparison
                x, y = convert_gps_to_xyz(lat, lon)
            
                gps_measurement = GPSMeasurement(timestamp_sec=msg.header.stamp.sec,
                                                timestamp_nsec=msg.header.stamp.nanosec,
                                                x=x,
                                                y=y,
                                                z=alt,
                                                position_cov=config_params_sbg.gps_position_cov)
                
                gps_measurement = gps_add_noise(gps_measurement) # Adding noise to corrupt RTK GPS measurement for testing

                inEKF_sbg_full_traj.gps_callback(gps_measurement)
                MEKF_sbg_full_traj.gps_callback(gps_measurement)
                inEKF_sbg_full_pose_full_traj.gps_callback(gps_measurement)

                inEKF_oak_full_traj.gps_callback(gps_measurement)
                MEKF_oak_full_traj.gps_callback(gps_measurement)
                inEKF_oak_full_pose_full_traj.gps_callback(gps_measurement)
   
            elif connection.topic == '/sbg/gps_hdt':
                # continue
                num_heading_msgs += 1
                if num_heading_msgs % 5 != 0:
                    continue
                msg = sbg_typestore.deserialize_cdr(ready_msg.data, ready_msg.connection.msgtype)
                timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                
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

                inEKF_sbg_full_traj.heading_callback(heading_meas)
                MEKF_sbg_full_traj.heading_callback(heading_meas)
                inEKF_oak_full_traj.heading_callback(heading_meas)
                MEKF_oak_full_traj.heading_callback(heading_meas)

                if last_horizon_observation is not None:
                    inEKF_sbg_full_pose_full_traj.orientation_callback_combined(heading_meas, last_horizon_observation)
                    inEKF_oak_full_pose_full_traj.orientation_callback_combined(heading_meas, last_horizon_observation)

            elif connection.topic == '/oak_d_lr_1/rgb/image_raw':
                # continue
                num_images += 1
                # if not num_images % 4 == 0: # TODO: Fix this!
                    # continue
                # msg = reader.deserialize(rawdata, connection.msgtype)
                timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

                cv_image = message_to_cvimage(msg, "bgr8")
                horizon_observation = camera_processor.lsd_image_callback(cv_image, debug=plot_images)
                if horizon_observation.valid:
                    last_horizon_observation = horizon_observation
                    inEKF_sbg_full_traj.image_callback(horizon_observation)
                    MEKF_sbg_full_traj.image_callback(horizon_observation)
                    inEKF_oak_full_traj.image_callback(horizon_observation)
                    MEKF_oak_full_traj.image_callback(horizon_observation)
                else:
                    last_horizon_observation = None
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
                vx = msg.velocity.x
                vy = msg.velocity.y
                vz = msg.velocity.z

                sbg_nav.append((timestamp, x, y, alt, vx, vy, vz))

        if exit_flag:
            filter_started = False
            first_lock = False
            num_heading_msgs = 0
            # Update all the trajectories with their paths
            exit_flag = False
            end_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9 #timestamp
            num_simulations += 1
            if num_simulations % 10 == 0:
                print(f"Finished Simulation {num_simulations}")
            sbg_inekf_mus.append(np.array(inEKF_sbg_full_traj.mus))
            sbg_timestamp = np.array(inEKF_sbg_full_traj.timestamps)
            sbg_timestamps.append(sbg_timestamp)
            oak_inekf_mus.append(np.array(inEKF_oak_full_traj.mus))
            oak_timestamp = np.array(inEKF_oak_full_traj.timestamps)
            oak_timestamps.append(oak_timestamp)
            sbg_mekf_mus.append(np.array(MEKF_sbg_full_traj.mus))
            oak_mekf_mus.append(np.array(MEKF_oak_full_traj.mus))
            gt_as_filter = sbg_ekf_to_filter(fuse_sbg_ekf_msgs(np.array(sbg_rpy), np.array(sbg_nav)))
            gt_poses.append(np.array(gt_as_filter.mus))
            gt_pose_timestamps.append(np.array(gt_as_filter.timestamps))

            # print(f"Finished Simulation {num_simulations}")
            # print(f"Length of SBG InEKF trajectory: {len(inEKF_sbg_full_traj.mus)}")
            # print(f"Length of OAK InEKF trajectory: {len(inEKF_oak_full_traj.mus)}\n")
        if num_simulations > 100: # For development purposes, limit to 10 simulations
            break
            
    for final_msg in buffer.flush(): # Clear the buffer for now
    # Process the last few bits of data if any
        pass

print("Done processing bag file.")
print(len(sbg_inekf_mus))

with open("real_world_convergence_data.pkl", "wb") as f:
    pickle.dump({
        "sbg_inekf_mus": sbg_inekf_mus,
        "oak_inekf_mus": oak_inekf_mus,
        "sbg_mekf_mus": sbg_mekf_mus,
        "oak_mekf_mus": oak_mekf_mus,
        "gt_poses": gt_poses,
        "sbg_timestamps": sbg_timestamps,
        "oak_timestamps": oak_timestamps,
        "gt_timestamps": gt_pose_timestamps,
    }, f)

plot_figure_6_rw_convergence(sbg_inekf_mus, oak_inekf_mus, sbg_mekf_mus, oak_mekf_mus, gt_poses, sbg_timestamps, oak_timestamps, gt_pose_timestamps, show=True)

# Next steps:
# 4. Make this plot and others look fancy!
# 5. Make timing figure 
