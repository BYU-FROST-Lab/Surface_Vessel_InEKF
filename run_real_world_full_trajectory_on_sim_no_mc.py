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

from helpers.function_helpers import compare_two_filters, convert_filter_mus_to_rpy_xyz, compute_position_mae, fuse_sbg_ekf_msgs, get_filter_error_metrics, imu_add_noise, gps_add_noise, heading_add_noise, add_initial_state_noise, plot_biases, plot_cov, plot_figure_3, plot_figure_4, plot_figure_5_bar_graph, plot_results, plot_results_comparison_gt, plot_results_cov, convert_gps_to_xyz, sbg_ekf_to_filter, plot_results_comparison_gt_cov
from src.core.jitter_buffer import JitterBuffer
from src.core.image_handler import ImageProcessor

from src.estimators.invariant_ekf import InEKF
from src.estimators.multiplicative_ekf import MEKF
from helpers.load_sbg_types import typestore as sbg_typestore

from helpers.params import HolooceanConfigParams, RealWorldOAKConfigParams, RealWorldSBGConfigParams

import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation

plot_images = False

# path_to_bag = "data/nav_and_forward_cam_seg"
path_to_bag = "/home/laser14/Documents/holoocean_ros_ws/src/holoocean-ros/holoocean_open_water_loop_waves_gps_offset"

# path_to_bag = "/home/laser14/Documents/inekf_hawaii/data/reordered_nav_and_forward_cam_seg_fixed_rollover"
num_msgs = 0
num_images = 0

np.random.seed(42) # Call this once at start of script to ensure repeatability

# # Parameters
JITTER_THRESHOLD = 700_000_000  # 0.7 second lookahead
config_params_sbg = HolooceanConfigParams()
camera_processor = ImageProcessor(config_params_sbg.mtx, config_params_sbg.dist, config_params_sbg.img_width, config_params_sbg.img_height, camera_height=config_params_sbg.camera_height)

config_params_sbg = RealWorldSBGConfigParams()
config_params_oak = RealWorldOAKConfigParams()

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

buffer = JitterBuffer(delay_ns=JITTER_THRESHOLD)
first_timestamp = None
filter_started = False
exit_flag = False
ground_truth_poses = []
sbg_nav = []
sbg_rpy = []

num_heading_msgs = 0
num_gps_msgs = 0

current_random_walk = np.zeros(6) # Random walk for IMU noise, initialized to zero

last_horizon_observation = None

with AnyReader([Path(path_to_bag)]) as reader:
    connections = [x for x in reader.connections]

    for connection, timestamp, rawdata in reader.messages(connections=connections):
        if connection.topic == "/imu/nav_sat_fix" or connection.topic == '/sbg/gps_hdt' or connection.topic == '/oak_d_lr_1/rgb/image_raw' or connection.topic == '/oak_d_lr_1/imu/data' or connection.topic == '/imu/data':
            msg = sbg_typestore.deserialize_cdr(rawdata, connection.msgtype)
            timestamp_ns = timestamp_ns = (msg.header.stamp.sec * 1_000_000_000) + msg.header.stamp.nanosec
        
        if not filter_started:
            # if connection.topic == "/holoocean/asv0/PoseSensor":
            if connection.topic == "/holoocean/asv0/PoseSensor":
                msg = sbg_typestore.deserialize_cdr(rawdata, connection.msgtype)
                timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                # lat, long, alt = msg.latitude, msg.longitude, msg.altitude
                # x, y = convert_gps_to_xyz(lat, long)
                x = msg.pose.pose.position.x
                y = msg.pose.pose.position.y
                alt = msg.pose.pose.position.z
                # vx = msg.twist.twist.linear.x
                # vy = msg.twist.twist.linear.y
                # vz = msg.twist.twist.linear.z
                orientation_quat=np.array([msg.pose.pose.orientation.x,
                                        msg.pose.pose.orientation.y,
                                        msg.pose.pose.orientation.z,
                                        msg.pose.pose.orientation.w]),
                rpy = Rotation.from_quat(orientation_quat).as_euler('xyz', degrees=False)[0]
                        
            
                # msg = reader.deserialize(rawdata, connection.msgtype)
                # timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

                if first_timestamp is None:
                    first_timestamp = timestamp
        
                # if timestamp - first_timestamp < 45:
                #     continue

                first_timestamp = timestamp
                # sbg_nav.append((timestamp, x, y, alt, vx, vy, vz))
                filter_started = True 
                # orientation_quat=np.array([msg.pose.pose.orientation.x,
                                        # msg.pose.pose.orientation.y,
                                        # msg.pose.pose.orientation.z,
                                        # msg.pose.pose.orientation.w]),
                # rpy = Rotation.from_quat(orientation_quat).as_euler('xyz', degrees=False)[0]
                # rpy = [5.65, -5.28, 159.6] 
                # rpy = [0, 0, 159.6] 
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
        # buffer.push(timestamp, connection, rawdata) # Messages pushed to jitter buffer
                                                    # Not needed for simulation, but keeping for real world data use.

        for ready_msg in buffer.pop_ready():
            try:
                msg = reader.deserialize(ready_msg.data, ready_msg.connection.msgtype)
            except:
                # TODO: Remove messages in bag that aren't needed in this experiment
                continue
            connection = ready_msg.connection
            if connection.topic == '/holoocean/asv0/IMUSensor':
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
                
                imu_measurement, current_random_walk = imu_add_noise(imu_measurement, config_params_sbg.imu_noise_params, current_random_walk) # Adding noise to IMU measurement for testing

                inEKF_sbg_full_traj.imu_callback(imu_measurement)
                inEKF_sbg_full_pose_full_traj.imu_callback(imu_measurement)
                MEKF_sbg_full_traj.imu_callback(imu_measurement)
                
                # if timestamp - first_timestamp < convergence_plot_length_seconds:
                #     inEKF_5hz_conv.imu_callback(imu_measurement)
                #     inEKF_no_horizon.imu_callback(imu_measurement)
                #     MEKF_5hz_conv.imu_callback(imu_measurement)
                #     MEKF_no_horizon.imu_callback(imu_measurement)

                if msg.header.stamp.sec - first_timestamp > simulation_length_s:
                    exit_flag = True

            # elif connection.topic == '/oak_d_lr_1/imu/data': # OAK IMU topic
                
            #     timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

            #     angular_vel = [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]
            #     linear_accel = [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]
            #     rot = config_params_oak.rot_sbg_to_oak_c_imu # Need to rotate IMU measurements into robot frame
            #     angular_vel = rot @ np.array(angular_vel)
            #     linear_accel = rot @ np.array(linear_accel)
                
            #     imu_measurement = ImuMeasurement(timestamp_sec=msg.header.stamp.sec,
            #                                 timestamp_nsec=msg.header.stamp.nanosec,
            #                                 gyro= angular_vel,
            #                                 accel = linear_accel)
                
            #     inEKF_oak_full_traj.imu_callback(imu_measurement)
            #     inEKF_oak_full_pose_full_traj.imu_callback(imu_measurement)
            #     MEKF_oak_full_traj.imu_callback(imu_measurement)
                
            #     # if timestamp - first_timestamp < convergence_plot_length_seconds:
            #     #     inEKF_5hz_conv.imu_callback(imu_measurement)
            #     #     inEKF_no_horizon.imu_callback(imu_measurement)
            #     #     MEKF_5hz_conv.imu_callback(imu_measurement)
            #     #     MEKF_no_horizon.imu_callback(imu_measurement)

            #     # if timestamp - first_timestamp > simulation_length_s:
            #     #     exit_flag = True


            elif connection.topic == "/holoocean/asv0/GPSSensor":
                # continue
                # msg = reader.deserialize(rawdata, connection.msgtype)
                num_gps_msgs += 1
                # if num_gps_msgs % 5 != 0:
                #     continue
                timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            
                # lat, lon, alt = msg.latitude, msg.longitude, msg.altitude-2.7 # IMU_nav_sat_fix and sbg_ekf_nav use different altitude references, so we need to align them for fair comparison
                # x, y = convert_gps_to_xyz(lat, lon)

                # position_covariance = msg.position_covariance
                # position_covariance = msg.position_accuracy
                gps_measurement = GPSMeasurement(timestamp_sec=msg.header.stamp.sec,
                                                timestamp_nsec=msg.header.stamp.nanosec,
                                            x=msg.pose.pose.position.x,
                                            y=msg.pose.pose.position.y,
                                            z=msg.pose.pose.position.z,
                                            position_cov=config_params_sbg.gps_position_cov)            
                # gps_measurement = GPSMeasurement(timestamp_sec=msg.header.stamp.sec,
                #                                 timestamp_nsec=msg.header.stamp.nanosec,
                #                                 x=x,
                #                                 y=y,
                #                                 z=alt,
                #                                 position_cov=config_params_sbg.gps_position_cov)
                
                gps_measurement = gps_add_noise(gps_measurement) # Adding noise to corrupt RTK GPS measurement for testing

                inEKF_sbg_full_traj.gps_callback(gps_measurement)
                MEKF_sbg_full_traj.gps_callback(gps_measurement)
                inEKF_sbg_full_pose_full_traj.gps_callback(gps_measurement)

                # inEKF_oak_full_traj.gps_callback(gps_measurement)
                # MEKF_oak_full_traj.gps_callback(gps_measurement)
                # inEKF_oak_full_pose_full_traj.gps_callback(gps_measurement)
                                
                # if timestamp - first_timestamp < convergence_plot_length_seconds:
                #     inEKF_5hz_conv.gps_callback(gps_measurement)
                #     inEKF_no_horizon.gps_callback(gps_measurement)
                #     MEKF_5hz_conv.gps_callback(gps_measurement)
                #     MEKF_no_horizon.gps_callback(gps_measurement)
            
            elif connection.topic == "/holoocean/asv0/RotationSensor":
                # continue
                num_heading_msgs += 1
                if num_heading_msgs % 5 != 0:
                    continue
                msg = sbg_typestore.deserialize_cdr(ready_msg.data, ready_msg.connection.msgtype)
                timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                
                yaw = msg.vector.z  # Assuming Z is yaw
                heading_meas = HeadingMeasurement(timestamp_sec=msg.header.stamp.sec,
                                                    timestamp_nsec=msg.header.stamp.nanosec,
                                                    heading=np.deg2rad(yaw),
                                                    heading_cov=config_params_sbg.heading_noise_var)
                    
                heading_meas = heading_add_noise(heading_meas)

                inEKF_sbg_full_traj.heading_callback(heading_meas)
                MEKF_sbg_full_traj.heading_callback(heading_meas)
                # inEKF_oak_full_traj.heading_callback(heading_meas)
                # MEKF_oak_full_traj.heading_callback(heading_meas)

                if last_horizon_observation is not None:
                    inEKF_sbg_full_pose_full_traj.orientation_callback_combined(heading_meas, last_horizon_observation)
                    # inEKF_oak_full_pose_full_traj.orientation_callback_combined(heading_meas, last_horizon_observation)

                # if timestamp - first_timestamp < convergence_plot_length_seconds:
                    # inEKF_5hz_conv.heading_callback(heading_meas)   
                    # inEKF_no_horizon.heading_callback(heading_meas)
                    # MEKF_5hz_conv.heading_callback(heading_meas)
                    # MEKF_no_horizon.heading_callback(heading_meas)

            elif connection.topic == "/holoocean/asv0/FrontCamera":
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
                    # inEKF_oak_full_traj.image_callback(horizon_observation)
                    # MEKF_oak_full_traj.image_callback(horizon_observation)
                else:
                    last_horizon_observation = None
                    # if num_images % 4 == 0:
                        # if timestamp - first_timestamp < convergence_plot_length_seconds:
                            # inEKF_5hz_conv.image_callback(horizon_observation)
                            # MEKF_5hz_conv.image_callback(horizon_observation)
                                    
            # elif connection.topic == '/sbg/ekf_quat':
            #     msg = sbg_typestore.deserialize_cdr(ready_msg.data, ready_msg.connection.msgtype)
            #     timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            #     quat = [msg.quaternion.x, msg.quaternion.y, msg.quaternion.z, msg.quaternion.w]
            #     rpy = Rotation.from_quat(quat).as_euler('xyz', degrees=True)
            #     sbg_rpy.append((timestamp, rpy[0], rpy[1], rpy[2]))

            # elif connection.topic == '/sbg/ekf_nav':
            #     msg = sbg_typestore.deserialize_cdr(ready_msg.data, ready_msg.connection.msgtype)
            #     timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            #     lat, long, alt = msg.latitude, msg.longitude, msg.altitude
            #     x, y = convert_gps_to_xyz(lat, long)
            #     vx = msg.velocity.x
            #     vy = msg.velocity.y
            #     vz = msg.velocity.z

            #     sbg_nav.append((timestamp, x, y, alt, vx, vy, vz))
            elif connection.topic == "/holoocean/asv0/PoseSensor": 
                # msg = reader.deserialize(rawdata, connection.msgtype)
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
                # self.ground_truth_poses.append((pose_measurement.timestamp, rpy, pose_measurement.position))
                ground_truth_poses.append((timestamp, rpy[0], rpy[1], rpy[2], pos[0], pos[1], pos[2]))


        if exit_flag:
            break

        
    
    # manager.plot_inekf_results()


    for final_msg in buffer.flush(): # Clear the buffer for now
    # Process the last few bits of data if any
        pass

gt_poses = sbg_ekf_to_filter(np.array(ground_truth_poses))

plot_cov(inEKF_sbg_full_traj, title="SBG InEKF Full Trajectory Covariance", show=False)
plot_cov(MEKF_sbg_full_traj, title="SBG MEKF Full Trajectory Covariance", show=False)
plot_cov(inEKF_sbg_full_pose_full_traj, title="SBG InEKF Full Pose Full Trajectory Covariance", show=False)
# plot_cov(inEKF_oak_full_traj, title="OAK InEKF Full Trajectory Covariance", show=False)
# plot_cov(MEKF_oak_full_traj, title="OAK MEKF Full Trajectory Covariance", show=False)
# plot_cov(inEKF_oak_full_pose_full_traj, title="OAK InEKF Full Pose Full Trajectory Covariance", show=False)


plot_results(inEKF_sbg_full_traj, gt_data=np.array(ground_truth_poses), title="SBG InEKF Full Trajectory", show=False)
plot_results(MEKF_sbg_full_traj, gt_data=np.array(ground_truth_poses), title="SBG MEKF Full Trajectory", show=False)
plot_results(inEKF_sbg_full_pose_full_traj, gt_data=np.array(ground_truth_poses), title="SBG InEKF Full Pose Full Trajectory", show=False)

# plot_results_comparison_gt_cov(inEKF_sbg_full_traj, inEKF_oak_full_traj, gt_poses, filter1_name="SBG InEKF", filter2_name="OAK InEKF", title="InEKF Full Trajectory Comparison with GT", show=False)
# plot_results_comparison_gt_cov(inEKF_sbg_full_pose_full_traj, inEKF_sbg_full_traj, gt_poses, filter1_name="SBG InEKF Full Pose", filter2_name="SBG InEKF", title="InEKF Full Trajectory Comparison with GT", show=False)
# plot_results_comparison_gt_cov(inEKF_oak_full_pose_full_traj, inEKF_oak_full_traj, gt_poses, filter1_name="OAK InEKF Full Pose", filter2_name="OAK InEKF", title="InEKF Full Trajectory Comparison with GT", show=False)
# plot_results_comparison_gt_cov(MEKF_sbg_full_traj, inEKF_sbg_full_traj, gt_poses, filter1_name="MEKF SBG", filter2_name="InEKF SBG", title="InEKF Full Trajectory Comparison with GT", show=False)
# plot_results_comparison_gt_cov(MEKF_oak_full_traj, inEKF_oak_full_traj, gt_poses, filter1_name="MEKF OAK", filter2_name="InEKF OAK", title="InEKF Full Trajectory Comparison with GT", show=False)

convergence_period_seconds = 20
# compare_two_filters(inEKF_sbg_full_traj, gt_poses, filter1_name="InEKF SBG Full Trajectory", filter2_name="GT", timeoffset=convergence_period_seconds)
# compare_two_filters(inEKF_oak_full_traj, gt_poses, filter1_name="InEKF OAK Full Trajectory", filter2_name="GT", timeoffset=convergence_period_seconds)

# compare_two_filters(inEKF_sbg_full_pose_full_traj, gt_poses, filter1_name="InEKF SBG Full Pose", filter2_name="GT", timeoffset=convergence_period_seconds)
# compare_two_filters(inEKF_oak_full_pose_full_traj, gt_poses, filter1_name="InEKF OAK Full Pose", filter2_name="GT", timeoffset=convergence_period_seconds)
# compare_two_filters(MEKF_sbg_full_traj, gt_poses, filter1_name="MEKF SBG", filter2_name="GT", timeoffset=convergence_period_seconds)
# compare_two_filters(MEKF_oak_full_traj, gt_poses, filter1_name="MEKF OAK", filter2_name="GT", timeoffset=convergence_period_seconds)

# Metrics are mean, std, and max of xy position error, z position error, roll error, pitch error, and yaw error
sbg_inekf_metrics = get_filter_error_metrics(inEKF_sbg_full_traj, gt_poses, timeoffset=convergence_period_seconds)
sbg_mekf_metrics = get_filter_error_metrics(MEKF_sbg_full_traj, gt_poses, timeoffset=convergence_period_seconds)
sbg_inekf_full_pose_metrics = get_filter_error_metrics(inEKF_sbg_full_pose_full_traj, gt_poses, timeoffset=convergence_period_seconds)
# oak_inekf_metrics = get_filter_error_metrics(inEKF_oak_full_traj, gt_poses, timeoffset=convergence_period_seconds)
# oak_mekf_metrics = get_filter_error_metrics(MEKF_oak_full_traj, gt_poses, timeoffset=convergence_period_seconds)
# oak_inekf_full_pose_metrics = get_filter_error_metrics(inEKF_oak_full_pose_full_traj, gt_poses, timeoffset=convergence_period_seconds) 


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
    sbg_inekf_metrics,
    sbg_mekf_metrics,
    sbg_inekf_full_pose_metrics,
    # oak_inekf_metrics,
    # oak_mekf_metrics,
    # oak_inekf_full_pose_metrics
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


