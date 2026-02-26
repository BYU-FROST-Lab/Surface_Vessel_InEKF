'''
This script evaluates the error and timing of the InEKF and MEKF across a full trajectory of real-world data.
The collected data lasts over 4 minutes, and the script takes about 4 minutes to run, processing 6 different filter configurations
(SBG InEKF, SBG MEKF, SBG InEKF with full orientation updates, OAK InEKF, OAK MEKF, OAK InEKF with full orientation updates).

This script generates figures 6 and 8 within the paper.
'''

from pathlib import Path
from rosbags.image import message_to_cvimage
from rosbags.highlevel import AnyReader
from matplotlib import pyplot as plt
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np
import time

import sys

core_path = os.path.abspath("./src/core")
sys.path.append(core_path)
try:
    import vision_cpp

except ImportError as e:
    print(f"❌ Could not find the module: {e}")
    print(f"Check if the .so file is in: {os.getcwd()}")
except Exception as e:
    print(f"❌ Something else went wrong: {e}")


from tqdm import tqdm
# Add the project root to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from helpers.function_helpers import fuse_sbg_ekf_msgs, get_filter_error_metrics, gps_add_noise, heading_add_noise, plot_figure_6, plot_figure_8_timing, convert_gps_to_xyz, sbg_ekf_to_filter
from src.core.jitter_buffer import JitterBuffer
from src.core.image_handler import ImageProcessor

from src.estimators.invariant_ekf import InEKF
from src.estimators.multiplicative_ekf import MEKF
from helpers.load_sbg_types import typestore as sbg_typestore

from helpers.params import RealWorldOAKConfigParams, RealWorldSBGConfigParams

import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation

plot_images = False

path_to_bag = "data/nav_and_forward_cam_seg_1"

num_msgs = 0
num_images = 0

image_function_timing = []
inekf_predict_timing = []
mekf_predict_timing = []
inekf_heading_update_timing = []
mekf_heading_update_timing = []
inekf_rp_update_timing = []
mekf_rp_update_timing = []
inekf_full_pose_update_timing = []

# # Parameters
JITTER_THRESHOLD = 700_000_000  # 0.7 second lookahead
config_params_sbg = RealWorldSBGConfigParams()
config_params_oak = RealWorldOAKConfigParams()
# camera_processor = ImageProcessor(config_params_sbg.mtx, config_params_sbg.dist, config_params_sbg.img_width, config_params_sbg.img_height, camera_height=config_params_sbg.camera_height)
camera_processor = vision_cpp.ImageProcessor(config_params_sbg.mtx, config_params_sbg.dist, config_params_sbg.img_width, config_params_sbg.img_height, config_params_sbg.camera_height)


from src.core.measurements import * # Import measurement types

convergence_plot_length_seconds = 10
simulation_length_s = 40

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

last_horizon_observation = None

with AnyReader([Path(path_to_bag)]) as reader:
    connections = [x for x in reader.connections]

    for connection, timestamp, rawdata in reader.messages(connections=connections):
        if connection.topic == "/imu/nav_sat_fix" or connection.topic == '/sbg/gps_hdt' or connection.topic == '/oak_d_lr_1/rgb/image_raw' or connection.topic == '/oak_d_lr_1/imu/data' or connection.topic == '/imu/data':
            msg = sbg_typestore.deserialize_cdr(rawdata, connection.msgtype)
            timestamp_ns = timestamp_ns = (msg.header.stamp.sec * 1_000_000_000) + msg.header.stamp.nanosec
        
        if not filter_started:
            if connection.topic == '/sbg/ekf_nav':
                msg = sbg_typestore.deserialize_cdr(rawdata, connection.msgtype)
                timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                lat, long, alt = msg.latitude, msg.longitude, msg.altitude
                x, y = convert_gps_to_xyz(lat, long)
                vx = msg.velocity.x
                vy = msg.velocity.y
                vz = msg.velocity.z

                if first_timestamp is None:
                    first_timestamp = timestamp

                first_timestamp = timestamp
                sbg_nav.append((timestamp, x, y, alt, vx, vy, vz))
                filter_started = True 

                rpy = [0, 0, 159.6] 
                init_state = np.array([np.deg2rad(rpy[0]), np.deg2rad(rpy[1]), np.deg2rad(rpy[2]), 0, 0, 0, x, y, alt])  # [roll, pitch, yaw, vx, vy, vz, x, y, z]

                inEKF_sbg_full_traj = InEKF(config_params_sbg.init_cov, config_params_sbg.imu_noise_params, init_state=init_state, first_timestamp=first_timestamp)
                MEKF_sbg_full_traj = MEKF(config_params_sbg.init_cov, config_params_sbg.imu_noise_params, init_state=init_state, first_timestamp=first_timestamp)
                inEKF_sbg_full_pose_full_traj = InEKF(config_params_sbg.init_cov, config_params_sbg.imu_noise_params, init_state=init_state, first_timestamp=first_timestamp)
                
                inEKF_oak_full_traj = InEKF(config_params_oak.init_cov, config_params_oak.imu_noise_params, init_state=init_state, first_timestamp=first_timestamp, update_rate=178)
                MEKF_oak_full_traj = MEKF(config_params_oak.init_cov, config_params_oak.imu_noise_params, init_state=init_state, first_timestamp=first_timestamp, update_rate=178)
                inEKF_oak_full_pose_full_traj = InEKF(config_params_oak.init_cov, config_params_oak.imu_noise_params, init_state=init_state, first_timestamp=first_timestamp, update_rate=178)
                    
            continue  # Skip processing until simulation is enabled

        try:
            msg = reader.deserialize(rawdata, connection.msgtype)
            timestamp_ns = (msg.header.stamp.sec * 1_000_000_000) + msg.header.stamp.nanosec
            buffer.push(timestamp_ns, connection, rawdata) # Messages pushed to jitter buffer
        except:
            continue

        for ready_msg in buffer.pop_ready():
            try:
                msg = reader.deserialize(ready_msg.data, ready_msg.connection.msgtype)
            except:
                # TODO: Remove messages in bag that aren't needed in this experiment
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
                start_time = time.time()
                inEKF_sbg_full_pose_full_traj.imu_callback(imu_measurement)
                inekf_predict_timing.append(time.time() - start_time)
                start_time = time.time()
                MEKF_sbg_full_traj.imu_callback(imu_measurement)
                mekf_predict_timing.append(time.time() - start_time)

                if msg.header.stamp.sec - first_timestamp > simulation_length_s:
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
                MEKF_oak_full_traj.imu_callback(imu_measurement)
                inEKF_oak_full_pose_full_traj.imu_callback(imu_measurement)


            elif connection.topic == "/sbg/gps_pos":
                num_gps_msgs += 1
                if num_gps_msgs % 5 != 0: # For this experiment, only using GPS at 1 HZ
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

                heading_meas = HeadingMeasurement(timestamp_sec=msg.header.stamp.sec,
                                                        timestamp_nsec=msg.header.stamp.nanosec,
                                                        heading=np.deg2rad(heading),
                                                        heading_cov=config_params_sbg.heading_noise_var)
                
                heading_meas = heading_add_noise(heading_meas)

                start_time = time.time()
                inEKF_sbg_full_traj.heading_callback(heading_meas)
                inekf_heading_update_timing.append(time.time() - start_time)

                start_time = time.time()
                MEKF_sbg_full_traj.heading_callback(heading_meas)
                mekf_heading_update_timing.append(time.time() - start_time)

                inEKF_oak_full_traj.heading_callback(heading_meas)
                MEKF_oak_full_traj.heading_callback(heading_meas)

                if last_horizon_observation is not None:
                    start_time = time.time()
                    inEKF_sbg_full_pose_full_traj.orientation_callback_combined(heading_meas, last_horizon_observation)
                    inekf_full_pose_update_timing.append(time.time() - start_time)

                    inEKF_oak_full_pose_full_traj.orientation_callback_combined(heading_meas, last_horizon_observation)

            elif connection.topic == '/oak_d_lr_1/rgb/image_raw':
                num_images += 1

                timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

                cv_image = message_to_cvimage(msg, "bgr8")
                start_time = time.time()
                horizon_observation = camera_processor.lsd_image_callback(cv_image, debug=plot_images)
                image_function_timing.append(time.time() - start_time)

                if horizon_observation.valid:
                    last_horizon_observation = horizon_observation
                    start_time = time.time()
                    inEKF_sbg_full_traj.image_callback(horizon_observation)
                    inekf_rp_update_timing.append(time.time() - start_time)

                    start_time = time.time()
                    MEKF_sbg_full_traj.image_callback(horizon_observation)
                    mekf_rp_update_timing.append(time.time() - start_time)

                    inEKF_oak_full_traj.image_callback(horizon_observation)
                    MEKF_oak_full_traj.image_callback(horizon_observation)

                else:
                    last_horizon_observation = None
      
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
        # if exit_flag:
        #     break
    for final_msg in buffer.flush(): # Clear the buffer for now
    # Process the last few bits of data if any
        pass

gt_poses = sbg_ekf_to_filter(fuse_sbg_ekf_msgs(np.array(sbg_rpy), np.array(sbg_nav)))

convergence_period_seconds = 20

# Metrics are mean, std, and max of xy position error, z position error, roll error, pitch error, and yaw error
sbg_inekf_metrics = get_filter_error_metrics(inEKF_sbg_full_traj, gt_poses, timeoffset=convergence_period_seconds)
sbg_mekf_metrics = get_filter_error_metrics(MEKF_sbg_full_traj, gt_poses, timeoffset=convergence_period_seconds)
sbg_inekf_full_pose_metrics = get_filter_error_metrics(inEKF_sbg_full_pose_full_traj, gt_poses, timeoffset=convergence_period_seconds)
oak_inekf_metrics = get_filter_error_metrics(inEKF_oak_full_traj, gt_poses, timeoffset=convergence_period_seconds)
oak_mekf_metrics = get_filter_error_metrics(MEKF_oak_full_traj, gt_poses, timeoffset=convergence_period_seconds)
oak_inekf_full_pose_metrics = get_filter_error_metrics(inEKF_oak_full_pose_full_traj, gt_poses, timeoffset=convergence_period_seconds) 


# TODO: Delete me after finalizing plotting
import pandas as pd
import pickle

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
df.to_csv('figure_data/filter_metrics_comparison.csv', index=False)
print("Saved comparison to figure_data/filter_metrics_comparison.csv")

plot_figure_6(df, show_plot=False)

# Save timing data to pickle file
timing_data = {
    'image_function_timing': image_function_timing,
    'inekf_predict_timing': inekf_predict_timing,
    'mekf_predict_timing': mekf_predict_timing,
    'inekf_heading_update_timing': inekf_heading_update_timing,
    'mekf_heading_update_timing': mekf_heading_update_timing,
    'inekf_rp_update_timing': inekf_rp_update_timing,
    'mekf_rp_update_timing': mekf_rp_update_timing,
    'inekf_full_pose_update_timing': inekf_full_pose_update_timing,
}

with open('figure_data/timing_data.pkl', 'wb') as f:
    pickle.dump(timing_data, f)

plot_figure_8_timing(image_function_timing, inekf_predict_timing, mekf_predict_timing, inekf_heading_update_timing, mekf_heading_update_timing, inekf_rp_update_timing, mekf_rp_update_timing, inekf_full_pose_update_timing, show=True)





