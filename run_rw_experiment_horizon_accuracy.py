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

from helpers.sim_helpers import convert_filter_mus_to_rpy_xyz, compute_position_mae, imu_add_noise, gps_add_noise, heading_add_noise, add_initial_state_noise, plot_cov, plot_figure_3, plot_figure_4, plot_results, plot_results_cov
from src.core.jitter_buffer import JitterBuffer
from src.core.image_handler import ImageProcessor

# from src.estimators.invariant_ekf import InEKF
# from src.estimators.multiplicative_ekf import MEKF

from helpers.load_sbg_types import typestore as sbg_typestore
import ahrs # TODO: Add to requirements.txt, used for Madgwick filter
import math


from helpers.sim_params import RealWorldSBGConfigParams

# from helpers.load_holoocean_ros_types import typestore as holoocean_typestore

import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation

# from sim_image_handler import image_handler
# from helpers.holoocean_constants import dist, mtx, unet_filepath, camera_height_sim as camera_height

'''
Topic: /holoocean/asv0/DVLSensorRange     | Type: holoocean_interfaces/msg/DVLSensorRange 
Topic: /holoocean/asv0/DVLSensorVelocity  | Type: geometry_msgs/msg/TwistWithCovarianceStamped 
Topic: /holoocean/asv0/DynamicsSensorIMU  | Type: sensor_msgs/msg/Imu 
Topic: /holoocean/asv0/DynamicsSensorOdom | Type: nav_msgs/msg/Odometry 
Topic: /holoocean/asv0/FrontCamera        | Type: sensor_msgs/msg/Image 
Topic: /holoocean/asv0/GPSSensor          | Type: nav_msgs/msg/Odometry 
Topic: /holoocean/asv0/IMUSensor          | Type: sensor_msgs/msg/Imu 
Topic: /holoocean/asv0/IMUSensorBias      | Type: geometry_msgs/msg/TwistWithCovarianceStamped 
Topic: /holoocean/asv0/LeftCamera         | Type: sensor_msgs/msg/Image 
Topic: /holoocean/asv0/RightCamera        | Type: sensor_msgs/msg/Image 
Topic: /holoocean/asv0/RotationSensor     | Type: geometry_msgs/msg/Vector3Stamped 
'''

plot_images = False

path_to_bag = "data/nav_and_forward_cam_seg"
num_msgs = 0

# # Parameters
JITTER_THRESHOLD = 1_000_000_000  # 1000ms lookahead
config_params = RealWorldSBGConfigParams()
camera_processor = ImageProcessor(config_params.mtx, config_params.dist, config_params.img_width, config_params.img_height, camera_height=config_params.camera_height)

from src.core.measurements import * # Import measurement types


num_imgs_stop = 50
tracked_rp = []  # Store tracked roll/pitch over time for analysis
sbg_rp = []  # Store SBG roll/pitch over time for analysis
sbg_imu_gyro_list = []
sbg_imu_accel_list = []
imu_rp = []  # Store IMU roll/pitch over time for analysis
oak_imu_timestamps = []
oak_imu_rp = []
oak_imu_gyro_list = []
oak_imu_accel_list = []
num_imgs = 0

buffer = JitterBuffer(delay_ns=JITTER_THRESHOLD)
first_timestamp = None
simulation_enabled = False
waiting_threshold = 5  # 15 seconds
exit_flag = False
ground_truth_poses = []
current_random_walk = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # Initial random walk for IMU noise
current_random_walk += np.random.normal(0, config_params.init_cov[9:15, 9:15].diagonal()**0.5, size=6)

last_horizon_observation = None

with AnyReader([Path(path_to_bag)]) as reader:
    connections = [x for x in reader.connections]

    for connection, timestamp, rawdata in reader.messages(connections=connections):
        buffer.push(timestamp, connection, rawdata) # Messages pushed to jitter buffer
                                                    # Not needed for simulation, but keeping for real world data use.

        for ready_msg in buffer.pop_ready():
            connection = ready_msg.connection

            if connection.topic == '/oak_d_lr_1/rgb/image_raw':
                msg = reader.deserialize(ready_msg.data, ready_msg.connection.msgtype)

                num_imgs += 1
                # msg = reader.deserialize(rawdata, connection.msgtype)
                timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

                cv_image = message_to_cvimage(msg, "bgr8")
                horizon_observation = camera_processor.lsd_image_callback(cv_image, debug=plot_images)
                tracked_rp.append((timestamp, np.rad2deg(horizon_observation.roll_rad), np.rad2deg(horizon_observation.pitch_rad)))
            elif connection.topic == '/sbg/ekf_quat':
                msg = sbg_typestore.deserialize_cdr(ready_msg.data, ready_msg.connection.msgtype)
                timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                quat = [msg.quaternion.x, msg.quaternion.y, msg.quaternion.z, msg.quaternion.w]
                rpy = Rotation.from_quat(quat).as_euler('xyz', degrees=True)
                sbg_rp.append((timestamp, rpy[0], rpy[1]))
            elif connection.topic == '/imu/data':
                msg = reader.deserialize(ready_msg.data, ready_msg.connection.msgtype)
                time = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
                # imu_gyro = [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]
                imu_accel = [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]
                sbg_imu_gyro_list.append([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])
                sbg_imu_accel_list.append(imu_accel)
                # Roll = ϕ = arctan2(ay, az)
                # Pitch = θ = arctan2(ax, gravity)
                roll = math.atan2(imu_accel[1], imu_accel[2]) * 180 / np.pi
                # pitch = math.atan2(imu_accel[0], 9.81) * 180 / np.pi
                pitch = math.atan2(-imu_accel[0], np.sqrt(imu_accel[1]**2 + imu_accel[2]**2)) * 180 / np.pi
                imu_rp.append((time, roll, pitch))
            
            elif connection.topic == '/oak_d_lr_1/imu/data':
                msg = reader.deserialize(ready_msg.data, ready_msg.connection.msgtype)
                timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                oak_imu_timestamps.append(timestamp)
                oak_imu_gyro_list.append([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])
                oak_imu_accel_list.append([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])

        if num_imgs > num_imgs_stop:
            break

    for final_msg in buffer.flush(): # Clear the buffer for now
    # Process the last few bits of data if any
        pass


print(f"Image messages: {num_imgs}")

# Plotting the results
tracked_rp = np.array(tracked_rp)[1:]
sbg_rp = np.array(sbg_rp)
imu_rp = np.array(imu_rp)

import ahrs

sbg_imu_accel_list = np.array(sbg_imu_accel_list)
sbg_imu_gyro_list = np.array(sbg_imu_gyro_list)

oak_imu_gyro_list = np.array(oak_imu_gyro_list)
oak_imu_accel_list = np.array(oak_imu_accel_list)
oak_imu_timestamps = np.array(oak_imu_timestamps)

# OAK IMU coordinate frame is in DES (Down East South) which is a 0, 90, 180 degree rotation about the y and z axes from ENU frame
# rot = Rotation.from_euler('xz', [-90, 180], degrees=True)
rot = np.array([[0, -1, 0],
                [0, 0, -1],
                [1, 0, 0]])
rot_sbg_to_oak_c = np.array([[-0.00364432, -0.00424146,  0.99998436],
                             [-0.99999098, -0.00216758, -0.00365354],
                             [ 0.00218304, -0.99998866, -0.00423352]])

rot_oak_c_to_oak_imu = np.array([[0.99998691, 0.00375827, -0.00347223],
                                 [-0.00377214, 0.99998489, -0.00399805],
                                 [ 0.00345715, 0.0040111, 0.99998598]])

# rot = rot_sbg_to_oak_c @ rot_oak_c_to_oak_imu.T

rot = rot_sbg_to_oak_c

for i in range(len(oak_imu_accel_list)):
    oak_imu_accel_list[i] = rot @ oak_imu_accel_list[i]
    oak_imu_accel_list[i][1] = oak_imu_accel_list[i][1] # + 1 # removing bias of 1 m/s^2 in Y direction
    oak_imu_gyro_list[i] = rot @ oak_imu_gyro_list[i]

for i in range(len(oak_imu_accel_list)):
    roll = math.atan2(oak_imu_accel_list[i][1], oak_imu_accel_list[i][2]) * 180 / np.pi
    # Pitch = θ = arctan2(ax, gravity)
    # pitch = math.atan2(imu_accel[0], 9.81) * 180 / np.pi
    pitch = math.atan2(-oak_imu_accel_list[i][0], np.sqrt(oak_imu_accel_list[i][1]**2 + oak_imu_accel_list[i][2]**2)) * 180 / np.pi
    oak_imu_rp.append((oak_imu_timestamps[i], roll, pitch))

oak_imu_rp = np.array(oak_imu_rp)

st = imu_rp[0,0]

# Madgwick filter implementation for IMU data
madgwick_sbg = ahrs.filters.Madgwick(gyr=sbg_imu_gyro_list,
                                 acc=sbg_imu_accel_list,
                                #  mag=simulated_data.magnetometers,
                                 frequency=200)

madgwick_sbg_quat = madgwick_sbg.Q
# quat is in (w, x, y, z) format, need to convert to (x, y, z, w)
madgwick_sbg_quat = np.array([[q[1], q[2], q[3], q[0]] for q in madgwick_sbg_quat])
madgwick_sbg_rpy = Rotation.from_quat(madgwick_sbg_quat).as_euler('xyz', degrees=True)

madgwick_oak = ahrs.filters.Madgwick(gyr=oak_imu_gyro_list,
                                 acc=oak_imu_accel_list,
                                #  mag=simulated_data.magnetometers,
                                 frequency=170)
madgwick_oak_quat = madgwick_oak.Q
# quat is in (w, x, y, z) format, need to convert to
madgwick_oak_quat = np.array([[q[1], q[2], q[3], q[0]] for q in madgwick_oak_quat])
madgwick_oak_rpy = Rotation.from_quat(madgwick_oak_quat).as_euler('xyz', degrees=True)

# ----- Comparison Metrics -----
# SBG IMU Gravity and Madgwick vs SBG EKF
sbg_madgwick_roll_errors = madgwick_sbg_rpy[:,0] - sbg_rp[1:,1]
sbg_madgwick_pitch_errors = madgwick_sbg_rpy[:,1] - sbg_rp[1:,2]
sbg_madgwick_roll_rmse = np.sqrt(np.mean(sbg_madgwick_roll_errors**2))
sbg_madgwick_pitch_rmse = np.sqrt(np.mean(sbg_madgwick_pitch_errors**2))
print(f"SBG IMU Madgwick vs SBG EKF - Roll RMSE: {sbg_madgwick_roll_rmse:.2f}, Pitch RMSE: {sbg_madgwick_pitch_rmse:.2f}")

sbg_gravity_roll_errors = imu_rp[:,1] - sbg_rp[1:,1]
sbg_gravity_pitch_errors = imu_rp[:,2] - sbg_rp[1:,2]
sbg_gravity_roll_rmse = np.sqrt(np.mean(sbg_gravity_roll_errors**2))
sbg_gravity_pitch_rmse = np.sqrt(np.mean(sbg_gravity_pitch_errors**2))
print(f"SBG IMU Gravity vs SBG EKF - Roll RMSE: {sbg_gravity_roll_rmse:.2f}, Pitch RMSE: {sbg_gravity_pitch_rmse:.2f}")

# OAK IMU Gravity and Madgwick vs SBG EKF
oak_madgwick_roll_errors = []
oak_madgwick_pitch_errors = []
oak_gravity_roll_errors = []
oak_gravity_pitch_errors = []

for i in range(len(oak_imu_rp)):
    oak_timestamp = oak_imu_rp[i,0]
    # Find closest sbg_rp timestamp
    idx = (np.abs(sbg_rp[:,0] - oak_timestamp)).argmin()
    oak_gravity_roll_errors.append(oak_imu_rp[i,1] - sbg_rp[idx,1])
    oak_gravity_pitch_errors.append(oak_imu_rp[i,2] - sbg_rp[idx,2])
    oak_madgwick_roll_errors.append(madgwick_oak_rpy[i,0] - sbg_rp[idx,1])
    oak_madgwick_pitch_errors.append(madgwick_oak_rpy[i,1] - sbg_rp[idx,2])

oak_madgwick_roll_rmse = np.sqrt(np.mean(np.array(oak_madgwick_roll_errors)**2))
oak_madgwick_pitch_rmse = np.sqrt(np.mean(np.array(oak_madgwick_pitch_errors)**2))
oak_gravity_roll_rmse = np.sqrt(np.mean(np.array(oak_gravity_roll_errors)**2))
oak_gravity_pitch_rmse = np.sqrt(np.mean(np.array(oak_gravity_pitch_errors)**2))
print(f"OAK IMU Madgwick vs SBG EKF - Roll RMSE: {oak_madgwick_roll_rmse:.2f}, Pitch RMSE: {oak_madgwick_pitch_rmse:.2f}")
print(f"OAK IMU Gravity vs SBG EKF - Roll RMSE: {oak_gravity_roll_rmse:.2f}, Pitch RMSE: {oak_gravity_pitch_rmse:.2f}")

horizon_pitch_errors, horizon_roll_errors = [], []
for i in range(len(tracked_rp)):
    horizon_timestamp = tracked_rp[i,0]
    # Find closest sbg_rp timestamp
    idx = (np.abs(sbg_rp[:,0] - horizon_timestamp)).argmin()
    horizon_roll_errors.append(tracked_rp[i,1] - sbg_rp[idx,1])
    horizon_pitch_errors.append(tracked_rp[i,2] - sbg_rp[idx,2])

horizon_roll_rmse = np.sqrt(np.mean(np.array(horizon_roll_errors)**2))
horizon_pitch_rmse = np.sqrt(np.mean(np.array(horizon_pitch_errors)**2))
print(f"Horizon vs SBG EKF - Roll RMSE: {horizon_roll_rmse:.2f}, Pitch RMSE: {horizon_pitch_rmse:.2f}")

lw = 3

import seaborn as sns
import matplotlib
matplotlib.rc("pdf", fonttype=42)
sns.set_context("paper")
sns.set_style("whitegrid")
sns.set_palette("colorblind")
c = sns.color_palette("colorblind")

import shutil
import matplotlib as mpl

cache_dir = mpl.get_cachedir()
shutil.rmtree(cache_dir)

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

from matplotlib.patches import Rectangle
from matplotlib.patches import ConnectionPatch

# --- Define Zoom Region (Roll) ---
ROLL_PITCH_XLIM = [2.75, 3.0]
ROLL_YLIM = [0, 3.5] # Custom Y-limits for the zoomed roll data

# --- Define Zoom Region (Pitch) ---
PITCH_XLIM = ROLL_PITCH_XLIM
ROLL_XLIM = ROLL_PITCH_XLIM
PITCH_YLIM = [0, 2.34] # Custom Y-limits for the zoomed pitch data
# --- End Dummy Data ---


zoomed_lw = 1.5
# Create a figure with the new small size
fig, axs = plt.subplots(
    2, 2,
    figsize=(3.54, 3), # <- Target Figure Size
    gridspec_kw={'width_ratios': [3, 1]},
    dpi=300,
    sharex='col' 
)

# Set a global font size for better consistency in a small figure
# -------------------------------------------------------------------------
SMALL_SIZE = 7
MEDIUM_SIZE = 7
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE - 1) # legend font size
plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title
# -------------------------------------------------------------------------
lw = 2.0  # Line width for plots

c = sns.color_palette("colorblind")
sbg_color =     c[4] #c[0]
horizon_color = c[8] #c[1]
sbg_imu_color = c[3] #c[2]
oak_imu_color = c[9] #c[9]
g_lw = 0.75
# sbg_gravity_color = c[2]
# sbg_madgwick_color = c[2]
# oak_gravity_color = c[9]
# oak_madgwick_color = c[9]

# --- ROW 1: Estimated Roll (Main and Zoomed) ---
ax1 = axs[0, 0]
ax1.plot(imu_rp[:, 0]-st, imu_rp[:, 1], label="SBG IMU Gravity", linewidth=g_lw, alpha=0.8, c=sbg_imu_color)
ax1.plot(oak_imu_rp[:, 0]-st, oak_imu_rp[:, 1], label="OAK IMU Gravity", linewidth=g_lw, alpha=0.8, c=oak_imu_color)
ax1.plot(sbg_rp[:, 0]-st, sbg_rp[:, 1], label="SBG QEKF", linewidth=lw, c=sbg_color)
ax1.plot(tracked_rp[:, 0]-st, tracked_rp[:, 1], label='Horizon Inferred', linewidth=lw, linestyle='dashed', c=horizon_color)
ax1.plot(imu_rp[:, 0]-st, madgwick_sbg_rpy[:, 0], label="SBG IMU Madgwick", linewidth=lw, c=sbg_imu_color, linestyle='dashed')
ax1.plot(oak_imu_timestamps-st, madgwick_oak_rpy[:, 0], label="OAK IMU Madgwick", linewidth=lw, linestyle='dashed', c=oak_imu_color)
ax1.grid(True, linestyle=':', alpha=0.5)
ax1.set_ylim(-30, 15)
ax1.set_xlim(-.1, 3.1)
ax1.yaxis.tick_left()
ax1.tick_params(axis='both', which='major', labelsize=SMALL_SIZE, pad=2)


# ax1.set_title('Estimated Roll (Full View)', fontsize=MEDIUM_SIZE) # Use slightly larger title font
ax1.set_ylabel('Roll (°)', fontsize=MEDIUM_SIZE, labelpad=0) # Use symbol for brevity

# Draw the Zoom Box on ax1 (Rectangle)
rect_roll = Rectangle(
    (ROLL_XLIM[0], ROLL_YLIM[0]),
    ROLL_XLIM[1] - ROLL_XLIM[0],
    ROLL_YLIM[1] - ROLL_YLIM[0],
    edgecolor='black', facecolor='none', linestyle='--', linewidth=zoomed_lw, zorder=10
)
ax1.add_patch(rect_roll)


# Zoomed Roll Plot (Top-Right)
ax_zoom_roll = axs[0, 1]
# ax_zoom_roll.plot(imu_rp[:, 0]-st, imu_rp[:, 1], linewidth=g_lw, alpha=0.8, c=c[2])
# ax_zoom_roll.plot(oak_imu_rp[:, 0]-st, oak_imu_rp[:, 1], linewidth=g_lw, alpha=0.8, c=c[9])
ax_zoom_roll.plot(sbg_rp[:, 0]-st, sbg_rp[:, 1], linewidth=lw, c=sbg_color)
ax_zoom_roll.plot(tracked_rp[:, 0]-st, tracked_rp[:, 1], linewidth=lw, linestyle='dashed', c=horizon_color)
ax_zoom_roll.plot(imu_rp[:, 0]-st, madgwick_sbg_rpy[:, 0], linewidth=lw, c=sbg_imu_color, linestyle='dashed')
ax_zoom_roll.plot(oak_imu_timestamps-st, madgwick_oak_rpy[:, 0], linewidth=lw, linestyle='dashed', c=oak_imu_color)
ax_zoom_roll.set_xlim(ROLL_XLIM)
ax_zoom_roll.set_ylim(ROLL_YLIM)
# ax_zoom_roll.set_title('Zoom: 0.8s to 1.2s', fontsize=MEDIUM_SIZE)
ax_zoom_roll.grid(True, linestyle=':', alpha=0.5)
ax_zoom_roll.yaxis.tick_right()
ax_zoom_roll.yaxis.set_label_position('right')
# ax_zoom_roll.set_ylabel('Roll (°)')
ax_zoom_roll.tick_params(labelleft=False)
ax_zoom_roll.tick_params(axis='both', which='major', labelsize=SMALL_SIZE, pad=2)

# Add a solid black border around the zoomed pitch plot
for spine in ax_zoom_roll.spines.values():
    spine.set_linewidth(zoomed_lw)  # Set the border thickness
    spine.set_edgecolor('black')  # Set the border color

# Add abels.append(f"{tick:.1f}")  # Smaller labels for 0.5 values


# Connector Lines (Roll)
con_top = ConnectionPatch(xyA=(ROLL_XLIM[1], ROLL_YLIM[1]), coordsA="data", axesA=ax1, xyB=(ROLL_XLIM[0], ROLL_YLIM[1]), coordsB="data", axesB=ax_zoom_roll, color="black", linestyle='--', linewidth=zoomed_lw)
fig.add_artist(con_top)
con_bot = ConnectionPatch(xyA=(ROLL_XLIM[1], ROLL_YLIM[0]), coordsA="data", axesA=ax1, xyB=(ROLL_XLIM[0], ROLL_YLIM[0]), coordsB="data", axesB=ax_zoom_roll, color="black", linestyle='--', linewidth=zoomed_lw)
fig.add_artist(con_bot)


# --- ROW 2: Estimated Pitch (Main and Zoomed) ---
ax2 = axs[1, 0]
ax2.grid(True, linestyle=':', alpha=0.5)
ax2.plot(imu_rp[:,0]-st, imu_rp[:,2], label="SBG IMU Gravity", linewidth=g_lw, alpha=0.8, c=sbg_imu_color)
ax2.plot(oak_imu_rp[:,0]-st, oak_imu_rp[:,2], label="OAK IMU Gravity", linewidth=g_lw, alpha=0.8, c=oak_imu_color)
ax2.plot(sbg_rp[:,0]-st, sbg_rp[:,2], label='SBG QEKF', linewidth=lw, c=sbg_color)
ax2.plot(tracked_rp[:, 0]-st, tracked_rp[:, 2], label='Horizon Inferred', linewidth=lw, linestyle='dashed', c=horizon_color)
ax2.plot(imu_rp[:,0]-st, madgwick_sbg_rpy[:,1], label="SBG IMU Madgwick", linewidth=lw, c=sbg_imu_color, linestyle='dashed')
ax2.plot(oak_imu_timestamps-st, madgwick_oak_rpy[:, 1], label="OAK IMU Madgwick", linewidth=lw, linestyle='dashed', c=oak_imu_color)
ax2.set_ylim([-22, 8])
ax2.set_xlim(-.1, 3.1)
ax2.yaxis.tick_left()
ax2.tick_params(axis='both', which='major', labelsize=SMALL_SIZE, pad=2)
# ax2.yaxis.set_tick_params(pad=0)  # Shift y-axis ticks slightly to the right
ax2.xaxis.tick_bottom()
# ax2.set_title('Estimated Pitch (Full View)', fontsize=MEDIUM_SIZE)
# ax2.set_xlabel('Time (s)      ', loc='right', fontsize=MEDIUM_SIZE) 
fig.text(0.5, 0.17, 'Time (s)', ha='center', fontsize=MEDIUM_SIZE)
ax2.set_ylabel('Pitch (°)', fontsize=MEDIUM_SIZE, labelpad=0)

# Draw the Zoom Box on ax2 (Rectangle)
rect_pitch = Rectangle(
    (PITCH_XLIM[0], PITCH_YLIM[0]),
    PITCH_XLIM[1] - PITCH_XLIM[0],
    PITCH_YLIM[1] - PITCH_YLIM[0],
    edgecolor='black', facecolor='none', linestyle='--', linewidth=zoomed_lw, zorder=10
)
ax2.add_patch(rect_pitch)

# Legend for the main plots
handles, labels = ax2.get_legend_handles_labels()
order = [2, 3, 4, 5, 0, 1]
# Use bbox_to_anchor to place the legend outside the subplot or adjust its position
# fig.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=2)
# fig.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=3, fontsize=SMALL_SIZE, frameon=False)
# fig.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=2, prop={'size': SMALL_SIZE}, handlelength=3, handleheight=1.25, handlelinewidth=2, frameon=False)
from matplotlib.legend_handler import HandlerLine2D
from matplotlib.lines import Line2D

# Define a simple custom handler that draws a Line2D object with a fixed linewidth
class HandlerThickLine(HandlerLine2D):
    def create_artists(self, legend, orig_handle, xdescent, ydescent,
                    width, height, fontsize, trans):
        # Create the standard line artists but override the linewidth
        line = super().create_artists(
            legend, orig_handle, xdescent, ydescent,
            width, height, fontsize, trans
        )
        line[0].set_linewidth(2)  # <-- Set your desired thickness here (e.g., 2)
        return line

# --- Now, modify your legend call ---
# You need to pass the custom handler for all lines (Line2D objects)
fig.legend(
    [handles[idx] for idx in order],
    [labels[idx] for idx in order],
    loc='lower center',
    bbox_to_anchor=(0.5, 0.0),
    ncol=2,
    prop={'size': SMALL_SIZE},
    handlelength=3,
    handleheight=1.25,
    frameon=False,
    # Pass the handler map to use your custom class for Line2D objects
    handler_map={Line2D: HandlerThickLine(numpoints=1)}
)

# Zoomed Pitch Plot (Bottom-Right)
ax_zoom_pitch = axs[1, 1]

ax_zoom_pitch.plot(sbg_rp[:,0]-st, sbg_rp[:,2], label='SBG QEKF', linewidth=lw, c=sbg_color)
ax_zoom_pitch.plot(tracked_rp[:, 0]-st, tracked_rp[:, 2], label='Horizon Inferred', linewidth=lw, linestyle='dashed', c=horizon_color)
ax_zoom_pitch.plot(imu_rp[:,0]-st, madgwick_sbg_rpy[:,1], label="SBG IMU Madgwick", linewidth=lw, c=sbg_imu_color, linestyle='dashed')
ax_zoom_pitch.plot(oak_imu_timestamps-st, madgwick_oak_rpy[:, 1], label="OAK IMU Madgwick", linewidth=lw, linestyle='dashed', c=oak_imu_color)
ax_zoom_pitch.set_xlim(PITCH_XLIM)
ax_zoom_pitch.set_ylim(PITCH_YLIM)
# ax_zoom_pitch.set_title('Zoom: 0.8s to 1.2s', fontsize=MEDIUM_SIZE)
# ax_zoom_pitch.set_xlabel('Time (s)')
ax_zoom_pitch.grid(True, linestyle=':', alpha=0.5)
ax_zoom_pitch.yaxis.tick_right()
ax_zoom_pitch.yaxis.set_label_position('right')
ax_zoom_pitch.xaxis.tick_bottom()
ax_zoom_pitch.tick_params(axis='both', which='major', labelsize=SMALL_SIZE, pad=2)

# Add a solid black border around the zoomed pitch plot
for spine in ax_zoom_pitch.spines.values():
    spine.set_linewidth(zoomed_lw)  # Set the border thickness
    spine.set_edgecolor('black')  # Set the border color

# ax_zoom_pitch.set_ylabel('Pitch (°)')
ax_zoom_pitch.tick_params(labelleft=False)
# Connector Lines (Pitch)
con_top_p = ConnectionPatch(xyA=(PITCH_XLIM[1], PITCH_YLIM[1]), coordsA="data", axesA=ax2, xyB=(PITCH_XLIM[0], PITCH_YLIM[1]), coordsB="data", axesB=ax_zoom_pitch, color="black", linestyle='--', linewidth=zoomed_lw)
fig.add_artist(con_top_p)
con_bot_p = ConnectionPatch(xyA=(PITCH_XLIM[1], PITCH_YLIM[0]), coordsA="data", axesA=ax2, xyB=(PITCH_XLIM[0], PITCH_YLIM[0]), coordsB="data", axesB=ax_zoom_pitch, color="black", linestyle='--', linewidth=zoomed_lw)
fig.add_artist(con_bot_p)


plt.tight_layout(rect=[0, 0.16, 1, 1], h_pad=0.05, w_pad=0.8) # Added bottom margin for legend, reduced vertical spacing
plt.savefig('horizon_comparison_test.pdf', format='pdf', bbox_inches='tight')
plt.show()