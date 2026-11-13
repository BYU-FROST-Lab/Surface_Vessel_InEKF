import math
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation
try:
    from src.core.measurements import ImuMeasurement
    from src.estimators.invariant_ekf import InEKF
except ImportError:
    pass 

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as lines
import pandas as pd

from matplotlib.lines import Line2D


def add_initial_state_noise(init_state, init_cov):
    """
    Add noise to the initial state based on the provided initial covariance.

    Args:
        init_state (np.ndarray): The original initial state vector.
        init_cov (np.ndarray): Diagonal covariance matrix for initial state noise.
        self.init_cov = np.diag([
            0.1, 0.1, 0.1,        # Rotation/Attitude (rad)
            0.5, 0.5, 0.5,        # Velocity (m/s)
            2.0, 2.0, 2.0,        # Position (m)
            np.deg2rad(0.05), np.deg2rad(0.05), np.deg2rad(0.05),  # Gyro Bias (rad/s)
            0.05, 0.05, 0.05         # Accel Bias (m/s^2)
        ])**2

    Returns:
        np.ndarray: New initial state with added noise.
    """
    std_devs = np.sqrt(np.diag(init_cov[:9, :9]))
    noise = np.random.normal(0, std_devs)
    noisy_init_state = init_state + noise

    return noisy_init_state

def heading_add_noise(heading_measurement):
    heading_std = math.sqrt(heading_measurement.heading_cov)
    noise = np.random.normal(0, heading_std)
    noisy_heading = heading_measurement.heading + noise
    noisy_heading_measurement = heading_measurement.__class__(
        timestamp_sec=heading_measurement.timestamp_sec,
        timestamp_nsec=heading_measurement.timestamp_nsec,
        heading=noisy_heading,
        heading_cov=heading_measurement.heading_cov,
        pitch=heading_measurement.pitch,
        pitch_cov=heading_measurement.pitch_cov
    )
    return noisy_heading_measurement

def gps_add_noise(gps_measurement, scaler=1.0):
    pos_cov = gps_measurement.position_cov
    std_devs = np.sqrt(np.diag(pos_cov)) * scaler
    noise = np.random.normal(0, std_devs)
    noisy_x = gps_measurement.x + noise[0]
    noisy_y = gps_measurement.y + noise[1]
    noisy_z = gps_measurement.z + noise[2]
    noisy_gps_measurement = gps_measurement.__class__(
        timestamp_sec=gps_measurement.timestamp_sec,
        timestamp_nsec=gps_measurement.timestamp_nsec,
        x=noisy_x,
        y=noisy_y,
        z=noisy_z,
        position_cov=pos_cov
    )
    return noisy_gps_measurement

def imu_add_noise(imu_measurement, Q, current_random_walk=None, delta_t=0.005):
    """
    Add noise to an IMU measurement based on the provided noise covariance matrix Q.
    Q is expected to be a diagonal covariance matrix with variances for:
       Order: [Rotation (3), Velocity (3), Position (3) (ignored), Gyro Bias (3), Accel Bias (3)]

    Args:
        imu_measurement (ImuMeasurement): The original IMU measurement.
        Q (np.ndarray): Diagonal covariance matrix for IMU noise.
        current_random_walk (list or np.ndarray): Current random walk values for gyro and accel biases.
        delta_t (float): Time step for random walk update.

    Returns:
        ImuMeasurement: New IMU measurement with added noise.
        np.ndarray: Updated random walk values.
    """

    # Extract standard deviations from the diagonal of Q
    std_devs = np.sqrt(np.diag(Q))*np.sqrt(delta_t) # Scale by sqrt of delta_t for proper noise scaling in discrete time

    # Gyro noise
    gyro_noise = np.random.normal(0, std_devs[0:3])
    # Accel noise
    accel_noise = np.random.normal(0, std_devs[3:6])

    if current_random_walk is not None:
        # Update random walk for gyro and accel biases
        gyro_random_walk = np.random.normal(0, std_devs[9:12]) * np.sqrt(delta_t)
        accel_random_walk = np.random.normal(0, std_devs[12:15]) * np.sqrt(delta_t)

        current_random_walk[0:3] += gyro_random_walk
        current_random_walk[3:6] += accel_random_walk

    noisy_gyro = imu_measurement.gyro + np.random.normal(0, std_devs[0:3]) + current_random_walk[0:3]
    noisy_accel = imu_measurement.accel + np.random.normal(0, std_devs[3:6]) + current_random_walk[3:6]

    # Create new ImuMeasurement with noisy data
    noisy_imu_measurement = ImuMeasurement(
        timestamp_sec=imu_measurement.timestamp_sec,
        timestamp_nsec=imu_measurement.timestamp_nsec,
        gyro=noisy_gyro,
        accel=noisy_accel
    )

    return noisy_imu_measurement, current_random_walk

def convert_gps_to_xyz(lat, lon, ref_latitude_deg=21.596611, ref_longitude_deg=-158.106222):
    """
    Convert latitude and longitude to local Cartesian coordinates (x, y).
    """
    # Convert degrees to radians
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    ref_lat_rad = math.radians(ref_latitude_deg)
    ref_lon_rad = math.radians(ref_longitude_deg)
    
    # Constants
    earth_radius = 6378137.0  # Earth's radius in meters
    
    # Calculate deltas
    delta_lat = lat_rad - ref_lat_rad
    delta_lon = lon_rad - ref_lon_rad
    
    # Approximate x and y using a local tangent plane
    x = delta_lon * earth_radius * math.cos(ref_lat_rad)  # Longitude scaling by cos(lat)
    y = delta_lat * earth_radius                          # Latitude scaling
    
    return x, y

def convert_filter_mus_to_rpy_xyz(filter):
    """
    Convert filter mus (list of 5x5 matrices) to roll, pitch, yaw and x, y, z positions.

    Args:
        filter (InEKF): An instance of the InEKF filter containing mus and timestamps.
    Returns:
        np.ndarray: Array of shape (N, 6) where each row is [roll, pitch, yaw, x, y, z].
    """

    t_rpy_xyz = []
    for timestamp, mu in zip(filter.timestamps, filter.mus):
        R_mat = mu[0:3, 0:3]
        rpy = Rotation.from_matrix(R_mat).as_euler('xyz', degrees=True)
        pos = mu[0:3, 4]
        t_rpy_xyz.append([timestamp, rpy[0], rpy[1], rpy[2], pos[0], pos[1], pos[2]])
    return np.array(t_rpy_xyz)

def compute_position_mae(filter_t_rpyxyz, gt_poses):
    # Values should not differ by more than 1 index
    if len(gt_poses) != len(filter_t_rpyxyz):
        if len(gt_poses) < len(filter_t_rpyxyz):
            if gt_poses[0,0] < filter_t_rpyxyz[0,0]:
                # Trim the start of filter_t_rpyxyz
                start_time = gt_poses[0,0]
                filter_t_rpyxyz = filter_t_rpyxyz[filter_t_rpyxyz[:,0]>=start_time]
            else:
                gt_poses = gt_poses[:-1]
        else:
            if filter_t_rpyxyz[0,0] < gt_poses[0,0]:
                # Trim the start of gt_poses
                start_time = filter_t_rpyxyz[0,0]
                gt_poses = gt_poses[gt_poses[:,0]>=start_time]
            else:
                filter_t_rpyxyz = filter_t_rpyxyz[:-1]

    filter_t_rpyxyz = filter_t_rpyxyz[(200*5):] # Skipping ahead 5 seconds to avoid initialization transient
    gt_poses = gt_poses[(200*5):]

    # Calculate XYZ position error
    mus_xy = filter_t_rpyxyz[:, 4:6]
    gt_xy = gt_poses[:, 4:6]
    error_xy = np.linalg.norm(mus_xy - gt_xy, axis=1)

    error_z = np.abs(filter_t_rpyxyz[:, 6] - gt_poses[:, 6])
    # Calculate RPY Error
    mus_rpy = filter_t_rpyxyz[:, 1:4]

    n = mus_rpy.shape[0]
    def wrap_angle(angle):
        return (angle + 180) % 360 - 180    
    
    error_r = wrap_angle(mus_rpy[:, 0] - gt_poses[:, 1])
    error_p = wrap_angle(mus_rpy[:, 1] - gt_poses[:, 2])
    error_y = wrap_angle(mus_rpy[:, 2] - gt_poses[:, 3])

    error_rpy = np.abs(np.vstack((error_r, error_p, error_y)).T)
    mean_error_xy = np.mean(error_xy)
    mean_error_z = np.mean(error_z)
    mean_error_rpy = np.mean(error_rpy, axis=0)

    return mean_error_xy, mean_error_z, mean_error_rpy

def fuse_sbg_ekf_msgs(sbg_attitude, sbg_position):
    sbg_fused = []
    pos_idx = 0
    for nav in sbg_position:
        nav_time = nav[0]
        # Find closest attitude message
        while pos_idx < len(sbg_attitude)-1 and sbg_attitude[pos_idx][0] < nav_time:
            pos_idx += 1
        if pos_idx == 0:
            closest_att = sbg_attitude[0]
        else:
            before = sbg_attitude[pos_idx-1]
            after = sbg_attitude[pos_idx]
            if abs(before[0] - nav_time) <= abs(after[0] - nav_time):
                closest_att = before
            else:
                closest_att = after
        sbg_fused.append((nav_time, closest_att[1], closest_att[2], closest_att[3], nav[1], nav[2], nav[3], nav[4], nav[5], nav[6]))

    sbg_fused = np.array(sbg_fused)
    for msg in sbg_fused:
        # msg[6] += 0.27  # Adjust z position for incorrect IMU offset (I say incorrect because the SBG tries to put it in the frame of the imu, not exactly sure)
        yaw = msg[3]
        msg[4] += 0.72 * math.cos(math.radians(yaw))  # Adjust x position for incorrect IMU offset
        msg[5] += 0.72 * math.sin(math.radians(yaw))  # Adjust y position for incorrect IMU offset

    return np.array(sbg_fused)

def sbg_ekf_to_filter(sbg_fused):
    timestamps = sbg_fused[:,0]
    mus = []
    for i in range(len(sbg_fused)):
        rpy = sbg_fused[i,1:4]
        R = Rotation.from_euler('xyz', rpy, degrees=True).as_matrix()
        pos = sbg_fused[i,4:7]
        mu = np.eye(5)
        mu[0:3,0:3] = R
        mu[0:3,4] = pos
        mus.append(mu)
    sbg_ekf = InEKF()
    sbg_ekf.timestamps = timestamps
    sbg_ekf.mus = mus

    return sbg_ekf

def get_filter_error_metrics(filter, gt_filter, timeoffset=0):
    timeoffset_idx1 = (np.abs(np.array(filter.timestamps) - (filter.timestamps[0] + timeoffset))).argmin()
    timeoffset_idx2 = (np.abs(np.array(gt_filter.timestamps) - (gt_filter.timestamps[0] + timeoffset))).argmin()

    stop_idx1 = len(filter.timestamps)
    stop_idx2 = len(gt_filter.timestamps)

    timestamps1 = np.array(filter.timestamps)[timeoffset_idx1:stop_idx1]
    timestamps2 = np.array(gt_filter.timestamps)[timeoffset_idx2:stop_idx2]

    mus1 = np.array(filter.mus)[timeoffset_idx1:stop_idx1]
    mus2 = np.array(gt_filter.mus)[timeoffset_idx2:stop_idx2]
    if len(timestamps1) != len(timestamps2):
        mus2_aligned = []
        for t in timestamps1:
            idx = (np.abs(timestamps2 - t)).argmin()
            mus2_aligned.append(mus2[idx])
        mus2 = np.array(mus2_aligned)

    xy_position_errors = np.linalg.norm(mus1[:, 0:2, 4] - mus2[:, 0:2, 4], axis=1)
    z_position_errors = np.abs(mus1[:, 2, 4] - mus2[:, 2, 4])

    roll_errors = []
    pitch_errors = []
    yaw_errors = []
    for i in range(timeoffset_idx1, len(timestamps1)):
        R1 = mus1[i, 0:3, 0:3]
        R2 = mus2[i, 0:3, 0:3]
        r1_rpy = Rotation.from_matrix(R1).as_euler('xyz')
        r2_rpy = Rotation.from_matrix(R2).as_euler('xyz')
        roll_errors.append(np.abs(np.rad2deg(r1_rpy[0] - r2_rpy[0])))
        pitch_errors.append(np.abs(np.rad2deg(r1_rpy[1] - r2_rpy[1])))
        yaw_errors.append(np.abs(np.rad2deg(r1_rpy[2] - r2_rpy[2])))
    roll_errors = np.array(roll_errors)
    pitch_errors = np.array(pitch_errors)
    yaw_errors = np.array(yaw_errors)

    return [np.mean(xy_position_errors), np.std(xy_position_errors), np.max(xy_position_errors), \
           np.mean(z_position_errors), np.std(z_position_errors), np.max(z_position_errors), \
           np.mean(roll_errors), np.std(roll_errors), np.max(roll_errors), \
           np.mean(pitch_errors), np.std(pitch_errors), np.max(pitch_errors), \
           np.mean(yaw_errors), np.std(yaw_errors), np.max(yaw_errors)]

def get_filter_error_convergence(filter, gt_filter, filter_timestamps, gt_timestamps, timeoffset=0):
    timeoffset_idx1 = (np.abs(np.array(filter_timestamps) - (filter_timestamps[0] + timeoffset))).argmin()
    timeoffset_idx2 = (np.abs(np.array(gt_timestamps) - (gt_timestamps[0] + timeoffset))).argmin()

    stop_idx1 = len(filter_timestamps)
    stop_idx2 = len(gt_timestamps)
    timestamps1 = np.array(filter_timestamps)[timeoffset_idx1:stop_idx1]
    timestamps2 = np.array(gt_timestamps)[timeoffset_idx2:stop_idx2]

    mus1 = np.array(filter)[timeoffset_idx1:stop_idx1]
    mus2 = np.array(gt_filter)[timeoffset_idx2:stop_idx2]
    if len(timestamps1) != len(timestamps2):
        mus2_aligned = []
        for t in timestamps1:
            idx = (np.abs(timestamps2 - t)).argmin()
            mus2_aligned.append(mus2[idx])
        mus2 = np.array(mus2_aligned)

    xy_position_errors = np.linalg.norm(mus1[:, 0:2, 4] - mus2[:, 0:2, 4], axis=1)
    z_position_errors = np.abs(mus1[:, 2, 4] - mus2[:, 2, 4])

    roll_errors = []
    pitch_errors = []
    yaw_errors = []
    for i in range(timeoffset_idx1, len(timestamps1)):
        R1 = mus1[i, 0:3, 0:3]
        R2 = mus2[i, 0:3, 0:3]
        r1_rpy = Rotation.from_matrix(R1).as_euler('xyz')
        r2_rpy = Rotation.from_matrix(R2).as_euler('xyz')
        roll_errors.append(np.rad2deg(r1_rpy[0] - r2_rpy[0]))
        pitch_errors.append(np.rad2deg(r1_rpy[1] - r2_rpy[1]))
        yaw_errors.append(np.rad2deg(r1_rpy[2] - r2_rpy[2]))
    roll_errors = np.array(roll_errors)
    pitch_errors = np.array(pitch_errors)
    yaw_errors = np.array(yaw_errors)

    return [xy_position_errors, z_position_errors, roll_errors, pitch_errors, yaw_errors]

def plot_figure_3(inekf_mae, mekf_mae, inekf_full_mae, show_plot=True):
    matplotlib.rc("pdf", fonttype=42)
    sns.set_context("paper")
    sns.set_style("whitegrid")
    sns.set_palette("colorblind")
    c = sns.color_palette("colorblind")

    # Make sure you install times & clear matplotlib cache
    # https://stackoverflow.com/a/49884009
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["mathtext.fontset"] = "stix"

    textwidth_pt = 495.0779 #505.89  # Replace with your actual \textwidth value in points
    textwidth_inches = textwidth_pt / 72.27

    states = ["Pos (XY)", "Pos (Z)", "Roll", "Pitch", "Yaw"] # Updated states for XY, Roll, Pitch, Yaw
    filters = ["InEKF", "MEKF", "InEKF Full\nOrientation"]

    # --- Prepare Data for Box Plots using Pandas DataFrame ---
    error_df_list = []

    state_errors_data = { # Dictionary to hold error data for each state and filter
        "Pos (XY)": {
            "InEKF": inekf_mae[:, 0],
            "MEKF": mekf_mae[:, 0],
            "InEKF Full\nOrientation": inekf_full_mae[:, 0],
        },
        "Pos (Z)": {
            "InEKF": inekf_mae[:, 1],
            "MEKF": mekf_mae[:, 1],
            "InEKF Full\nOrientation": inekf_full_mae[:, 1],
        },
        "Roll": {
            "InEKF": (inekf_mae[:, 2]), # Roll to degrees
            "MEKF": (mekf_mae[:, 2]),
            "InEKF Full\nOrientation": (inekf_full_mae[:, 2]),
        },
        "Pitch": {
            "InEKF": (inekf_mae[:, 3]), # Pitch to degrees
            "MEKF": (mekf_mae[:, 3]),
            "InEKF Full\nOrientation": (inekf_full_mae[:, 3]),
        },
        "Yaw": {
            "InEKF": (inekf_mae[:, 4]), # Yaw to degrees
            "MEKF": (mekf_mae[:, 4]),
            "InEKF Full\nOrientation": (inekf_full_mae[:, 4]),
        }
    }

    for state_name in states:
        df_list_state = [] # temporary list to hold dataframes for each filter in this state
        for filter_name in filters:
            state_error_values = state_errors_data[state_name][filter_name]
            df_filter = pd.DataFrame({'Filter': [filter_name] * len(state_error_values),
                                        'State': [state_name] * len(state_error_values),
                                        'Error Value': state_error_values})
            df_list_state.append(df_filter)
        error_df_list.append(pd.concat(df_list_state, ignore_index=True))

    fig = plt.figure(figsize=(textwidth_inches, 2))
    gs = gridspec.GridSpec(1, 6, figure=fig, width_ratios=[1, 1.01, -0.1, 1.01, 1, 1]) # adjust width ratios

    # Create the subplots using the GridSpec
    axes = [fig.add_subplot(gs[0, i]) for i in [0, 1, 3, 4, 5]] # Skip column 2

    # Share y-axis for the first two subplots
    axes[1].sharey(axes[0])

    # Share y-axis for the last three subplots
    axes[3].sharey(axes[2])
    axes[4].sharey(axes[2]) # Or axes[4].sharey(axes[3])

    # Adjust subplot spacing
    fig.subplots_adjust(left=0.068, wspace=0.3)  # Adjust left margin & spacing between plots

    # Ensure filters match the data exactly
    filters = ["InEKF", "MEKF", "InEKF Full\nOrientation"]
    for i, state_name in enumerate(states):
        ax = axes[i]
        ax.yaxis.labelpad = 0 # Adjust this value to control the padding
        ax.yaxis.grid(True, linestyle='-', linewidth=1, color='gray')
        state_df = error_df_list[i]
        
        show_legend = True if i == 0 else False

        sns.boxplot(
            x='Filter', 
            y='Error Value', 
            hue='Filter',
            data=state_df, 
            linewidth=1, 
            ax=ax,
            hue_order=filters, 
            dodge=False, 
            width=0.9, 
            showfliers=False, 
            legend=show_legend
        )

        # Fix tick label warning: Explicitly set tick positions
        if state_name in ["Roll", "Pitch", "Yaw"]:
            ax.set_ylim(0, 1)
        else:
            ax.set_ylim(0.2, 2)
        ax.set_yticks(ax.get_yticks())  # Ensures labels align correctly
        ax.tick_params(axis='y', which='major', pad=-2) # Adjust the 'pad' value
        ax.set_yticklabels([f'{y:.1f}' for y in ax.get_yticks()], fontsize=8)


        ax.set_title(f'{state_name}', fontsize=8, fontweight='bold')
        if state_name == "Roll":
            ax.set_ylabel('degrees', fontsize=8)
        elif i == 0:
            ax.set_ylabel('meters', fontsize=8)
        else:
            ax.set_ylabel('')
        ax.set_xticklabels([])
        ax.set_xlabel('')

        if i == 0:
            ax.legend(fontsize=8, loc='upper center',)

    fig.text(0.01, 0.5, 'Trajectory Mean Abs. Error', va='center', rotation='vertical', fontsize=8, fontweight='bold')
    plt.savefig("figures/simulation_MAE.pdf", bbox_inches='tight')
    if show_plot:
        plt.show()

def plot_figure_4(inekf_5hz_conv_trajectories, inekf_no_horizon_trajectories, mekf_5hz_conv_trajectories, mekf_no_horizon_trajectories, gt_data, show_plot=True):
    matplotlib.rc("pdf", fonttype=42)
    sns.set_context("paper")
    sns.set_style("whitegrid")
    sns.set_palette("colorblind")
    c = sns.color_palette("colorblind")

    # Make sure you install times & clear matplotlib cache
    # https://stackoverflow.com/a/49884009
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["mathtext.fontset"] = "stix"

    textwidth_pt = 495.0779 #505.89  # Replace with your actual \textwidth value in points
    textwidth_inches = textwidth_pt / 72.27

    filter_names = ["MEKF", "INEKF"]
    state_names = ["Roll (°)", "Pitch (°)", "Yaw (°)", "X (m)", "Y (m)", "Z (m)"]  

    num_filters = len(filter_names)
    num_states = len(state_names)

    timesteps_to_plot = inekf_5hz_conv_trajectories.shape[1]  # Adjust as needed

    plot_data = {"Cam": {"INEKF": inekf_5hz_conv_trajectories[:,:,1:], "MEKF": mekf_5hz_conv_trajectories[:,:,1:]},
                "Yaw": {"INEKF": inekf_no_horizon_trajectories[:,:,1:], "MEKF": mekf_no_horizon_trajectories[:,:,1:]}}

    ymin = [-180, -90, -180, -15, -25, -40]
    ymax = [180,   90,  180, 35, 25,  20]


    ytick_fontsize=6

    fig, axes = plt.subplots(nrows=4, ncols=num_states, figsize=(textwidth_inches, int(textwidth_inches/1.5))) #, hspace=0.2, wspace=0.3) #textwidth_inches

    for col_idx, state_name in enumerate(state_names):
        # Plot Cam data (rows 0 and 1)
        for row_idx, filter_name in enumerate(filter_names):
            ax_cam = axes[row_idx, col_idx]
            data_cam = plot_data["Cam"][filter_name][:, :timesteps_to_plot, col_idx].copy() # Use a copy to avoid modifying original data
            if col_idx in [0, 1, 2]:
                for sim in range(data_cam.shape[0]):
                    diff = np.diff(data_cam[sim, :])
                    jump_indices = np.where(np.abs(diff) > 270)[0] # Detect jumps larger than 180 degrees
                    for index in jump_indices:
                        data_cam[sim, index + 1] = np.nan # Insert NaN to break the line

            ax_cam.plot(data_cam.T, alpha=0.8, linewidth=0.25) # Plot the transpose if needed to match dimensions

            gt_state_data = gt_data[:timesteps_to_plot, col_idx+1].copy()
            if col_idx in [0, 1, 2]:
                diff_gt = np.diff(gt_state_data)
                jump_indices_gt = np.where(np.abs(diff_gt) > 270)[0]
                for index in jump_indices_gt:
                    gt_state_data[index + 1] = np.nan
            ax_cam.plot(gt_state_data, 'k--', label='Ground Truth' if row_idx == num_filters - 1 and col_idx == num_states - 1 else "")
        

            if row_idx == 0:
                ax_cam.set_title(state_name)
            if col_idx == 0:
                ax_cam.set_ylabel(f"{filter_name}", fontsize=8, labelpad=-4)
            ax_cam.set_ylim([ymin[col_idx], ymax[col_idx]])
            if col_idx == 0 or col_idx == 2:  # Roll and Yaw
                ax_cam.set_yticks([-180, -90, 0, 90, 180])
            elif col_idx == 1:  # Pitch
                ax_cam.set_yticks([-90, -45, 0, 45, 90])
            elif col_idx in [3, 4, 5]:  # X, Y, Z
                yticks_positions = np.linspace(ymin[col_idx], ymax[col_idx], 5, dtype=int)
                ax_cam.set_yticks(yticks_positions)
                # To ensure integer labels (if appropriate for your data range):
                ax_cam.set_yticklabels([f'{int(y)}' for y in yticks_positions])
            
            ax_cam.set_yticklabels(ax_cam.get_yticks(), fontsize=ytick_fontsize)

            ax_cam.grid(True)
            # Remove x-axis labels for all rows except the last one (index 3)
            ax_cam.set_xticklabels([])
            xticks_positions = [0, 1000, 2000] 

            ax_cam.set_xticks(xticks_positions)
            ax_cam.tick_params(axis='y', which='major', pad=-3)
            ax_cam.set_xlim(left=None, right=2000)


        # Plot Yaw data (rows 2 and 3)
        for row_idx, filter_name in enumerate(filter_names):
            ax_yaw = axes[row_idx + 2, col_idx]
            data_yaw = plot_data["Yaw"][filter_name][:, :timesteps_to_plot, col_idx].copy()
            if col_idx in [0, 1, 2]:
                for sim in range(data_yaw.shape[0]):
                    diff = np.diff(data_yaw[sim, :])
                    jump_indices = np.where(np.abs(diff) > 270)[0]
                    for index in jump_indices:
                        data_yaw[sim, index + 1] = np.nan

            ax_yaw.plot(data_yaw.T, alpha=0.8, linewidth=0.25) # Plot the transpose if needed to match dimensions

            gt_state_data = gt_data[:timesteps_to_plot, col_idx+1].copy()
            if col_idx in [0, 1, 2]:
                diff_gt = np.diff(gt_state_data)
                jump_indices_gt = np.where(np.abs(diff_gt) > 270)[0]
                for index in jump_indices_gt:
                    gt_state_data[index + 1] = np.nan
            ax_yaw.plot(gt_state_data, 'k--', label='Ground Truth' if row_idx == num_filters - 1 and col_idx == num_states - 1 else "")

            if col_idx == 0:
                ax_yaw.set_ylabel(f"{filter_name}", fontsize=8, labelpad=-4)
            ax_yaw.set_ylim([ymin[col_idx], ymax[col_idx]])
            if col_idx == 0 or col_idx == 2:  # Roll and Yaw
                ax_yaw.set_yticks([-180, -90, 0, 90, 180])
            elif col_idx == 1:  # Pitch
                ax_yaw.set_yticks([-90, -45, 0, 45, 90])
            elif col_idx in [3, 4, 5]:  # X, Y, Z
                yticks_positions = np.linspace(ymin[col_idx], ymax[col_idx], 5, dtype=int)
                ax_yaw.set_yticks(yticks_positions)
                ax_yaw.set_yticklabels([f'{int(y)}' for y in yticks_positions])
            
            ax_yaw.set_yticklabels(ax_yaw.get_yticks(), fontsize=ytick_fontsize)

            ax_yaw.grid(True)
            xticks_positions = [0, 1000, 2000]
            ax_yaw.set_xticks(xticks_positions)
            if row_idx + 2 < 3: # Check if the current row is not the last row (index 3)
                ax_yaw.set_xticklabels([])
            elif row_idx + 2 == 3:
                # Convert time steps to seconds
                xticks_labels = ['0', '5', '10']
                ax_yaw.set_xticklabels(xticks_labels, fontsize=ytick_fontsize)
                labels = ax_yaw.get_xticklabels()
                labels[-1].set_ha('right') # Or try 'center'
                ax_yaw.tick_params(axis='x', which='major', pad=-2.5)
            
            ax_yaw.tick_params(axis='y', which='major', pad=-3)
            ax_yaw.set_xlim(left=None, right=2000)

    fig.text(0.001, 0.7, 'With Horizon Meas. (4 Hz)', va='center', rotation='vertical', fontsize=8)#, fontweight='bold')
    fig.text(0.001, 0.25, 'Without Horizon Meas.', va='center', rotation='vertical', fontsize=8)#, fontweight='bold')

    fig.text(0.5, 0.01, 'Time (s)', ha='center', fontsize=8)

    # Adjust layout
    plt.subplots_adjust(hspace=0.01, wspace=0.02) # Adjust these values as needed

    plt.tight_layout(rect=[0, 0.03, 1, 0.95], h_pad=0.2, w_pad=0.025)

    ### Iterate through lower subplot and create a whitespace
    # Get the bounding box of the last "Cam" subplot (you can choose any column)
    bbox_cam_bottom = axes[1, 0].get_position()
    cam_bottom_y = bbox_cam_bottom.y0

    # Get the bounding box of the first "Yaw" subplot
    bbox_yaw_top = axes[2, 0].get_position()
    yaw_top_y = bbox_yaw_top.y1

    # Calculate the current vertical distance (can be negative if overlapping)
    current_gap = yaw_top_y - cam_bottom_y

    # Define the desired gap (as a fraction of the figure height - experiment with this value)
    desired_gap = 0
    # Calculate the additional shift needed for the "Yaw" subplots
    additional_shift = desired_gap - current_gap
    # Iterate through the "Yaw" subplots (rows 2 and 3) and shift their positions
    for i in range(2, 4):  # Rows 2 and 3
        for j in range(axes.shape[1]):
            ax = axes[i, j]
            pos = ax.get_position().bounds  # [x, y, width, height]
            new_y = pos[1] - additional_shift # Shift downwards in figure coordinates
            ax.set_position([pos[0], new_y, pos[2], pos[3]])


    # Get the bounding boxes of the Cam and Yaw axes to find the vertical gap
    bbox_cam = axes[1, 0].get_position() # Bottom row of Cam plots
    bbox_yaw = axes[2, 0].get_position() # Top row of Yaw plots

    # Calculate the y-coordinate for the horizontal line (midpoint between the two rows)
    y_line = (bbox_cam.y0 + bbox_yaw.y1) / 2


    # Create a Line2D artist spanning the width of the subplots
    line = lines.Line2D([bbox_cam.x0-0.05, 0.978], [y_line, y_line], color='gray', linewidth=0.7, linestyle='--')
    # Add the line to the figure
    fig.add_artist(line)
    plt.savefig("figures/sim_convergence_plot.pdf", bbox_inches='tight', pad_inches=0)
    if show_plot:
        plt.show()

def plot_figure_6(df, show_plot=True):
    # --- Style Setup ---
    plt.rc("pdf", fonttype=42)
    sns.set_context("paper")
    sns.set_style("whitegrid")
    
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["mathtext.fontset"] = "stix"

    # % The column width is: 241.53897pt
    # % The total text width is: 495.07794pt
    textwidth_pt = 495.0779
    textwidth_inches = textwidth_pt / 72.27
    
    states = ["Pos (XY)", "Pos (Z)", "Roll", "Pitch", "Yaw"]
    csv_map = {
        "Pos (XY)": ["pos_mean", "pos_std", "pos_max"],
        "Pos (Z)": ["z_mean", "z_std", "z_max"],
        "Roll": ["roll_mean", "roll_std", "roll_max"],
        "Pitch": ["pitch_mean", "pitch_std", "pitch_max"],
        "Yaw": ["yaw_mean", "yaw_std", "yaw_max"]
    }

    filter_labels = {
        "sbg_inekf": "SBG InEKF (L)",
        "oak_inekf": "OAK InEKF (L)",
        "sbg_inekf_right": "SBG InEKF (R)",
        "oak_inekf_right": "OAK InEKF (R)",
        "sbg_mekf": "SBG MEKF",
        "oak_mekf": "OAK MEKF",
        "sbg_inekf_full_pose": "SBG InEKF FO",
        "oak_inekf_full_pose": "OAK InEKF FO",
    }
    filter_names = df["filter_name"].tolist()
    filter_display_names = [filter_labels[name] for name in filter_names]
    colors = sns.color_palette("colorblind", n_colors=len(filter_display_names))

    # --- Create Figure & GridSpec ---
    # Increased height slightly to accommodate the legend underneath
    fig = plt.figure(figsize=(textwidth_inches, 2.8)) 
    gs = gridspec.GridSpec(1, 6, figure=fig, width_ratios=[1, 1.01, 0.05, 1, 1, 1]) 
    axes = [fig.add_subplot(gs[0, i]) for i in [0, 1, 3, 4, 5]]

    for i, state_name in enumerate(states):
        ax = axes[i]
        cols = csv_map[state_name]
        
        means = df[cols[0]].values
        stds = df[cols[1]].values
        maxs = df[cols[2]].values
        x_pos = np.arange(len(filter_display_names))

        # Plot Mean Bars + Std Dev Error Bars
        ax.bar(x_pos, means, yerr=stds, capsize=2, color=colors, 
               edgecolor='black', linewidth=0.5, alpha=0.8)
        
        # Plot Max Error Markers
        ax.scatter(x_pos, maxs, color='red', marker='x', s=12, linewidth=0.7, zorder=3)

        # Title and Formatting
        ax.set_title(state_name, fontsize=10, fontweight='bold')
        ax.set_xticks([]) 
        ax.yaxis.grid(True, linestyle='-', linewidth=0.5, color='lightgray')
        
        # Axis Scaling & Labels
        if state_name in ["Roll", "Pitch", "Yaw"]:
            ax.set_ylim(0, max(3.0, 1.05 * max(maxs.max(), (means + stds).max())))
            
            # 1. Define all positions where you want a "tick" line
            tick_positions = np.arange(0, 3.5, 0.5)
            
            # 2. Create labels: only show the number if it's an integer, otherwise ""
            tick_labels = [f"{int(x)}" if x % 1 == 0 else "" for x in tick_positions]
            
            if ax.get_ylim()[1] == 3.0:
                ax.set_yticks(tick_positions)
                ax.set_yticklabels(tick_labels)
            if i == 2:
                ax.set_ylabel('degrees', fontsize=10)
        else:
            # Position scaling (adjust if your max position error is > 6m)
            ax.set_ylim(0, max(5.5, 1.05 * max(maxs.max(), (means + stds).max())))
            if i == 0:
                ax.set_ylabel('meters', fontsize=10)

        ax.tick_params(axis='y', labelsize=10, pad=1)

    # --- Legend Construction ---
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors[j], label=filter_display_names[j], edgecolor='black', linewidth=0.7) 
        for j in range(len(filter_display_names))
    ]

    legend_by_name = dict(zip(filter_names, legend_elements))
    legend_elements = [legend_by_name[name] for name in filter_labels if name in legend_by_name]
    
    # Add the Max Error 'x' to the legend
    legend_elements.append(Line2D([0], [0], marker='x', color='red', label='Max Error', 
                                  markersize=5, linestyle='None', markeredgewidth=1))

    # Place legend centered underneath the subplots
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.0),
               ncol=4, fontsize=10, frameon=False, handletextpad=0.5, columnspacing=1.0)

    # Adjust layout to make room for labels and legend
    plt.subplots_adjust(left=0.07, right=0.98, top=0.85, bottom=0.29, wspace=0.45)
    
    plt.savefig("figures/filter_comparison_bars.pdf", bbox_inches='tight')
    if show_plot:
        plt.show()

def report_real_world_convergence(
    filter_runs, reference_poses, reference_timestamps,
    angle_tol_deg=5.0, position_tol_m=1.0, dwell_s=0.5,
):
    """Report settling statistics and plot signed errors over completed restarts."""
    if not filter_runs:
        print("No completed restarts to evaluate.")
        return

    names = list(filter_runs[0])
    fig, axes = plt.subplots(
        len(names), 6, figsize=(12, 1.35 * len(names)), sharex=True, sharey="col",
        layout="constrained", squeeze=False,
    )
    colors = sns.color_palette("colorblind", len(names))

    settling_times = {name: [] for name in filter_runs[0]}
    final_errors = {name: [] for name in filter_runs[0]}
    for filters, ref, ref_times in zip(filter_runs, reference_poses, reference_timestamps):
        ref_angles = np.unwrap(
            Rotation.from_matrix(ref[:, :3, :3]).as_euler("xyz"), axis=0
        )
        ref_states = np.column_stack((np.rad2deg(ref_angles), ref[:, :3, 4]))
        for row, name in enumerate(names):
            estimator = filters[name]
            times = np.asarray(estimator.timestamps)
            overlap = (times >= ref_times[0]) & (times <= ref_times[-1])
            if not overlap.any():
                continue

            poses = np.asarray(estimator.mus)[overlap]
            angles = Rotation.from_matrix(poses[:, :3, :3]).as_euler("xyz", degrees=True)
            states = np.column_stack((angles, poses[:, :3, 4]))
            aligned_reference = np.column_stack([
                np.interp(times[overlap], ref_times, state) for state in ref_states.T
            ])
            errors = states - aligned_reference
            errors[:, :3] = (errors[:, :3] + 180) % 360 - 180
            absolute_errors = np.abs(errors)
            final_errors[name].append(absolute_errors[-1])

            within = (absolute_errors <= [angle_tol_deg] * 3 + [position_tol_m] * 3).all(axis=1)
            elapsed = times[overlap] - times[0]
            angle_errors = errors[:, :3]
            angle_errors[1:][np.abs(np.diff(angle_errors, axis=0)) > 180] = np.nan
            for ax, error in zip(axes[row], errors.T):
                ax.plot(elapsed, error, color=colors[row], alpha=0.4, linewidth=0.7)

            # A settled run stays within tolerance through its last matched sample.
            violations = np.flatnonzero(~within)
            start = violations[-1] + 1 if violations.size else 0
            if start < len(elapsed) and elapsed[-1] - elapsed[start] >= dwell_s:
                settling_times[name].append(elapsed[start])

    rows = []
    for name, times in settling_times.items():
        errors = final_errors[name]
        median_errors = np.median(errors, axis=0) if errors else [np.nan] * 6
        rows.append([
            name, len(errors), len(times), len(times) / len(filter_runs),
            np.median(times) if times else np.nan, *median_errors,
        ])

    summary = pd.DataFrame(rows, columns=[
        "Filter", "Matched", "Settled", "Rate", "Median settle (s)",
        "Roll (°)", "Pitch (°)", "Yaw (°)", "X (m)", "Y (m)", "Z (m)",
    ])
    print(f"\nConvergence against SBG EKF across {len(filter_runs)} restarts")
    print(f"Settled if each angle stays within {angle_tol_deg:g}° and each position axis "
          f"within {position_tol_m:g} m through the last matched sample for at least {dwell_s:g} s.")
    print("State columns show median final absolute error. Median settling time uses settled runs.")
    print(summary.to_string(index=False, na_rep="—", formatters={"Rate": "{:.0%}".format},
                            float_format="{:.3f}".format))
    for row, name in enumerate(names):
        axes[row, 0].set_ylabel(name, fontsize=9)
        for ax in axes[row]:
            ax.axhline(0, color="0.35", linestyle="--", linewidth=0.7)
            ax.grid(alpha=0.25)
    for ax, label in zip(axes[0], summary.columns[-6:]):
        ax.set_title(label)
    fig.suptitle("Pose convergence relative to SBG EKF")
    fig.supxlabel("Time since filter initialization (s)")
    fig.savefig("figures/real_world_state_convergence.pdf")
    plt.show()
    return summary


def plot_figure_7_rw_convergence(sbg_inekf_mus, oak_inekf_mus, sbg_mekf_mus, oak_mekf_mus, 
                                 gt_poses, sbg_timestamps, oak_timestamps, gt_timestamps, show=True):
    
    # 1. Compute errors 
    # Each entry in the dict becomes a list of arrays. Each array is [5, t_sim]
    filter_data = {
        'SBG\nMEKF': [get_filter_error_convergence(m, g, s_t, g_t) for m, g, s_t, g_t in zip(sbg_mekf_mus, gt_poses, sbg_timestamps, gt_timestamps)],
        'SBG\nInEKF': [get_filter_error_convergence(m, g, s_t, g_t) for m, g, s_t, g_t in zip(sbg_inekf_mus, gt_poses, sbg_timestamps, gt_timestamps)],
        'OAK\nMEKF': [get_filter_error_convergence(m, g, o_t, g_t) for m, g, o_t, g_t in zip(oak_mekf_mus, gt_poses, oak_timestamps, gt_timestamps)],
        'OAK\nInEKF': [get_filter_error_convergence(m, g, o_t, g_t) for m, g, o_t, g_t in zip(oak_inekf_mus, gt_poses, oak_timestamps, gt_timestamps)]
    }

    row_names = ['SBG\nMEKF', 'SBG\nInEKF', 'OAK\nMEKF', 'OAK\nInEKF']
    col_names = ['Roll Error (°)', 'Pitch Error (°)', 'Yaw Error (°)']
    
    ###
    matplotlib.rc("pdf", fonttype=42)
    sns.set_context("paper")
    sns.set_style("whitegrid")
    sns.set_palette("colorblind")
    c = sns.color_palette("colorblind")

    # Make sure you install times & clear matplotlib cache
    # https://stackoverflow.com/a/49884009
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["mathtext.fontset"] = "stix"

    textwidth_pt = 241.53897 #505.89  # Replace with your actual \textwidth value in points
    textwidth_inches = textwidth_pt / 72.27

    sns.set_style("whitegrid")
    
    fig, axs = plt.subplots(4, 3, figsize=(textwidth_inches, int(textwidth_inches)), dpi=300, sharex=True, constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0, h_pad=0, hspace=0, wspace=0.05) # Negative wspace pulls columns in
    c = sns.color_palette("colorblind")

    for row_idx, filter_name in enumerate(row_names):
        runs = filter_data[filter_name] # List of arrays, each [5, t_sim]
        
        # Select the correct timestamps for this filter group
        if "SBG" in filter_name:
            # Assuming sbg_timestamps is a list of arrays corresponding to simulation runs
            current_timestamps_list = sbg_timestamps 
        else:
            current_timestamps_list = oak_timestamps

        for col_idx in range(3):
            ax = axs[row_idx, col_idx]

            # Set Y-Limits
            if col_idx < 2:
                ax.set_ylim(-16, 16)
                if col_idx == 0:
                    ax.set_yticks([-10, 0, 10])  # Set custom y-ticks
                    ax.set_yticklabels(['-10', '0', '10'], fontsize=9)  # Label the y-ticks
                if col_idx == 1:
                    if col_idx == 1:
                        ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False, labelright=False)
            else:
                ax.set_ylim(-330, 30)
                ax.set_yticks([0, -180])  # Set custom y-ticks
                ax.set_yticklabels(['0', '-180'], fontsize=9)  # Label the y-ticks

            ax.tick_params(axis='x', pad=-3) # Decrease padding between tick and graph
            ax.tick_params(axis='y', pad=-3)  

            ax.set_xticks(np.arange(0, 3, 1))  # Set ticks at 0, 1, 2
            ax.set_xticklabels(['0', '', '2'], fontsize=9)  # Label only 0 and 2

            # 1. Iterate through each simulation individually
            for sim_idx, run_data in enumerate(runs):
                t_raw = current_timestamps_list[sim_idx]
                t_norm = t_raw - t_raw[0] # Normalize to start at 0
                
                # Extract the specific error row (R=2, P=3, Y=4)
                y_vals = run_data[col_idx + 2]

                mask = t_norm <= 2.0 
                t_plot = t_norm[mask]
                y_plot = y_vals[mask]

                ax.plot(t_plot, y_plot, alpha=0.8, linewidth=0.25)

            # Formatting
            ax.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
            ax.set_xlim(0, 2.0) # Force the X-axis to 2 seconds for uniform comparison
            for label in ax.get_xticklabels():
                if label.get_text() == '0':
                    label.set_horizontalalignment('left')
                if label.get_text() == '2':
                    label.set_horizontalalignment('right')

            if row_idx == 0:
                ax.set_title(col_names[col_idx], fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(f"{filter_name}", fontsize=10)
    
    axs[3, 1].set_xlabel("Time (s)", fontsize=10, labelpad=-2)  # Set xlabel only for the middle bottom

    if show:
        plt.savefig("figures/real_world_convergence.pdf")
        plt.show()

    return

# A helper function to filter a single array
def prune_outliers(data, std_devs=3):
    data = np.array(data)
    mean = np.mean(data)
    std = np.std(data)
    # Boolean mask for values within the range
    return data[np.abs(data - mean) <= (std_devs * std)]

def plot_figure_8_timing(
    image_function_timing, inekf_predict_timing, mekf_predict_timing,
    inekf_heading_update_timing, mekf_heading_update_timing,
    inekf_rp_update_timing, mekf_rp_update_timing, inekf_full_pose_update_timing,
    show=True, *, inekf_right_predict_timing=(), inekf_right_rp_update_timing=(),
):
    print("Timing Results (s) for roll pitch estimation from Horizon:")
    print(f"Average time for image callback function was {np.mean(image_function_timing)*1000:.4f} ms, with a standard deviation of {np.std(image_function_timing)*1000:.4f} ms over {len(image_function_timing)} runs.")
    print(f"This equates to an average frequency of {1/np.mean(image_function_timing):.2f} Hz for the image processing callback.")

    print(f"Size of predict function timings: {len(inekf_predict_timing)}")
    print(f"Size of heading update function timings: {len(inekf_heading_update_timing)}")
    print(f"Size of roll/pitch update function timings: {len(inekf_rp_update_timing)}")
    print(f"Size of full Orientation update function timings: {len(inekf_full_pose_update_timing)}")

    data_list = []
    
    # Structure data
    groups = [
        ('Predict', 'InEKF (L)', inekf_predict_timing),
        ('Predict', 'MEKF', mekf_predict_timing),
        ('Heading\nUpdate', 'InEKF (L)', inekf_heading_update_timing),
        ('Heading\nUpdate', 'MEKF', mekf_heading_update_timing),
        ('Roll/Pitch\nUpdate', 'InEKF (L)', inekf_rp_update_timing),
        ('Roll/Pitch\nUpdate', 'MEKF', mekf_rp_update_timing),
        ('Full Ori.\nUpdate', 'InEKF FO', inekf_full_pose_update_timing),
    ]

    groups.extend([
        ('Predict', 'InEKF (R)', inekf_right_predict_timing),
        ('Roll/Pitch\nUpdate', 'InEKF (R)', inekf_right_rp_update_timing),
    ])
    pruned_groups = [
       (label, model, prune_outliers(timing_data))
        for label, model, timing_data in groups
        if len(timing_data)
    ]   

    for task, filter_type, timings in pruned_groups:
        for t in timings:
            data_list.append({'Task': task, 'Filter': filter_type, 'Time (ms)': t * 1000})

    df = pd.DataFrame(data_list)

    # 1. IJRR / IEEE Single-Column Specifications
    # 3.5 inches is the standard column width. 
    # Height is reduced to keep the aspect ratio tight.
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams['pdf.fonttype'] = 42
    
    FIG_WIDTH = 241.53897 / 72.27  # Convert from points to inches
    FIG_HEIGHT = FIG_WIDTH * 0.7     # Adjust height for a more compact
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=300)
    sns.set_style("whitegrid", {"grid.linestyle": ":"})
    ax.grid(True, which='major', axis='y', linestyle=':', linewidth=0.8, alpha=0.7)
    # 2. Plot with adjusted widths for small scale
    ax = sns.violinplot(
        x='Task', 
        y='Time (ms)', 
        hue='Filter', 
        data=df, 
        split=False,
        inner=None, 
        palette=dict(zip(
            ["InEKF (L)", "MEKF", "InEKF FO", "InEKF (R)"],
            sns.color_palette("colorblind", 4),
        )),
        linewidth=0.8,
        cut=0,
        density_norm='width',
        width=0.8,        # Increases the footprint of the violin on the x-axis
        ax=ax
    )

    ax.set_ylim(0, max(7, 1.05 * df["Time (ms)"].max()))

    ax.set_ylabel("Execution Time (ms)", fontsize=10)
    ax.set_xlabel("", fontsize=8)
    
    # Compact Tick Labels
    plt.xticks(rotation=0, ha='center', fontsize=10)
    ax.tick_params(axis='x', which='major', pad=7)
    
    plt.yticks(fontsize=10)
    
    # Compact Legend
    ax.legend(loc='upper right', fontsize=8, ncol=2, frameon=True)
    plt.tight_layout(pad=0.5)

    if show:
        plt.savefig("figures/timing_violin.pdf", bbox_inches='tight', pad_inches=0.1)
        plt.show()

    return ax


import pickle as pkl
if __name__ == "__main__":
    try:
        data = np.load("figure_data/inekf_holoocean_simulation_results.npz", allow_pickle=True)
        inekf_full_traj_maes = data["inekf_full_traj_maes"]
        inekf_5hz_conv_trajectories = data["inekf_5hz_conv_trajectories"]
        inekf_no_horizon_trajectories = data["inekf_no_horizon_trajectories"]
        inEKF_full_pose_full_traj_maes = data["inEKF_full_pose_full_traj_maes"]
        mekf_full_traj_maes = data["mekf_full_traj_maes"]
        mekf_5hz_conv_trajectories = data["mekf_5hz_conv_trajectories"]
        mekf_no_horizon_trajectories = data["mekf_no_horizon_trajectories"]
        gt_data = data["gt_data"]

        plot_figure_3(inekf_full_traj_maes, mekf_full_traj_maes, inEKF_full_pose_full_traj_maes, show_plot=False)
        plot_figure_4(inekf_5hz_conv_trajectories, inekf_no_horizon_trajectories, mekf_5hz_conv_trajectories, mekf_no_horizon_trajectories, gt_data)
    except Exception as e:
        print(f"Error loading simulation data for figure 4 and 5: {e}")
        print("Make sure the file 'figure_data/inekf_holoocean_simulation_results.npz' exists and is in the correct format.")
        print("You can also generate this data by running 'run_sim_experiments_fig3_fig4.py'.")
        

    try:
        df_loaded = pd.read_csv('figure_data/filter_metrics_comparison.csv')
        plot_figure_6(df_loaded, show_plot=False)
    except Exception as e:
        print(f"Error loading data for Figure 6: {e}")
        print("Make sure the file 'figure_data/filter_metrics_comparison.csv' exists and is in the correct format.")

    try:
        # Plotting figure 7 convergence for real-world data
        with open("figure_data/real_world_convergence_data.pkl", "rb") as f:
            convergence_data = pkl.load(f) 
            sbg_inekf_mus = convergence_data["sbg_inekf_mus"]
            oak_inekf_mus = convergence_data["oak_inekf_mus"]
            sbg_mekf_mus = convergence_data["sbg_mekf_mus"]
            oak_mekf_mus = convergence_data["oak_mekf_mus"]
            gt_poses = convergence_data["gt_poses"]
            sbg_timestamps = convergence_data["sbg_timestamps"]
            oak_timestamps = convergence_data["oak_timestamps"]
            gt_timestamps = convergence_data["gt_timestamps"]

        plot_figure_7_rw_convergence(sbg_inekf_mus, oak_inekf_mus, sbg_mekf_mus, oak_mekf_mus, gt_poses, sbg_timestamps, oak_timestamps, gt_timestamps, show=False)
    except Exception as e:
        print(f"Error loading real-world convergence data for figure 7: {e}")
        print("Make sure the file 'figure_data/real_world_convergence_data.pkl' exists and is in the correct format.")

    try:
        with open("figure_data/timing_data.pkl", "rb") as f:
            timing_data = pkl.load(f)

        plot_figure_8_timing(**timing_data, show=False)
    except Exception as e:
        print(f"Error loading timing data for figure 8: {e}")
        print("Make sure the file 'timing_data.pkl' exists and is in the correct format.")

    plt.show()