from copyreg import pickle
import math
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation
from sklearn.metrics import max_error
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


    # Combine all state DataFrames into one
    # combined_error_df = pd.concat(error_df_list, ignore_index=True)

    # Same code block as before except trying to make it fit in latex sizing
    # --- Define Custom Colors (Blue, Red, Green) ---

    # --- Create 5 Subplots (1 row, 5 columns) ---
    # fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(textwidth_inches, 2))  # Increased width for better spacing
    fig = plt.figure(figsize=(textwidth_inches, 2))
    gs = gridspec.GridSpec(1, 6, figure=fig, width_ratios=[1, 1.01, -0.1, 1.01, 1, 1]) # adjust width ratios

    # Create the subplots using the GridSpec
    axes = [fig.add_subplot(gs[0, i]) for i in [0, 1, 3, 4, 5]] # Skip column 2

    # Share y-axis for the first two subplots
    axes[1].sharey(axes[0])

    # Share y-axis for the last three subplots
    axes[3].sharey(axes[2])
    axes[4].sharey(axes[2]) # Or axes[4].sharey(axes[3])
    # Remove redundant labels
    # Turn off ticks on the shared y-axes
    # plt.setp(axes[1].get_yticklabels(), visible=False)
    # plt.setp(axes[3].get_yticklabels(), visible=False)
    # plt.setp(axes[4].get_yticklabels(), visible=False)


    # Adjust subplot spacing if needed (for left/right margins)
    # fig.subplots_adjust(left=0.07, right=0.98)
    # custom_colors = ["blue", "red", "green"]

    # Adjust subplot spacing
    # fig.subplots_adjust(left=0.068, right=0.98, wspace=0.3)  # Adjust left margin & spacing between plots
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
        # ax.set_ylabel('degrees' if state_name in ["Roll", "Pitch", "Yaw"] else 'meters', fontsize=10)
        ax.set_xticklabels([])
        ax.set_xlabel('')

        if i == 0:
            ax.legend(fontsize=8, loc='upper center',)
        # Get the colors used in seaborn
        # colors = sns.color_palette()

    fig.text(0.01, 0.5, 'Trajectory Mean Abs. Error', va='center', rotation='vertical', fontsize=8, fontweight='bold')
    plt.savefig("figures/simulation_MAE.pdf", bbox_inches='tight')
    if show_plot:
        plt.show()

def plot_figure_4(inekf_5hz_conv_trajectories, inekf_no_horizon_trajectories, mekf_5hz_conv_trajectories, mekf_no_horizon_trajectories, gt_data, show_plot=True):
    # gt_data = [timestamp, roll, pitch, yaw, init_xyz[0], init_xyz[1], init_xyz[2], vel[0], vel[1], vel[2]]
    
    # inekf_5hz_conv_trajectories = data["inekf_5hz_conv_trajectories"]
    # inekf_no_horizon_trajectories = data["inekf_no_horizon_trajectories"]
    # mekf_5hz_conv_trajectories = data["mekf_5hz_conv_trajectories"]
    # mekf_no_horizon_trajectories = data["mekf_no_horizon_trajectories"]

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
    # state_names = ["Roll", "Pitch", "Yaw", "X", "Y", "Z"]
    # state_names = ["Roll (deg)", "Pitch (deg)", "Yaw (deg)", "X (m)", "Y (m)", "Z (m)"]
    state_names = ["Roll (°)", "Pitch (°)", "Yaw (°)", "X (m)", "Y (m)", "Z (m)"]

    
    # state_names = ["Roll", "Pitch", "Yaw", "Vel X", "Vel Y", "Vel Z", "X", "Y", "Z"]

    num_filters = len(filter_names)
    num_states = len(state_names)

    timesteps_to_plot = inekf_5hz_conv_trajectories.shape[1]  # Adjust as needed

    plot_data = {"Cam": {"INEKF": inekf_5hz_conv_trajectories[:,:,1:], "MEKF": mekf_5hz_conv_trajectories[:,:,1:]},
                "Yaw": {"INEKF": inekf_no_horizon_trajectories[:,:,1:], "MEKF": mekf_no_horizon_trajectories[:,:,1:]}}

    # ymin = [-180, -90, -180, 12, 640, -25]
    # ymax = [180,   90,  180, 44, 698,  25]
    ymin = [-180, -90, -180, -15, -25, -40]
    ymax = [180,   90,  180, 35, 25,  20]


    ytick_fontsize=6

    # fig, axes = plt.subplots(nrows=4, ncols=num_states, figsize=(24, 16)) #textwidth_inches
    fig, axes = plt.subplots(nrows=4, ncols=num_states, figsize=(textwidth_inches, int(textwidth_inches/1.5))) #, hspace=0.2, wspace=0.3) #textwidth_inches

    # fig.suptitle("Comparison of Cam and Yaw Data", fontsize=8, fontweight='bold')

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
                # gt_state_data = np.rad2deg(gt_state_data) # recent change
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
            # if row_idx < 1:
            ax_cam.set_xticklabels([])
            # xticks_positions = [0, 5 * 1000, 10 * 1000]  # 0, 500, 1000 timesteps
            xticks_positions = [0, 1000, 2000]  # 0, 500, 1000 timesteps

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
                # gt_state_data = np.rad2deg(gt_state_data) # recent change
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
                # To ensure integer labels (if appropriate for your data range):
                ax_yaw.set_yticklabels([f'{int(y)}' for y in yticks_positions])
            
            ax_yaw.set_yticklabels(ax_yaw.get_yticks(), fontsize=ytick_fontsize)

            ax_yaw.grid(True)
            # Remove x-axis labels for all rows except the last one (index 3)
            xticks_positions = [0, 1000, 2000]  # 0, 500, 1000 timesteps
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
    desired_gap = 0 # Try values like 0.05, 0.1, 0.15, etc.
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
        heading_cov=heading_measurement.heading_cov
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

    # Add noise and bias to the original measurements
    # noisy_gyro = imu_measurement.gyro + gyro_noise + current_random_walk[0:3]
    # noisy_accel = imu_measurement.accel + accel_noise + current_random_walk[3:6]

    # Debugging stuff right now
    noisy_gyro = imu_measurement.gyro + np.random.normal(0, std_devs[0:3]) + current_random_walk[0:3]  # TODO: Turn on once simulation is fixed
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
# def compare_results(filter, gt_data):
    
    # mus = np.array(filter.mus)
    # sigmas = np.array(filter.sigmas)
    # timestamps = np.array(filter.timestamps)

    # gt_data = gt_data[10:] # Skipping ahead 1 second. FIXME: Remove this later
    # gt_mus = gt_data[:, 1:10]  
    # gt_timestamps = gt_data[:, 0]

    # Find closest timestamp to each gt_timestamp
    # closest_indices = []
    # for gt_time in gt_poses[:, 0]:
    #     closest_index = np.argmin(np.abs(filter_t_rpyxyz[:, 0] - gt_time))
    #     closest_indices.append(closest_index)
    # filter_t_rpyxyz = filter_t_rpyxyz[closest_indices]

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
    # gt_rpy = Rotation.from_euler('xyz', gt_mus[:, 0:3]).as_matrix()

    n = mus_rpy.shape[0]
    # rotation_errors = np.zeros(n)
    # for i in range(n):
        # rotation_errors[i] = rotation_error_lie(mus_rpy[i], gt_rpy[i])
    def wrap_angle(angle):
        # return (angle + np.pi) % (2 * np.pi) - np.pi
        return (angle + 180) % 360 - 180    
    
    error_r = wrap_angle(mus_rpy[:, 0] - gt_poses[:, 1])
    error_p = wrap_angle(mus_rpy[:, 1] - gt_poses[:, 2])
    error_y = wrap_angle(mus_rpy[:, 2] - gt_poses[:, 3])

        # rotation_errors[i] = [error_r, error_p, error_y]
        # abs value
    error_rpy = np.abs(np.vstack((error_r, error_p, error_y)).T)
    mean_error_xy = np.mean(error_xy)
    mean_error_z = np.mean(error_z)
    mean_error_rpy = np.mean(error_rpy, axis=0)

    return mean_error_xy, mean_error_z, mean_error_rpy

def plot_imu_accel_gyro(imu_vals, gyro_vals):
    fig, axs = plt.subplots(4, 1, figsize=(10, 12))
    # Plot timesteps to show gaps in measurements
    imu_times = [val[0] for val in imu_vals]
    axs[0].plot(imu_times, np.ones_like(imu_times), '|', label='IMU Timesteps')
    axs[0].set_title('IMU Timesteps (Gaps in Measurements)')
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Presence')
    axs[0].legend()
    axs[0].grid()

    # Plot IMU X acceleration
    axs[1].plot(imu_times, [val[1] for val in imu_vals], label='X Acceleration')
    axs[1].set_title('IMU X Acceleration')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Acceleration (m/s²)')
    axs[1].legend()
    axs[1].grid()

    # Plot IMU Y acceleration
    axs[2].plot(imu_times, [val[2] for val in imu_vals], label='Y Acceleration')
    axs[2].set_title('IMU Y Acceleration')
    axs[2].set_xlabel('Time (s)')
    axs[2].set_ylabel('Acceleration (m/s²)')
    axs[2].legend()
    axs[2].grid()

    # Plot IMU Z acceleration
    axs[3].plot(imu_times, [val[3] for val in imu_vals], label='Z Acceleration')
    axs[3].set_title('IMU Z Acceleration')
    axs[3].set_xlabel('Time (s)')
    axs[3].set_ylabel('Acceleration (m/s²)')
    axs[3].legend()
    axs[3].grid()
    plt.tight_layout()

    fig, axs = plt.subplots(3, 1, figsize=(10, 12))
    gyro_times = [val[0] for val in gyro_vals]

    # Plot Gyro X angular velocity
    axs[0].plot(gyro_times, [val[1] for val in gyro_vals], label='X Angular Velocity')
    axs[0].set_title('Gyro X Angular Velocity')
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Angular Velocity (rad/s)')
    axs[0].legend()
    axs[0].grid()

    # Plot Gyro Y angular velocity
    axs[1].plot(gyro_times, [val[2] for val in gyro_vals], label='Y Angular Velocity')
    axs[1].set_title('Gyro Y Angular Velocity')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Angular Velocity (rad/s)')
    axs[1].legend()
    axs[1].grid()

    # Plot Gyro Z angular velocity
    axs[2].plot(gyro_times, [val[3] for val in gyro_vals], label='Z Angular Velocity')
    axs[2].set_title('Gyro Z Angular Velocity')
    axs[2].set_xlabel('Time (s)')
    axs[2].set_ylabel('Angular Velocity (rad/s)')
    axs[2].legend()
    axs[2].grid()
    plt.tight_layout()

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
        msg[6] += 0.27  # Adjust z position for incorrect IMU offset (I say incorrect because the SBG tries to put it in the frame of the imu, not exactly sure)
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
        # vel = sbg_fused[i,7:10]
        mu = np.eye(5)
        mu[0:3,0:3] = R
        # mu[0:3,3] = vel
        mu[0:3,4] = pos
        mus.append(mu)
    sbg_ekf = InEKF()
    sbg_ekf.timestamps = timestamps
    sbg_ekf.mus = mus

    return sbg_ekf


def plot_data(title="Add Title", data=[], data_labels=[]):
    """
    Plot the data with the given title and labels, aligning them on a common x-axis
    representing the same time period.

    Args:
        title (str): Title of the plot.
        data (list): List of 1D numpy arrays or lists of data to plot.
        data_labels (list): List of labels for the data.
    """
    if not data:
        print("No data provided for plotting.")
        return
    
    max_len = 0
    for d in data:
        max_len = max(max_len, len(d))

    plt.figure()
    for i, d in enumerate(data):
        n_points = len(d)
        if n_points > 0:
            # Create a normalized time axis from 0 to max_len
            normalized_time = np.linspace(0, max_len, n_points)
            plt.plot(normalized_time, d, label=data_labels[i])
        else:
            print(f"Data for {data_labels[i]} is empty.")

    plt.title(title)
    plt.xlabel('Time (Normalized)')
    plt.ylabel('Value')
    plt.legend()
    plt.grid()

def plot_results_comparison_gt_cov(filter1, filter2, filter3, title="", show=True, gps_timestamps=None, gps_altitude=None, body_frame_vel=False, imu_rpy=None, heading_measurements=None, rp_meas=None, filter1_name="1", filter2_name="2", filter3_name="3"):
    # if gps_timestamps is not None:
        # gps_timestamps -= gps_timestamps[0]

    # if rp_meas is not None:
        # rp_meas[:,0] -= rp_meas[0,0]

    # if heading_measurements is not None:
        # heading_measurements[:,0] -= heading_measurements[0,0]

    mus1 = np.array(filter1.mus)
    sigmas1 = np.array(filter1.sigmas)
    timestamps1 = np.array(filter1.timestamps)
    first_timestamp = timestamps1[0]
    timestamps1 = timestamps1 - first_timestamp  # Normalize time to start at 0
    covariances1 = np.array(filter1.sigmas)[:, :9, :9] 

    # Attitude Sigmas (Roll, Pitch, Yaw) - Indices 0, 1, 2
    att_vars1 = covariances1[:, 0:3, 0:3].diagonal(axis1=1, axis2=2)
    att_sigma1 = np.sqrt(att_vars1)
    
    # Velocity Sigmas (VX, VY, VZ) - Indices 3, 4, 5
    vel_vars1 = covariances1[:, 3:6, 3:6].diagonal(axis1=1, axis2=2)
    vel_sigma1 = np.sqrt(vel_vars1)
    
    # Position Sigmas (X, Y, Z) - Indices 6, 7, 8
    pos_vars1 = covariances1[:, 6:9, 6:9].diagonal(axis1=1, axis2=2)
    pos_sigma1 = np.sqrt(pos_vars1)


    mus2 = np.array(filter2.mus)
    sigmas2 = np.array(filter2.sigmas)
    timestamps2 = np.array(filter2.timestamps)
    timestamps2 = timestamps2 - first_timestamp  # Normalize time to start at 0
    covariances2 = np.array(filter2.sigmas)[:, :9, :9] 

    # Attitude Sigmas (Roll, Pitch, Yaw) - Indices 0, 1, 2
    att_vars2 = covariances2[:, 0:3, 0:3].diagonal(axis1=1, axis2=2)
    att_sigma2 = np.sqrt(att_vars2)
    
    # Velocity Sigmas (VX, VY, VZ) - Indices 3, 4, 5
    vel_vars2 = covariances2[:, 3:6, 3:6].diagonal(axis1=1, axis2=2)
    vel_sigma2 = np.sqrt(vel_vars2)
    
    # Position Sigmas (X, Y, Z) - Indices 6, 7, 8
    pos_vars2 = covariances2[:, 6:9, 6:9].diagonal(axis1=1, axis2=2)
    pos_sigma2 = np.sqrt(pos_vars2)


    mus3 = np.array(filter3.mus)
    sigmas3 = np.array(filter3.sigmas)
    timestamps3 = np.array(filter3.timestamps)
    timestamps3 = timestamps3 - first_timestamp  # Normalize time to start at 0
    
    if gps_timestamps is not None:
        gps_timestamps = gps_timestamps - first_timestamp

    if rp_meas is not None:
        rp_meas[:,0] -= first_timestamp

    if heading_measurements is not None:
        heading_measurements[:,0] -= first_timestamp

    # Plot the results
    fig, axs = plt.subplots(3, 3)
    fig.suptitle(title)

    error_fill_kwargs1 = {'alpha': 0.2, 'color': 'red', 'label': '$\pm 1\sigma$ Bound 1'}
    error_fill_kwargs2 = {'alpha': 0.2, 'color': 'blue', 'label': '$\pm 1\sigma$ Bound 2'}

    # plot xyz position
    axs[0, 0].plot(timestamps1, mus1[:, 0, 4], marker='x', linestyle='-', markersize=3, color='r', label=filter1_name)
    axs[0, 0].plot(timestamps2, mus2[:, 0, 4], marker='x', linestyle='-', markersize=3, color='b', label=filter2_name)
    axs[0, 0].plot(timestamps3, mus3[:, 0, 4], marker='x', linestyle='-', markersize=3, color='g', label=filter3_name)
    axs[0, 0].fill_between(timestamps1, mus1[:, 0, 4] - pos_sigma1[:, 0], mus1[:, 0, 4] + pos_sigma1[:, 0], **error_fill_kwargs1)
    axs[0, 0].fill_between(timestamps2, mus2[:, 0, 4] - pos_sigma2[:, 0], mus2[:, 0, 4] + pos_sigma2[:, 0], **error_fill_kwargs2)

    axs[0, 0].set_title('x')

    axs[1, 0].plot(timestamps1, mus1[:, 1, 4], marker='x', linestyle='-', markersize=3, color='r', label=filter1_name)
    axs[1, 0].plot(timestamps2, mus2[:, 1, 4], marker='x', linestyle='-', markersize=3, color='b', label=filter2_name)
    axs[1, 0].plot(timestamps3, mus3[:, 1, 4], marker='x', linestyle='-', markersize=3, color='g', label=filter3_name)
    axs[1, 0].fill_between(timestamps1, mus1[:, 1, 4] - pos_sigma1[:, 1], mus1[:, 1, 4] + pos_sigma1[:, 1], **error_fill_kwargs1)
    axs[1, 0].fill_between(timestamps2, mus2[:, 1, 4] - pos_sigma2[:, 1], mus2[:, 1, 4] + pos_sigma2[:, 1], **error_fill_kwargs2)
    axs[1, 0].set_title('y')

    axs[2, 0].plot(timestamps1, mus1[:, 2, 4], marker='x', linestyle='-', markersize=3, color='r', label=filter1_name)
    axs[2, 0].plot(timestamps2, mus2[:, 2, 4], marker='x', linestyle='-', markersize=3, color='b', label=filter2_name)
    axs[2, 0].plot(timestamps3, mus3[:, 2, 4], marker='x', linestyle='-', markersize=3, color='g', label=filter3_name)
    
    axs[2, 0].fill_between(timestamps1, mus1[:, 2, 4] - pos_sigma1[:, 2], mus1[:, 2, 4] + pos_sigma1[:, 2], **error_fill_kwargs1)
    axs[2, 0].fill_between(timestamps2, mus2[:, 2, 4] - pos_sigma2[:, 2], mus2[:, 2, 4] + pos_sigma2[:, 2], **error_fill_kwargs2)
    axs[2, 0].set_title('z')

    if gps_altitude is not None:
        # axs[2, 0].plot(gps_timestamps, gps_altitude, marker='x', markersize=3, color='g', label='GPS Measurement')
        gps_altitudes = np.array(gps_altitude)
        axs[0, 0].scatter(gps_timestamps, gps_altitude[:,0], color='g', label='GPS Measurement')
        axs[1, 0].scatter(gps_timestamps, gps_altitude[:,1], color='g', label='GPS Measurement')
        axs[2, 0].scatter(gps_timestamps, gps_altitude[:,2], color='g', label='GPS Measurement')

    R = Rotation.from_matrix(mus1[:, 0:3, 0:3])
    rpy = R.as_euler('xyz')
    rpy2 = Rotation.from_matrix(mus2[:, 0:3, 0:3]).as_euler('xyz')
    rpy3 = Rotation.from_matrix(mus3[:, 0:3, 0:3]).as_euler('xyz')

    axs[0, 1].plot(timestamps1, np.rad2deg(rpy[:, 0]), marker='x', linestyle='-', markersize=3, color='r')
    if imu_rpy is not None:
        axs[0, 1].plot(imu_rpy[:, 0], (imu_rpy[:, 1]), marker='x', linestyle='-', markersize=3, color='b', label='IMU RPY')
    axs[0, 1].plot(timestamps2, np.rad2deg(rpy2[:, 0]), marker='x', linestyle='-', markersize=3, color='b')
    axs[0, 1].plot(timestamps3, np.rad2deg(rpy3[:, 0]), marker='x', linestyle='-', markersize=3, color='g', label=filter3_name)
    if rp_meas is not None:
        axs[0, 1].scatter(rp_meas[:,0], rp_meas[:,1], color='g', label='Image Roll Measurement')

    axs[0, 1].fill_between(timestamps1, np.rad2deg(rpy[:, 0]) - np.rad2deg(att_sigma1[:, 0]), np.rad2deg(rpy[:, 0]) + np.rad2deg(att_sigma1[:, 0]), **error_fill_kwargs1)
    axs[0, 1].fill_between(timestamps2, np.rad2deg(rpy2[:, 0]) - np.rad2deg(att_sigma2[:, 0]), np.rad2deg(rpy2[:, 0]) + np.rad2deg(att_sigma2[:, 0]), **error_fill_kwargs2)
    
    axs[0, 1].set_title('roll')
    axs[1, 1].plot(timestamps1, np.rad2deg(rpy[:, 1]), marker='x', linestyle='-', markersize=3, color='r')
    axs[1, 1].plot(timestamps2, np.rad2deg(rpy2[:, 1]), marker='x', linestyle='-', markersize=3, color='b')
    axs[1, 1].plot(timestamps3, np.rad2deg(rpy3[:, 1]), marker='x', linestyle='-', markersize=3, color='g', label=filter3_name)
    if imu_rpy is not None:
        axs[1, 1].plot(imu_rpy[:, 0], (imu_rpy[:, 2]), marker='x', linestyle='-', markersize=3, color='b', label='IMU RPY')
    if rp_meas is not None:
        axs[1, 1].scatter(rp_meas[:,0], rp_meas[:,2], color='g', label='Image Pitch Measurement')

    axs[1, 1].fill_between(timestamps1, np.rad2deg(rpy[:, 1]) - np.rad2deg(att_sigma1[:, 1]), np.rad2deg(rpy[:, 1]) + np.rad2deg(att_sigma1[:, 1]), **error_fill_kwargs1)
    axs[1, 1].fill_between(timestamps2, np.rad2deg(rpy2[:, 1]) - np.rad2deg(att_sigma2[:, 1]), np.rad2deg(rpy2[:, 1]) + np.rad2deg(att_sigma2[:, 1]), **error_fill_kwargs2)
    
    axs[1, 1].set_title('pitch')
    axs[2, 1].plot(timestamps1, np.rad2deg(rpy[:, 2]), marker='x', linestyle='-', markersize=3, color='r')
    axs[2, 1].plot(timestamps2, np.rad2deg(rpy2[:, 2]), marker='x', linestyle='-', markersize=3, color='b')
    axs[2, 1].plot(timestamps3, np.rad2deg(rpy3[:, 2]), marker='x', linestyle='-', markersize=3, color='g', label=filter3_name)
    if imu_rpy is not None:
        axs[2, 1].plot(imu_rpy[:, 0], (imu_rpy[:, 3]), marker='x', linestyle='-', markersize=3, color='b', label='IMU RPY')
        axs[2, 1].legend()
    if heading_measurements is not None:        
        axs[2, 1].scatter(heading_measurements[:,0], heading_measurements[:,1], color='g', label='Heading Measurement')
    
    axs[2, 1].fill_between(timestamps1, np.rad2deg(rpy[:, 2]) - np.rad2deg(att_sigma1[:, 2]), np.rad2deg(rpy[:, 2]) + np.rad2deg(att_sigma1[:, 2]), **error_fill_kwargs1)
    axs[2, 1].fill_between(timestamps2, np.rad2deg(rpy2[:, 2]) - np.rad2deg(att_sigma2[:, 2]), np.rad2deg(rpy2[:, 2]) + np.rad2deg(att_sigma2[:, 2]), **error_fill_kwargs2)
    axs[2, 1].set_title('yaw')

    # Plot velocities
    velocities = mus1[:, 0:3, 3]
    velocities2 = mus2[:, 0:3, 3]
    velocities3 = mus3[:, 0:3, 3]
    if body_frame_vel:
        rotation_matrices = mus1[:, 0:3, 0:3]
        # Transform world-frame velocities into the body frame: v_body = Rᵀ * v_world
        velocities = np.einsum('ijk,ik->ij', rotation_matrices.transpose(0, 2, 1), velocities)

    axs[0, 2].plot(timestamps1, velocities[:, 0], marker='x', linestyle='-', markersize=3, color='r')
    axs[0, 2].plot(timestamps2, velocities2[:, 0], marker='x', linestyle='-', markersize=3, color='b')
    axs[0, 2].plot(timestamps3, velocities3[:, 0], marker='x', linestyle='-', markersize=3, color='g', label=filter3_name)
    axs[0, 2].fill_between(timestamps1, velocities[:, 0] - vel_sigma1[:, 0], velocities[:, 0] + vel_sigma1[:, 0], **error_fill_kwargs1)
    axs[0, 2].fill_between(timestamps2, velocities2[:, 0] - vel_sigma2[:, 0], velocities2[:, 0] + vel_sigma2[:, 0], **error_fill_kwargs2)

    axs[0, 2].set_title('vx')
    axs[1, 2].plot(timestamps1, velocities[:, 1], marker='x', linestyle='-', markersize=3, color='r')
    axs[1, 2].plot(timestamps2, velocities2[:, 1], marker='x', linestyle='-', markersize=3, color='b')
    axs[1, 2].plot(timestamps3, velocities3[:, 1], marker='x', linestyle='-', markersize=3, color='g', label=filter3_name)
    
    axs[1, 2].fill_between(timestamps1, velocities[:, 1] - vel_sigma1[:, 1], velocities[:, 1] + vel_sigma1[:, 1], **error_fill_kwargs1)
    axs[1, 2].fill_between(timestamps2, velocities2[:, 1] - vel_sigma2[:, 1], velocities2[:, 1] + vel_sigma2[:, 1], **error_fill_kwargs2)
    axs[1, 2].set_title('vy')

    axs[2, 2].plot(timestamps1, velocities[:, 2], marker='x', linestyle='-', markersize=3, color='r')
    axs[2, 2].plot(timestamps2, velocities2[:, 2], marker='x', linestyle='-', markersize=3, color='b')
    axs[2, 2].plot(timestamps3, velocities3[:, 2], marker='x', linestyle='-', markersize=3, color='g', label=filter3_name)
    axs[2, 2].fill_between(timestamps1, velocities[:, 2] - vel_sigma1[:, 2], velocities[:, 2] + vel_sigma1[:, 2], **error_fill_kwargs1)
    axs[2, 2].fill_between(timestamps2, velocities2[:, 2] - vel_sigma2[:, 2], velocities2[:, 2] + vel_sigma2[:, 2], **error_fill_kwargs2)
    axs[2, 2].set_title('vz')

    for ax in axs.flat:
        ax.grid(True)
    axs[0, 0].legend()    

    # Handling pass by reference instead of value, need to add back the first timestamp
    # timestamps = timestamps + first_timestamp  # Normalize time to start at 0
    # gt_data[:, 0] = gt_data[:, 0] - timestamps[0] #gt_data[0, 0]  # Normalize GT time to start at 0
    if gps_timestamps is not None:
        gps_timestamps += first_timestamp
    if rp_meas is not None:
        rp_meas[:,0] += first_timestamp
    if heading_measurements is not None:
        heading_measurements[:,0] += first_timestamp
    if show:
        plt.show()

def plot_results_cov(filter, gt_data=None, title="", show=True, gps_timestamps=None, 
                 gps_altitude=None, body_frame_vel=False, imu_rpy=None, 
                 heading_measurements=None, rp_meas=None):
    """
    Plots the filter state estimates (mu) against ground truth (gt_data) 
    and includes 1-sigma error bounds from the covariance matrices.

    Args:
        filter (object): Object containing mus, sigmas, and timestamps.
        gt_data (np.ndarray): Ground truth data (time, roll, pitch, yaw, x, y, z, vx, vy, vz).
        title (str): Title for the figure.
        show (bool): Whether to call plt.show().
        covariances (np.ndarray, optional): Array of N x 9 x 9 covariance matrices (P).
    """
    covariances = np.array(filter.sigmas)[:, :9, :9] 
    # covariances = None
    # --- 1. Data Extraction and Time Normalization ---
    mus = np.array(filter.mus)
    # sigmas = np.array(filter.sigmas) # Not used, replaced by covariances
    timestamps = np.array(filter.timestamps)
    first_timestamp = timestamps[0]
    timestamps = timestamps - first_timestamp  # Normalize time to start at 0
    # gt_data[:, 0] = gt_data[:, 0] - timestamps[0] #gt_data[0, 0]  # Normalize GT time to start at 0
    gt_data[:, 0] -= first_timestamp
    if gps_timestamps is not None:
        gps_timestamps -= first_timestamp
    if rp_meas is not None:
        rp_meas[:,0] -= first_timestamp
    if heading_measurements is not None:
        heading_measurements[:,0] -= first_timestamp

    # --- 2. Sigma Extraction for Error Bounds (New Logic) ---
    if covariances is not None:
        covariances = np.array(covariances)
        # Extract the diagonal variance elements and take the square root to get sigma (std dev)
        
        # Attitude Sigmas (Roll, Pitch, Yaw) - Indices 0, 1, 2
        att_vars = covariances[:, 0:3, 0:3].diagonal(axis1=1, axis2=2)
        att_sigma = np.sqrt(att_vars)
        
        # Velocity Sigmas (VX, VY, VZ) - Indices 3, 4, 5
        vel_vars = covariances[:, 3:6, 3:6].diagonal(axis1=1, axis2=2)
        vel_sigma = np.sqrt(vel_vars)
        
        # Position Sigmas (X, Y, Z) - Indices 6, 7, 8
        pos_vars = covariances[:, 6:9, 6:9].diagonal(axis1=1, axis2=2)
        pos_sigma = np.sqrt(pos_vars)
        
    # --- 3. Calculate Attitude and Velocity Estimates ---
    R = Rotation.from_matrix(mus[:, 0:3, 0:3])
    rpy = R.as_euler('xyz')
    deg_rpy = np.rad2deg(rpy)
    
    # Velocity calculation
    velocities = mus[:, 0:3, 3]
    if body_frame_vel:
        rotation_matrices = mus[:, 0:3, 0:3]
        # Transform world-frame velocities into the body frame: v_body = Rᵀ * v_world
        velocities = np.einsum('ijk,ik->ij', rotation_matrices.transpose(0, 2, 1), velocities)

    # --- 4. Plotting Setup ---
    fig, axs = plt.subplots(3, 3, figsize=(18, 12))
    fig.suptitle(title, fontsize=16)

    # Define common plot properties
    error_fill_kwargs = {'alpha': 0.2, 'color': 'gray', 'label': '$\pm 1\sigma$ Bound'}

    # State extraction for plotting ease
    x_est, y_est, z_est = mus[:, 0, 4], mus[:, 1, 4], mus[:, 2, 4]
    
    # --- 5. Plot XYZ Position (Column 0) ---
    # time,              roll,          pitch,             yaw,      x,      y,     z,      vx,     vy,     vz,       roll_std,      pitch_std,      yaw_std,     x_std,    y_std,   z_std,    vx_std,   vy_std,   vz_std
    # nav_time, closest_att[1], closest_att[2], closest_att[3], nav[1], nav[2], nav[3], nav[4], nav[5], nav[6], closest_att[4], closest_att[5], closest_att[6], nav[7], nav[8], nav[9], nav[10], nav[11], nav[12]))

    # Extend timestamps array if needed to match gt_data length
    if len(timestamps) < len(x_est):
        timestamps = np.append(timestamps, timestamps[-1] + 0.005)

    # X Position
    axs[0, 0].plot(timestamps, x_est, marker='x', linestyle='-', markersize=3, color='r', label='Estimate')
    axs[0, 0].plot(gt_data[:, 0], gt_data[:, 4], marker='x', linestyle='-', markersize=3, color='b', label='SBG Estimate')
    axs[0, 0].fill_between(gt_data[:, 0], gt_data[:,4] - gt_data[:,13], gt_data[:,4] + gt_data[:,13], alpha=0.2, color='gray', label='SBG $\pm 1\sigma$ Bound')
    if covariances is not None:
        axs[0, 0].fill_between(timestamps, x_est - pos_sigma[:, 0], x_est + pos_sigma[:, 0], **error_fill_kwargs)
    axs[0, 0].set_title('x')
    
    # Y Position
    axs[1, 0].plot(timestamps, y_est, marker='x', linestyle='-', markersize=3, color='r')
    axs[1, 0].plot(gt_data[:, 0], gt_data[:, 5], marker='x', linestyle='-', markersize=3, color='b')
    axs[1, 0].fill_between(gt_data[:, 0], gt_data[:,5] - gt_data[:,14], gt_data[:,5] + gt_data[:,14], alpha=0.2, color='gray', label='SBG $\pm 1\sigma$ Bound')
    if covariances is not None:
        axs[1, 0].fill_between(timestamps, y_est - pos_sigma[:, 1], y_est + pos_sigma[:, 1], **error_fill_kwargs)
    axs[1, 0].set_title('y')
    
    # Z Position
    axs[2, 0].plot(timestamps, z_est, marker='x', linestyle='-', markersize=3, color='r')
    axs[2, 0].plot(gt_data[:, 0], gt_data[:, 6], marker='x', linestyle='-', markersize=3, color='b')
    axs[2, 0].fill_between(gt_data[:, 0], gt_data[:,6] - gt_data[:,15], gt_data[:,6] + gt_data[:,15], alpha=0.2, color='gray', label='SBG $\pm 1\sigma$ Bound')

    if covariances is not None:
        axs[2, 0].fill_between(timestamps, z_est - pos_sigma[:, 2], z_est + pos_sigma[:, 2], **error_fill_kwargs)
    axs[2, 0].set_title('z')

    # GPS Altitude (Assumes gps_altitude is [x, y, z] measurements)
    if gps_altitude is not None:
        axs[0, 0].scatter(gps_timestamps, gps_altitude[:,0], color='g', s=15, label='GPS X Measurement')
        axs[1, 0].scatter(gps_timestamps, gps_altitude[:,1], color='g', s=15, label='GPS Y Measurement')
        axs[2, 0].scatter(gps_timestamps, gps_altitude[:,2], color='g', s=15, label='GPS Z Measurement')

    # --- 6. Plot RPY Attitude (Column 1) ---

    # Roll (rpy[:, 0])
    axs[0, 1].plot(timestamps, deg_rpy[:, 0], marker='x', linestyle='-', markersize=3, color='r')
    axs[0, 1].plot(gt_data[:, 0], gt_data[:, 1], marker='x', linestyle='-', markersize=3, color='b')
    axs[0, 1].fill_between(gt_data[:, 0], gt_data[:,1] - gt_data[:,10], gt_data[:,1] + gt_data[:,10], alpha=0.2, color='gray', label='SBG $\pm 1\sigma$ Bound')
    if covariances is not None:
        deg_att_sigma = np.rad2deg(att_sigma) # Convert error bound to degrees
        axs[0, 1].fill_between(timestamps, deg_rpy[:, 0] - deg_att_sigma[:, 0], deg_rpy[:, 0] + deg_att_sigma[:, 0], **error_fill_kwargs)

    if imu_rpy is not None:
        axs[0, 1].plot(imu_rpy[:, 0], (imu_rpy[:, 1]), marker='x', linestyle='-', markersize=3, color='b', label='IMU RPY')
    if rp_meas is not None:
        axs[0, 1].scatter(rp_meas[:,0], rp_meas[:,1], color='g', s=10, label='Image Roll Measurement')
    axs[0, 1].set_title('roll (deg)')
    
    # Pitch (rpy[:, 1])
    axs[1, 1].plot(timestamps, deg_rpy[:, 1], marker='x', linestyle='-', markersize=3, color='r')
    axs[1, 1].plot(gt_data[:, 0], gt_data[:, 2], marker='x', linestyle='-', markersize=3, color='b')
    axs[1, 1].fill_between(gt_data[:, 0], gt_data[:,2] - gt_data[:,11], gt_data[:,2] + gt_data[:,11], alpha=0.2, color='gray', label='SBG $\pm 1\sigma$ Bound')
    if covariances is not None:
        axs[1, 1].fill_between(timestamps, deg_rpy[:, 1] - deg_att_sigma[:, 1], deg_rpy[:, 1] + deg_att_sigma[:, 1], **error_fill_kwargs)

    if imu_rpy is not None:
        axs[1, 1].plot(imu_rpy[:, 0], (imu_rpy[:, 2]), marker='x', linestyle='-', markersize=3, color='b', label='IMU RPY')
    if rp_meas is not None:
        axs[1, 1].scatter(rp_meas[:,0], rp_meas[:,2], color='g', s=10, label='Image Pitch Measurement')
    axs[1, 1].set_title('pitch (deg)')
    
    # Yaw (rpy[:, 2])
    axs[2, 1].plot(timestamps, deg_rpy[:, 2], marker='x', linestyle='-', markersize=3, color='r')
    axs[2, 1].plot(gt_data[:, 0], gt_data[:, 3], marker='x', linestyle='-', markersize=3, color='b')
    axs[2, 1].fill_between(gt_data[:, 0], gt_data[:,3] - gt_data[:,12], gt_data[:,3] + gt_data[:,12], alpha=0.2, color='gray', label='SBG $\pm 1\sigma$ Bound')
    if covariances is not None:
        axs[2, 1].fill_between(timestamps, deg_rpy[:, 2] - deg_att_sigma[:, 2], deg_rpy[:, 2] + deg_att_sigma[:, 2], **error_fill_kwargs)

    if imu_rpy is not None:
        axs[2, 1].plot(imu_rpy[:, 0], (imu_rpy[:, 3]), marker='x', linestyle='-', markersize=3, color='b', label='IMU RPY')
        axs[2, 1].legend()
    if heading_measurements is not None:
        axs[2, 1].scatter(heading_measurements[:,0], heading_measurements[:,1], color='g', s=10, label='Heading Measurement')
    axs[2, 1].set_title('yaw (deg)')

    # --- 7. Plot Velocities (Column 2) ---

    # VX
    axs[0, 2].plot(timestamps, velocities[:, 0], marker='x', linestyle='-', markersize=3, color='r')
    axs[0, 2].plot(gt_data[:, 0], gt_data[:, 7], marker='x', linestyle='-', markersize=3, color='b')
    axs[0, 2].fill_between(gt_data[:, 0], gt_data[:,7] - gt_data[:,16], gt_data[:,7] + gt_data[:,16], alpha=0.2, color='gray', label='SBG $\pm 1\sigma$ Bound')
    if covariances is not None:
        axs[0, 2].fill_between(timestamps, velocities[:, 0] - vel_sigma[:, 0], velocities[:, 0] + vel_sigma[:, 0], **error_fill_kwargs)
    axs[0, 2].set_title('vx')
    
    # VY
    axs[1, 2].plot(timestamps, velocities[:, 1], marker='x', linestyle='-', markersize=3, color='r')
    axs[1, 2].plot(gt_data[:, 0], gt_data[:, 8], marker='x', linestyle='-', markersize=3, color='b')
    axs[1, 2].fill_between(gt_data[:, 0], gt_data[:,8] - gt_data[:,17], gt_data[:,8] + gt_data[:,17], alpha=0.2, color='gray', label='SBG $\pm 1\sigma$ Bound')
    if covariances is not None:
        axs[1, 2].fill_between(timestamps, velocities[:, 1] - vel_sigma[:, 1], velocities[:, 1] + vel_sigma[:, 1], **error_fill_kwargs)
    axs[1, 2].set_title('vy')
    
    # VZ
    axs[2, 2].plot(timestamps, velocities[:, 2], marker='x', linestyle='-', markersize=3, color='r')
    axs[2, 2].plot(gt_data[:, 0], gt_data[:, 9], marker='x', linestyle='-', markersize=3, color='b')
    axs[2, 2].fill_between(gt_data[:, 0], gt_data[:,9] - gt_data[:,18], gt_data[:,9] + gt_data[:,18], alpha=0.2, color='gray', label='SBG $\pm 1\sigma$ Bound')
    if covariances is not None:
        axs[2, 2].fill_between(timestamps, velocities[:, 2] - vel_sigma[:, 2], velocities[:, 2] + vel_sigma[:, 2], **error_fill_kwargs)
    axs[2, 2].set_title('vz')

    # --- 8. Final Touches ---
    for ax in axs.flat:
        ax.grid(True)
        ax.set_xlabel('Time (s)')
    
    # Combine legends from all subplots to prevent duplication of the error bound label
    handles, labels = axs[0, 0].get_legend_handles_labels()
    # Find and remove the duplicate error bound labels
    unique_labels = {}
    for h, l in zip(handles, labels):
        unique_labels[l] = h
    
    # Put the main legend on the first subplot
    axs[0, 0].legend(unique_labels.values(), unique_labels.keys(), loc='upper right')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for suptitle

    # Handling pass by reference instead of value, need to add back the first timestamp
    timestamps = timestamps + first_timestamp  # Normalize time to start at 0
    # gt_data[:, 0] = gt_data[:, 0] - timestamps[0] #gt_data[0, 0]  # Normalize GT time to start at 0
    gt_data[:, 0] += first_timestamp
    if gps_timestamps is not None:
        gps_timestamps += first_timestamp
    if rp_meas is not None:
        rp_meas[:,0] += first_timestamp
    if heading_measurements is not None:
        heading_measurements[:,0] += first_timestamp

    if show:
        plt.show()

def plot_results(filter, gt_data=None, title="", show=True, gps_timestamps=None, gps_altitude=None, body_frame_vel=False, imu_rpy=None, heading_measurements=None, rp_meas=None, horizon_dict=None):
    # if gps_timestamps is not None:
        # gps_timestamps -= gps_timestamps[0]
    timestamps = np.array(filter.timestamps)

    if rp_meas is not None:
        rp_meas[:,0] -= rp_meas[0,0]

    if heading_measurements is not None:
        heading_measurements[:,0] -= heading_measurements[0,0]

    if horizon_dict is not None:
        horizon_t_rp = []
        for timestamp in timestamps:
            if timestamp in horizon_dict:
                h_roll = horizon_dict[timestamp].roll_rad
                h_pitch = horizon_dict[timestamp].pitch_rad
                horizon_t_rp.append((timestamp, np.rad2deg(h_roll), np.rad2deg(h_pitch)))
        horizon_t_rp = np.array(horizon_t_rp)
        horizon_t_rp[:,0] -= timestamps[0]
        # h_timestamp = horizon_dict.keys()
        # h_values = horizon_dict.values()
        # h_timestamps = np.array(list(h_timestamp)) - list(h_timestamp)[0]
        # h_values = np.array(list(h_values.roll_rad, list(h_values.pitch_rad)))
        # h_values = np.rad2deg(h_values)

    mus = np.array(filter.mus)
    sigmas = np.array(filter.sigmas)
    timestamps = timestamps - timestamps[0]  # Normalize time to start at 0
    gt_data[:,0] = gt_data[:,0] - gt_data[0,0]  # Normalize GT time to start at 0

    # Plot the results
    fig, axs = plt.subplots(3, 3)
    fig.suptitle(title)

    # plot xyz position
    axs[0, 0].plot(timestamps, mus[:, 0, 4], marker='x', linestyle='-', markersize=3, color='r', label='Estimate')
    axs[0, 0].plot(gt_data[:, 0], gt_data[:, 4], marker='x', linestyle='-', markersize=3, color='b', label='SBG Estimate')
    axs[0, 0].set_title('x')
    
    axs[1, 0].plot(timestamps, mus[:, 1, 4], marker='x', linestyle='-', markersize=3, color='r')
    axs[1, 0].plot(gt_data[:, 0], gt_data[:, 5], marker='x', linestyle='-', markersize=3, color='b')
    axs[1, 0].set_title('y')
    
    axs[2, 0].plot(timestamps, mus[:, 2, 4], marker='x', linestyle='-', markersize=3, color='r')
    axs[2, 0].plot(gt_data[:, 0], gt_data[:, 6], marker='x', linestyle='-', markersize=3, color='b')
    axs[2, 0].set_title('z')

    if gps_altitude is not None:
        # axs[2, 0].plot(gps_timestamps, gps_altitude, marker='x', markersize=3, color='g', label='GPS Measurement')
        gps_altitudes = np.array(gps_altitude)
        axs[0, 0].scatter(gps_timestamps, gps_altitude[:,0], color='g', label='GPS Measurement')
        axs[1, 0].scatter(gps_timestamps, gps_altitude[:,1], color='g', label='GPS Measurement')
        axs[2, 0].scatter(gps_timestamps, gps_altitude[:,2], color='g', label='GPS Measurement')

    R = Rotation.from_matrix(mus[:, 0:3, 0:3])
    rpy = R.as_euler('xyz')
    axs[0, 1].plot(timestamps, np.rad2deg(rpy[:, 0]), marker='x', linestyle='-', markersize=3, color='r')
    if horizon_dict is not None:
        axs[0, 1].scatter(horizon_t_rp[:,0], (horizon_t_rp[:, 1]), color='g', label='Horizon RPY')
    axs[0, 1].plot(gt_data[:, 0], gt_data[:, 1], marker='x', linestyle='-', markersize=3, color='b')
    if rp_meas is not None:
        axs[0, 1].scatter(rp_meas[:,0], rp_meas[:,1], color='g', label='Image Roll Measurement')

    axs[0, 1].set_title('roll')
    axs[1, 1].plot(timestamps, np.rad2deg(rpy[:, 1]), marker='x', linestyle='-', markersize=3, color='r')
    axs[1, 1].plot(gt_data[:, 0], gt_data[:, 2], marker='x', linestyle='-', markersize=3, color='b')
    if horizon_dict is not None:
        axs[1, 1].scatter(horizon_t_rp[:,0], (horizon_t_rp[:, 2]), color='g', label='Horizon RPY')
    if rp_meas is not None:
        axs[1, 1].scatter(rp_meas[:,0], rp_meas[:,2], color='g', label='Image Pitch Measurement')

    axs[1, 1].set_title('pitch')
    axs[2, 1].plot(timestamps, np.rad2deg(rpy[:, 2]), marker='x', linestyle='-', markersize=3, color='r')
    axs[2, 1].plot(gt_data[:, 0], gt_data[:, 3], marker='x', linestyle='-', markersize=3, color='b')
    if imu_rpy is not None:
        axs[2, 1].plot(imu_rpy[:, 0], (imu_rpy[:, 3]), marker='x', linestyle='-', markersize=3, color='b', label='IMU RPY')
        axs[2, 1].legend()
    if heading_measurements is not None:
        axs[2, 1].scatter(heading_measurements[:,0], heading_measurements[:,1], color='g', label='Heading Measurement')
    axs[2, 1].set_title('yaw')

    # Plot velocities
    velocities = mus[:, 0:3, 3]
    if body_frame_vel:
        rotation_matrices = mus[:, 0:3, 0:3]
        # Transform world-frame velocities into the body frame: v_body = Rᵀ * v_world
        velocities = np.einsum('ijk,ik->ij', rotation_matrices.transpose(0, 2, 1), velocities)

    axs[0, 2].plot(timestamps, velocities[:, 0], marker='x', linestyle='-', markersize=3, color='r')
    # axs[0, 2].plot(gt_data[:, 0], gt_data[:, 7], marker='x', linestyle='-', markersize=3, color='b')
    axs[0, 2].set_title('vx')
    axs[1, 2].plot(timestamps, velocities[:, 1], marker='x', linestyle='-', markersize=3, color='r')
    # axs[1, 2].plot(gt_data[:, 0], gt_data[:, 8], marker='x', linestyle='-', markersize=3, color='b')
    axs[1, 2].set_title('vy')
    axs[2, 2].plot(timestamps, velocities[:, 2], marker='x', linestyle='-', markersize=3, color='r')
    # axs[2, 2].plot(gt_data[:, 0], gt_data[:, 9], marker='x', linestyle='-', markersize=3, color='b')
    axs[2, 2].set_title('vz')

    for ax in axs.flat:
        ax.grid(True)
    axs[0, 0].legend()    

    if show:
        plt.show()

#SBG InEKF
#SBG MEKF
#SBG InEKF Full Pose
#OAK InEKF
#OAK MEKF
#OAK InEKF Full Pose

# def plot_figure_5(inEKF_sbg_full_traj, MEKF_sbg_full_traj, inEKF_sbg_full_pose_full_traj, inEKF_oak_full_traj, MEKF_oak_full_traj, inEKF_oak_full_pose_full_traj, gt_poses, show_plot=True):
#     # Metrics are mean, std, and max of xy position error, z position error, roll error, pitch error, and yaw error
#     sbg_inekf_metrics = get_filter_error_metrics(inEKF_sbg_full_traj, gt_poses)
#     sbg_mekf_metrics = get_filter_error_metrics(MEKF_sbg_full_traj, gt_poses)
#     sbg_inekf_full_pose_metrics = get_filter_error_metrics(inEKF_sbg_full_pose_full_traj, gt_poses)
#     oak_inekf_metrics = get_filter_error_metrics(inEKF_oak_full_traj, gt_poses)
#     oak_mekf_metrics = get_filter_error_metrics(MEKF_oak_full_traj, gt_poses)
#     oak_inekf_full_pose_metrics = get_filter_error_metrics(inEKF_oak_full_pose_full_traj, gt_poses)

def plot_figure_5_bar_graph(df, show_plot=True):
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

    # Filter names for the legend
    filter_display_names = [
        "SBG InEKF", "SBG MEKF", "SBG InEKF FO", 
        "OAK InEKF", "OAK MEKF", "OAK InEKF FO"
    ]
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
            ax.set_ylim(0, 3.0)
            
            # 1. Define all positions where you want a "tick" line
            tick_positions = np.arange(0, 3.5, 0.5)
            
            # 2. Create labels: only show the number if it's an integer, otherwise ""
            # We use % 1 == 0 to check if the number is a whole integer
            tick_labels = [f"{int(x)}" if x % 1 == 0 else "" for x in tick_positions]
            
            ax.set_yticks(tick_positions)
            ax.set_yticklabels(tick_labels)
            if i == 2:
                ax.set_ylabel('degrees', fontsize=10)
        else:
            # Position scaling (adjust if your max position error is > 6m)
            ax.set_ylim(0, 5.5) 
            if i == 0:
                ax.set_ylabel('meters', fontsize=10)

        ax.tick_params(axis='y', labelsize=10, pad=1)

    # --- Legend Construction ---
    # Create colored patches for filters
    # --- Updated Legend Elements ---
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors[j], label=filter_display_names[j], edgecolor='black', linewidth=0.7) 
        for j in range(len(filter_display_names))
    ]

    # When you call the legend, you can control the patch shape/size:
    # ax.legend(
    #     handles=legend_elements, 
    #     loc='lower center', 
    #     ncol=3, 
    #     handlelength=1.5, # Controls the width of the patch
    #     handleheight=1.0, # Controls the height of the patch
    #     frameon=False
    # )
    # legend_elements = [Line2D([0], [0], color=colors[j], lw=5, label=filter_display_names[j]) 
                    #    for j in range(len(filter_display_names))]
    
    new_order = [0, 3, 1, 4, 2, 5]  # Desired order of filters in the legend
    legend_elements = [legend_elements[i] for i in new_order]
    
    # Add the Max Error 'x' to the legend
    legend_elements.append(Line2D([0], [0], marker='x', color='red', label='Max Error', 
                                  markersize=5, linestyle='None', markeredgewidth=1))

    # Place legend centered underneath the subplots
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.0),
               ncol=4, fontsize=10, frameon=False, handletextpad=0.5, columnspacing=1.0)

    # Adjust layout to make room for labels and legend
    plt.subplots_adjust(left=0.07, right=0.98, top=0.85, bottom=0.22, wspace=0.45)
    
    plt.savefig("figures/filter_comparison_bars.pdf", bbox_inches='tight')
    if show_plot:
        plt.show()

def get_filter_error_metrics(filter, gt_filter, timeoffset=0):
    timeoffset_idx1 = (np.abs(np.array(filter.timestamps) - (filter.timestamps[0] + timeoffset))).argmin()
    timeoffset_idx2 = (np.abs(np.array(gt_filter.timestamps) - (gt_filter.timestamps[0] + timeoffset))).argmin()

    stop_idx1 = len(filter.timestamps)
    stop_idx2 = len(gt_filter.timestamps)

    timestamps1 = np.array(filter.timestamps)[timeoffset_idx1:stop_idx1]
    timestamps2 = np.array(gt_filter.timestamps)[timeoffset_idx2:stop_idx2]

    mus1 = np.array(filter.mus)[timeoffset_idx1:stop_idx1]
    mus2 = np.array(gt_filter.mus)[timeoffset_idx2:stop_idx2]
    # print("#"*40)
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


def compare_two_filters(filter1, filter2, filter1_name="Filter 1", filter2_name="Filter 2", timeoffset=0, stop_time=None):
    timeoffset_idx1 = (np.abs(np.array(filter1.timestamps) - (filter1.timestamps[0] + timeoffset))).argmin()
    timeoffset_idx2 = (np.abs(np.array(filter2.timestamps) - (filter2.timestamps[0] + timeoffset))).argmin()

    if stop_time is not None:
        stop_idx1 = (np.abs(np.array(filter1.timestamps) - (filter1.timestamps[0] + stop_time))).argmin()
        stop_idx2 = (np.abs(np.array(filter2.timestamps) - (filter2.timestamps[0] + stop_time))).argmin()
        timeoffset_idx1 = min(timeoffset_idx1, stop_idx1)
        timeoffset_idx2 = min(timeoffset_idx2, stop_idx2)
    else:
        stop_idx1 = len(filter1.timestamps)
        stop_idx2 = len(filter2.timestamps)

    timestamps1 = np.array(filter1.timestamps)[timeoffset_idx1:stop_idx1]
    timestamps2 = np.array(filter2.timestamps)[timeoffset_idx2:stop_idx2]

    mus1 = np.array(filter1.mus)[timeoffset_idx1:stop_idx1]
    mus2 = np.array(filter2.mus)[timeoffset_idx2:stop_idx2]
    print("#"*40)
    if len(timestamps1) != len(timestamps2):
        # print("Filters have different number of timestamps, cannot compare (yet, will implement interpolation later).")
        print(f"{filter1_name} has {len(timestamps1)} timestamps, {filter2_name} has {len(timestamps2)} timestamps.")
        mus2_aligned = []
        for t in timestamps1:
            idx = (np.abs(timestamps2 - t)).argmin()
            mus2_aligned.append(mus2[idx])
        mus2 = np.array(mus2_aligned)

    xy_position_errors = np.linalg.norm(mus1[:, 0:2, 4] - mus2[:, 0:2, 4], axis=1)
    z_position_errors = np.abs(mus1[:, 2, 4] - mus2[:, 2, 4])
    max_xy_error = np.max(xy_position_errors)
    mean_xy_error = np.mean(xy_position_errors)
    median_xy_error = np.median(xy_position_errors)
    print(f"Max Position Error between {filter1_name} and {filter2_name}: {max_xy_error:.4f} m, and occurs at time {timestamps1[np.argmax(xy_position_errors)]-filter1.timestamps[0]:.2f} s")
    print(f"Median Position Error between {filter1_name} and {filter2_name}: {median_xy_error:.4f} m")
    print(f"Mean Position Error between {filter1_name} and {filter2_name}: {mean_xy_error:.4f} m, with stddev {np.std(xy_position_errors):.4f} m")
    print("\n")
    print(f"Max Z Position Error between {filter1_name} and {filter2_name}: {np.max(z_position_errors):.4f} m, and occurs at time {timestamps1[np.argmax(z_position_errors)]-filter1.timestamps[0]:.2f} s")
    print(f"Median Z Position Error between {filter1_name} and {filter2_name}: {np.median(z_position_errors):.4f} m")
    print(f"Mean Z Position Error between {filter1_name} and {filter2_name}: {np.mean(z_position_errors):.4f} m, with stddev {np.std(z_position_errors):.4f} m")
    # return
    attitude_errors = []
    for i in range(len(timestamps1)):
        R1 = mus1[i, 0:3, 0:3]
        R2 = mus2[i, 0:3, 0:3]
        R_diff = R1 @ R2.T
        angle = np.arccos((np.trace(R_diff) - 1) / 2)
        attitude_errors.append(np.rad2deg(angle))
    attitude_errors = np.array(attitude_errors)
    max_att_error = np.max(attitude_errors)
    mean_att_error = np.mean(attitude_errors)
    print("\n")
    print(f"Max Attitude Error between {filter1_name} and {filter2_name}: {max_att_error:.4f} deg and occurs at time {timestamps1[np.argmax(attitude_errors)]-filter1.timestamps[0]:.2f} s")
    print(f"Median Attitude Error between {filter1_name} and {filter2_name}: {np.median(attitude_errors):.4f} deg")
    print(f"Mean Attitude Error between {filter1_name} and {filter2_name}: {mean_att_error:.4f} deg")

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
    print(f"Max Roll Error between {filter1_name} and {filter2_name}: {np.max(roll_errors):.4f} deg and occurs at time {timestamps1[np.argmax(roll_errors)]-filter1.timestamps[0]:.2f} s")
    print(f"Mean Roll Error between {filter1_name} and {filter2_name}: {np.mean(roll_errors):.4f} deg, with stddev {np.std(roll_errors):.4f} deg")
    print("\n")
    print(f"Max Pitch Error between {filter1_name} and {filter2_name}: {np.max(pitch_errors):.4f} deg and occurs at time {timestamps1[np.argmax(pitch_errors)]-filter1.timestamps[0]:.2f} s")
    print(f"Mean Pitch Error between {filter1_name} and {filter2_name}: {np.mean(pitch_errors):.4f} deg, with stddev {np.std(pitch_errors):.4f} deg")
    print("\n")
    print(f"Max Yaw Error between {filter1_name} and {filter2_name}: {np.max(yaw_errors):.4f} deg and occurs at time {timestamps1[np.argmax(yaw_errors)]-filter1.timestamps[0]:.2f} s")
    print(f"Mean Yaw Error between {filter1_name} and {filter2_name}: {np.mean(yaw_errors):.4f} deg, with stddev {np.std(yaw_errors):.4f} deg")
    print("\n")
    return


def plot_results_comparison(filter1, filter2, title="", show=True, gps_timestamps=None, gps_altitude=None, body_frame_vel=False, imu_rpy=None, heading_measurements=None, rp_meas=None, filter1_name="1", filter2_name="2"):
    # if gps_timestamps is not None:
        # gps_timestamps -= gps_timestamps[0]

    # if rp_meas is not None:
        # rp_meas[:,0] -= rp_meas[0,0]

    # if heading_measurements is not None:
        # heading_measurements[:,0] -= heading_measurements[0,0]

    mus1 = np.array(filter1.mus)[1:]
    sigmas1 = np.array(filter1.sigmas)
    timestamps1 = np.array(filter1.timestamps)
    first_timestamp = timestamps1[0]
    timestamps1 = timestamps1 - first_timestamp  # Normalize time to start at 0
    
    mus2 = np.array(filter2.mus)
    sigmas2 = np.array(filter2.sigmas)
    timestamps2 = np.array(filter2.timestamps)
    timestamps2 = timestamps2 - first_timestamp  # Normalize time to start at 0
   
    if gps_timestamps is not None:
        gps_timestamps = gps_timestamps - first_timestamp

    if rp_meas is not None:
        rp_meas[:,0] -= first_timestamp

    if heading_measurements is not None:
        heading_measurements[:,0] -= first_timestamp

    # Plot the results
    fig, axs = plt.subplots(3, 3)
    fig.suptitle(title)

    # plot xyz position
    axs[0, 0].plot(timestamps1, mus1[:, 0, 4], marker='x', linestyle='-', markersize=3, color='r', label=filter1_name)
    axs[0, 0].plot(timestamps2, mus2[:, 0, 4], marker='x', linestyle='-', markersize=3, color='b', label=filter2_name)
    axs[0, 0].set_title('x')

    axs[1, 0].plot(timestamps1, mus1[:, 1, 4], marker='x', linestyle='-', markersize=3, color='r', label=filter1_name)
    axs[1, 0].plot(timestamps2, mus2[:, 1, 4], marker='x', linestyle='-', markersize=3, color='b', label=filter2_name)
    axs[1, 0].set_title('y')

    axs[2, 0].plot(timestamps1, mus1[:, 2, 4], marker='x', linestyle='-', markersize=3, color='r', label=filter1_name)
    axs[2, 0].plot(timestamps2, mus2[:, 2, 4], marker='x', linestyle='-', markersize=3, color='b', label=filter2_name)
    axs[2, 0].set_title('z')

    if gps_altitude is not None:
        # axs[2, 0].plot(gps_timestamps, gps_altitude, marker='x', markersize=3, color='g', label='GPS Measurement')
        gps_altitudes = np.array(gps_altitude)
        axs[0, 0].scatter(gps_timestamps, gps_altitude[:,0], color='g', label='GPS Measurement')
        axs[1, 0].scatter(gps_timestamps, gps_altitude[:,1], color='g', label='GPS Measurement')
        axs[2, 0].scatter(gps_timestamps, gps_altitude[:,2], color='g', label='GPS Measurement')

    R = Rotation.from_matrix(mus1[:, 0:3, 0:3])
    rpy = R.as_euler('xyz')
    rpy2 = Rotation.from_matrix(mus2[:, 0:3, 0:3]).as_euler('xyz')

    axs[0, 1].plot(timestamps1, np.rad2deg(rpy[:, 0]), marker='x', linestyle='-', markersize=3, color='r')
    if imu_rpy is not None:
        axs[0, 1].plot(imu_rpy[:, 0], (imu_rpy[:, 1]), marker='x', linestyle='-', markersize=3, color='b', label='IMU RPY')
    axs[0, 1].plot(timestamps2, np.rad2deg(rpy2[:, 0]), marker='x', linestyle='-', markersize=3, color='b')
    if rp_meas is not None:
        axs[0, 1].scatter(rp_meas[:,0], rp_meas[:,1], color='g', label='Image Roll Measurement')

    axs[0, 1].set_title('roll')
    axs[1, 1].plot(timestamps1, np.rad2deg(rpy[:, 1]), marker='x', linestyle='-', markersize=3, color='r')
    axs[1, 1].plot(timestamps2, np.rad2deg(rpy2[:, 1]), marker='x', linestyle='-', markersize=3, color='b')
    if imu_rpy is not None:
        axs[1, 1].plot(imu_rpy[:, 0], (imu_rpy[:, 2]), marker='x', linestyle='-', markersize=3, color='b', label='IMU RPY')
    if rp_meas is not None:
        axs[1, 1].scatter(rp_meas[:,0], rp_meas[:,2], color='g', label='Image Pitch Measurement')

    axs[1, 1].set_title('pitch')
    axs[2, 1].plot(timestamps1, np.rad2deg(rpy[:, 2]), marker='x', linestyle='-', markersize=3, color='r')
    axs[2, 1].plot(timestamps2, np.rad2deg(rpy2[:, 2]), marker='x', linestyle='-', markersize=3, color='b')
    if imu_rpy is not None:
        axs[2, 1].plot(imu_rpy[:, 0], (imu_rpy[:, 3]), marker='x', linestyle='-', markersize=3, color='b', label='IMU RPY')
        axs[2, 1].legend()
    # axs[2, 1].scatter(heading_measurements[:,0], heading_measurements[:,1], color='g', label='Heading Measurement')
    axs[2, 1].set_title('yaw')

    # Plot velocities
    velocities = mus1[:, 0:3, 3]
    velocities2 = mus2[:, 0:3, 3]
    if body_frame_vel:
        rotation_matrices = mus1[:, 0:3, 0:3]
        # Transform world-frame velocities into the body frame: v_body = Rᵀ * v_world
        velocities = np.einsum('ijk,ik->ij', rotation_matrices.transpose(0, 2, 1), velocities)

    axs[0, 2].plot(timestamps1, velocities[:, 0], marker='x', linestyle='-', markersize=3, color='r')
    axs[0, 2].plot(timestamps2, velocities2[:, 0], marker='x', linestyle='-', markersize=3, color='b')
    axs[0, 2].set_title('vx')
    axs[1, 2].plot(timestamps1, velocities[:, 1], marker='x', linestyle='-', markersize=3, color='r')
    axs[1, 2].plot(timestamps2, velocities2[:, 1], marker='x', linestyle='-', markersize=3, color='b')
    axs[1, 2].set_title('vy')
    axs[2, 2].plot(timestamps1, velocities[:, 2], marker='x', linestyle='-', markersize=3, color='r')
    axs[2, 2].plot(timestamps2, velocities2[:, 2], marker='x', linestyle='-', markersize=3, color='b')
    axs[2, 2].set_title('vz')

    for ax in axs.flat:
        ax.grid(True)
    axs[0, 0].legend()    

    # Handling pass by reference instead of value, need to add back the first timestamp
    # timestamps = timestamps + first_timestamp  # Normalize time to start at 0
    # gt_data[:, 0] = gt_data[:, 0] - timestamps[0] #gt_data[0, 0]  # Normalize GT time to start at 0
    if gps_timestamps is not None:
        gps_timestamps += first_timestamp
    if rp_meas is not None:
        rp_meas[:,0] += first_timestamp
    if heading_measurements is not None:
        heading_measurements[:,0] += first_timestamp

    if show:
        plt.show()

def plot_results_comparison_gt(filter1, filter2, filter3, title="", show=True, gps_timestamps=None, gps_altitude=None, body_frame_vel=False, imu_rpy=None, heading_measurements=None, rp_meas=None, filter1_name="1", filter2_name="2", filter3_name="3"):
    # if gps_timestamps is not None:
        # gps_timestamps -= gps_timestamps[0]

    # if rp_meas is not None:
        # rp_meas[:,0] -= rp_meas[0,0]

    # if heading_measurements is not None:
        # heading_measurements[:,0] -= heading_measurements[0,0]

    mus1 = np.array(filter1.mus)
    sigmas1 = np.array(filter1.sigmas)
    timestamps1 = np.array(filter1.timestamps)
    first_timestamp = timestamps1[0]
    timestamps1 = timestamps1 - first_timestamp  # Normalize time to start at 0
    
    mus2 = np.array(filter2.mus)
    sigmas2 = np.array(filter2.sigmas)
    timestamps2 = np.array(filter2.timestamps)
    timestamps2 = timestamps2 - first_timestamp  # Normalize time to start at 0
   
    mus3 = np.array(filter3.mus)
    sigmas3 = np.array(filter3.sigmas)
    timestamps3 = np.array(filter3.timestamps)
    timestamps3 = timestamps3 - first_timestamp  # Normalize time to start at 0
    
    if gps_timestamps is not None:
        gps_timestamps = gps_timestamps - first_timestamp

    if rp_meas is not None:
        rp_meas[:,0] -= first_timestamp

    if heading_measurements is not None:
        heading_measurements[:,0] -= first_timestamp

    # Plot the results
    fig, axs = plt.subplots(3, 3)
    fig.suptitle(title)

    # plot xyz position
    axs[0, 0].plot(timestamps1, mus1[:, 0, 4], marker='x', linestyle='-', markersize=3, color='r', label=filter1_name)
    axs[0, 0].plot(timestamps2, mus2[:, 0, 4], marker='x', linestyle='-', markersize=3, color='b', label=filter2_name)
    axs[0, 0].plot(timestamps3, mus3[:, 0, 4], marker='x', linestyle='-', markersize=3, color='g', label=filter3_name)
    axs[0, 0].set_title('x')

    axs[1, 0].plot(timestamps1, mus1[:, 1, 4], marker='x', linestyle='-', markersize=3, color='r', label=filter1_name)
    axs[1, 0].plot(timestamps2, mus2[:, 1, 4], marker='x', linestyle='-', markersize=3, color='b', label=filter2_name)
    axs[1, 0].plot(timestamps3, mus3[:, 1, 4], marker='x', linestyle='-', markersize=3, color='g', label=filter3_name)
    axs[1, 0].set_title('y')

    axs[2, 0].plot(timestamps1, mus1[:, 2, 4], marker='x', linestyle='-', markersize=3, color='r', label=filter1_name)
    axs[2, 0].plot(timestamps2, mus2[:, 2, 4], marker='x', linestyle='-', markersize=3, color='b', label=filter2_name)
    axs[2, 0].plot(timestamps3, mus3[:, 2, 4], marker='x', linestyle='-', markersize=3, color='g', label=filter3_name)
    axs[2, 0].set_title('z')

    if gps_altitude is not None:
        # axs[2, 0].plot(gps_timestamps, gps_altitude, marker='x', markersize=3, color='g', label='GPS Measurement')
        gps_altitudes = np.array(gps_altitude)
        axs[0, 0].scatter(gps_timestamps, gps_altitude[:,0], color='g', label='GPS Measurement')
        axs[1, 0].scatter(gps_timestamps, gps_altitude[:,1], color='g', label='GPS Measurement')
        axs[2, 0].scatter(gps_timestamps, gps_altitude[:,2], color='g', label='GPS Measurement')

    R = Rotation.from_matrix(mus1[:, 0:3, 0:3])
    rpy = R.as_euler('xyz')
    rpy2 = Rotation.from_matrix(mus2[:, 0:3, 0:3]).as_euler('xyz')
    rpy3 = Rotation.from_matrix(mus3[:, 0:3, 0:3]).as_euler('xyz')

    axs[0, 1].plot(timestamps1, np.rad2deg(rpy[:, 0]), marker='x', linestyle='-', markersize=3, color='r')
    if imu_rpy is not None:
        axs[0, 1].plot(imu_rpy[:, 0], (imu_rpy[:, 1]), marker='x', linestyle='-', markersize=3, color='b', label='IMU RPY')
    axs[0, 1].plot(timestamps2, np.rad2deg(rpy2[:, 0]), marker='x', linestyle='-', markersize=3, color='b')
    axs[0, 1].plot(timestamps3, np.rad2deg(rpy3[:, 0]), marker='x', linestyle='-', markersize=3, color='g', label=filter3_name)
    if rp_meas is not None:
        axs[0, 1].scatter(rp_meas[:,0], rp_meas[:,1], color='g', label='Image Roll Measurement')

    axs[0, 1].set_title('roll')
    axs[1, 1].plot(timestamps1, np.rad2deg(rpy[:, 1]), marker='x', linestyle='-', markersize=3, color='r')
    axs[1, 1].plot(timestamps2, np.rad2deg(rpy2[:, 1]), marker='x', linestyle='-', markersize=3, color='b')
    axs[1, 1].plot(timestamps3, np.rad2deg(rpy3[:, 1]), marker='x', linestyle='-', markersize=3, color='g', label=filter3_name)
    if imu_rpy is not None:
        axs[1, 1].plot(imu_rpy[:, 0], (imu_rpy[:, 2]), marker='x', linestyle='-', markersize=3, color='b', label='IMU RPY')
    if rp_meas is not None:
        axs[1, 1].scatter(rp_meas[:,0], rp_meas[:,2], color='g', label='Image Pitch Measurement')

    axs[1, 1].set_title('pitch')
    axs[2, 1].plot(timestamps1, np.rad2deg(rpy[:, 2]), marker='x', linestyle='-', markersize=3, color='r')
    axs[2, 1].plot(timestamps2, np.rad2deg(rpy2[:, 2]), marker='x', linestyle='-', markersize=3, color='b')
    axs[2, 1].plot(timestamps3, np.rad2deg(rpy3[:, 2]), marker='x', linestyle='-', markersize=3, color='g', label=filter3_name)
    if imu_rpy is not None:
        axs[2, 1].plot(imu_rpy[:, 0], (imu_rpy[:, 3]), marker='x', linestyle='-', markersize=3, color='b', label='IMU RPY')
        axs[2, 1].legend()
    if heading_measurements is not None:        
        axs[2, 1].scatter(heading_measurements[:,0], heading_measurements[:,1], color='g', label='Heading Measurement')
    axs[2, 1].set_title('yaw')

    # Plot velocities
    velocities = mus1[:, 0:3, 3]
    velocities2 = mus2[:, 0:3, 3]
    velocities3 = mus3[:, 0:3, 3]
    if body_frame_vel:
        rotation_matrices = mus1[:, 0:3, 0:3]
        # Transform world-frame velocities into the body frame: v_body = Rᵀ * v_world
        velocities = np.einsum('ijk,ik->ij', rotation_matrices.transpose(0, 2, 1), velocities)

    axs[0, 2].plot(timestamps1, velocities[:, 0], marker='x', linestyle='-', markersize=3, color='r')
    axs[0, 2].plot(timestamps2, velocities2[:, 0], marker='x', linestyle='-', markersize=3, color='b')
    axs[0, 2].plot(timestamps3, velocities3[:, 0], marker='x', linestyle='-', markersize=3, color='g', label=filter3_name)
    axs[0, 2].set_title('vx')
    axs[1, 2].plot(timestamps1, velocities[:, 1], marker='x', linestyle='-', markersize=3, color='r')
    axs[1, 2].plot(timestamps2, velocities2[:, 1], marker='x', linestyle='-', markersize=3, color='b')
    axs[1, 2].plot(timestamps3, velocities3[:, 1], marker='x', linestyle='-', markersize=3, color='g', label=filter3_name)
    axs[1, 2].set_title('vy')
    axs[2, 2].plot(timestamps1, velocities[:, 2], marker='x', linestyle='-', markersize=3, color='r')
    axs[2, 2].plot(timestamps2, velocities2[:, 2], marker='x', linestyle='-', markersize=3, color='b')
    axs[2, 2].plot(timestamps3, velocities3[:, 2], marker='x', linestyle='-', markersize=3, color='g', label=filter3_name)
    axs[2, 2].set_title('vz')

    for ax in axs.flat:
        ax.grid(True)
    axs[0, 0].legend()    

    # Handling pass by reference instead of value, need to add back the first timestamp
    # timestamps = timestamps + first_timestamp  # Normalize time to start at 0
    # gt_data[:, 0] = gt_data[:, 0] - timestamps[0] #gt_data[0, 0]  # Normalize GT time to start at 0
    if gps_timestamps is not None:
        gps_timestamps += first_timestamp
    if rp_meas is not None:
        rp_meas[:,0] += first_timestamp
    if heading_measurements is not None:
        heading_measurements[:,0] += first_timestamp
    if show:
        plt.show()

def plot_results_convergence_rp(filter1, filter2, filter3, title="", show=True, heading_measurements=None, rp_meas=None, filter1_name="1", filter2_name="2", filter3_name="3"):
    import seaborn as sns
    import matplotlib
    matplotlib.rc("pdf", fonttype=42)
    sns.set_context("paper")
    sns.set_style("whitegrid")
    sns.set_palette("colorblind")
    c = sns.color_palette("colorblind")

    # Make sure you install times & clear matplotlib cache
    # https://stackoverflow.com/a/49884009
    # plt.rcParams["font.family"] = "Times New Roman"
    # plt.rcParams["mathtext.fontset"] = "stix"
    # Source - https://stackoverflow.com/q
    # Posted by dylkot, modified by community. See post 'Timeline' for change history
    # Retrieved 2025-11-13, License - CC BY-SA 3.0


    # plt.rcParams['font.family'] = ['serif']
    # plt.rcParams['font.serif'] = ['Times New Roman']

    import shutil
    import matplotlib as mpl

    # cache_dir = mpl.get_cachedir()
    # shutil.rmtree(cache_dir)

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    
    SMALL_SIZE = 7
    MEDIUM_SIZE = 9
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE) # legend font size
    plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title
    
    stop_idx = 2 * 200 # 2 seconds at 200 Hz
    
    mus1 = np.array(filter1.mus)
    sigmas1 = np.array(filter1.sigmas)
    timestamps1 = np.array(filter1.timestamps)
    first_timestamp = timestamps1[0]
    timestamps1 = timestamps1 - first_timestamp  # Normalize time to start at 0
    
    mus2 = np.array(filter2.mus)
    sigmas2 = np.array(filter2.sigmas)
    timestamps2 = np.array(filter2.timestamps)
    timestamps2 = timestamps2 - first_timestamp  # Normalize time to start at 0
   
    mus3 = np.array(filter3.mus)
    sigmas3 = np.array(filter3.sigmas)
    timestamps3 = np.array(filter3.timestamps)
    timestamps3 = timestamps3 - first_timestamp  # Normalize time to start at 0
 
    if rp_meas is not None:
        rp_meas[:,0] -= first_timestamp

    if heading_measurements is not None:
        heading_measurements[:,0] -= first_timestamp

    # Plot the results
    fig, axs = plt.subplots(3, 1, figsize=(3.54, 3), dpi=300, constrained_layout=True)
    # fig.suptitle(title)

    R = Rotation.from_matrix(mus1[:, 0:3, 0:3])
    rpy = R.as_euler('xyz')
    rpy2 = Rotation.from_matrix(mus2[:, 0:3, 0:3]).as_euler('xyz')
    rpy3 = Rotation.from_matrix(mus3[:, 0:3, 0:3]).as_euler('xyz')

    lw= 2
    lw_m = 2.5

    axs[0].plot(timestamps3[:stop_idx], np.rad2deg(rpy3[:stop_idx, 0]), linestyle='-', linewidth=lw, color=c[4], label=filter3_name)
    axs[0].plot(timestamps1[:stop_idx], np.rad2deg(rpy[:stop_idx, 0]), linestyle='-', linewidth=lw_m, color=c[3], label=filter1_name)
    axs[0].plot(timestamps2[:stop_idx], np.rad2deg(rpy2[:stop_idx, 0]), linestyle='--', linewidth=lw, color=c[0], label=filter2_name)
    if rp_meas is not None:
        axs[0].scatter(rp_meas[:,0], rp_meas[:,1], color='g', label='Image Roll Measurement')
    # axs[0].set_title('roll')
    axs[0].set_ylabel("Roll (°)")
    axs[0].get_yaxis().set_label_coords(-0.085, 0.5)  # Adjust label position
    axs[1].set_ylabel("Pitch (°)")
    axs[1].get_yaxis().set_label_coords(-0.085, 0.5)  # Align with Roll
    axs[2].set_ylabel("Yaw (°)")
    axs[2].get_yaxis().set_label_coords(-0.085, 0.5)  # Align with Roll and Pitch

    axs[0].tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)

    axs[1].plot(timestamps3[:stop_idx], np.rad2deg(rpy3[:stop_idx, 1]), linestyle='-', linewidth=lw, color=c[4], label=filter3_name)
    axs[1].plot(timestamps1[:stop_idx], np.rad2deg( rpy[:stop_idx, 1]), linestyle='-', linewidth=lw_m, color=c[3], label=filter1_name)
    axs[1].plot(timestamps2[:stop_idx], np.rad2deg(rpy2[:stop_idx, 1]), linestyle='--', linewidth=lw, color=c[0], label=filter2_name)
    if rp_meas is not None:
        axs[1].scatter(rp_meas[:,0], rp_meas[:,2], color='g', label='Image Pitch Measurement')
    # axs[1].set_title('pitch')
    # axs[1].ylabel.set_size(14)
    # axs[1].set_ylabel("Pitch (°)")
    axs[1].tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)

    axs[2].plot(timestamps3[:stop_idx], np.rad2deg(rpy3[:stop_idx, 2]), linestyle='-', linewidth=lw, color=c[4], label=filter3_name)
    axs[2].plot(timestamps1[:stop_idx], np.rad2deg(rpy[:stop_idx, 2]), linestyle='-', linewidth=lw_m, color=c[3], label=filter1_name)
    axs[2].plot(timestamps2[:stop_idx], np.rad2deg(rpy2[:stop_idx, 2]), linestyle='--', linewidth=lw, color=c[0], label=filter2_name)
    # if heading_measurements is not None:        
        # axs[2, 1].scatter(heading_measurements[:,0], heading_measurements[:,1], color='g', label='Heading Measurement')
    # axs[2, 1].set_title('yaw')
    # axs[2].set_ylabel("Yaw (°)")
    axs[2].set_xlabel("Time (S)")

    for ax in axs.flat:
        ax.grid(True)
        ax.tick_params(axis='y', which='major', labelsize=SMALL_SIZE, pad=-1)
    axs[2].legend()    

    if show:
        plt.savefig("figures/convergence_rp.pdf")
        plt.show()
        

def plot_cov(filter, title="", show=True):
    sigmas = np.array(filter.sigmas)
    timestamps = np.array(filter.timestamps)
    timestamps = timestamps - timestamps[0]  # Normalize time to start at 0


    # Plot the results
    fig, axs = plt.subplots(3, 5)
    fig.suptitle(title)
    fig.supxlabel('Covariance values for R,P,Y, Vx,Vy,Vz, Px,Py,Pz\nValues in degrees, m/s, m')
    axs[0, 0].plot(timestamps, np.sqrt(np.rad2deg(sigmas[:, 0, 0])), marker='x', linestyle='-', markersize=3, color='r')
    axs[0, 0].set_title('Cov Roll')
    axs[1, 0].plot(timestamps, np.sqrt(np.rad2deg(sigmas[:, 1, 1])), marker='x', linestyle='-', markersize=3, color='r')
    axs[1, 0].set_title('Cov Pitch')
    axs[2, 0].plot(timestamps, np.sqrt(np.rad2deg(sigmas[:, 2, 2])), marker='x', linestyle='-', markersize=3, color='r')
    axs[2, 0].set_title('Cov Yaw')
    axs[0, 1].plot(timestamps, np.sqrt(sigmas[:, 3, 3]), marker='x', linestyle='-', markersize=3, color='r')
    axs[0, 1].set_title('Cov Vx')
    axs[1, 1].plot(timestamps, np.sqrt(sigmas[:, 4, 4]), marker='x', linestyle='-', markersize=3, color='r')
    axs[1, 1].set_title('Cov Vy')
    axs[2, 1].plot(timestamps, np.sqrt(sigmas[:, 5, 5]), marker='x', linestyle='-', markersize=3, color='r')
    axs[2, 1].set_title('Cov Vz')
    
    axs[0, 2].plot(timestamps, np.sqrt(sigmas[:, 6, 6]), marker='x', linestyle='-', markersize=3, color='r')
    axs[0, 2].set_title('Cov Px')
    axs[1, 2].plot(timestamps, np.sqrt(sigmas[:, 7, 7]), marker='x', linestyle='-', markersize=3, color='r')
    axs[1, 2].set_title('Cov Py')
    axs[2, 2].plot(timestamps, np.sqrt(sigmas[:, 8, 8]), marker='x', linestyle='-', markersize=3, color='r')
    axs[2, 2].set_title('Cov Pz')

    axs[0, 3].plot(timestamps, np.sqrt(sigmas[:, 9, 9]), marker='x', linestyle='-', markersize=3, color='r')
    axs[0, 3].set_title('Cov Gyro Bias x')
    axs[1, 3].plot(timestamps, np.sqrt(sigmas[:, 10, 10]), marker='x', linestyle='-', markersize=3, color='r')
    axs[1, 3].set_title('Cov Gyro Bias y')
    axs[2, 3].plot(timestamps, np.sqrt(sigmas[:, 11, 11]), marker='x', linestyle='-', markersize=3, color='r')
    axs[2, 3].set_title('Cov Gyro Bias z')

    axs[0, 4].plot(timestamps, np.sqrt(sigmas[:, 12, 12]), marker='x', linestyle='-', markersize=3, color='r')
    axs[0, 4].set_title('Cov Accel Bias x')
    axs[1, 4].plot(timestamps, np.sqrt(sigmas[:, 13, 13]), marker='x', linestyle='-', markersize=3, color='r')
    axs[1, 4].set_title('Cov Accel Bias y')
    axs[2, 4].plot(timestamps, np.sqrt(sigmas[:, 14, 14]), marker='x', linestyle='-', markersize=3, color='r')
    axs[2, 4].set_title('Cov Accel Bias z')


    for ax in axs.flat:
        ax.grid(True)

    if show:
        plt.show()

def plot_biases(filter, title="", show=True):
    biases = np.array(filter.biases)
    timestamps = np.array(filter.timestamps)
    timestamps = timestamps - timestamps[0]  # Normalize time to start at 0

    # Plot the results
    fig, axs = plt.subplots(2, 3)
    fig.suptitle(title)
    
    axs[0, 0].plot(timestamps, np.rad2deg(biases[:, 0, 0]), marker='x', linestyle='-', markersize=3, color='r')
    axs[0, 0].set_title('Gyro Bias x (deg)')
    
    axs[0, 1].plot(timestamps, np.rad2deg(biases[:, 0, 1]), marker='x', linestyle='-', markersize=3, color='r')
    axs[0, 1].set_title('Gyro Bias y (deg)')

    axs[0, 2].plot(timestamps, np.rad2deg(biases[:, 0, 2]), marker='x', linestyle='-', markersize=3, color='r')
    axs[0, 2].set_title('Gyro Bias z (deg)')


    axs[1, 0].plot(timestamps, biases[:, 1, 0], marker='x', linestyle='-', markersize=3, color='r')
    axs[1, 0].set_title('Accel Bias x')

    axs[1, 1].plot(timestamps, biases[:, 1, 1], marker='x', linestyle='-', markersize=3, color='r')
    axs[1, 1].set_title('Accel Bias y')

    axs[1, 2].plot(timestamps, biases[:, 1, 2], marker='x', linestyle='-', markersize=3, color='r')
    axs[1, 2].set_title('Accel Bias z')

    for ax in axs.flat:
        ax.grid(True)
    
    if show:
        plt.show()

def compare_filter_against_gps(filter, gps_measuresments, filter_name="", timeoffset=0.0, show=True, offset_idx=0, stop_idx = 185):
    # timeoffset_idx1 = (np.abs(np.array(filter1.timestamps) - (filter1.timestamps[0] + timeoffset))).argmin()

    # if stop_idx is not None:
        # stop_idx1 = (np.abs(np.array(filter1.timestamps) - (filter1.timestamps[0] + stop_time))).argmin()
        # stop_idx2 = (np.abs(np.array(filter2.timestamps) - (filter2.timestamps[0] + stop_time))).argmin()
        # timeoffset_idx1 = min(timeoffset_idx1, stop_idx1)
        # timeoffset_idx2 = min(timeoffset_idx2, stop_idx2)
    # else:
        # stop_idx1 = len(filter1.timestamps)
        # stop_idx2 = len(filter2.timestamps)
    # gps_idx = np.abs(np.array(gps_measuresments[:,0]) - timeoffset).argmin()
    # gps_stop_idx = np.abs(np.array(gps_measuresments[:,0]) - (gps_measuresments[:,0] + stop_idx)).argmin()

    gps_idx = int(timeoffset * 4.77)
    gps_stop_idx = int(stop_idx * 4.77)
    gps_timestamps = gps_measuresments[gps_idx:gps_stop_idx,0]
    gps_positions = gps_measuresments[gps_idx:gps_stop_idx,1:3]

    mus = np.array(filter.mus)
    timestamps = np.array(filter.timestamps)

    prev_idx = np.searchsorted(timestamps, gps_timestamps, side='left') - offset_idx
    
    matched_positions = np.zeros_like(gps_positions)
    for i in range(2):
        matched_positions[:, i] = mus[prev_idx, i, 4]

    position_errors = np.linalg.norm(matched_positions - gps_positions, axis=1)
    mean_error = np.mean(position_errors)
    median_error = np.median(position_errors)
    max_error = np.max(position_errors)

    '''
    # Interpolate filter positions to GPS timestamps
    interp_positions = np.zeros_like(gps_positions)
    for i in range(3):
        interp_positions[:, i] = np.interp(gps_timestamps, timestamps, mus[:, i, 4])

    position_errors = np.linalg.norm(interp_positions - gps_positions, axis=1)
    mean_error = np.mean(position_errors)
    max_error = np.max(position_errors)
    '''
    print(f"{filter_name} - Mean XY Position Error vs GPS: {mean_error:.4f} m")
    print(f"{filter_name} - Median XY Position Error vs GPS: {median_error:.4f} m")
    print(f"{filter_name} - Max XY Position Error vs GPS: {max_error:.4f} m")
    print("\n")
    # Plot position errors over time
    plt.figure()
    plt.plot(gps_timestamps, position_errors, marker='x', linestyle='-', markersize=3, color='r')
    plt.title(f"{filter_name} - Position Error vs GPS Over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Position Error (m)")
    plt.grid(True)
    if show:
        plt.show()


def get_filter_error_convergence(filter, gt_filter, filter_timestamps, gt_timestamps, timeoffset=0):
    # if filter_timestamps is None:
    #     filter_timestamps = np.array(filter.timestamps)
    # if gt_timestamps is None:
    #     gt_timestamps = np.array(gt_filter.timestamps)
        
    timeoffset_idx1 = (np.abs(np.array(filter_timestamps) - (filter_timestamps[0] + timeoffset))).argmin()
    timeoffset_idx2 = (np.abs(np.array(gt_timestamps) - (gt_timestamps[0] + timeoffset))).argmin()

    stop_idx1 = len(filter_timestamps)
    stop_idx2 = len(gt_timestamps)
    timestamps1 = np.array(filter_timestamps)[timeoffset_idx1:stop_idx1]
    timestamps2 = np.array(gt_timestamps)[timeoffset_idx2:stop_idx2]

    mus1 = np.array(filter)[timeoffset_idx1:stop_idx1]
    mus2 = np.array(gt_filter)[timeoffset_idx2:stop_idx2]
    # print("#"*40)
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
        # roll_errors.append(np.abs(np.rad2deg(r1_rpy[0] - r2_rpy[0])))
        # pitch_errors.append(np.abs(np.rad2deg(r1_rpy[1] - r2_rpy[1])))
        # yaw_errors.append(np.abs(np.rad2deg(r1_rpy[2] - r2_rpy[2])))
        roll_errors.append(np.rad2deg(r1_rpy[0] - r2_rpy[0]))
        pitch_errors.append(np.rad2deg(r1_rpy[1] - r2_rpy[1]))
        yaw_errors.append(np.rad2deg(r1_rpy[2] - r2_rpy[2]))
    roll_errors = np.array(roll_errors)
    pitch_errors = np.array(pitch_errors)
    yaw_errors = np.array(yaw_errors)

    return [xy_position_errors, z_position_errors, roll_errors, pitch_errors, yaw_errors]

def plot_figure_6_rw_convergence(sbg_inekf_mus, oak_inekf_mus, sbg_mekf_mus, oak_mekf_mus, 
                                 gt_poses, sbg_timestamps, oak_timestamps, gt_timestamps, show=True):
    
    # 1. Compute errors 
    # Each entry in the dict becomes a list of arrays. Each array is [5, t_sim]
    filter_data = {
        'SBG\nMEKF': [get_filter_error_convergence(m, g, s_t, g_t) for m, g, s_t, g_t in zip(sbg_mekf_mus, gt_poses, sbg_timestamps, gt_timestamps)],
        'SBG\nInEKF': [get_filter_error_convergence(m, g, s_t, g_t) for m, g, s_t, g_t in zip(sbg_inekf_mus, gt_poses, sbg_timestamps, gt_timestamps)],
        'OAK\nMEKF': [get_filter_error_convergence(m, g, o_t, g_t) for m, g, o_t, g_t in zip(oak_mekf_mus, gt_poses, oak_timestamps, gt_timestamps)],
        'OAK\nInEKF': [get_filter_error_convergence(m, g, o_t, g_t) for m, g, o_t, g_t in zip(oak_inekf_mus, gt_poses, oak_timestamps, gt_timestamps)]
    }
    # dummy_vals_sbg = [get_filter_error_convergence(m, g, s_t, g_t) for m, g, s_t, g_t in zip(sbg_mekf_mus, gt_poses, sbg_timestamps, gt_timestamps)]
    # dummy_vals_oak = [get_filter_error_convergence(m, g, o_t, g_t) for m, g, o_t, g_t in zip(oak_mekf_mus, gt_poses, oak_timestamps, gt_timestamps)]
    # filter_data = {
    #     'SBG\nMEKF': dummy_vals_sbg,
    #     'SBG\nInEKF': dummy_vals_sbg,
    #     'OAK\nMEKF': dummy_vals_oak,
    #     'OAK\nInEKF': dummy_vals_oak
    # }


    desired_stop = 400 
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

    ###
    sns.set_style("whitegrid")
    # plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    
    fig, axs = plt.subplots(4, 3, figsize=(textwidth_inches, int(textwidth_inches)), dpi=300, sharex=True, constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0, h_pad=0, hspace=0, wspace=0.05) # Negative wspace pulls columns in
    c = sns.color_palette("colorblind")
    # 2. Set a baseline tight spacing
    # plt.subplots_adjust(wspace=0.05, hspace=0.1)

    # # 3. Manually nudge columns 1 and 2 to the right to create a gap after col 0
    # for row_idx in range(4):
    #     # Get current position of Column 1
    #     pos1 = axs[row_idx, 1].get_position()
    #     pos2 = axs[row_idx, 2].get_position()
        
    #     # Nudge them right by 0.02 units of figure width
    #     axs[row_idx, 1].set_position([pos1.x0 + 0.02, pos1.y0, pos1.width, pos1.height])
    #     axs[row_idx, 2].set_position([pos2.x0 + 0.02, pos2.y0, pos2.width, pos2.height])


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

            # def wrap_val(vals):
            #     # Standard wrap to [-180, 180]
            #     return (vals + 180) % 360 - 180

            # Set Y-Limits
            if col_idx < 2:
                ax.set_ylim(-16, 16)
                if col_idx == 0:
                    ax.set_yticks([-10, 0, 10])  # Set custom y-ticks
                    ax.set_yticklabels(['-10', '0', '10'], fontsize=9)  # Label the y-ticks
                if col_idx == 1:
                    # ax.set_ylim(-16, 16)
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
                # run_data is [5, t_sim]
                # Grab the corresponding timestamp array for this specific sim
                t_raw = current_timestamps_list[sim_idx]
                t_norm = t_raw - t_raw[0] # Normalize to start at 0
                
                # Extract the specific error row (R=2, P=3, Y=4)
                y_vals = run_data[col_idx + 2]

                # 2. Slice both t and y to your desired window (e.g., 2 seconds)
                # This handles cases where one sim might be 1.9s and another 2.1s
                mask = t_norm <= 2.0 
                t_plot = t_norm[mask]
                y_plot = y_vals[mask]

                # if col_idx == 2: # Wrap Yaw
                #     y_plot = wrap_val(y_plot)

                # 3. Plot using the actual timestamps
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
    
    # for ax in axs[3, :]:
    #     ax.set_xlabel("Time (s)", fontsize=10, fontweight='bold')
    axs[3, 1].set_xlabel("Time (s)", fontsize=10, labelpad=-2)  # Set xlabel only for the middle bottom

    if show:
        plt.savefig("figures/real_world_convergence.pdf")
        plt.show()

    return

def plot_figure_timing(image_function_timing, inekf_predict_timing, mekf_predict_timing, inekf_heading_update_timing, mekf_heading_update_timing, inekf_rp_update_timing, mekf_rp_update_timing, inekf_full_pose_update_timing, show=True):    
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
        ('Predict', 'InEKF', inekf_predict_timing),
        ('Predict', 'MEKF', mekf_predict_timing),
        ('Heading\nUpdate', 'InEKF', inekf_heading_update_timing),
        ('Heading\nUpdate', 'MEKF', mekf_heading_update_timing),
        ('Roll/Pitch\nUpdate', 'InEKF', inekf_rp_update_timing),
        ('Roll/Pitch\nUpdate', 'MEKF', mekf_rp_update_timing),
        ('Full Ori.\nUpdate', 'InEKF', inekf_full_pose_update_timing),
    ]

    for task, filter_type, timings in groups:
        for t in timings:
            data_list.append({'Task': task, 'Filter': filter_type, 'Time (ms)': t * 1000})

    df = pd.DataFrame(data_list)

    # 1. IJRR / IEEE Single-Column Specifications
    # 3.5 inches is the standard column width. 
    # Height is reduced to keep the aspect ratio tight.
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['sans-serif'] + plt.rcParams['font.serif']
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
        split=True, 
        inner=None, 
        palette="colorblind",
        linewidth=0.8,
        density_norm='width',
        # scale="width",    # Makes all violins have the same peak width
        width=0.8,        # Increases the footprint of the violin on the x-axis
        ax=ax
    )

    new_color = sns.color_palette("colorblind")[2] 
    ax.collections[6].set_facecolor(new_color)



    ax.set_ylim(0, 5) # Focusing on the 0-15ms range where the bulk of math happens

    # 4. Academic Formatting
    ax.set_ylabel("Execution Time (ms)", fontsize=10)
    ax.set_xlabel("", fontsize=8)
    
    # Compact Tick Labels
    plt.xticks(rotation=0, ha='center', fontsize=10)
    ax.tick_params(axis='x', which='major', pad=7)
    
    plt.yticks(fontsize=10)
    
    # Compact Legend
    from matplotlib.patches import Patch
    fo_patch = Patch(facecolor=new_color, edgecolor='black', label='InEKF FO', linewidth=0.7)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(fo_patch)
    labels.append("InEKF FO")
    # ax.legend(handles=handles, labels=labels, title="", fontsize=10, loc='upper right', frameon=True)
    ax.legend(handles=handles, labels=labels, loc='upper right', frameon=True)
    plt.tight_layout(pad=0.5)

    if show:
        plt.savefig("figures/timing_violin.pdf", bbox_inches='tight', pad_inches=0.1)
        plt.show()

    return ax

import pickle as pkl
if __name__ == "__main__":
    data = np.load("inekf_holoocean_simulation_results_15x15.npz", allow_pickle=True)
    inekf_full_traj_maes = data["inekf_full_traj_maes"]
    inekf_5hz_conv_trajectories = data["inekf_5hz_conv_trajectories"]
    inekf_no_horizon_trajectories = data["inekf_no_horizon_trajectories"]
    inEKF_full_pose_full_traj_maes = data["inEKF_full_pose_full_traj_maes"]
    mekf_full_traj_maes = data["mekf_full_traj_maes"]
    mekf_5hz_conv_trajectories = data["mekf_5hz_conv_trajectories"]
    mekf_no_horizon_trajectories = data["mekf_no_horizon_trajectories"]
    gt_data = data["gt_data"]


    ## np.savez("inekf_holoocean_simulation_results_convergence.npz",
    ##     inekf_5hz_conv_trajectories=np.array(inekf_5hz_conv_trajectories, dtype=object),
    ##     inekf_no_horizon_trajectories=np.array(inekf_no_horizon_trajectories, dtype=object),
    ##     mekf_5hz_conv_trajectories=np.array(mekf_5hz_conv_trajectories, dtype=object),
    ##     mekf_no_horizon_trajectories=np.array(mekf_no_horizon_trajectories, dtype=object),
    ##     gt_data=np.array(ground_truth_poses)
    ##     )

    ## data = np.load("inekf_holoocean_simulation_results_convergence.npz", allow_pickle=True)
    ## inekf_5hz_conv_trajectories = data["inekf_5hz_conv_trajectories"]
    ## inekf_no_horizon_trajectories = data["inekf_no_horizon_trajectories"]
    ## mekf_5hz_conv_trajectories = data["mekf_5hz_conv_trajectories"]
    ## mekf_no_horizon_trajectories = data["mekf_no_horizon_trajectories"]
    ## gt_data = data["gt_data"]


    # plot_figure_3(inekf_full_traj_maes, mekf_full_traj_maes, inEKF_full_pose_full_traj_maes, show_plot=True)
    # plot_figure_4(inekf_5hz_conv_trajectories, inekf_no_horizon_trajectories, mekf_5hz_conv_trajectories, mekf_no_horizon_trajectories, gt_data)

    # # Load the data
    # df_loaded = pd.read_csv('filter_metrics_comparison.csv')

    # # Example: Accessing specific data for plotting
    # plot_figure_5_bar_graph(df_loaded, show_plot=True)


    ## Plotting figure 6 convergence for real-world data
    with open("real_world_convergence_data.pkl", "rb") as f:
        convergence_data = pkl.load(f)
    
    sbg_inekf_mus = convergence_data["sbg_inekf_mus"]
    oak_inekf_mus = convergence_data["oak_inekf_mus"]
    sbg_mekf_mus = convergence_data["sbg_mekf_mus"]
    oak_mekf_mus = convergence_data["oak_mekf_mus"]
    gt_poses = convergence_data["gt_poses"]
    sbg_timestamps = convergence_data["sbg_timestamps"]
    oak_timestamps = convergence_data["oak_timestamps"]
    gt_timestamps = convergence_data["gt_timestamps"]

    plot_figure_6_rw_convergence(sbg_inekf_mus, oak_inekf_mus, sbg_mekf_mus, oak_mekf_mus, gt_poses, sbg_timestamps, oak_timestamps, gt_timestamps, show=True)

    ### Real world convergence with GPS comparison
    # with open("timing_data.pkl", "rb") as f:
    #     timing_data = pkl.load(f)

    # image_function_timing = timing_data["image_function_timing"]
    # inekf_predict_timing = timing_data["inekf_predict_timing"]
    # mekf_predict_timing = timing_data["mekf_predict_timing"]
    # inekf_heading_update_timing = timing_data["inekf_heading_update_timing"]
    # mekf_heading_update_timing = timing_data["mekf_heading_update_timing"]
    # inekf_rp_update_timing = timing_data["inekf_rp_update_timing"]
    # mekf_rp_update_timing = timing_data["mekf_rp_update_timing"]
    # inekf_full_pose_update_timing = timing_data["inekf_full_pose_update_timing"]
  
    # plot_figure_timing(image_function_timing, inekf_predict_timing, mekf_predict_timing, inekf_heading_update_timing, mekf_heading_update_timing, inekf_rp_update_timing, mekf_rp_update_timing, inekf_full_pose_update_timing, show=True)