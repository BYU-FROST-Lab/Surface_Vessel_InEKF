'''
Unused functions that may be useful for future analysis or plotting.
Most of these functions were used when debugging and analyzing the data
before making finalized plots and functions found in function_helpers.py.

These functions were originally in function_helpers.py, but were moved here to keep function_helpers.py 
clean and focused on the final versions of the functions used for plotting and analysis in the paper.
'''

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
