# Start by reading in: Topic: /wamv/sensors/position/ground_truth_odometry | Type: nav_msgs/msg/Odometry | Count: 220 | Serialization Format: cdr
import numpy as np
from scipy.spatial.transform import Rotation as Rotation
from scipy.linalg import expm, logm
from numpy.linalg import inv
import sys, os
import math
import time

from matplotlib import pyplot as plt

import warnings 

from src.core.measurements import ImuMeasurement, GPSMeasurement, HeadingMeasurement, HorizonObservation
from helpers.robot_system import WAMVASV

import warnings 

class MEKF:
    # def __init__(self, init_quat, init_xyz, init_vel, first_timestamp=0, est_bias=True, R_orientation_sen=None, init_sigma=sbg_Q, Q=sbg_Q, init_bias=False):
    def __init__(self, init_sigma=None, Q=None, init_state=None, first_timestamp=0, est_bias=True, update_rate=200):   
        self.est_bias = est_bias # Can turn of Bias estimation
        self.sys = WAMVASV(Q)
        self.update_rate = update_rate
               
        # Initialize state, covariance, and noise parameters here
        if init_state is not None:
            mu0 = init_state
        else:
            mu0 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]) 

        if mu0.shape ==(9,):
            # mu0 = expm(sys.carat(mu0))
            mu0 = np.block([[Rotation.from_euler('xyz', mu0[:3]).as_matrix(), mu0[3:6].reshape(-1,1), mu0[6:9].reshape(-1,1)],
                            [np.zeros((2,3)), np.eye(2)]])
            
        self.mus = [mu0]
        self.timestamps = [first_timestamp]

        I = np.eye(3)
        zero = np.zeros((3,3))
        self.heading_H = np.block([[-I, zero, zero, zero, zero]]) # Measurement matrix, measurement is rpy
        self.heading_H_T = self.heading_H.T
        
        self.sigmas = [init_sigma]
        
        self.biases = [np.zeros((2,3))]

        self.prev_time = None
        self.last_gyro = None

    @property
    def mu(self):
        return self.mus[-1]
    
    @property
    def sigma(self):
        return self.sigmas[-1]
    
    @property
    def bias(self):
        return self.biases[-1]

    # QEKF Propagate motion
    # Process IMU data for prediction step
    def imu_callback(self, imu_measurement: ImuMeasurement):
        """Runs prediction step of Quaternion-EKF. (Multiplicative EKF)

        Args:
            u       (k ndarray) : control taken at this step
            u[0]                : angular velocity
            u[1]                : acceleration
            dt                  : time step
        
        Returns:
            mu    (nxn ndarray) : Propagated state
            mu[:3,:3]           : Rotation Matrix
            mu[:3,3]            : Velocity
            mu[:3,4]            : Position (z, what we care about the most because of the depth measurement, is mu[2,4])

            sigma (nxn ndarray) : Propagated covariances"""
        
        # T = np.diag([1, -1, -1]) # Convert NWU to NED

        current_time = imu_measurement.timestamp_sec + imu_measurement.timestamp_nsec / 1e9
        
        linear_accel = imu_measurement.accel
        angular_velocity = imu_measurement.gyro

        if self.prev_time is None:
            dt = 1/self.update_rate
        else:
            dt = current_time - self.prev_time
        self.prev_time = current_time
        # dt = 1/100 # 100 Hz

        angular_velocity = np.array([angular_velocity[0], angular_velocity[1], angular_velocity[2]])
        linear_accel = np.array([linear_accel[0], linear_accel[1], linear_accel[2]])

        # Convert to ENU to NED
        # linear_accel[0], linear_accel[1] = linear_accel[1], -linear_accel[0]

        u = np.array([angular_velocity, linear_accel])
    
        # Update the state
        # mu_bar = self.sys.f_lie(self.state, u, dt, self.bias)
        mu_bar = self.sys.f_lie(self.mu, u, dt, self.bias)

        # make propagation matrix
        zero = np.zeros((3,3))
        I = np.eye(3)

        # Get A matrix
        w_cross = self.sys.cross(angular_velocity - self.bias[0])
        a_cross = self.sys.cross(linear_accel - self.bias[1])
        R = mu_bar[:3,:3]

        # Left error A matrix
        self.expA = expm(np.block([[  -w_cross,  zero,  zero,    -I,  zero],
                                   [-R@a_cross,  zero,  zero,  zero,    -R],
                                   [      zero,     I,  zero,  zero,  zero],
                                   [      zero,  zero,  zero,  zero,  zero],
                                   [      zero,  zero,  zero,  zero,  zero]])*dt)
        
        # Multiply elements of Q by dt and dt^2, as appropriate 
        Qdt = self.sys.Q * dt
        # Qdt[:6,:6] = Qdt[:6,:6] * dt

        Q_mod = 1 # dt/(1/self.update_rate) # For situations where dt is not constant because measurements are dropped
        Qdt = Qdt * Q_mod # For situations where dt is not constant because measurements are dropped
        
        # Update Covarariance
        sigma_bar = self.expA @ self.sigma @ self.expA.T + Qdt
        # if (sigma_bar[5][5] < self.sigmas[-1][5][5]):
            # print("Warning: Z velocity variance decreased")
            # warnings.warn("Warning: Z velocity variance decreased")
        
        # self.state = mu_bar

        # save for use later
        self.mus.append( mu_bar )
        self.biases.append( self.bias )
        self.sigmas.append( sigma_bar )
        self.timestamps.append(current_time)

        # self.orientation_callback(msg) # moved function call to run filter script

    def get_rot_error_y(self, yaw_bel, yaw_meas):
        R_yaw_error = self.rot_matrix_2d(yaw_bel).T @ self.rot_matrix_2d(yaw_meas)
        error = np.array(logm(R_yaw_error)[0,1]) # might need to switch this!
        return error
    
    def heading_callback(self, heading_measurement: HeadingMeasurement): # Kalliyan approach for DGPS heading
        # w = msg.orientation.w
        # x = msg.orientation.x
        # y = msg.orientation.y
        # z = msg.orientation.z

        # rpy = Rotation.from_quat([x, y, z, w]).as_matrix()

        # 90 degree CCW rotation matrix about Z-axis
        # R_z_90 = np.array([
            # [0, -1, 0],
            # [1,  0, 0],
            # [0,  0, 1]
        # ])
        # rpy = R_z_90 @ rpy # Rotate 90 degrees CCW about Z-axis

        # x_axis_meas = rpy.T[:,0]

        world_x = np.array([1, 0, 0])
        # yaw_dgps_meas = np.arccos(np.dot(x_axis_meas, world_x) / (np.linalg.norm(x_axis_meas) * np.linalg.norm(world_x)))
        # yaw_dgps_meas = Rotation.from_matrix(rpy).as_euler('xyz')[2]
        yaw_dgps_meas = heading_measurement.heading
        # heading_acc = heading_measurement.heading_cov
        

        R_hat = self.mu[:3,:3]
        # x_axis_est = R_hat.T[:,0]
        # yaw_dgps_est = np.arccos(np.dot(x_axis_est, world_x) / (np.linalg.norm(x_axis_est) * np.linalg.norm(world_x)))
        yaw_dgps_est = Rotation.from_matrix(R_hat).as_euler('xyz')[2]

        # eps = 1e-6

        H_num = np.zeros((1,9))
        # for i in range(3):
        #     delta_step = np.zeros(9)
        #     delta_step[i] = eps
        #     delta_step_forward = np.block([[Rotation.from_euler('xyz', delta_step[:3]).as_matrix(), delta_step[3:6].reshape(-1,1), delta_step[6:9].reshape(-1,1)],
        #                     [np.zeros((2,3)), np.eye(2)]])
            
        #     delta_step_backward = np.block([[Rotation.from_euler('xyz', -delta_step[:3]).as_matrix(), -delta_step[3:6].reshape(-1,1), -delta_step[6:9].reshape(-1,1)],
        #                     [np.zeros((2,3)), np.eye(2)]])
            
        #     pose_forward = self.mu @ delta_step_forward
        #     pose_backward = self.mu @ delta_step_backward

        #     # dgps_forward = np.arccos(np.dot(pose_forward.T[:3,0], world_x) / (np.linalg.norm(pose_forward.T[:3,0]) * np.linalg.norm(world_x)))
        #     # dgps_backward = np.arccos(np.dot(pose_backward.T[:3,0], world_x) / (np.linalg.norm(pose_backward.T[:3,0]) * np.linalg.norm(world_x)))

        #     dgps_forward = Rotation.from_matrix(pose_forward[:3,:3]).as_euler('xyz')[2]
        #     dgps_backward = Rotation.from_matrix(pose_backward[:3,:3]).as_euler('xyz')[2]

        #     error_forward = self.get_rot_error_y(dgps_forward, yaw_dgps_est)
        #     error_backward = self.get_rot_error_y(dgps_backward, yaw_dgps_est)

        #     hdot = (error_forward - error_backward) / (2*eps)
        #     H_num[:,i] = hdot
        H_num[0,2] = 1
        H = np.block([H_num, np.zeros((1,6))]) # need to add bias

        R_yaw_error = self.rot_matrix_2d(yaw_dgps_est).T @ self.rot_matrix_2d(yaw_dgps_meas)

        error = np.array([logm(R_yaw_error)[1,0]])

        # V_log = logm(z_hat.T @ z) # innovation is measured pitch - belief pitch
        V = error # innovation is measured pitch - belief pitch
        # V_log = logm(z@z_hat.T) # innovation is measured pitch - belief pitch
        
        # make our special measurement covariance
        S = H @ self.sigma @ H.T + heading_measurement.heading_cov #**2 # + self.sys.R[-1, -1]

        K = self.sigma @ H.T @ inv(S) # Kalman gain

        K_V = K @ V
        
        # Update the state
        self.mus[-1][:3,:3] = self.mu[:3,:3] @ expm( self.sys.cross( K_V[:3] ) )
        self.mus[-1][:3,3] += K_V[3:6] # update velocity
        self.mus[-1][:3,4] += K_V[6:9] # update position
        if self.est_bias:
            self.biases[-1] = self.bias + ( K_V[9:] ).reshape((2,3)) # update bias
        
        self.sigmas[-1] = (np.eye(15) - K @ H) @ self.sigma # Update the covariance
        

    def gps_callback(self, gps_measurement: GPSMeasurement):

        z = np.array([gps_measurement.x, gps_measurement.y, gps_measurement.z])

        zero = np.zeros((3,3))
        I = np.eye(3)

        H = np.block([[zero, zero, I, zero, zero]]) # Measurement matrix, measurement is xyz for now
        z_hat = self.mu[:3,4] # Predicted measurement
        V = z - z_hat # Innovation

        # make our special measurement covariance
        Rot = self.mu[:3,:3]
        # S = H @ self.sigma @ H.T + Rot.T @ gps_measurement.position_cov @ Rot # FIXME: Does this need to be rotated?
        S = H @ self.sigma @ H.T + gps_measurement.position_cov # FIXME: Does this need to be rotated?


        K = self.sigma @ H.T @ inv(S) # Kalman gain

        K_V = K @ V
        
        # Update the state
        self.mus[-1][:3,:3] = self.mu[:3,:3] @ expm( self.sys.cross( K_V[:3] ) )
        self.mus[-1][:3,3] += K_V[3:6] # update velocity
        self.mus[-1][:3,4] += K_V[6:9] # update position
        if self.est_bias:
            self.biases[-1] = self.bias + ( K_V[9:] ).reshape((2,3)) # update bias
        
        self.sigmas[-1] = (np.eye(15) - K @ H) @ self.sigma # Update the covariance
        
        return
        # self.publish_state()

    # def gps_callback_qekf(self, x, y, alt, R_pos_sensor):

    #     z = np.array([x, y, alt])
        
    #     zero = np.zeros((3,3))
    #     I = np.eye(3)

    #     H = np.block([[zero, zero, I, zero, zero]]) # Measurement matrix, measurement is xyz for now
    #     z_hat = self.mu[:3,4] # Predicted measurement
    #     V = z - z_hat # Innovation

    #     # make our special measurement covariance
    #     R = self.mu[:3,:3]
    #     S = H @ self.sigma @ H.T + R.T @ R_pos_sensor @ R # FIXME: Does this need to be rotated?
    #     S = H @ self.sigma @ H.T + R.T @ R_pos_sensor @ R # FIXME: Does this need to be rotated?

    #     K = self.sigma @ H.T @ inv(S) # Kalman gain

    #     K_V = K @ V
        
    #     # Update the state
    #     self.mus[-1][:3,:3] = self.mu[:3,:3] @ expm( self.sys.cross( K_V[:3] ) )
    #     self.mus[-1][:3,3] += K_V[3:6] # update velocity
    #     self.mus[-1][:3,4] += K_V[6:9] # update position
    #     if self.sim_bias:
    #         self.biases[-1] = self.bias + ( K_V[9:] ).reshape((2,3)) # update bias
        
    #     self.sigmas[-1] = (np.eye(15) - K @ H) @ self.sigma # Update the covariance
        
    #     return
    #     # self.publish_state()

    def update_depth(self, z, sim_bias=True):
        """Runs correction step of Quaternion-EKF. (Multiplicative EKF)

        Args:
            z (m ndarray): measurement at this step, corresponds to mu[2,4]

        Returns:
            mu    (nxn ndarray) : Corrected state
            sigma (nxn ndarray) : Corrected covariances"""
        # make H matrices
        zero    = np.zeros((1,3))
        I       = np.eye(3)    
        H = np.block([zero, zero, I[-1,:], zero, zero])
        # make innovation
        z_hat = self.mu[2,4]
        V = z - z_hat

        # make our special measurement covariance
        R = self.mu[:3,:3] # Rotation Matrix
        meas_cov = inv(H@self.sigma@H.T + self.sys.R[-1,-1])

        K = self.sigma @ H.T @ meas_cov
        K_state = K[:9]
        K_bias = K[9:]

        K_V = (K * V).flatten()
        # Update the state
        self.mus[-1][:3,:3] = self.mu[:3,:3] @ expm( self.sys.cross( K_V[:3] ) )
        self.mus[-1][:3,3] += K_V[3:6]
        self.mus[-1][:3,4] += K_V[6:9]
        # Update the bias'
        if sim_bias:
            self.biass[-1] = self.bias + ( K_V[9:] ).reshape((2,3))
        # Update the covariance
        self.sigmas[-1] = (np.eye(15) - K @ H) @ self.sigma

        return self.mu, self.sigma


    def image_callback_orig(self, msg):
        w = msg.pose.orientation.w
        x = msg.pose.orientation.x
        y = msg.pose.orientation.y
        z = msg.pose.orientation.z

        rpy = Rotation.from_quat([x, y, z, w]).as_euler('xyz')
        roll = rpy[0]
        pitch = rpy[1]
        
        prev_rpy = Rotation.from_matrix(self.mus[-1][:3,:3]).as_euler('xyz')
        yaw = prev_rpy[2]       
        # yaw = self.last_yaw

        meas_R = Rotation.from_euler('xyz', [roll, pitch, yaw])

        # This likely needs updating.
        # meas_R = Rotation.from_quat(quat)
        z = meas_R.as_matrix()
        
        R_bel = self.mu[:3,:3]
        
        zero = np.zeros((3,3))
        I = np.eye(3)

        H = np.block([[-I, zero, zero, zero, zero]]) # Measurement matrix, measurement is rpy
        z_hat = R_bel # Predicted measurement
        
        # V_log = logm(z_hat.T @ z) # innovation is measured pitch - belief pitch
        V_log = logm(np.linalg.inv(z)@z_hat) # innovation is measured pitch - belief pitch
        # V_log = logm(z@z_hat.T) # innovation is measured pitch - belief pitch
        V = np.array([V_log[2, 1], V_log[0, 2], V_log[1,0]])#.reshape(3,1) # Extract the log of the rotation matrix using the Vee operator
        

        # make our special measurement covariance
        S = H @ self.sigma @ H.T + self.R_orientation_sensor # + self.sys.R[-1, -1]

        K = self.sigma @ H.T @ inv(S) # Kalman gain

        K_V = K @ V
        
        # Update the state
        self.mus[-1][:3,:3] = self.mu[:3,:3] @ expm( self.sys.cross( K_V[:3] ) )
        self.mus[-1][:3,3] += K_V[3:6] # update velocity
        self.mus[-1][:3,4] += K_V[6:9] # update position
        if self.est_bias:
            self.biases[-1] = self.bias + ( K_V[9:] ).reshape((2,3)) # update bias
        
        self.sigmas[-1] = (np.eye(15) - K @ H) @ self.sigma # Update the covariance
        
        return rpy
        # self.publish_state()

    def rot_matrix_2d(self, angle_radians):
        """
        Generates a 2x2 rotation matrix for a given angle in degrees.

        Args:
            angle_degrees: The rotation angle in degrees.

        Returns:
            numpy.ndarray: A 2x2 NumPy array representing the rotation matrix.
        """
        cos_theta = np.cos(angle_radians)
        sin_theta = np.sin(angle_radians)
        rotation_matrix = np.array([
            [cos_theta, -sin_theta],
            [sin_theta,  cos_theta]
        ])
        return rotation_matrix

    # def get_rot_error_rp(self, R_bel, R_meas):
    #     rp_meas = Rotation.from_matrix(R_meas).as_euler('xyz')[0:2]
    #     roll_meas = rp_meas[0]
    #     pitch_meas = rp_meas[1]
        
    #     rp_est = Rotation.from_matrix(R_bel).as_euler('xyz')[0:2]
    #     roll_est = rp_est[0]
    #     pitch_est = rp_est[1]
        
    #     R_roll_error = self.rot_matrix_2d(roll_est).T @ self.rot_matrix_2d(roll_meas)
    #     R_pitch_error = self.rot_matrix_2d(pitch_est).T @ self.rot_matrix_2d(pitch_meas)

    #     error = np.array([logm(R_roll_error)[0,1], logm(R_pitch_error)[0,1]])
    #     return error

    def get_rot_error_rp(self, R_bel, R_meas):
        """
        Calculates the roll/pitch measurement error based on 3D rotation error
        projected to roll/pitch space.

        Args:
            R_bel: 3x3 NumPy rotation matrix - Belief rotation.
            R_meas: 3x3 NumPy rotation matrix - Measurement rotation.

        Returns:
            numpy.ndarray: A 2x1 NumPy array representing the roll/pitch error.
        """
        R_error_so3 = R_meas.T @ R_bel  # Rotation from belief to measurement
        error_logm = (logm(R_error_so3)) # 3D error vector in so(3)
        error_so3_vec = np.array([error_logm[2, 1], error_logm[0, 2], error_logm[1, 0]])  # Extract the log of the rotation matrix using the Vee operator

        # Project to roll/pitch space (XYZ Euler convention assumed for error state):
        error_rp = error_so3_vec[0:2] # Take roll (x-axis) and pitch (y-axis) components

        return error_rp


    def image_callback(self, rp_measurement: HorizonObservation): # Kalliyan approach
        
        # rpy = Rotation.from_quat([x, y, z, w]).as_euler('xyz')
        roll_meas = rp_measurement.roll_rad
        pitch_meas = rp_measurement.pitch_rad
        
        # rp_est = Rotation.from_matrix(self.mus[-1][:3,:3]).as_euler('xyz')[0:2]
        # roll_est = rp_est[0]
        # pitch_est = rp_est[1]

        R = self.mu[:3,:3]

        eps = 1e-6

        H_num = np.zeros((2,3))
        # for i in range(3):
        #     delta_step = np.zeros(9)
        #     delta_step[i] = eps
        #     delta_step_forward = np.block([[Rotation.from_euler('xyz', delta_step[:3]).as_matrix(), delta_step[3:6].reshape(-1,1), delta_step[6:9].reshape(-1,1)],
        #                     [np.zeros((2,3)), np.eye(2)]])
            
        #     delta_step_backward = np.block([[Rotation.from_euler('xyz', -delta_step[:3]).as_matrix(), -delta_step[3:6].reshape(-1,1), -delta_step[6:9].reshape(-1,1)],
        #                     [np.zeros((2,3)), np.eye(2)]])
            
        #     pose_forward = self.mu @ delta_step_forward
        #     pose_backward = self.mu @ delta_step_backward

        #     error_forward = self.get_rot_error_rp(R, pose_forward[:3,:3])
        #     error_backward = self.get_rot_error_rp(R, pose_backward[:3,:3])

        #     hdot = (error_forward - error_backward) / (2*eps)
        #     H_num[:,i] = hdot
        H_num[0,0] = -1
        H_num[1,1] = -1
        H = np.block([H_num, np.zeros((2,12))]) # need to add bias
        # H[:,2] = 0 # zero out yaw influence
        # R_roll_error = self.rot_matrix_2d(roll_est).T @ self.rot_matrix_2d(roll_meas)
        # R_pitch_error = self.rot_matrix_2d(pitch_est).T @ self.rot_matrix_2d(pitch_meas)

        # error = np.array([logm(R_roll_error)[1,0], logm(R_pitch_error)[1,0]])
        wamv_yaw = Rotation.from_matrix(R).as_euler('xyz')[2]
        error = self.get_rot_error_rp(R, Rotation.from_euler('xyz', [roll_meas, pitch_meas, wamv_yaw]).as_matrix())
        # V_log = logm(z_hat.T @ z) # innovation is measured pitch - belief pitch
        V = error # innovation is measured pitch - belief pitch
        # V_log = logm(z@z_hat.T) # innovation is measured pitch - belief pitch
        
        # measurement covariance
        R_orientation_sensor = np.diag([rp_measurement.roll_var, rp_measurement.pitch_var]) # **2
        S = H @ self.sigma @ H.T + R_orientation_sensor
        
        # d2 = V.T @ inv(S) @ V
        # if d2 > 16: # 95% confidence interval for 2D chi-squared
        #     print(f"Outlier detected in horizon measurement with Mahalanobis distance squared: {d2:.2f} at timestep {self.timestamps[-1]-self.timestamps[0]:.2f}s")
        #     return

         # + self.sys.R[-1, -1]
        
        K = self.sigma @ H.T @ inv(S) # Kalman gain

        K_V = K @ V
        
        # Update the state
        self.mus[-1][:3,:3] = self.mu[:3,:3] @ expm( self.sys.cross( K_V[:3] ) )
        self.mus[-1][:3,3] += K_V[3:6] # update velocity
        self.mus[-1][:3,4] += K_V[6:9] # update position
        if self.est_bias:
            self.biases[-1] = self.bias + ( K_V[9:] ).reshape((2,3)) # update bias
        
        self.sigmas[-1] = (np.eye(15) - K @ H) @ self.sigma # Update the covariance
        
        return 
    
    def image_callback_other(self, msg):
        w = msg.pose.orientation.w
        x = msg.pose.orientation.x
        y = msg.pose.orientation.y
        z = msg.pose.orientation.z

        rpy = Rotation.from_quat([x, y, z, w]).as_euler('xyz')
        roll = rpy[0]
        pitch = rpy[1]
        
        prev_rpy = Rotation.from_matrix(self.mus[-1][:3,:3]).as_euler('xyz')
        # yaw = prev_rpy[2]       
        # yaw = self.last_yaw

        meas_R = Rotation.from_euler('xyz', [roll, pitch, 0])

        # This likely needs updating.
        # meas_R = Rotation.from_quat(quat)
        z = meas_R.as_matrix()
        
        R_bel = self.mu[:3,:3]
        R_bel_yaw = Rotation.from_matrix(R_bel).as_euler('xyz')[2]
        R_bel_yaw = Rotation.from_euler('xyz', [0, 0, R_bel_yaw]).as_matrix()

        R_bel_no_yaw = R_bel_yaw.T @ R_bel # Extracting yaw to only get roll pitch measurement
        
        zero = np.zeros((3,3))
        I = np.eye(3)

        H = np.block([[-I, zero, zero, zero, zero]]) # Measurement matrix, measurement is rpy
        z_hat = R_bel_no_yaw # Predicted measurement
        
        # V_log = logm(z_hat.T @ z) # innovation is measured pitch - belief pitch
        V_log = logm(np.linalg.inv(z)@z_hat) # innovation is measured pitch - belief pitch
        # V_log = logm(z@z_hat.T) # innovation is measured pitch - belief pitch
        V = np.array([V_log[2, 1], V_log[0, 2], V_log[1,0]])#.reshape(3,1) # Extract the log of the rotation matrix using the Vee operator
        

        # make our special measurement covariance
        R_cov = np.diag([self.R_orientation_sensor[0,0], self.R_orientation_sensor[1,1], 0])
        S = H @ self.sigma @ H.T + R_cov # + self.sys.R[-1, -1]
        # S = H @ self.sigma @ H.T + R_orientation_sensor # + self.sys.R[-1, -1]

        K = self.sigma @ H.T @ inv(S) # Kalman gain

        K_V = K @ V
        
        # Update the state
        self.mus[-1][:3,:3] = self.mu[:3,:3] @ expm( self.sys.cross( K_V[:3] ) )
        self.mus[-1][:3,3] += K_V[3:6] # update velocity
        self.mus[-1][:3,4] += K_V[6:9] # update position
        if self.est_bias:
            self.biases[-1] = self.bias + ( K_V[9:] ).reshape((2,3)) # update bias
        
        self.sigmas[-1] = (np.eye(15) - K @ H) @ self.sigma # Update the covariance
        
        return rpy

def main(args=None):    
    pass
