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

class InEKF:
    # def __init__(self, init_quat, init_xyz, init_vel, first_timestamp=0, est_bias=True, R_orientation_sen=None, init_sigma=sbg_Q, Q=sbg_Q, init_bias=False):
    def __init__(self, init_sigma=None, Q=None, init_state=None, first_timestamp=0, est_bias=True, update_rate=200, left=True):

        self.est_bias = est_bias # Can turn off Bias estimation
        self.sys = WAMVASV(Q)
        self.update_rate = update_rate

        self.left = left # If true, use left-invariant error definition, else right-invariant error definition

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
        
        if init_sigma is not None and not self.left:
            # The configured initial covariance is in body-frame (left-error)
            # coordinates. Map pose errors into the right-error coordinates
            # used by propagation and correction. Bias errors stay additive.
            transform = np.eye(15)
            transform[:9, :9] = self.sys.adjoint(mu0)
            init_sigma = transform @ init_sigma @ transform.T

        self.sigmas = [init_sigma]
        
        self.biases = [np.zeros((2,3))]

        # Attempting to preserve the first state for plotting later.
        # self.mus.append(self.mus[-1])
        # self.sigmas.append(self.sigmas[-1])
        # self.biases.append(self.biases[-1])
        # self.timestamps.append(first_timestamp)

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

    # InEKF Propagate motion
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
        
        current_time = imu_measurement.timestamp_sec + imu_measurement.timestamp_nsec / 1e9

        linear_accel = imu_measurement.accel
        angular_velocity = imu_measurement.gyro

        if self.prev_time is None:
            dt = 1/self.update_rate
            # self.timestamps.append(current_time-dt)
            # return
        else:
            dt = current_time - self.prev_time
        self.prev_time = current_time
        # dt = 1/100 # 100 Hz

        angular_velocity = np.array([angular_velocity[0], angular_velocity[1], angular_velocity[2]])
        linear_accel = np.array([linear_accel[0], linear_accel[1], linear_accel[2]])
        self.last_gyro = angular_velocity
        # Convert to ENU to NED
        # linear_accel[0], linear_accel[1] = linear_accel[1], -linear_accel[0]

        u = np.array([angular_velocity, linear_accel])
    
        # Update the state
        mu_bar = self.sys.f_lie(self.mu, u, dt, self.bias)

        # make adjoint
        I = np.eye(6)
        zero = np.zeros((9,6))
        adj_X = np.block([[self.sys.adjoint(mu_bar), zero],
                        [                  zero.T,    I]])
        
        # make propagation matrix
        zero = np.zeros((3,3))
        I = np.eye(3)

        # Get A matrix
        if self.left:
            w_cross = self.sys.cross(angular_velocity - self.bias[0]) # The self.bias was originally excluded
            a_cross = self.sys.cross(linear_accel - self.bias[1]) # CHECKME: Should bias be excluded here?

            # Left error A matrix
            self.expA = expm(np.block([[  -w_cross,      zero,      zero,    -I,  zero],
                                    [  -a_cross,  -w_cross,      zero,  zero,    -I],
                                    [      zero,         I,  -w_cross,  zero,  zero],
                                    [      zero,      zero,      zero,  zero,  zero],
                                    [      zero,      zero,      zero,  zero,  zero]])*dt)

            Q_mod = 1 # dt/(1/self.update_rate) # For situations where dt is not constant because measurements are dropped
            sigma_bar = self.expA @ self.sigma @ self.expA.T + self.expA @ (Q_mod * self.sys.Q) @ self.expA.T * dt

        else:            
            R = mu_bar[:3,:3]
            g_cross = self.sys.cross(np.array([0, 0, -9.8]))
            self.expA = expm(np.block([[   zero, zero, zero,         -R, zero],
                                    [g_cross, zero, zero, -adj_X[3:6,0:3],   -R],
                                    [   zero,    I, zero, -adj_X[6:9,0:3], zero],
                                    [   zero, zero, zero,       zero, zero],
                                    [   zero, zero, zero,       zero, zero]])*dt)

            sigma_bar = self.expA @ (self.sigma + adj_X @ self.sys.Q @ adj_X.T * dt) @ self.expA.T

        # save for use later
        self.mus.append( mu_bar )
        self.biases.append( self.bias )
        self.sigmas.append( sigma_bar )
        self.timestamps.append(current_time)

        # self.orientation_callback(msg)

    # def heading_callback(self, heading_measurement: HeadingMeasurement):
    #     """Runs Update step for a heading only measurement (rotation about Z-axis)

    #     :heading: rotation about world Z-axis in degrees
    #     :heading_acc: std_dev of the heading measurement in degrees
        
    #     :return:None
    #     """
    #     R_hat = self.mu[:3,:3]        
        
    #     heading = heading_measurement.heading
    #     pitch = heading_measurement.pitch
    
    #     yaw_dgps_meas = heading # TODO: Update name to not say yaw_dgps_meas
    #     so3_rotation_yaw_dgps_meas = Rotation.from_euler('z', yaw_dgps_meas, degrees=False).as_matrix()

    #     veh_yaw = Rotation.from_matrix(R_hat).as_euler('xyz')[2]
    #     # veh_yaw_only = Rotation.from_euler('xyz', [0, 0, veh_yaw]).as_matrix()
    #     def rotation_matrix_z(yaw):
    #         c = np.cos(yaw)
    #         s = np.sin(yaw)
    #         return np.array([[c, -s, 0],
    #                         [s,  c, 0],
    #                         [0,  0, 1]])
        
    #     veh_yaw_only = rotation_matrix_z(veh_yaw)

    #     veh_no_yaw = veh_yaw_only.T @ R_hat
    #     z = so3_rotation_yaw_dgps_meas @ veh_no_yaw

    #     z_hat = veh_yaw_only
    #     z = so3_rotation_yaw_dgps_meas

    #     # zero = np.zeros((3,3))
    #     # I = np.eye(3)
    #     # H = np.block([[-I, zero, zero, zero, zero]]) # Measurement matrix, measurement is rpy
        
    #     # V_log = logm(np.linalg.inv(z)@z_hat) # innovation is measured pitch^-1 @ belief pitch
    #     V_log = logm(z.T@z_hat) # innovation is measured pitch - belief pitch
    #     V = np.array([V_log[2, 1], V_log[0, 2], V_log[1,0]]) # Extract the log of the rotation matrix using the Vee operator

    #     # make our special measurement covariance
    #     invR = np.zeros((3,3))
    #     invR[2][2] = 1/heading_measurement.heading_cov #**2

    #     # sig_til = inv(H@self.sigma@H.T)
    #     sig_til = inv(self.heading_H @ self.sigma @ self.heading_H_T)
    #     S_inv = sig_til - sig_til@inv(R_hat.T@invR@R_hat + sig_til)@sig_til

    #     # K = self.sigma @ H.T @ S_inv # Kalman gain
    #     K = self.sigma @ self.heading_H_T @ S_inv # Kalman gain

    #     K_V = K @ V # K is (15,3) V is (3,)
        
    #     # Update the state
    #     self.mus[-1][:3,:3] = self.mu[:3,:3] @ expm( self.sys.cross( K_V[:3] ) )
    #     self.mus[-1][:3,3] += K_V[3:6] # update velocity
    #     self.mus[-1][:3,4] += K_V[6:9] # update position
        
    #     if self.est_bias:
    #         self.biases[-1] = self.bias + ( K_V[9:] ).reshape((2,3)) # update bias
        
    #     self.sigmas[-1] = (np.eye(15) - K @ self.heading_H) @ self.sigma # Update the covariance
    #     return 0

    def heading_callback_vec(self, heading_measurement: HeadingMeasurement):
            psi = heading_measurement.heading   # Assuming radians (convert with np.deg2rad if in degrees)
            theta = heading_measurement.pitch   # Assuming radians

            # 1. Convert heading (psi) and pitch (theta) to body X-axis unit vector in world frame
            # y_world = R_z(psi) @ R_y(theta) @ [1, 0, 0]^T
            sy, cy = np.sin(psi), np.cos(psi)
            st, ct = np.sin(theta), np.cos(theta)

            y_x = cy * ct
            y_y = sy * ct
            y_z = -st

            rotation = self.mu[:3, :3]
            forward = np.array([1.0, 0.0, 0.0])

            # Linearize the angular noise at the predicted direction. The
            # unit-vector innovation only has two first-order components.
            _, pitch, yaw = Rotation.from_matrix(rotation).as_euler("xyz")
            cp, sp = np.cos(pitch), np.sin(pitch)
            cy, sy = np.cos(yaw), np.sin(yaw)
            direction_jacobian = np.array([
                [-sy * cp, -cy * sp],
                [ cy * cp, -sy * sp],
                [     0.0,      -cp],
            ])
            tangent = np.eye(3)[1:]
            noise_jacobian = tangent @ rotation.T @ direction_jacobian
            angular_cov = np.diag([heading_measurement.heading_cov, heading_measurement.pitch_cov])
            noise_cov = noise_jacobian @ angular_cov @ noise_jacobian.T

            H = np.zeros((2, 15))
            H[:, :3] = -tangent @ self.sys.cross(forward)
            if not self.left:
                H[:, :3] = H[:, :3] @ rotation.T

            cross_cov = self.sigma @ H.T
            K = np.linalg.solve(H @ cross_cov + noise_cov, cross_cov.T).T
            measured_direction = np.array([y_x, y_y, y_z])
            V = (tangent @ (rotation.T @ measured_direction - forward)).reshape(2, 1)

            if self.left:
                self.mus[-1] = self.mu @ expm(self.sys.carat((K[:9] @ V).squeeze()))
            else:
                self.mus[-1] = expm(self.sys.carat((K[:9] @ V).squeeze())) @ self.mu

            if self.est_bias:
                self.biases[-1] = self.bias + ((K[9:] @ V).squeeze()).reshape((2, 3))

            correction = np.eye(15) - K @ H
            posterior_cov = correction @ self.sigma @ correction.T + K @ noise_cov @ K.T
            self.sigmas[-1] = (posterior_cov + posterior_cov.T) * 0.5
            return y_x, y_y, y_z

    def gps_callback(self, gps_measurement: GPSMeasurement):
        rotation = self.mu[:3, :3]
        position = self.mu[:3, 4]
        residual = np.array([gps_measurement.x, gps_measurement.y, gps_measurement.z]) - position
        H = np.zeros((3, 15))
        H[:, 6:9] = np.eye(3)
        noise_cov = gps_measurement.position_cov

        if self.left:
            # X exp(delta): position error is R delta_p. Use a body-frame residual.
            residual = rotation.T @ residual
            noise_cov = rotation.T @ noise_cov @ rotation
        else:
            # exp(delta) X: position error is delta_p + delta_theta × p.
            H[:, :3] = -self.sys.cross(position)

        prior_cov = self.sigma
        cross_cov = prior_cov @ H.T
        S = H @ cross_cov + noise_cov
        K = np.linalg.solve(S, cross_cov.T).T
        V = residual.reshape(3, 1)

        if self.left:
            self.mus[-1] = self.mu @ expm( self.sys.carat( (K[:9] @ V).squeeze() ))
        else:
            self.mus[-1] = expm( self.sys.carat( (K[:9] @ V).squeeze() )) @ self.mu

        
        if self.est_bias:
            self.biases[-1] = self.bias + ( (K[9:] @ V).squeeze() ).reshape((2,3)) # update bias
        
        correction = np.eye(15) - K @ H
        posterior_cov = correction @ prior_cov @ correction.T + K @ noise_cov @ K.T
        self.sigmas[-1] = (posterior_cov + posterior_cov.T) * 0.5
        return gps_measurement.x, gps_measurement.y, gps_measurement.z
   

    # TODO: Convert this to body frame right message update
    # def gps_velocity_callback(self, vel_measurement: DvlMeasurement):
    #     Rot = self.mu[:3,:3]
    #     zero = np.zeros((3,3))
    #     I = np.eye(3)
    #     H = np.block([[zero, I, zero, zero, zero]]) # Measurement matrix, measurement is vel_xyz for now

    #     # R_pos_sensor = np.diag([0.1, 0.1, 0.5])**2
    #     S = H @ self.sigma @ H.T + Rot.T @ vel_measurement.velocity_cov @ Rot

    #     K = self.sigma @ H.T @ inv(S)

    #     zero = np.zeros((3,2))
    #     II = np.block([[I, zero]])
    #     # Z = np.array([[x, y, alt-z_offset_ellipsoid, 0, 1]]).T
    #     Z = np.array([[vel_measurement.velocity[0], vel_measurement.velocity[1], vel_measurement.velocity[2], 1, 0]]).T

    #     # Z = np.array([[x, y, 0, 0, 1]]).T


    #     # Z[1] = -Z[1] # Flip y to match simulator
    #     # Z[2] = -Z[2] # Flip z to match simulator

    #     V = II @ inv(self.mu) @ Z

    #     self.mus[-1] = self.mu @ expm( self.sys.carat( (K[:9] @ V).squeeze() ))
    #     if self.est_bias:
    #         self.biases[-1] = self.bias + ( (K[9:] @ V).squeeze() ).reshape((2,3)) # update bias
        
    #     self.sigmas[-1] = (np.eye(15) - K @ H) @ self.sigma # Update the covariance
    #     # return x, y, alt-z_offset_ellipsoid
    #     return 
    
    # def dvl_callback(self, dvl_measurement: DvlMeasurement):
    #     # self.sys.dvl_r is a rotation matrix from DVL to body frame
    #     # self.sys.dvl_p is a translation vector from DVL to body frame
    #     # z = self.sys.dvl_r@z + self.sys.dvl_p@(self.last_gyro - self.bias[0])
    #     z = dvl_measurement.velocity

    #     z = z + (self.last_gyro - self.bias[0]) # Handling lever arm

    #     # make H matrices
    #     zero    = np.zeros((3,3))
    #     I       = np.eye(3)    
    #     H = np.block([zero, I, zero, zero, zero])

    #     I = np.eye(6)
    #     zero = np.zeros((9,6))
    #     # convert H1 to Left measurement
    #     H = H@np.block([[self.sys.adjoint(self.mu), zero],
    #                     [                     zero.T,    I]])

    #     # put our measurements into full form
    #     z = np.array([z[0], z[1], z[2], -1, 0])

    #     # make innovation
    #     V = (self.mu @ z)[:3]

    #     R = self.mu[:3,:3]

    #     # make our special measurement covariance
    #     meas_cov   = inv(H@self.sigma@H.T + R@dvl_measurement.velocity_cov@R.T)

    #     K = self.sigma @ H.T @ meas_cov
    #     K_state = K[:9]
    #     K_bias = K[9:]
    #     self.mus[-1] = self.mu @ expm( self.sys.carat(K_state @ V) )
        
    #     if self.est_bias:
    #         self.biases[-1] = self.bias + ( K_bias@V ).reshape((2,3))
    #     self.sigmas[-1] = (np.eye(15) - K @ H) @ self.sigma

    #     return

    def image_callback_gravity_vec(self, rp_measurement: HorizonObservation):
            phi = rp_measurement.roll_rad
            theta = rp_measurement.pitch_rad

            # 1. Convert roll/pitch to gravity unit vector in body frame: y = R_x(-phi) R_y(-theta) [0, 0, 1]^T
            sp, cp = np.sin(phi), np.cos(phi)
            st, ct = np.sin(theta), np.cos(theta)

            y_x = -st
            y_y = sp * ct
            y_z = cp * ct

            # 2. Forward Euler uncertainty (2x2) to 3D vector space (3x3)
            J_rp = np.array([
                [0.0,      -ct],
                [cp * ct,  -sp * st],
                [-sp * ct, -cp * st]
            ])
            Sigma_rp = np.diag([rp_measurement.roll_var, rp_measurement.pitch_var])
            
            # Radial regularization so 3x3 innovation covariance S is invertible
            R_cov_body = J_rp @ Sigma_rp @ J_rp.T + 1e-6 * np.eye(3)

            # 3. Reference gravity vector in world frame (b) and measurement vector (Z)
            b = np.array([[0.0, 0.0, 1.0, 0.0, 0.0]]).T
            b_cross = self.sys.cross(b[:3].squeeze())

            Rot = self.mu[:3, :3]

            zero = np.zeros((3, 3))
            I = np.eye(3)
            
            # Right-Invariant Measurement Matrix H
            H = np.block([[b_cross, zero, zero, zero, zero]])

            if self.left:
                H = H @ np.block([[self.sys.adjoint(self.mu), np.zeros((9, 6))],
                                [np.zeros((6, 9)),          np.eye(6)]])

            S = H @ self.sigma @ H.T + Rot @ R_cov_body @ Rot.T

            K = self.sigma @ H.T @ inv(S)

            zero_3x2 = np.zeros((3, 2))
            II = np.block([[I, zero_3x2]])
            Z = np.array([[y_x, y_y, y_z, 0.0, 0.0]]).T

            # Right-Invariant Innovation
            V = II @ (self.mu @ Z - b)

            if self.left:
                self.mus[-1] = self.mu @ expm(self.sys.carat((K[:9] @ V).squeeze()))
            else:
                self.mus[-1] = expm(self.sys.carat((K[:9] @ V).squeeze())) @ self.mu

            if self.est_bias:
                self.biases[-1] = self.bias + ((K[9:] @ V).squeeze()).reshape((2, 3))

            self.sigmas[-1] = (np.eye(15) - K @ H) @ self.sigma
            return y_x, y_y, y_z
  
    def image_callback(self, rp_measurement: HorizonObservation): # image callback with yaw removed for seafaring frame WORKING! CORRECT, USE ME!
        r, p, y = Rotation.from_matrix(self.mu[:3,:3]).as_euler('xyz')

        R_cam_measurement = self.rotation_matrix_from_roll_pitch(rp_measurement.roll_rad, rp_measurement.pitch_rad)
        # R_cam_measurement = Rotation.from_euler('xyz', [roll, pitch, y]).as_matrix()

        wamv_yaw_only = Rotation.from_euler('xyz', [0, 0, y]).as_matrix()
        R_cam_estimate = wamv_yaw_only.T @ self.mu[:3,:3] # Converting estimate to seafaring frame

        z = R_cam_measurement
        
        R_bel = R_cam_estimate
        
        zero = np.zeros((3,3))
        I = np.eye(3)
        H = np.block([[-I, zero, zero, zero, zero]]) # Measurement matrix, measurement is rpy

        z_hat = R_bel # Predicted measurement
        
        V_log = logm(np.linalg.inv(z)@z_hat) # innovation is measured pitch - belief pitch
        V = np.array([V_log[2, 1], V_log[0, 2], V_log[1,0]])#.reshape(3,1) # Extract the log of the rotation matrix using the Vee operator


        # make our special measurement covariance
        invR = np.zeros((3,3))
        invR[0][0] = 1/(rp_measurement.roll_var) #**2)
        invR[1][1] = 1/(rp_measurement.pitch_var) #**2)

        sig_til = inv(H@self.sigma@H.T)

        sig_til_no_inv = H@self.sigma@H.T
        # alt_approach
        S_inv = sig_til - sig_til@inv(sig_til_no_inv @ (R_bel.T@invR@R_bel)+np.eye(3))

        K = self.sigma @ H.T @ S_inv #inv(S) # Kalman gain
        K_V = K @ V
        
        # Update the state
        self.mus[-1][:3,:3] = self.mu[:3,:3] @ expm( self.sys.cross( K_V[:3] ) )
        self.mus[-1][:3,3] += K_V[3:6] # update velocity
        self.mus[-1][:3,4] += K_V[6:9] # update position
        
        if self.est_bias:
            self.biases[-1] = self.bias + ( K_V[9:] ).reshape((2,3)) # update bias
        
        self.sigmas[-1] = (np.eye(15) - K @ H) @ self.sigma # Update the covariance
        return 0
    
    def rotation_matrix_from_roll_pitch(self, roll, pitch):
        R_x = np.array([[1, 0, 0],
                        [0, np.cos(roll), -np.sin(roll)],
                        [0, np.sin(roll), np.cos(roll)]])

        R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                        [0, 1, 0],
                        [-np.sin(pitch), 0, np.cos(pitch)]])

        R = np.dot(R_y, R_x) # or R = R_y @ R_x in Python 3.5+
        return R

    ''' TODO: Update this to use new measurement structure '''
    # Original callback with pose estimated by magnetometer
    def orientation_callback_combined(self, heading_measurement: HeadingMeasurement, rp_measurement: HorizonObservation): 
        # quat = np.array([msg_imu.orientation.x,
        #                     msg_imu.orientation.y,
        #                     msg_imu.orientation.z,
        #                     msg_imu.orientation.w])
        # mag_orientation = Rotation.from_quat(quat).as_matrix()

        # # 90-degree CCW rotation matrix about Z-axis
        # R_z_90 = np.array([
        #     [0, -1, 0],
        #     [1,  0, 0],
        #     [0,  0, 1]
        # ])
        # mag_orientation = R_z_90 @ mag_orientation # Rotate 90 degrees CCW about Z-axis

        # yaw_dgps_meas = Rotation.from_matrix(mag_orientation).as_euler('xyz')[2]
        
        z = Rotation.from_euler('xyz', [rp_measurement.roll_rad, rp_measurement.pitch_rad, heading_measurement.heading]).as_matrix()
        # z = mag_orientation

        R_bel = self.mu[:3,:3]
        
        zero = np.zeros((3,3))
        I = np.eye(3)

        H = np.block([[-I, zero, zero, zero, zero]]) # Measurement matrix, measurement is xyz for now
        z_hat = R_bel # Predicted measurement
        
        # V_log = logm(z_hat.T @ z) # innovation is measured pitch - belief pitch
        # V_log = logm(z.T@z_hat) # innovation is measured pitch - belief pitch
        # V_log = logm(z_hat.T @ z)
        V_log = logm(z.T@z_hat)
        # V_log = logm(z@z_hat.T) # innovation is measured pitch - belief pitch
        V = np.array([V_log[2, 1], V_log[0, 2], V_log[1,0]])#.reshape(3,1) # Extract the log of the rotation matrix using the Vee operator
        
        # make our special measurement covariance
        R_error = np.diag([rp_measurement.roll_var, rp_measurement.pitch_var, heading_measurement.heading_cov]) # **2
        S = H @ self.sigma @ H.T + R_bel.T @ R_error @ R_bel # + self.sys.R[-1, -1]

        K = self.sigma @ H.T @ inv(S) # Kalman gain

        K_V = K @ V
        
        # Update the state
        # self.mus[-1] = self.mu @ expm( self.sys.carat( K[:9] @ V ))
        self.mus[-1][:3,:3] = self.mu[:3,:3] @ expm( self.sys.cross( K_V[:3] ) )
        self.mus[-1][:3,3] += K_V[3:6] # update velocity
        self.mus[-1][:3,4] += K_V[6:9] # update position
        self.biases[-1] = self.bias + ( K[9:] @ V ).reshape((2,3)) # update bias
        
        self.sigmas[-1] = (np.eye(15) - K @ H) @ self.sigma # Update the covariance

def main(args=None):
    pass