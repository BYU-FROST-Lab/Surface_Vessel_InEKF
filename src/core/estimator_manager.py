import numpy as np
from scipy.spatial.transform import Rotation


from src.estimators.invariant_ekf import InEKF as InvariantEKF
from helpers.function_helpers import *

class EstimatorManager:
    def __init__(self, config, map_data=None):
        # GT Pose at time 5.01s: Pos(26.05, 663.99, 0.98), RPY(3.48, -5.21, 55.31)        
        # init_state = np.array([np.deg2rad(3.48), np.deg2rad(-5.21), np.deg2rad(55.31), 0, 0, 0, 26.05, 663.99, 0.98])  # [roll, pitch, yaw, vx, vy, vz, x, y, z]
        init_state = np.array([np.deg2rad(0), np.deg2rad(0), np.deg2rad(0), 0, 0, 0, 26.05, 663.99, 0.98])  # [roll, pitch, yaw, vx, vy, vz, x, y, z]

        # [24.0, 657.0, 0.0],
        # "rotation": [0,0,90]
        self.inekf = InvariantEKF(config.init_cov, config.imu_noise_params, init_state=init_state)
        # self.pf = ParticleFilter()
        # self.fg = FactorGraphManager()
        self.localized = False
        self.ground_truth_poses = []  # For evaluation purposes

        # Plotting values
        self.imu_measurements = []
        self.print_pose = True

    def process_imu(self, imu_measurement):
        self.inekf.imu_callback(imu_measurement)
        self.imu_measurements.append(imu_measurement)
        return

    def process_gps(self, gps_measurement):
        # self.inekf.gps_callback(gps_measurement) # Intenitionally not handling GPS
        return
    
    def process_horizon(self, horizon_measurement):
        self.inekf.image_callback(horizon_measurement)
        return
    
    def process_heading(self, heading_measurement):
        self.inekf.heading_callback(heading_measurement)
        return
    
    def process_gt_pose(self, pose_measurement):
        timestamp = pose_measurement.timestamp_sec + pose_measurement.timestamp_nsec * 1e-9
        rpy = Rotation.from_quat([pose_measurement.orientation[0],
                                 pose_measurement.orientation[1],
                                 pose_measurement.orientation[2],
                                 pose_measurement.orientation[3]]).as_euler('xyz', degrees=True)
        pos = pose_measurement.position
        # self.ground_truth_poses.append((pose_measurement.timestamp, rpy, pose_measurement.position))
        self.ground_truth_poses.append((timestamp, rpy[0], rpy[1], rpy[2], pos[0], pos[1], pos[2]))
        if self.print_pose:
            print(f"GT Pose at time {timestamp:.2f}s: Pos({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}), RPY({rpy[0]:.2f}, {rpy[1]:.2f}, {rpy[2]:.2f})")
            self.print_pose = False  # Only print once for debugging
        return
    
    def get_InEKF_attitude_estimate(self):
        return self.inekf.mu[0:3, 0:3]
    
    def plot_inekf_results(self):
        gt = sbg_ekf_to_filter(np.array(self.ground_truth_poses))
        # compare_two_filters(self.inekf, gt, filter1_name="InEKF", filter2_name="GT")
        plot_results_comparison(self.inekf, gt, filter1_name="InEKF", filter2_name="GT")
        # def plot_results_comparison(filter1, filter2, title="", show=True, gps_timestamps=None, gps_altitude=None, body_frame_vel=False, imu_rpy=None, heading_measurements=None, rp_meas=None, filter1_name="1", filter2_name="2"):

    def plot_imu_measurements(self):
        plot_imu_data(self.imu_measurements)

    # def process_batch(self, messages):
    #     """Processes a list of chronologically sorted messages."""
    #     for msg in messages:
    #         # 1. ALWAYS update the InEKF for dead-reckoning
    #         if msg.topic in ['/imu', '/dvl']:
    #             self.inekf.predict(msg)

    #         # 2. Logic for Global Localization (PF)
    #         if not self.localized:
    #             if msg.topic == '/camera/coastline':
    #                 # self.pf.update(msg)
    #                 if self.pf.is_converged():
    #                     # THE SNAP: Seed the graph
    #                     # self.fg.initialize(self.pf.get_mean(), self.inekf.get_pose())
    #                     self.localized = True

    #         # 3. Logic for SLAM (GTSAM)
    #         else:
    #             if msg.topic == '/camera/coastline':
    #                 # Use InEKF as the 'initial guess' for the new GTSAM node
    #                 current_guess = self.inekf.get_pose()
    #                 # self.fg.add_observation(msg, current_guess)