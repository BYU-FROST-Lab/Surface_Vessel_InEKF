import numpy as np

Radius_Earth = 6378137.0  # in meters

class HolooceanConfigParams:
    def __init__(self):
        # Example Time Step (e.g., IMU runs at 200 Hz)
        update_rate = 200.0  # Hz
        Delta_t = 1/update_rate  # seconds

        # SBG IMU Params
        # Q_g = 0.0012482515368315754 # from Kalibr report
        # Q_a = 0.0077783394070347215
        ## Gyro Bias (driven by Gyro Random Walk)
        # Q_bg = 3.905118591654977e-06   # rad/s/sqrt(s)
        ## Accel Bias (driven by Accelerometer Random Walk)
        # Q_ba = 1.0549217075926282e-05 # m/s^2/sqrt(s)

        # Oak IMU Params
        # Q_g = 0.006154494275600531
        # Q_a = 0.02995700050383457
        # Q_bg = 4.455742618201502e-06
        # Q_ba = 0.00021405239609180207
        
        # # SBG Params
        # Q_g = 0.0012482515368315754 # from Kalibr report
        # Q_a = 0.0077783394070347215
        # # Gyro Bias (driven by Gyro Random Walk)
        # Q_bg = 3.905118591654977e-06 * np.sqrt(Delta_t) #np.sqrt(update_rate)   # rad/s/sqrt(s)
        # # Accel Bias (driven by Accelerometer Random Walk)
        # Q_ba = 1.0549217075926282e-05 * np.sqrt(Delta_t) # m/s^2/sqrt(s)
        
        
        # Q_g = 0.0012482515368315754/np.sqrt(Delta_t) # from Kalibr report
        # Q_a = 0.0077783394070347215/np.sqrt(Delta_t) # Converting back to continuous time noise density
        # # Gyro Bias (driven by Gyro Random Walk)
        # Q_bg = 3.905118591654977e-06 #* np.sqrt(Delta_t) #np.sqrt(update_rate)   # rad/s/sqrt(s)
        # # Accel Bias (driven by Accelerometer Random Walk)
        # Q_ba = 1.0549217075926282e-05 #* np.sqrt(Delta_t) # m/s^2/sqrt(s)


        # Current HoloOcean Params
        # Q_g = 0.01
        # Q_a = 0.01
        # Q_bg = 0.002 #4.455742618201502e-06
        # Q_ba = 0.00021405239609180207
        
        # Oak IMU Params
        Q_g = 0.006154494275600531/np.sqrt(Delta_t)
        Q_a = 0.02995700050383457/np.sqrt(Delta_t)
        Q_bg = 4.455742618201502e-06 #* np.sqrt(Delta_t) # np.sqrt(update_rate)
        Q_ba = 0.00021405239609180207 #* np.sqrt(Delta_t) # np.sqrt(update_rate)

        Q_p = 0.0 # Standard practice for IMU mechanization state

        # Order: [Rotation (3), Velocity (3), Position (3), Gyro Bias (3), Accel Bias (3)]
        Q_diag = np.array([
            Q_g, Q_g, Q_g,    # Rotation/Attitude
            Q_a, Q_a, Q_a,    # Velocity
            Q_p, Q_p, Q_p,    # Position (Zero)
            Q_bg, Q_bg, Q_bg,  # Gyro Bias
            Q_ba, Q_ba, Q_ba   # Accel Bias
        ])
        
        self.imu_noise_params = np.diag(Q_diag)**2
        
        # self.init_cov = np.diag([
        #     0.1, 0.1, 0.1,        # Rotation/Attitude (rad)
        #     0.5, 0.5, 0.5,        # Velocity (m/s)
        #     2.0, 2.0, 2.0,        # Position (m)
        #     np.deg2rad(0.5), np.deg2rad(0.5), np.deg2rad(0.5),  # Gyro Bias (rad/s)
        #     0.005, 0.005, 0.005         # Accel Bias (m/s^2)
        # ])**2
        # self.init_cov = np.diag([
        #     np.deg2rad(2), np.deg2rad(2), np.deg2rad(60),        # Rotation/Attitude (rad)
        #     0.5, 0.5, 0.5,        # Velocity (m/s)
        #     1.0, 1.0, 1.0,        # Position (m)
        #     # Q_bg*np.sqrt(time), Q_bg*np.sqrt(time), Q_bg*np.sqrt(time),  # Gyro Bias (rad/s)
        #     np.deg2rad(0.0005), np.deg2rad(0.0005), np.deg2rad(0.0005),  # Gyro Bias (rad/s)
        #     # Q_ba*np.sqrt(time), Q_ba*np.sqrt(time), Q_ba*np.sqrt(time)
        #     0.0005, 0.0005, 0.0005         # Accel Bias (m/s^2)
        # ])**2
        self.init_cov = np.diag([
            np.deg2rad(2), np.deg2rad(2), np.deg2rad(60),        # Rotation/Attitude (rad)
            0.5, 0.5, 0.5,        # Velocity (m/s)
            1.0, 1.0, 1.0,        # Position (m)
            # Q_bg*np.sqrt(time), Q_bg*np.sqrt(time), Q_bg*np.sqrt(time),  # Gyro Bias (rad/s)
            np.deg2rad(0.0005), np.deg2rad(0.0005), np.deg2rad(0.0005),  # Gyro Bias (rad/s)
            # Q_ba*np.sqrt(time), Q_ba*np.sqrt(time), Q_ba*np.sqrt(time)
            0.005, 0.005, 0.005         # Accel Bias (m/s^2)
        ])**2

        # FIXME: Use better initial covariance later

        self.init_cov_high = np.diag([
            np.deg2rad(60), np.deg2rad(60), np.deg2rad(90),        # Rotation/Attitude (rad)
            4.0, 4.0, 4.0,        # Velocity (m/s)
            4.0, 4.0, 1.0,
            np.deg2rad(0.5), np.deg2rad(0.5), np.deg2rad(0.5),  # Gyro Bias (rad/s)
            0.005, 0.005, 0.005        # Accel Bias (m/s^2)
        ])**2

        # np.diag([np.deg2rad(30)*modifier, np.deg2rad(30)*modifier, np.deg2rad(30)*modifier, 2*modifier, 2*modifier, 2*modifier, 2*modifier, 2*modifier, 0.5*modifier, 0,0,0,0,0,0])**2


        self.img_width = 768
        self.img_height = 480
        self.dist = np.array([0.0, 0.0, 0.0, 0.0])  # Perfect distortion coefficients for simulation
        FoV = 90  # [deg]
        f = self.img_width / (2 * np.tan((FoV * np.pi / 180) / 2))
        self.mtx = np.array([
            [f, 0.0, self.img_width/2], 
            [0.0, f, self.img_height/2], 
            [0.0, 0.0, 1.0]
        ])

        self.camera_height = 1.57  # meters

        self.heading_noise_var = np.deg2rad(1) ** 2 # radians
        self.gps_position_cov = np.diag([1.0, 1.0, 2.0]) ** 2  # meters

class RealWorldSBGConfigParams:
    def __init__(self):
        # Example Time Step (e.g., IMU runs at 200 Hz)
        update_rate = 200.0  # Hz
        Delta_t = 1/update_rate  # seconds

        # SBG IMU Params
        time =  np.sqrt(1303) # seconds since boot, only applicable for this real world dataset

        Q_g = 0.0012482515368315754/np.sqrt(Delta_t) # from Kalibr report
        Q_a = 0.0077783394070347215/np.sqrt(Delta_t) # Converting back to continuous time noise density
        # Gyro Bias (driven by Gyro Random Walk)
        Q_bg = 3.905118591654977e-06 #* np.sqrt(Delta_t) #np.sqrt(update_rate)   # rad/s/sqrt(s)
        # Accel Bias (driven by Accelerometer Random Walk)
        Q_ba = 1.0549217075926282e-05 #* np.sqrt(Delta_t) # m/s^2/sqrt(s)

        Q_p = 0.0 # Standard practice for IMU mechanization state

        # Order: [Rotation (3), Velocity (3), Position (3), Gyro Bias (3), Accel Bias (3)]
        Q_diag = np.array([
            Q_g, Q_g, Q_g,    # Rotation/Attitude
            Q_a, Q_a, Q_a,    # Velocity
            Q_p, Q_p, Q_p,    # Position (Zero)
            Q_bg, Q_bg, Q_bg,  # Gyro Bias
            Q_ba, Q_ba, Q_ba   # Accel Bias
        ])
        
        self.imu_noise_params = np.diag(Q_diag)**2
        
        self.init_cov = np.diag([
            np.deg2rad(2), np.deg2rad(2), np.deg2rad(60),        # Rotation/Attitude (rad)
            0.5, 0.5, 0.5,        # Velocity (m/s)
            1.0, 1.0, 1.0,        # Position (m)
            # Q_bg*np.sqrt(time), Q_bg*np.sqrt(time), Q_bg*np.sqrt(time),  # Gyro Bias (rad/s)
            np.deg2rad(0.005), np.deg2rad(0.005), np.deg2rad(0.005),  # Gyro Bias (rad/s)
            # Q_ba*np.sqrt(time), Q_ba*np.sqrt(time), Q_ba*np.sqrt(time)
            0.05, 0.05, 0.05         # Accel Bias (m/s^2)
        ])**2
        # FIXME: Use better initial covariance later

        self.init_cov_high = np.diag([
            np.deg2rad(5), np.deg2rad(5), np.deg2rad(60),        # Rotation/Attitude (rad)
            0.5, 0.5, 0.5,        # Velocity (m/s)
            1.0, 1.0, 1.0,        # Position (m)
            # Q_bg*np.sqrt(time), Q_bg*np.sqrt(time), Q_bg*np.sqrt(time),  # Gyro Bias (rad/s)
            np.deg2rad(0.005), np.deg2rad(0.005), np.deg2rad(0.005),  # Gyro Bias (rad/s)
            # Q_ba*np.sqrt(time), Q_ba*np.sqrt(time), Q_ba*np.sqrt(time)
            0.05, 0.05, 0.05         # Accel Bias (m/s^2)
        ])**2

        # np.diag([np.deg2rad(30)*modifier, np.deg2rad(30)*modifier, np.deg2rad(30)*modifier, 2*modifier, 2*modifier, 2*modifier, 2*modifier, 2*modifier, 0.5*modifier, 0,0,0,0,0,0])**2


        self.img_width = 768
        self.img_height = 480
        self.mtx = np.array([[458.1709103887057, 0, 388.71610934773304],
                        [0, 457.99950520503074, 230.87195190807373],
                        [0, 0, 1]])
        self.dist = np.array([0.03738763890606355, -0.07371669274499101, -0.00024887228653412924, 0.0005885934089097064])

        self.camera_height = 1.2 #1.2  # meters

        self.rot_sbg_to_oak_c_imu = np.array([[-0.00364432, -0.00424146,  0.99998436],
                                              [-0.99999098, -0.00216758, -0.00365354],
                                              [ 0.00218304, -0.99998866, -0.00423352]])

        self.heading_noise_var = np.deg2rad(1) ** 2 # radians
        self.gps_position_cov = np.diag([1.0, 1.0, 2.0]) ** 2  # meters
        # self.gps_position_cov = np.diag([2.0, 2.0, 4.0]) ** 2  # meters
        # self.gps_position_cov = np.diag([0.5, 0.5, 1.0]) ** 2  # meters
        # self.gps_position_cov = np.diag([0.1, 0.1, 0.25]) ** 2  # meters


class RealWorldOAKConfigParams:
    def __init__(self):
        # Example Time Step (e.g., IMU runs at 200 Hz)
        update_rate = 178.0  # Hz
        Delta_t = 1/update_rate  # seconds

        # Oak IMU Params
        Q_g = 0.006154494275600531/np.sqrt(Delta_t)
        Q_a = 0.02995700050383457/np.sqrt(Delta_t)
        Q_bg = 4.455742618201502e-06 #* np.sqrt(Delta_t) # np.sqrt(update_rate)
        Q_ba = 0.00021405239609180207 #* np.sqrt(Delta_t) # np.sqrt(update_rate)

        Q_p = 0.0 # Standard practice for IMU mechanization state

        # Order: [Rotation (3), Velocity (3), Position (3), Gyro Bias (3), Accel Bias (3)]
        Q_diag = np.array([
            Q_g, Q_g, Q_g,    # Rotation/Attitude
            Q_a, Q_a, Q_a,    # Velocity
            Q_p, Q_p, Q_p,    # Position (Zero)
            Q_bg, Q_bg, Q_bg,  # Gyro Bias
            Q_ba, Q_ba, Q_ba   # Accel Bias
        ])
        
        self.imu_noise_params = np.diag(Q_diag)**2
        

        time = np.sqrt(1303) # seconds since boot, only applicable for this real world dataset

        # self.init_cov = np.diag([
        #     np.deg2rad(2), np.deg2rad(2), np.deg2rad(60),        # Rotation/Attitude (rad)
        #     0.5, 0.5, 0.5,        # Velocity (m/s)
        #     2.0, 2.0, 2.0,        # Position (m)
        #     # Q_bg*np.sqrt(time), Q_bg*np.sqrt(time), Q_bg*np.sqrt(time),  # Gyro Bias (rad/s)
        #     # Q_ba*np.sqrt(time), Q_ba*np.sqrt(time), Q_ba*np.sqrt(time)         # Accel Bias (m/s^2)
        #     np.deg2rad(0.005), np.deg2rad(0.005), np.deg2rad(0.005),  # Gyro Bias (rad/s)
        #     0.01, 0.07, 0.01         # Accel Bias (m/s^2)
        # ])**2
        self.init_cov = np.diag([
            np.deg2rad(2), np.deg2rad(2), np.deg2rad(60),        # Rotation/Attitude (rad)
            0.5, 0.5, 0.5,        # Velocity (m/s)
            1.0, 1.0, 1.0,        # Position (m)
            # Q_bg*np.sqrt(time), Q_bg*np.sqrt(time), Q_bg*np.sqrt(time),  # Gyro Bias (rad/s)
            np.deg2rad(0.05), np.deg2rad(0.05), np.deg2rad(0.05),  # Gyro Bias (rad/s)
            # Q_ba*np.sqrt(time), Q_ba*np.sqrt(time), Q_ba*np.sqrt(time)
            0.5, 0.5, 0.5         # Accel Bias (m/s^2)
        ])**2

        self.init_cov_high = np.diag([
            np.deg2rad(5), np.deg2rad(5), np.deg2rad(60),        # Rotation/Attitude (rad)
            0.5, 0.5, 0.5,        # Velocity (m/s)
            1.0, 1.0, 1.0,        # Position (m)
            # Q_bg*np.sqrt(time), Q_bg*np.sqrt(time), Q_bg*np.sqrt(time),  # Gyro Bias (rad/s)
            np.deg2rad(0.05), np.deg2rad(0.05), np.deg2rad(0.05),  # Gyro Bias (rad/s)
            # Q_ba*np.sqrt(time), Q_ba*np.sqrt(time), Q_ba*np.sqrt(time)
            0.5, 0.5, 0.5         # Accel Bias (m/s^2)
        ])**2

        # self.init_cov_high = np.diag([
        #     np.deg2rad(60), np.deg2rad(60), np.deg2rad(60),        # Rotation/Attitude (rad)
        #     4.0, 4.0, 4.0,        # Velocity (m/s)
        #     4.0, 4.0, 1.0,
        #     np.deg2rad(0.5), np.deg2rad(0.5), np.deg2rad(0.5),  # Gyro Bias (rad/s)
        #     0.005, 0.005, 0.005        # Accel Bias (m/s^2)
        # ])**2

        self.img_width = 768
        self.img_height = 480
        self.mtx = np.array([[458.1709103887057, 0, 388.71610934773304],
                        [0, 457.99950520503074, 230.87195190807373],
                        [0, 0, 1]])
        self.dist = np.array([0.03738763890606355, -0.07371669274499101, -0.00024887228653412924, 0.0005885934089097064])

        self.camera_height = 1.2 #1.2  # meters

        self.rot_sbg_to_oak_c_imu = np.array([[-0.00364432, -0.00424146,  0.99998436],
                                              [-0.99999098, -0.00216758, -0.00365354],
                                              [ 0.00218304, -0.99998866, -0.00423352]])

        self.heading_noise_var = np.deg2rad(1) ** 2 # radians
        # self.gps_position_cov = np.diag([2.0, 2.0, 4.0]) ** 2  # meters
        self.gps_position_cov = np.diag([1.0, 1.0, 2.0]) ** 2  # meters
        # self.gps_position_cov = np.diag([0.1, 0.1, 0.25]) ** 2  # meters

