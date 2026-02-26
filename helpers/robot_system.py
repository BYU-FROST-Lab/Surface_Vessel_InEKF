import numpy as np
from scipy.linalg import expm
from numpy.linalg import inv
import warnings 
from scipy.spatial.transform import Rotation as Rotation


class WAMVASV:
    def __init__(self, Q):
        """Our system for a quadcopter, with IMU measurements as controls. 

        Args:
            filename      (str) : Where the recorded data has been stored
            Q   (15,15 ndarray) : Covariance of noise on state
            R     (6,6 ndarray) : Covariance of noise on measurements
            dvl_p   (3 ndarray) : Translation of DVL from IMU
            dvl_r (3,3 ndarray) : Rotation matrix of DVL from IMU"""
        self.Q = Q
        # self.R = R
        
        # self.Hz = self.data['ticks']
        # self.T = self.data['x'].shape[0]
        # local velocity
        # self.b1 = np.array([0, 0, 0, -1, 0])
        # global depth
        # self.b2 = np.array([0, 0, 0, 0, 1])

        # convert z1 noise into the correct frame
        # self.R[:3,:3] = self.dvl_r@self.R[:3,:3]@self.dvl_r.T + self.dvl_p@(self.Q[0:3,0:3] + self.Q[9:12,9:12])@self.dvl_p.T

    def f_lie(self, state, u, dt, bias, gravity=True):
        """Propagates state forward in Lie Group. Used for IEKF.

        Args:
            state (5,5 ndarray) : X_n of model in Lie Group
            u     (2,3 ndarray) : U_n of model (IMU measurements)
            bias  (2,3 ndarray) : Estimated bias

        Returns:
            X_{n+1} (5,5 ndarray)"""
        # transform u into the right representation
        # get stuff we need
        # u[1][0] *= -1 # flipping x accel
        # u[1][1] *= -1 # flipping y accel

        # g = np.zeros(3) # Trying to figure out errors
        R_90_ccw = Rotation.from_euler('z', 90, degrees=True).as_matrix()
        if gravity:
            g = np.array([0,0,-9.8])
        else:
            g = np.zeros(3)

        R = state[:3,:3]
        v = state[:3,3]
        p = state[:3,4]

        # Direct Manipulation
        omega = u[0] - bias[0]
        a = u[1] - bias[1]
        ## put it together
        Rnew = R @ expm( self.cross( omega*dt ))
        # Rnew = R + np.eye(3) + self.cross( omega*dt )

        vnew = v + (R@a + g)*dt
        pnew = p + v*dt + (R@a + g)*dt**2/2
        og_X = np.block([[Rnew, vnew.reshape(-1,1), pnew.reshape(-1,1)],
                        [np.zeros((2,3)), np.eye(2)]])

        # U composition
        # dR = expm( self.cross( omega*dt ))
        # dR = np.eye(3) + self.cross( omega*dt )

        # dv = (a + g)*dt
        # dp = 0.5 * (a+g)*dt**2
        # U = np.eye(5)
        # U[:3,:3] = dR
        # U[:3,3] = dv.reshape(-1)
        # U[:3,4] = dp.reshape(-1)
        # new_x = state @ U

        # R_rot = R @ R_90_ccw # Trying to figure out errors
        # a[0], a[1] = a[1], -a[0] # Trying to figure out errors
        # a[0], a[1] = -a[1], a[0] # Trying to figure out errors
        # a[1] *= -1 # Trying to figure out errors
        # a = np.zeros(3) # Trying to figure out errors
        
        # vnew = v + (a + g)*dt # Trying to figure out errors # Shouldn't need this because imu is now in NWU frame
        # pnew = p + v*dt + (a + g)*dt**2/2 # Trying to figure out errors

        # return new_x
        return og_X

    @staticmethod
    def cross(x):
        """Moves a 3 vector into so(3)

        Args:
            x (3 ndarray) : Parametrization of Lie Algebra

        Returns:
            x (3,3 ndarray) : Element of so(3)"""
        return np.array([[   0, -x[2],  x[1]],
                        [ x[2],     0, -x[0]],
                        [-x[1],  x[0],     0]])

    @staticmethod
    def carat(xi):
        """Moves a 9 vector to the Lie Algebra se_2(3).

        Args:
            xi (9 ndarray) : Parametrization of Lie algebra

        Returns:
            xi^ (5,5 ndarray) : Element in Lie Algebra se_2(3)"""
        w_cross = WAMVASV.cross(xi[0:3])
        v       = xi[3:6].reshape(-1,1)
        p       = xi[6:9].reshape(-1,1)
        return np.block([[w_cross, v, p],
                         [np.zeros((2,5))]])
    
    @staticmethod
    def adjoint(xi):
        """Takes adjoint of element in SE_2(3)

        Args:
            xi (5,5 ndarray) : Element in Lie Group

        Returns:
            Ad_xi (9,9 ndarray) : Adjoint in SE_2(3)"""
        R = xi[:3,:3]
        v_cross = WAMVASV.cross(xi[:3,3])
        p_cross = WAMVASV.cross(xi[:3,4])
        zero    = np.zeros((3,3))
        return np.block([[        R, zero, zero],
                         [v_cross@R,    R, zero],
                         [p_cross@R, zero,   R]])
