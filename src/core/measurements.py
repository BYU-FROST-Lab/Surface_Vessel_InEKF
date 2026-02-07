# This file defines all the measurement data structures used in the SLAM system.
# The purpose of moving away from ROS message types is to have a clean separation
# between the SLAM core logic and the ROS middleware, facilitating testing and easing
# the transition to C++ at a later date.

from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class ImuMeasurement:
    timestamp_sec: int
    timestamp_nsec: int
    gyro: np.ndarray   # Shape (3,)
    accel: np.ndarray  # Shape (3,)

@dataclass(frozen=True)
class GPSMeasurement: # GPS measurement converted to XYZ
    timestamp_sec: int
    timestamp_nsec: int
    x: float
    y: float
    z: float
    position_cov: np.ndarray  # Shape (3, 3)

@dataclass(frozen=True)
class HeadingMeasurement:
    timestamp_sec: int
    timestamp_nsec: int
    heading: float  # In radians
    heading_cov: float  # Variance

@dataclass(frozen=True)
class HorizonObservation:
    valid: bool
    timestamp_sec: int
    timestamp_nsec: int
    roll_rad: float
    pitch_rad: float
    roll_var: float  # Variance
    pitch_var: float  # Variance

# Might use for robot state estimation tracking
@dataclass(frozen=True)
class NavStateMeasurement:
    timestamp_sec: int
    timestamp_nsec: int
    position: np.ndarray  # Shape (3,)
    velocity: np.ndarray  # Shape (3,)
    orientation: np.ndarray  # Quaternion as (x, y, z, w)
    state_cov: np.ndarray  # Shape (9, 9) for position, velocity, orientation covariance

# Used for ground truth
@dataclass(frozen=True)
class PoseMeasurement:
    timestamp_sec: int
    timestamp_nsec: int
    position: np.ndarray  # Shape (3,)
    orientation: np.ndarray  # Quaternion as (x, y, z, w)
    pose_cov: np.ndarray  # Shape (6, 6) for position and orientation covariance
    