import numpy as np
from dataclasses import dataclass

@dataclass
class VehicleMotionData:
    time_data: np.ndarray
    velocity_magnitude_data: np.ndarray
    velocity_magnitude_dot_data: np.ndarray
    heading_angle_data: np.ndarray
    heading_angular_rate_data: np.ndarray
    control_angle_data: np.ndarray = None
    control_angular_rate_data: np.ndarray = None

    def __post_init__(self):
        if self.control_angle_data is not None:
            ctrl_angle_length = len(self.control_angle_data)
        else:
            ctrl_angle_length = len(self.time_data)
        if self.control_angular_rate_data is not None:
            ctrl_angular_rate_length = len(self.control_angular_rate_data)
        else:
            ctrl_angular_rate_length = len(self.time_data)

        if len(self.time_data) != len(self.velocity_magnitude_data) or \
                len(self.velocity_magnitude_data) != len(self.velocity_magnitude_dot_data) or \
                len(self.velocity_magnitude_dot_data) != len(self.heading_angle_data) or \
                len(self.heading_angle_data) != len(self.heading_angular_rate_data) or\
                len(self.heading_angular_rate_data) != ctrl_angle_length or \
                ctrl_angle_length != ctrl_angular_rate_length:
            raise Exception("Vehicle Motion data lengths are not equal")