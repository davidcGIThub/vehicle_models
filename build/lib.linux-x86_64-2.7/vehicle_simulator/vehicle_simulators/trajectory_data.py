import numpy as np
from dataclasses import dataclass

@dataclass
class TrajectoryData:
    location_data: np.ndarray
    velocity_data: np.ndarray
    acceleration_data: np.ndarray
    jerk_data: np.ndarray
    time_data: np.ndarray
    curvature_data: np.ndarray = None
    angular_rate_data: np.ndarray = None
    centripetal_acceleration_data: np.ndarray = None
    longitudinal_acceleration_data: np.ndarray = None

    def __post_init__(self):
        location_dimension = np.shape(self.location_data)[0]
        velocity_dimension = np.shape(self.velocity_data)[0]
        acceleration_dimension = np.shape(self.acceleration_data)[0]
        jerk_dimension = np.shape(self.jerk_data)[0]
        location_length = np.shape(self.location_data)[1]
        velocity_length = np.shape(self.velocity_data)[1]
        acceleration_length = np.shape(self.acceleration_data)[1]
        jerk_length = np.shape(self.jerk_data)[1]
        time_length = len(self.time_data)
        if location_dimension != velocity_dimension or \
                velocity_dimension != acceleration_dimension or \
                acceleration_dimension != jerk_dimension:
            raise Exception("Trajectory data dimensions are not equal")
        if location_length != velocity_length or \
                velocity_length != acceleration_length or \
                acceleration_length != jerk_length or \
                jerk_length !=  time_length:
            raise Exception("Trajectory data lengths are not equal")
        cross_term_data = self.__calculate_cross_term_data()
        dot_product_data = self.__calculate_dot_product_term_data()
        velocity_magnitude_data = self.__calculate_velocity_magnitude_data()
        velocity_magnitude_data[velocity_magnitude_data < 8e-10] = 1
        self.longitudinal_acceleration_data = dot_product_data/velocity_magnitude_data
        self.centripetal_acceleration_data = cross_term_data/velocity_magnitude_data
        self.angular_rate_data = cross_term_data/velocity_magnitude_data**2
        self.curvature_data = cross_term_data/velocity_magnitude_data**3
    
    def __calculate_cross_term_data(self):
        cross_product_norm = np.abs(np.cross(self.velocity_data.T, self.acceleration_data.T).flatten())
        return cross_product_norm
    
    def __calculate_dot_product_term_data(self):
        dot_product_term = np.sum(self.acceleration_data*self.velocity_data,0)
        return dot_product_term
    
    def __calculate_velocity_magnitude_data(self):
        return np.linalg.norm(self.velocity_data,2,0)