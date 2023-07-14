import numpy as np
from vehicle_simulator.vehicle_controllers.bspline_evaluator import BsplineEvaluator

class FixedWingSplinePathFollower:

    def __init__(self, order, distance_gain = 1, path_gain = 1, feedforward_gain = 2, feedforward_distance = 5):
        self._order = order
        self._path_gain = path_gain
        self._distance_gain = distance_gain
        self._feedforward_distance = feedforward_distance
        self._feedforward_gain = feedforward_gain
        self._spline_evaluator = BsplineEvaluator(self._order)

    def get_commands(self, control_points, position, desired_airspeed):
        scale_factor = 1
        closest_point, closest_velocity_vector, closest_acceleration_vector = \
            self._spline_evaluator.get_closest_point_and_derivatives(control_points, scale_factor, position) 
        direction_desired = self.get_desired_direction_vector(closest_point, position,
            closest_velocity_vector, closest_acceleration_vector, desired_airspeed)
        course_angle_command = np.arctan2(direction_desired.item(1), direction_desired.item(0))
        climb_rate_command = desired_airspeed * (-direction_desired.item(2))
        airspeed_command = desired_airspeed
        phi_feedforward = 0
        return np.array([course_angle_command, climb_rate_command, airspeed_command, phi_feedforward])

    def get_desired_direction_vector(self, closest_point, position, closest_velocity_vector, 
                                     closest_acceleration_vector, desired_airspeed):
        path_vector = closest_velocity_vector/np.linalg.norm(closest_velocity_vector)
        path_change_vector = closest_acceleration_vector/np.linalg.norm(closest_acceleration_vector)
        distance_vector = closest_point.flatten() - position.flatten()
        distance = np.linalg.norm(distance_vector,2)
        if distance < self._feedforward_distance:
            desired_direction_vector = distance_vector*self._distance_gain + \
                path_vector.flatten()*desired_airspeed*self._path_gain + path_change_vector.flatten() * self._feedforward_gain
        else:
            desired_direction_vector = distance_vector*self._distance_gain + \
                path_vector.flatten()*desired_airspeed*self._path_gain
        desired_direction_vector = desired_direction_vector/ np.linalg.norm(desired_direction_vector)
        return desired_direction_vector
    
    def get_order(self):
        return self._order

    

# evaluate the spline ahead of time so don't have to evaluate each time.

# add feedforward term when within tolerance of path (seek position ahead)

# add code for when outside the transition region???

#need to trim parts of path when get past certian parts of it ## maybe part of path manager???

# if the closest point is the endpoint no velocity gain.

