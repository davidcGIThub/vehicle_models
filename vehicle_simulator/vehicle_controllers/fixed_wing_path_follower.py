import numpy as np
from vehicle_simulator.vehicle_controllers.bspline_evaluator import BsplineEvaluator

class FixedWingSplinePathFollower:

    def __init__(self, order, distance_p_gain = 2, distance_i_gain = 0.1, distance_d_gain = 2,
                  path_direction_gain = 2, feedforward_gain = 0.5, feedforward_distance = 5, start_position = np.zeros(3)):
        self._order = order
        self._path_direction_gain = path_direction_gain
        self._distance_p_gain = distance_p_gain
        self._distance_i_gain = distance_i_gain
        self._distance_d_gain = distance_d_gain
        self._feedforward_distance = feedforward_distance
        self._feedforward_gain = feedforward_gain
        self._spline_evaluator = BsplineEvaluator(self._order)
        self._integrator_term = np.zeros(3)
        self._prev_position = start_position

    def get_commands_from_bspline(self, control_points, vehicle_position, desired_airspeed, dt):
        scale_factor = 1
        closest_positon, closest_velocity, closest_acceleration = \
            self._spline_evaluator.get_closest_point_and_derivatives(control_points, scale_factor, vehicle_position) 
        closest_curvature = np.linalg.norm(np.cross(closest_velocity.flatten(), closest_acceleration.flatten())) \
                            / np.linalg.norm(closest_velocity.flatten(),2)**3
        commands = self.get_commands_from_closest_point(closest_positon, closest_velocity, 
                                                        closest_acceleration, closest_curvature,
                                                        vehicle_position, desired_airspeed, dt)
        return commands


    def get_commands_from_closest_point(self, position, velocity, acceleration, curvature, vehicle_position, desired_speed, dt):
        direction_desired = self.get_desired_direction_vector(position, vehicle_position, velocity, acceleration, curvature, dt)
        course_angle_command = np.arctan2(direction_desired.item(1), direction_desired.item(0))
        climb_rate_command = desired_speed * (-direction_desired.item(2))
        airspeed_command = desired_speed
        phi_feedforward = 0
        return np.array([course_angle_command, climb_rate_command, airspeed_command, phi_feedforward])

    def get_desired_direction_vector(self, closest_point, vehicle_position, closest_velocity_vector, 
                                     closest_acceleration_vector, curvature, dt):
        norm_vel = np.linalg.norm(closest_velocity_vector)
        path_direction_vector = (closest_velocity_vector/norm_vel).flatten()
        if (np.linalg.norm(closest_acceleration_vector) < 0.00001 or curvature == 0):
            path_change_vector = np.zeros(3)
        else:
            accel_proj_onto_vel = np.dot(closest_acceleration_vector.flatten(), path_direction_vector) * path_direction_vector
            path_change_vector = closest_acceleration_vector.flatten() - accel_proj_onto_vel.flatten()
            path_change_vector = path_change_vector/np.linalg.norm(path_change_vector)
        error_vector = closest_point.flatten() - vehicle_position.flatten()
        error_direction = error_vector / np.linalg.norm(error_vector)
        prev_error_vector = closest_point.flatten() - self._prev_position.flatten()
        distance = np.linalg.norm(error_vector,2)
        derivative_vector = (vehicle_position.flatten() - self._prev_position.flatten())/dt
        derivative_vector_onto_path_direction = np.dot(derivative_vector.flatten(), path_direction_vector) * path_direction_vector
        derivative_vector_lateral = (derivative_vector - derivative_vector_onto_path_direction).flatten()
        
        if distance < self._feedforward_distance:
            self._integrator_term += dt*(error_vector + prev_error_vector)/2
            # integrator_vector = np.dot(self._integrator_term, error_direction)*error_direction
            integrator_vector = self._integrator_term
            # self._integrator_term += np.dot(dt*(error_vector + prev_error_vector)/2, error_direction)*error_direction
            # integrator_vector = self._integrator_term
            desired_direction_vector = \
                  error_vector.flatten()              *  self._distance_p_gain \
                + integrator_vector.flatten()         *  self._distance_i_gain \
                - derivative_vector_lateral.flatten() *  self._distance_d_gain \
                + path_direction_vector.flatten()*self._path_direction_gain \
                + path_change_vector.flatten() * self._feedforward_gain * curvature
        else:
            desired_direction_vector = \
                  error_vector.flatten() *  self._distance_p_gain \
                - derivative_vector_lateral.flatten() *  self._distance_d_gain \
                + path_direction_vector.flatten()*self._path_direction_gain
            self._integrator_term = np.zeros(3)
        desired_direction_vector = desired_direction_vector/ np.linalg.norm(desired_direction_vector)
        self._prev_position = vehicle_position
        return desired_direction_vector
    
    def get_order(self):
        return self._order
    

    def get_commands_from_dubins(self, position_points, tangent_points, perpindicular_points, curvature_points,
                                 vehicle_position, desired_airspeed, dt):
        distances = np.linalg.norm(vehicle_position.flatten()[:,None] - position_points,2,0)
        closest_point_index = np.argmin(distances)
        closest_position = position_points[:,closest_point_index].flatten()[:,None]
        closest_velocity = tangent_points[:,closest_point_index].flatten()[:,None]
        closest_acceleration = perpindicular_points[:,closest_point_index].flatten()[:,None]
        closest_curvature = curvature_points[closest_point_index]
        commands = self.get_commands_from_closest_point(closest_position, closest_velocity,
                                                         closest_acceleration, closest_curvature,
                                                         vehicle_position, desired_airspeed, dt)
        return commands

    

# evaluate the spline ahead of time so don't have to evaluate each time.

# add feedforward term when within tolerance of path (seek position ahead)

# add code for when outside the transition region???

#need to trim parts of path when get past certian parts of it ## maybe part of path manager???

# if the closest point is the endpoint no velocity gain.

