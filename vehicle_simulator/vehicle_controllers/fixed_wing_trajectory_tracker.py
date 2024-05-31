import numpy as np
from vehicle_simulator.vehicle_controllers.bspline_evaluator import BsplineEvaluator
from vehicle_simulator.vehicle_models.fixed_wing_model import FixedWingParameters

class FixedWingTrajectoryTracker:

    def __init__(self, order, p_gain = 2, i_gain = 0.1, d_gain = 2,
                  feedforward_tolerance = 5, integrator_tolerance = 5, start_position = np.zeros(3),
                  fixed_wing_parameters = FixedWingParameters()):
        self._order = order
        self._p_gain = p_gain
        self._i_gain = i_gain
        self._d_gain = d_gain
        self._feedforward_tolerance = feedforward_tolerance
        self._integrator_tolerance = integrator_tolerance
        self._spline_evaluator = BsplineEvaluator(self._order)
        self._integrator_term = np.zeros(3)
        self._prev_position = start_position
        self._prev_time = 0
        self._fixed_wing_parameters = fixed_wing_parameters

    def get_commands_from_bspline(self, control_points, scale_factor, start_knot, vehicle_state, t):
        position_traj, velocity_traj, acceleration_traj = self._spline_evaluator.get_position_and_derivatives(
            control_points, start_knot, scale_factor, t)
        commands = self.get_commands(position_traj, velocity_traj, acceleration_traj, vehicle_state, t)
        self._prev_time = t
        return commands

    def get_commands(self, position_traj, velocity_traj, acceleration_traj, vehicle_state, t):
        dt = t - self._prev_time
        vehicle_position = vehicle_state.flatten()[0:3]
        vehicle_velocity_magnitude = np.linalg.norm(vehicle_state.flatten()[3:6])
        desired_velocity_vector, distance_error_magnitude = \
            self.get_desired_velocity_vector(position_traj, velocity_traj, vehicle_position, dt)
        # roll_feedforward = self.get_roll_feedforward(velocity_traj, acceleration_traj, \
        #                                              vehicle_velocity_magnitude, distance_error_magnitude)
        # throttle_feedforward = self.get_throttle_feedforward( velocity_traj, acceleration_traj, \
        #                                                      vehicle_velocity_magnitude, distance_error_magnitude)
        roll_feedforward = 0
        throttle_feedforward = 0
        course_angle_command = np.arctan2(desired_velocity_vector.item(1), desired_velocity_vector.item(0))
        airspeed_command = np.linalg.norm(desired_velocity_vector)
        climb_rate_command =  -desired_velocity_vector.item(2)
        self._prev_position = vehicle_position
        return np.array([course_angle_command, climb_rate_command, airspeed_command, \
                         roll_feedforward, throttle_feedforward])

    def get_desired_velocity_vector(self, position_traj, velocity_traj, vehicle_position, dt):
        path_direction = velocity_traj.flatten()/np.linalg.norm(velocity_traj)
        error_vector = position_traj.flatten() - vehicle_position.flatten()
        prev_error_vector = position_traj.flatten() - self._prev_position.flatten()
        distance_error_magnitude = np.linalg.norm(error_vector)
        if dt == 0:
            derivative_vector_lateral = np.zeros(3)
        else:
            derivative_vector = (vehicle_position.flatten() - self._prev_position.flatten())/dt
            derivative_vector_onto_path_direction = np.dot(derivative_vector.flatten(), path_direction) * path_direction
            derivative_vector_lateral = (derivative_vector - derivative_vector_onto_path_direction).flatten()
        if distance_error_magnitude < self._integrator_tolerance:
            self._integrator_term += dt*(error_vector + prev_error_vector)/2
            desired_velocity_vector = \
                  error_vector.flatten()              *  self._p_gain \
                + self._integrator_term.flatten()     *  self._i_gain \
                - derivative_vector_lateral.flatten() *  self._d_gain \
                + velocity_traj.flatten()
        else:
            desired_velocity_vector = \
                  error_vector.flatten()              *  self._p_gain \
                - derivative_vector_lateral.flatten() *  self._d_gain \
                + velocity_traj.flatten()
            self._integrator_term = np.zeros(3)
        return desired_velocity_vector, distance_error_magnitude

    def get_roll_feedforward(self, velocity_traj, acceleration_traj, vehicle_velocity, distance_error_magnitude):
        gravity = self._fixed_wing_parameters.gravity
        planar_velocity = velocity_traj.flatten()[0:2]
        planar_acceleration = acceleration_traj.flatten()[0:2]
        cross_product = np.cross(planar_velocity, planar_acceleration)
        planar_radius = np.linalg.norm(planar_velocity,2)**3 \
            / np.linalg.norm(cross_product)
        direction = np.sign(cross_product.item(0))
        if distance_error_magnitude < self._feedforward_tolerance:
            roll_feedforward = direction * np.arctan(vehicle_velocity** 2 / gravity / planar_radius)
        else:
            roll_feedforward = 0
        return roll_feedforward
    
    def get_throttle_feedforward(self, velocity_traj, acceleration_traj, vehicle_velocity, distance_error_magnitude):
        if distance_error_magnitude < self._feedforward_tolerance:
            # Trajectory Data
            path_direction = velocity_traj.flatten()/np.linalg.norm(velocity_traj)
            tangential_acceleration = np.dot(acceleration_traj.flatten(),path_direction)
            # Gather parameters
            mass = self._fixed_wing_parameters.mass
            Va = vehicle_velocity
            C_T2 = self._fixed_wing_parameters.C_T2
            rho = self._fixed_wing_parameters.rho
            D_prop = self._fixed_wing_parameters.D_prop
            C_Q2 = self._fixed_wing_parameters.C_Q2
            KQ = self._fixed_wing_parameters.KQ
            R_motor = self._fixed_wing_parameters.R_motor
            i0 = self._fixed_wing_parameters.i0
            V_max = self._fixed_wing_parameters.V_max
            # Calculations
            thrust_prop = mass*np.abs(tangential_acceleration)
            c = rho*C_T2*(D_prop**2)*(Va**2) - thrust_prop
            v_in = (c - C_Q2*rho*np.power(D_prop, 3)*(Va**2) - KQ*i0) * (-R_motor/KQ)
            throttle_ffwd = v_in * np.sign(tangential_acceleration) / V_max 
        else:
            throttle_ffwd = 0
        return throttle_ffwd

    def get_order(self):
        return self._order