"""
Unicycle Kinematic Trajectory Tracker Class
"""

from debugpy import trace_this_thread
import numpy as np
from scipy.optimize import fsolve

class UnicycleKinematicTrajectoryTracker:

    def __init__(self, 
                 dt = 0.1,
                 kp_p = 1, 
                 kp_i = 1,
                 kv_p = 1,
                 ktheta_p = 1, 
                 v_max = 7,
                 omega_max = np.pi/4,
                 tolerance = 0.05):
        self._dt = dt
        self._kp_p = kp_p
        self._kp_i = kp_i
        self._kv_p = kv_p
        self._ktheta_p = ktheta_p
        self._v_max = v_max 
        self._omega_max = omega_max
        self._tolerance = tolerance
        self._previous_commands = np.array([0,0])
        self._previous_states = np.array([0,0,0.0,0.0,0.0,0.0])
        self._previous_desired_states = np.array([0.0,0.0,0.0,0.0,0.0,0.0])
        self._integrators = np.array([0.0,0.0])

    def trajectory_tracker(self, states, states_desired):
        x_vel_command, y_vel_command = self.get_xy_velocity_command(states, states_desired)
        velocity_command = self.get_velocity_command(x_vel_command, y_vel_command,states)
        theta_des = self.get_theta_desired(x_vel_command, y_vel_command)
        angular_vel_command = self.p_angular_control(states, theta_des)
        self._previous_commands = np.array([velocity_command, angular_vel_command])
        self._previous_states = states
        self._previous_desired_states = states_desired
        return velocity_command, angular_vel_command

    def set_previous_states(self, states):
        self._previous_states = states

    def set_previous_desired_states(self, states):
        self._previous_desired_states = states

    def set_previous_commands(self, commands):
        self._previous_commands = commands

    def get_xy_velocity_command(self,states,states_desired):
        x = states[0]
        y = states[1]
        x_des = states_desired[0]
        y_des = states_desired[1]
        x_dot_des = states_desired[3]
        y_dot_des = states_desired[4]
        # x_p_control_term = self.p_control(x, x_des, self._kp_p)
        # y_p_control_term = self.p_control(y, y_des, self._kp_p)
        x_prev = self._previous_states[0]
        y_prev = self._previous_states[1]
        x_des_prev = self._previous_desired_states[0]
        y_des_prev = self._previous_desired_states[1]
        x_p_control_term = self.pi_control(x, x_prev, x_des, x_des_prev, self._kp_p, self._kp_i, 0)
        y_p_control_term = self.pi_control(y, y_prev, y_des, y_des_prev, self._kp_p, self._kp_i, 1)
        x_vel_command = x_p_control_term + x_dot_des*self._kv_p
        y_vel_command = y_p_control_term + y_dot_des*self._kv_p
        return x_vel_command, y_vel_command

    def get_velocity_command(self,x_vel_command, y_vel_command, states):
        theta = states[2]
        desired_velocity_vector = np.array([x_vel_command, y_vel_command])
        current_direction = np.array([np.cos(theta), np.sin(theta)])
        command_velocity_vector = np.dot(current_direction, desired_velocity_vector)
        vel_command = np.linalg.norm(command_velocity_vector)
        vel_command_sat = np.clip( vel_command, 0 , self._v_max)
        return vel_command_sat

    def get_theta_desired(self, x_vel_command,y_vel_command):
        return np.arctan2(y_vel_command,x_vel_command)

    def p_control(self, state, desired_state, gain):
        error = desired_state - state
        command = error * gain
        return command

    def pi_control(self, state, previous_state, desired_state, previous_desired_state, p_gain, i_gain, integrator_index):
        p_term = self.p_control(state, desired_state, p_gain)
        error = desired_state - state
        error_prev = previous_desired_state - previous_state
        integrator = self._integrators[integrator_index]
        integrator_vector = np.zeros(2)
        integrator_vector[integrator_index] = integrator
        command = p_term + integrator
        # if (self._previous_commands[0] > self._v_max) and (np.sign(error + error_prev) == np.sign(integrator)):
        #     pass
        if np.abs(error) < self._tolerance:
            self._integrators[integrator_index] = 0
        else:
            new_integrator = (self._dt * i_gain / 2) * (error + error_prev) + integrator
            self._integrators[integrator_index] = new_integrator
        # print("new_integrator: " , new_integrator)
        # print("integrators: " , self._integrators)
        return command

    def p_angular_control(self,states, theta_des):
        theta = states[2]
        theta_error = self.find_closest_angle_and_direction(theta,theta_des)
        angular_vel_command = theta_error*self._ktheta_p
        # angular_vel_command = self.low_pass_filter(angular_vel_command, self._previous_commands[1], 0.8)
        angular_vel_command_sat = np.clip(angular_vel_command , -self._omega_max , self._omega_max)
        print("angular_vel_command_sat: " , angular_vel_command_sat)
        return angular_vel_command_sat

    def find_closest_angle_and_direction(self, angle, desired_angle):
        direction = np.sign(np.arctan2( np.sin(desired_angle - angle) , np.cos(desired_angle - angle)))
        angle = (np.pi - np.abs(np.abs(desired_angle-angle) - np.pi) )
        return angle*direction

    def low_pass_filter(self, value, value_prev, alpha):
        alpha = np.clip(alpha,0,1)
        new_value = value*alpha + (1-alpha)*value_prev
        return new_value
    

        



        

        
