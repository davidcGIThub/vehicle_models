"""
Unicycle Kinematic Controller Class
"""

import numpy as np
from scipy.optimize import fsolve

class UnicycleKinematicWaypointTracker:

    def __init__(self, 
                 kp_xy = 1, 
                 kd_xy = 1,
                 kp_theta = 1,
                 kd_theta = 1, 
                 v_max = 7,
                 omega_max = np.pi/4,
                 tolerance = 0.05):
        self._kp_xy = kp_xy
        self._kd_xy = kd_xy
        self._kp_theta = kp_theta
        self._kd_theta = kd_theta
        self._v_max = v_max 
        self._omega_max = omega_max
        self._tolerance = tolerance
        self._
        self._prev_v = 0
        self._prev_omega = 0

    def waypoint_tracker(self, states, states_previous, states_desired, time_step):
        x = states[0]
        y = states[1]
        theta = states[2]
        x_prev = states_previous[0]
        y_prev = states_previous[1]
        theta_prev = states[2]
        x_des = states_desired[0]
        y_des = states_desired[1]
        x_vel_command = self.pd_position_control(x, x_prev, x_des, time_step)
        y_vel_command = self.pd_position_control(y, y_prev, y_des, time_step)
        vel_command = np.sqrt(x_vel_command**2 + y_vel_command**2)
        vel_command_sat = np.clip( vel_command, 0 , self._v_max)
        theta_des = np.arctan2(y_vel_command,x_vel_command)
        angular_vel_command = self.pd_angular_control(theta,theta_prev,theta_des,time_step)
        return vel_command_sat, angular_vel_command

    def pd_position_control(self, pos, previous_pos, desired_pos, time_step):
        error = desired_pos - pos
        derivative = (pos - previous_pos)/time_step
        vel_command = error * self._kp_xy - derivative*self._kd_xy
        return vel_command

    def pd_angular_control(self,theta,theta_prev,theta_des,time_step):
        theta_error = self.find_closest_angle_and_direction(theta,theta_des)
        theta_deriv = self.find_closest_angle_and_direction(theta_prev,theta)/time_step
        angular_vel_command = theta_error*self._kp_theta - theta_deriv*self._kd_theta
        angular_vel_command_sat = np.clip(angular_vel_command , -self._omega_max , self._omega_max)
        return angular_vel_command_sat


    # def pd_control(self, x, y, theta, delta, x_des_states, y_des_states):
    #     theta = np.arctan2( np.sin(theta) , np.cos(theta))
    #     x_des = x_des_states[0]
    #     x_des_dot = x_des_states[1]
    #     x_des_ddot = x_des_states[2]
    #     y_des = y_des_states[0]
    #     y_des_dot = y_des_states[1]
    #     y_des_ddot = y_des_states[2]
    #     x_dot_c = self._kp_xy*(x_des - x) + x_des_dot * self._kd_xy
    #     y_dot_c = self._kp_xy*(y_des - y) + y_des_dot * self._kd_xy
    #     v_c = np.clip( np.sqrt(x_dot_c**2 + y_dot_c**2), 0 , self._v_max)
    #     theta_des = np.arctan2(y_dot_c,x_dot_c)
    #     theta_des_dot = (x_des_dot*y_des_ddot - y_des_dot*x_des_ddot) / (x_des_dot**2 + y_des_dot**2)
    #     sign = self.find_turn_direction(theta,theta_des)
    #     theta_com_dot = self._kp_theta* sign * (np.pi - np.abs(np.abs(theta_des-theta) - np.pi) ) + theta_des_dot * self._kd_theta 
    #     def theta_dot_equation(delta):
    #         beta = np.arctan2(self._l * np.tan(delta) , self._L)
    #         f = v_c*np.cos(beta)*np.tan(delta)/self._L - theta_com_dot 
    #         return f
    #     delta_d = np.clip(fsolve(theta_dot_equation, delta)[0],-self._delta_max,self._delta_max)
    #     delta_dot_c = np.sign(delta_d - delta) * (np.pi - np.abs(np.abs(delta_d-delta) - np.pi) ) * self._kp_delta
    #     return v_c, delta_dot_c

    def find_closest_angle_and_direction(self, angle, desired_angle):
        direction = np.sign(np.arctan2( np.sin(desired_angle - angle) , np.cos(desired_angle - angle)))
        angle = (np.pi - np.abs(np.abs(desired_angle-angle) - np.pi) )
        return angle*direction
    

        



        

        
