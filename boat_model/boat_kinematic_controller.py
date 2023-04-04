"""
Boat Kinematic Controller Class
"""
from matplotlib.text import get_rotation
import numpy as np
from scipy.optimize import fsolve
import time

class BoatKinematicController:

    def __init__(self, 
                 kpos_xy = 1, 
                 kvel_xy = 1,
                 k_ang_theta = 1,
                 k_ang_rate_theta = 1, 
                 kpos_delta = 1,
                 a_max = 5,
                 v_max = 7,
                 delta_max = np.pi/2,
                 v_coast = 0.2):
        self._kpos_xy = kpos_xy
        self._kvel_xy = kvel_xy
        self._k_ang_theta = k_ang_theta
        self._k_ang_rate_theta = k_ang_rate_theta
        self._kpos_delta = kpos_delta
        self.a_max = a_max
        self._v_max = v_max 
        self._delta_max = delta_max
        self._v_coast = v_coast

    def pos_control(self, x, y, theta, x_dot, y_dot, x_des, y_des, tolerance = 0.1):
        x_error = (x_des - x)
        y_error = (y_des - y)
        proximity = np.sqrt((x_error)**2  + (y_error)**2)
        if proximity > tolerance:
            x_dot_d = self._kpos_xy*(x_des - x)
            y_dot_d = self._kpos_xy*(y_des - y)
            theta_d = np.arctan2(y_dot_d,x_dot_d)
            in_line = np.cos(self.get_angle_difference(theta_d, theta))
            in_line = np.clip(in_line,0,1)
            velocity_des = np.sqrt(x_dot_d**2 + y_dot_d**2)
            velocity_des = np.max((self._v_coast, velocity_des*in_line))
            velocity = np.sqrt(x_dot**2 + y_dot**2)
            vel_error = velocity_des - velocity
            accel_c = np.clip(vel_error, -self.a_max, self.a_max)
            sign = self.find_turn_direction(theta,theta_d)
            theta_dot_d = self._k_ang_theta* sign * self.get_angle_difference(theta_d, theta)
            temp = np.clip(theta_dot_d*velocity_des, -1, 1)
            delta_c = -np.arcsin(temp)
            if velocity < self._v_coast:
                delta_c = 0
        else:
            accel_c = 0
            delta_c = 0
        return accel_c, delta_c

    def pd_control(self, x, y, theta, delta, x_des_states, y_des_states):
        theta = np.arctan2( np.sin(theta) , np.cos(theta))
        x_des = x_des_states[0]
        x_des_dot = x_des_states[1]
        x_des_ddot = x_des_states[2]
        y_des = y_des_states[0]
        y_des_dot = y_des_states[1]
        y_des_ddot = y_des_states[2]
        x_dot_c = self._kpos_xy*(x_des - x) + x_des_dot * self._kvel_xy
        y_dot_c = self._kpos_xy*(y_des - y) + y_des_dot * self._kvel_xy
        v_c = np.clip( np.sqrt(x_dot_c**2 + y_dot_c**2), 0 , self._v_max)
        theta_des = np.arctan2(y_dot_c,x_dot_c)
        theta_des_dot = (x_des_dot*y_des_ddot - y_des_dot*x_des_ddot) / (x_des_dot**2 + y_des_dot**2)
        sign = self.find_turn_direction(theta,theta_des)
        theta_com_dot = self._kpos_theta* sign * (np.pi - np.abs(np.abs(theta_des-theta) - np.pi) ) + theta_des_dot * self._kvel_theta 
        def theta_dot_equation(delta):
            beta = np.arctan2(self._l * np.tan(delta) , self._L)
            f = v_c*np.cos(beta)*np.tan(delta)/self._L - theta_com_dot 
            return f
        delta_d = np.clip(fsolve(theta_dot_equation, delta)[0],-self._delta_max,self._delta_max)
        delta_dot_c = np.sign(delta_d - delta) * (np.pi - np.abs(np.abs(delta_d-delta) - np.pi) ) * self._kpos_delta
        return v_c, delta_dot_c
    
    def get_angle_difference(self, angle_1, angle_2):
        return (np.pi - np.abs(np.abs(angle_1-angle_2) - np.pi) )

    def find_turn_direction(self, angle, desired_angle):
        return np.sign(np.arctan2( np.sin(desired_angle - angle) , np.cos(desired_angle - angle)))
    

        



        

        
