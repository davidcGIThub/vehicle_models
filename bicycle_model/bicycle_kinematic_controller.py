"""
Bicycle Kinematic Controller Class
"""
from matplotlib.text import get_rotation
import numpy as np
from scipy.optimize import fsolve

class BicycleKinematicController:

    def __init__(self, 
                 kp_xy = 1, 
                 kd_xy = 1,
                 kp_theta = 1,
                 kd_theta = 1, 
                 kp_delta = 1,
                 v_max = 7,
                 delta_max = np.pi/4,
                 l = 0.5,
                 L = 1):
        self._kp_xy = kp_xy
        self._kd_xy = kd_xy
        self._kp_theta = kp_theta
        self._kd_theta = kd_theta
        self._kp_delta = kp_delta
        self._v_max = v_max 
        self._delta_max = delta_max
        self._l = l
        self._L = L

    def p_control(self, x, y, theta, delta, x_des, y_des, tolerance = 0.1):
        proximity = np.sqrt((x_des - x)**2  + (y_des - y)**2)
        if proximity > tolerance:
            x_dot_c = self._kp_xy*(x_des - x)
            y_dot_c = self._kp_xy*(y_des-y)
            v_c = np.clip( np.sqrt(x_dot_c**2 + y_dot_c**2), 0 , self._v_max)
            theta_d = np.arctan2(y_dot_c,x_dot_c)
            sign = self.find_turn_direction(theta,theta_d)
            theta_dot_c = self._kp_theta* sign * (np.pi - np.abs(np.abs(theta_d-theta) - np.pi) )
            def theta_dot_equation(delta):
                beta = np.arctan2(self._l * np.tan(delta) , self._L)
                f = v_c*np.cos(beta)*np.tan(delta)/self._L - theta_dot_c
                return f
            delta_d = np.clip(fsolve(theta_dot_equation, delta)[0], - self._delta_max,self._delta_max)
            delta_dot_c = np.sign(delta_d - delta) * (np.pi - np.abs(np.abs(delta_d-delta) - np.pi) ) * self._kp_delta
        else:
            v_c = 0
            delta_dot_c = 0
        return v_c, delta_dot_c

    def pd_control(self, x, y, theta, delta, x_des_states, y_des_states):
        theta = np.arctan2( np.sin(theta) , np.cos(theta))
        x_des = x_des_states[0]
        x_des_dot = x_des_states[1]
        x_des_ddot = x_des_states[2]
        y_des = y_des_states[0]
        y_des_dot = y_des_states[1]
        y_des_ddot = y_des_states[2]
        x_dot_c = self._kp_xy*(x_des - x) + x_des_dot * self._kd_xy
        y_dot_c = self._kp_xy*(y_des - y) + y_des_dot * self._kd_xy
        v_c = np.clip( np.sqrt(x_dot_c**2 + y_dot_c**2), 0 , self._v_max)
        theta_des = np.arctan2(y_dot_c,x_dot_c)
        theta_des_dot = (x_des_dot*y_des_ddot - y_des_dot*x_des_ddot) / (x_des_dot**2 + y_des_dot**2)
        sign = self.find_turn_direction(theta,theta_des)
        theta_com_dot = self._kp_theta* sign * (np.pi - np.abs(np.abs(theta_des-theta) - np.pi) ) + theta_des_dot * self._kd_theta 
        def theta_dot_equation(delta):
            beta = np.arctan2(self._l * np.tan(delta) , self._L)
            f = v_c*np.cos(beta)*np.tan(delta)/self._L - theta_com_dot 
            return f
        delta_d = np.clip(fsolve(theta_dot_equation, delta)[0],-self._delta_max,self._delta_max)
        delta_dot_c = np.sign(delta_d - delta) * (np.pi - np.abs(np.abs(delta_d-delta) - np.pi) ) * self._kp_delta
        return v_c, delta_dot_c

    def find_turn_direction(self, angle, desired_angle):
        return np.sign(np.arctan2( np.sin(desired_angle - angle) , np.cos(desired_angle - angle)))
    

        



        

        
