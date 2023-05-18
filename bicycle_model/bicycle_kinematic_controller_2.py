"""
Bicycle Kinematic Controller Class
"""
from matplotlib.text import get_rotation
import numpy as np
from scipy.optimize import fsolve

class BicycleKinematicController:

    def __init__(self, 
                 k_pos = 1, 
                 k_vel = 1,
                 k_theta = 1,
                 k_delta = 1,
                 v_max = 7,
                 delta_max = np.pi/4,
                 lr = 0.5,
                 L = 1):
        self._k_pos = k_pos
        self._k_vel = k_vel
        self._k_theta = k_theta
        self._k_delta = k_delta
        self._v_max = v_max
        self._delta_max = delta_max
        self._lr = lr
        self._L = L

    def model_predictive_controller(self, states, desired_states):
        # current states
        x = states[0,0]
        y = states[0,1]
        theta = states[0,2]
        delta = states[0,3]
        x_dot = states[1,0]
        y_dot = states[1,1]
        theta_dot = states[1,2]
        #desired_states
        x_path = desired_states[0,0]
        y_path = desired_states[0,1]
        x_dot_path = desired_states[1,0]
        y_dot_path = desired_states[1,1]
        x_ddot_path = desired_states[2,0]
        y_ddot_path = desired_states[2,1]
        #command computations
        x_pos_error = x_path - x
        y_pos_error = y_path - y
        x_vel_des = x_pos_error*self._k_pos + x_dot_path
        y_vel_des = y_pos_error*self._k_pos + y_dot_path
        x_accel_des = (x_vel_des - x_dot) * self._k_vel + x_ddot_path
        y_accel_des = (y_vel_des - y_dot) * self._k_vel + y_ddot_path
        vel_vec_des = np.array([x_vel_des,y_vel_des])
        vel_des = np.linalg.norm(vel_vec_des)
        accel_vec_des = np.array([x_accel_des,y_accel_des])
        vel_dot_command = np.dot(vel_vec_des,accel_vec_des) / vel_des
        chi_des = np.arctan2(y_vel_des, x_vel_des)
        beta_des = self.get_closest_angle(theta, chi_des) * self.find_turn_direction(theta, chi_des)
        delta_des = np.clip(np.arctan2(self._L*np.tan(beta_des), self._lr), -self._delta_max , self._delta_max) 
        beta_dot_des = (y_accel_des - vel_dot_command*np.sin(beta_des + theta))/(vel_des*np.cos(beta_des + theta)) - theta_dot
        delta_dot_des = beta_dot_des*((self._lr**2)*np.tan(delta_des)**2 + self._L**2)*(np.cos(delta_des)**2)/(self._L*self._lr)
        delta_error = self.get_closest_angle(delta, delta_des) * np.sign(delta_des - delta)
        delta_dot_command = delta_dot_des + delta_error * self._k_delta
        return vel_dot_command, delta_dot_command

    def find_turn_direction(self, angle, desired_angle):
        return np.sign(np.arctan2( np.sin(desired_angle - angle) , np.cos(desired_angle - angle)))
    
    def get_closest_angle(self, angle, angle_des):
        (np.pi - np.abs(np.abs(angle_des-angle) - np.pi) )

    

        



        

        
