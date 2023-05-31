"""
Boat Kinematic Controller Class
"""
from matplotlib.text import get_rotation
import numpy as np
from scipy.optimize import fsolve
import time

class BoatKinematicController:

    def __init__(self, 
                 c_r = 2/np.pi,
                 c_b = 0.1,
                 k_pos = 1, 
                 k_vel = 1,
                 k_accel = 1,
                 k_theta = 1,
                 k_theta_dot = 1, 
                 max_vel = 7,
                 max_vel_dot = 10,
                 max_delta = np.pi/2,
                 max_delta_dot = 1,
                 turn_vel = 0.5):
        self._c_r = c_r
        self._c_b = c_b
        self._k_pos = k_pos
        self._k_vel = k_vel
        self._k_accel = k_accel
        self._k_theta = k_theta
        self._k_theta_dot = k_theta_dot
        self._max_vel = max_vel
        self._max_vel_dot = max_vel_dot
        self._max_delta = max_delta
        self._max_delta_dot = max_delta_dot
        self._turn_vel = turn_vel 

    def mpc_control_vel_input(self, states, trajectory_states):
        # Get current states
        x = states[0,0]
        y = states[0,1]
        theta = states[0,2]
        delta = states[0,3]
        x_dot = states[1,0]
        y_dot = states[1,1]
        # Get Desired Trajectory States
        x_traj = trajectory_states[0,0]
        y_traj = trajectory_states[0,1]
        x_dot_traj = trajectory_states[1,0]
        y_dot_traj = trajectory_states[1,1]
        x_ddot_traj = trajectory_states[2,0]
        y_ddot_traj = trajectory_states[2,1]
        theta_traj = np.arctan2(y_dot_traj, x_dot_traj)
        vel_hat_traj = np.array([np.cos(theta_traj), np.sin(theta_traj)])
        # Evaluate vel control input
        x_pos_error = x_traj - x
        y_pos_error = y_traj - y
        x_vel_des = x_pos_error*self._k_pos + x_dot_traj
        y_vel_des = y_pos_error*self._k_pos + y_dot_traj
        x_accel_des = (x_vel_des - x_dot) * self._k_vel + x_ddot_traj
        y_accel_des = (y_vel_des - y_dot) * self._k_vel + y_ddot_traj
        vel_vec_des = np.array([x_vel_des,y_vel_des])
        vel_des = np.linalg.norm(vel_vec_des)
        min_vel = np.min((vel_des, self._turn_vel))
        vel_dir = np.array([np.cos(theta),np.sin(theta)])
        vel_com = np.max( (np.dot(vel_vec_des, vel_dir), min_vel) )
        vel_vec = np.array[x_dot,y_dot]
        vel = np.linalg.norm(vel_vec)
        accel_traj = np.array([x_ddot_traj,y_ddot_traj])
        vel_dot_traj = np.dot(accel_traj, vel_hat_traj)
        vel_dot_des = (vel_com - vel)*self._k_vel + vel_dot_traj
        vel_dot_com = np.clip(vel_dot_des, -self._max_vel_dot, self._max_vel_dot)
        # angular acceleration computation
        theta_des = np.arctan2(y_vel_des,x_vel_des)
        theta_error = self.find_angle_error(theta,theta_des)
        theta_dot_des = (x_vel_des*y_accel_des - y_vel_des*x_accel_des)/(x_vel_des**2 + y_vel_des**2)
        theta_dot_com = theta_error*self._k_theta + theta_dot_des
        delta_des = -np.arcsin(theta_dot_com*())
        return vel_dot_com, delta_dot_command
    
    def get_angle_difference(self, angle_1, angle_2):
        return (np.pi - np.abs(np.abs(angle_1-angle_2) - np.pi) )

    def find_turn_direction(self, angle, desired_angle):
        return np.sign(np.arctan2( np.sin(desired_angle - angle) , np.cos(desired_angle - angle)))
    

        



        

        
