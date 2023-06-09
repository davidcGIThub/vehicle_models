"""
Unicycle Kinematic Trajectory Tracker Class
"""

from debugpy import trace_this_thread
import numpy as np
from scipy.optimize import fsolve

class UnicycleTrajectoryTracker:

    def __init__(self, 
                 k_pos = 1, 
                 k_vel = 1,
                 k_accel = 1,
                 k_theta = 1,
                 location_fwd_tol = 1,
                 heading_ffwd_tol = 0.3,
                 max_vel = 7,
                 max_vel_dot = 10,
                 max_theta_dot = 5,
                 max_theta_ddot = 5):
        self._k_pos = k_pos
        self._k_vel = k_vel
        self._k_accel = k_accel
        self._k_theta = k_theta
        self._max_vel = max_vel
        self._max_vel_dot = max_vel_dot
        self._max_theta_dot = max_theta_dot 
        self._max_theta_ddot = max_theta_ddot 
        self._location_fwd_tol = location_fwd_tol,
        self._heading_ffwd_tol = heading_ffwd_tol,

    def mpc_control_accel_input(self, inputs, states, trajectory_states):
        #### Data Extraction ####
        # current states
        x = states[0,0]
        y = states[0,1]
        theta = states[0,2]
        x_dot = states[1,0]
        y_dot = states[1,1]
        theta_dot = states[1,2]
        # desired trajectory states
        x_traj = trajectory_states[0,0]
        y_traj = trajectory_states[0,1]
        x_dot_traj = trajectory_states[1,0]
        y_dot_traj = trajectory_states[1,1]
        x_ddot_traj = trajectory_states[2,0]
        y_ddot_traj = trajectory_states[2,1]
        #### longitudinal acceleration computation ####
        x_error = x_traj - x
        y_error = y_traj - y
        x_dot_error = x_dot_traj - x_dot
        y_dot_error = y_dot_traj - y_dot
        x_ddot_des = x_error*self._k_pos + x_dot_error*self._k_vel
        y_ddot_des = y_error*self._k_pos + y_dot_error*self._k_vel
        accel_vec_des = np.array([x_ddot_des, y_ddot_des])
        theta = np.arctan2(y_dot, x_dot)
        vel_hat = np.array([np.cos(theta), np.sin(theta)])
        vel_dot_des = np.dot(accel_vec_des, vel_hat)
        # Feedforward velocity dot
        theta_traj = np.arctan2(y_dot_traj, x_dot_traj)
        heading_error = np.abs(self.find_angle_error(theta, theta_traj))
        location_error = np.sqrt(x_error**2 + y_error**2)
        vel_hat_traj = np.array([np.cos(theta_traj), np.sin(theta_traj)])
        accel_vec_traj = np.array([x_ddot_traj,y_ddot_traj])
        vel_dot_ffwd = np.dot(accel_vec_traj, vel_hat_traj)
        if location_error < self._location_fwd_tol and heading_error < self._heading_ffwd_tol:
            vel_dot_com = vel_dot_ffwd + vel_dot_des
        else:
            vel_dot_com = vel_dot_des
        vel_dot_com_sat = np.clip(vel_dot_com, -self._max_vel_dot, self._max_vel_dot)
        #### angular acceleration computation ####
        # desired angular acceleration
        x_dot_des = x_error*self._k_pos + x_dot_traj
        y_dot_des = y_error*self._k_pos + y_dot_traj
        theta_des = np.arctan2(y_dot_des,x_dot_des)
        theta_error = self.find_angle_error(theta,theta_des)
        theta_dot_des = theta_error*self._k_theta
        theta_dot_ffwd = (x_dot_traj*y_ddot_traj - y_dot_traj*x_ddot_traj)/(x_dot_traj**2 + y_dot_traj**2)
        if location_error < self._location_fwd_tol and heading_error < self._heading_ffwd_tol:
            theta_dot_com = theta_dot_ffwd + theta_dot_des
        else:
            theta_dot_com = theta_dot_des
        theta_dot_com_sat = np.clip(theta_dot_com, -self._max_theta_dot, self._max_theta_dot)
        return vel_dot_com_sat, theta_dot_com_sat 

    def mpc_control_vel_input(self, inputs, states, trajectory_states):
        #### Data Extraction ####
        # current states
        x = states[0,0]
        y = states[0,1]
        theta = states[0,2]
        x_dot = states[1,0]
        y_dot = states[1,1]
        theta_dot = states[1,2]
        # desired trajectory states
        x_traj = trajectory_states[0,0]
        y_traj = trajectory_states[0,1]
        x_dot_traj = trajectory_states[1,0]
        y_dot_traj = trajectory_states[1,1]
        x_ddot_traj = trajectory_states[2,0]
        y_ddot_traj = trajectory_states[2,1]
        #### longitudinal acceleration computation ####
        x_error = x_traj - x
        y_error = y_traj - y
        x_dot_des = x_error*self._k_pos + x_dot_traj
        y_dot_des = y_error*self._k_pos + y_dot_traj
        vel_vec_des = np.array([x_dot_des, y_dot_des])
        theta = np.arctan2(y_dot, x_dot)
        vel_hat = np.array([np.cos(theta), np.sin(theta)])
        vel_des = np.dot(vel_vec_des, vel_hat)
        vel_com_sat = np.clip(vel_des, -self._max_vel, self._max_vel)
        #### angular acceleration computation ####
        # desired angular acceleration
        theta_des = np.arctan2(y_dot_des,x_dot_des)
        theta_error = self.find_angle_error(theta,theta_des)
        theta_dot_des = theta_error*self._k_theta
        theta_dot_ffwd = (x_dot_traj*y_ddot_traj - y_dot_traj*x_ddot_traj)/(x_dot_traj**2 + y_dot_traj**2)
        theta_traj = np.arctan2(y_dot_traj, x_dot_traj)
        heading_error = np.abs(self.find_angle_error(theta, theta_traj))
        location_error = np.sqrt(x_error**2 + y_error**2)
        if location_error < self._location_fwd_tol and heading_error < self._heading_ffwd_tol:
            theta_dot_com = theta_dot_ffwd + theta_dot_des
        else:
            theta_dot_com = theta_dot_des
        theta_dot_com_sat = np.clip(theta_dot_com, -self._max_theta_dot, self._max_theta_dot)
        return vel_com_sat, theta_dot_com_sat 
    
    def find_angle_error(self, angle, desired_angle):
        angle_error = self.find_turn_direction(angle, desired_angle) * self.get_closest_angle(angle, desired_angle)
        return angle_error 
    
    def find_turn_direction(self, angle, desired_angle):
        sign_direction =  np.sign(np.arctan2( np.sin(desired_angle - angle) , np.cos(desired_angle - angle)))
        return sign_direction
    
    def get_closest_angle(self, angle, angle_des):
        closest_angle = (np.pi - np.abs(np.abs(angle_des-angle) - np.pi) )
        return closest_angle
    

        



        

        
