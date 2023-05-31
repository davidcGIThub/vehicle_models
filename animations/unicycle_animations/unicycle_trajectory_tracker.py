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
                 k_theta_dot = 1, 
                 max_vel = 7,
                 max_vel_dot = 10,
                 max_theta_dot = 1):
        self._k_pos = k_pos
        self._k_vel = k_vel
        self._k_accel = k_accel
        self._k_theta = k_theta
        self._k_theta_dot = k_theta_dot
        self._max_vel = max_vel
        self._max_vel_dot = max_vel_dot
        self._max_theta_dot = max_theta_dot 

    def mpc_control_accel_input(self, states, trajectory_states):
        # Get current states
        x = states[0,0]
        y = states[0,1]
        theta = states[0,2]
        x_dot = states[1,0]
        y_dot = states[1,1]
        theta_dot = states[1,2]
        x_ddot = states[2,0]
        y_ddot = states[2,1]
        # Get Desired Trajectory States
        x_traj = trajectory_states[0,0]
        y_traj = trajectory_states[0,1]
        x_dot_traj = trajectory_states[1,0]
        y_dot_traj = trajectory_states[1,1]
        x_ddot_traj = trajectory_states[2,0]
        y_ddot_traj = trajectory_states[2,1]
        x_dddot_traj = trajectory_states[3,0]
        y_dddot_traj = trajectory_states[3,1]
        # Evaluate accel control input
        x_pos_error = x_traj - x
        y_pos_error = y_traj - y
        x_vel_des = x_pos_error*self._k_pos + x_dot_traj
        y_vel_des = y_pos_error*self._k_pos + y_dot_traj
        x_accel_des = (x_vel_des - x_dot) * self._k_vel + x_ddot_traj
        y_accel_des = (y_vel_des - y_dot) * self._k_vel + y_ddot_traj
        vel_vec_des = np.array([x_vel_des,y_vel_des])
        vel_dir = np.array([np.cos(theta),np.sin(theta)])
        vel_command = np.dot(vel_vec_des, vel_dir)
        vel = np.sqrt(x_dot**2 + y_dot**2)
        accel_vec_traj = np.array([x_ddot_traj,y_ddot_traj])
        theta_traj = np.arctan2(y_dot_traj, x_dot_traj)
        vel_hat_traj = np.array([np.cos(theta_traj), np.sin(theta_traj)]) #* np.sqrt(x_dot_traj**2 + y_dot_traj**2)
        vel_dot_traj = np.dot(accel_vec_traj, vel_hat_traj)
        vel_dot_des = (vel_command - vel)*self._k_vel + vel_dot_traj
        vel_dot_command = np.clip(vel_dot_des, -self._max_vel_dot, self._max_vel_dot)
        # angular acceleration computation
        theta_des = np.arctan2(y_vel_des,x_vel_des)
        theta_error = self.find_angle_error(theta,theta_des)
        theta_dot_des = (x_vel_des*y_accel_des - y_vel_des*x_accel_des)/(x_vel_des**2 + y_vel_des**2)
        theta_dot_command = theta_error*self._k_theta + theta_dot_des
        x_jerk_des = (x_accel_des - x_ddot) * self._k_accel + x_dddot_traj
        y_jerk_des = (y_accel_des - y_ddot) * self._k_accel + y_dddot_traj
        theta_ddot_des = ((x_vel_des**2 + y_vel_des**2)* (y_jerk_des*x_vel_des - x_jerk_des*y_vel_des) + \
            2*(x_accel_des*y_vel_des - x_vel_des*y_accel_des)*(x_vel_des*x_accel_des + y_vel_des*y_accel_des)) / \
            (x_vel_des**2 + y_vel_des**2)**2
        theta_ddot_command = (theta_dot_command - theta_dot) * self._k_theta_dot + theta_ddot_des
        print(" ")
        print("theta: " , theta*180/np.pi)
        print("theta_des: " , theta_des*180/np.pi)
        print("theta_dot_c: , ", theta_dot_command)
        return vel_dot_command, theta_ddot_command
    
    def mpc_control_vel_input(self, states, trajectory_states):
        # Get current states
        x = states[0,0]
        y = states[0,1]
        theta = states[0,2]
        x_dot = states[1,0]
        y_dot = states[1,1]
        # Get Desired Trajectory States
        x_traj = trajectory_states[0,0]
        y_traj = trajectory_states[0,1]
        x_dot_traj = trajectory_states[1,0]
        y_dot_traj = trajectory_states[1,1]
        x_ddot_traj = trajectory_states[2,0]
        y_ddot_traj = trajectory_states[2,1]
        # Evaluate vel control input
        x_pos_error = x_traj - x
        y_pos_error = y_traj - y
        x_vel_des = x_pos_error*self._k_pos + x_dot_traj
        y_vel_des = y_pos_error*self._k_pos + y_dot_traj
        x_accel_des = (x_vel_des - x_dot) * self._k_vel + x_ddot_traj
        y_accel_des = (y_vel_des - y_dot) * self._k_vel + y_ddot_traj
        vel_vec_des = np.array([x_vel_des,y_vel_des])
        vel_dir = np.array([np.cos(theta),np.sin(theta)])
        vel_command = np.dot(vel_vec_des, vel_dir)
        # angular acceleration computation
        theta_des = np.arctan2(y_vel_des,x_vel_des)
        theta_error = self.find_angle_error(theta,theta_des)
        theta_dot_des = (x_vel_des*y_accel_des - y_vel_des*x_accel_des)/(x_vel_des**2 + y_vel_des**2)
        theta_dot_command = theta_error*self._k_theta + theta_dot_des
        return vel_command, theta_dot_command
    
    def find_angle_error(self, angle, desired_angle):
        angle_error = self.find_turn_direction(angle, desired_angle) * self.get_closest_angle(angle, desired_angle)
        return angle_error 
    
    def find_turn_direction(self, angle, desired_angle):
        sign_direction =  np.sign(np.arctan2( np.sin(desired_angle - angle) , np.cos(desired_angle - angle)))
        return sign_direction
    
    def get_closest_angle(self, angle, angle_des):
        closest_angle = (np.pi - np.abs(np.abs(angle_des-angle) - np.pi) )
        return closest_angle
    

        



        

        
