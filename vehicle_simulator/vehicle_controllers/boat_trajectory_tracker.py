"""
Boat Kinematic Controller Class
"""
from matplotlib.text import get_rotation
import numpy as np
from scipy.optimize import fsolve
import time

class BoatTrajectoryTracker:

    def __init__(self, 
                 c_r = 2/np.pi,
                 c_b = 0.01,
                 k_pos = 1, 
                 k_vel = 1,
                 k_theta = 1,
                 k_delta = 1,
                 max_vel = 7,
                 max_vel_dot = 10,
                 max_delta = np.pi/2,
                 turn_vel = 1.2,
                 location_fwd_tol = 2,
                 heading_ffwd_tol = 0.3):
        self._c_r = c_r
        self._c_b = c_b
        self._k_pos = k_pos
        self._k_vel = k_vel
        self._k_theta = k_theta
        self._k_delta = k_delta
        self._max_vel = max_vel
        self._max_vel_dot = max_vel_dot
        self._max_delta = max_delta
        self._turn_vel = turn_vel 
        self._location_fwd_tol = location_fwd_tol,
        self._heading_ffwd_tol = heading_ffwd_tol,
    
    def mpc_control_accel_input(self, inputs, states, trajectory_states):
        # Get current states
        x = states[0,0]
        y = states[0,1]
        theta = states[0,2]
        x_dot = states[1,0]
        y_dot = states[1,1]
        # inputs
        delta = inputs[0,1]
        vel = inputs[0,0]
        # Get Desired Trajectory States
        x_traj = trajectory_states[0,0]
        y_traj = trajectory_states[0,1]
        x_dot_traj = trajectory_states[1,0]
        y_dot_traj = trajectory_states[1,1]
        x_ddot_traj = trajectory_states[2,0]
        y_ddot_traj = trajectory_states[2,1]
        x_dddot_traj = trajectory_states[3,0]
        y_dddot_traj = trajectory_states[3,1]
        #### Longitudinal Acceleration ####
        # Desired Longitudinal Acceleration
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
        # vel_dot feedforward #
        theta_traj = np.arctan2(y_dot_traj, x_dot_traj)
        heading_error = np.abs(self.find_angle_error(theta, theta_traj))
        location_error = np.sqrt(x_error**2 + y_error**2)
        accel_vec_traj = np.array([x_ddot_traj,y_ddot_traj])
        vel_hat_traj = np.array([np.cos(theta_traj), np.sin(theta_traj)])
        vel_dot_ffwd = np.dot(accel_vec_traj, vel_hat_traj)
        # vel dot command #
        if location_error < self._location_fwd_tol and heading_error < self._heading_ffwd_tol:
            vel_dot_com = vel_dot_ffwd + vel_dot_des
        else:
            vel_dot_com = vel_dot_des
        #### Rudder Turn Rate #####
        # desired rudder turn rate
        x_dot_des = x_error*self._k_pos + x_dot_traj
        y_dot_des = y_error*self._k_pos + y_dot_traj
        theta_des = np.arctan2(y_dot_des,x_dot_des)
        theta_error = self.find_angle_error(theta, theta_des)
        theta_dot_des = theta_error*self._k_theta
        delta_des = -np.arcsin(np.clip(theta_dot_des*(vel+self._c_b)/(self._c_r*np.arctan(vel**2)), -1, 1))
        delta_com = np.clip(delta_des, -self._max_delta, self._max_delta)
        delta_error = self.find_angle_error(delta, delta_com)
        delta_dot_des = delta_error*self._k_delta
        # feedforward rudder turn rate
        vel_traj = np.sqrt(x_dot_traj**2 + y_dot_traj**2)
        vel_dot_traj = np.dot(accel_vec_traj, vel_hat_traj)
        theta_dot_traj = (x_dot_traj*y_ddot_traj - y_dot_traj*x_ddot_traj)/(x_dot_traj**2 + y_dot_traj**2)
        delta_traj = -np.arcsin(np.clip(theta_dot_traj*(vel_traj + self._c_b)/(self._c_r*np.arctan(vel_traj**2)),-1,1))
        theta_ddot_traj = ((x_dot_traj**2 + y_dot_traj**2)*(y_dddot_traj*x_dot_traj - x_dddot_traj*y_dot_traj) \
                + 2*(x_ddot_traj*y_dot_traj-x_dot_traj*y_ddot_traj)*(x_dot_traj*x_ddot_traj + y_dot_traj*y_ddot_traj)) \
                /(x_dot_traj**2 + y_dot_traj**2)**2
        if delta_traj >= self._max_delta or delta_traj <= -self._max_delta:
            delta_dot_ffwd = -theta_ddot_traj
        else:
            term_1 = self._c_r*vel_dot_traj*np.sin(delta_traj)*np.arctan(vel_traj**2)/(self._c_b+vel_traj)**2
            term_2 = 2*self._c_r*vel_traj*vel_dot_traj*np.sin(delta_traj)/((vel_traj**4 + 1)*(self._c_b + vel_traj))
            term_3 = (self._c_b + vel_traj)/(self._c_r*np.cos(delta_traj)*np.arctan(vel_traj**2))
            delta_dot_ffwd = (term_1 - term_2 - theta_ddot_traj)*term_3
        if location_error < self._location_fwd_tol and heading_error < self._heading_ffwd_tol:
            delta_dot_com = delta_dot_ffwd + delta_dot_des
        else:
            delta_dot_com = delta_dot_des
        return vel_dot_com, delta_dot_com
    
    # def mpc_control_accel_input(self, inputs, states, trajectory_states):
    #     # Get current states
    #     x = states[0,0]
    #     y = states[0,1]
    #     theta = states[0,2]
    #     delta = inputs[0,1]
    #     x_dot = states[1,0]
    #     y_dot = states[1,1]
    #     # Get Desired Trajectory States
    #     x_traj = trajectory_states[0,0]
    #     y_traj = trajectory_states[0,1]
    #     x_dot_traj = trajectory_states[1,0]
    #     y_dot_traj = trajectory_states[1,1]
    #     x_ddot_traj = trajectory_states[2,0]
    #     y_ddot_traj = trajectory_states[2,1]
    #     x_dddot_traj = trajectory_states[3,0]
    #     y_dddot_traj = trajectory_states[3,1]
    #     #### Error Terms ####
    #     x_pos_error = x_traj - x
    #     y_pos_error = y_traj - y
    #     theta_traj = np.arctan2(y_dot_traj, x_dot_traj)
    #     heading_error = np.abs(self.find_angle_error(theta, theta_traj))
    #     location_error = np.sqrt(x_pos_error**2 + y_pos_error**2)
    #     #### Longitudinal Acceleration ####
    #     # Desired Longitudinal Acceleration
    #     x_dot_des =  x_dot_traj + x_pos_error*self._k_pos
    #     y_dot_des = y_dot_traj + y_pos_error*self._k_pos
    #     vel_vec_des = np.array([x_dot_des, y_dot_des])
    #     vel_traj = np.sqrt(x_dot_traj**2 + y_dot_traj**2)
    #     vel_min = np.min((self._turn_vel, vel_traj))
    #     vel_hat = np.array([np.cos(theta), np.sin(theta)])
    #     vel_com = np.max((np.dot(vel_vec_des,vel_hat), vel_min))
    #     vel = np.sqrt(x_dot**2 + y_dot**2)
    #     vel_dot_des = (vel_com - vel)*self._k_vel
    #     # feedforward long acceleration
    #     vel_hat_traj = np.array([np.cos(theta_traj), np.sin(theta_traj)])
    #     accel_vec_traj = np.array([x_ddot_traj, y_ddot_traj])
    #     vel_dot_ffwd = np.dot(accel_vec_traj, vel_hat_traj)
    #     if location_error < self._location_fwd_tol and heading_error < self._heading_ffwd_tol:
    #         vel_dot_com = vel_dot_ffwd + vel_dot_des 
    #     else:
    #         vel_dot_com = vel_dot_des 
    #     #### Rudder Turn Rate #####
    #     # desired rudder turn rate
    #     theta_des = np.arctan2(y_dot_des, x_dot_des)
    #     theta_error = self.find_angle_error(theta, theta_des)
    #     theta_dot_des = theta_error*self._k_theta
    #     delta_des = -np.arcsin(np.clip(theta_dot_des*(vel+self._c_b)/(self._c_r*np.arctan(vel**2)), -1, 1))
    #     delta_com = np.clip(delta_des, -self._max_delta, self._max_delta)
    #     delta_error = self.find_angle_error(delta, delta_com)
    #     delta_dot_des = delta_error*self._k_delta
    #     # feedforward rudder turn rate
    #     vel_dot_traj = np.dot(accel_vec_traj, vel_hat_traj)
    #     theta_dot_traj = (x_dot_traj*y_ddot_traj - y_dot_traj*x_ddot_traj)/(x_dot_traj**2 + y_dot_traj**2)
    #     delta_traj = -np.arcsin(np.clip(theta_dot_traj*(vel_traj + self._c_b)/(self._c_r*np.arctan(vel_traj**2)),-1,1))
    #     term_1 = self._c_r*vel_dot_traj*np.sin(delta_traj)*np.arctan(vel_traj**2)/(self._c_b+vel_traj)**2
    #     term_2 = 2*self._c_r*vel_traj*vel_dot_traj*np.sin(delta_traj)/((vel_traj**4 + 1)*(self._c_b + vel_traj))
    #     theta_ddot_traj = ((x_dot_traj**2 + y_dot_traj**2)*(y_dddot_traj*x_dot_traj - x_dddot_traj*y_dot_traj) \
    #         + 2*(x_ddot_traj*y_dot_traj-x_dot_traj*y_ddot_traj)*(x_dot_traj*x_ddot_traj + y_dot_traj*y_ddot_traj)) \
    #         /(x_dot_traj**2 + y_dot_traj**2)**2
    #     term_3 = (self._c_b + vel_traj)/(self._c_r*np.cos(delta_traj)*np.arctan(vel_traj**2))
    #     delta_dot_ffwd = (term_1 - term_2 - theta_ddot_traj)*term_3
    #     if location_error < self._location_fwd_tol and heading_error < self._heading_ffwd_tol:
    #         delta_dot_com = delta_dot_ffwd + delta_dot_des
    #     else:
    #         delta_dot_com = delta_dot_des
    #     return vel_dot_com, delta_dot_com

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
        x_dddot_traj = trajectory_states[3,0]
        y_dddot_traj = trajectory_states[3,1]
        #### Error Terms ####
        x_pos_error = x_traj - x
        y_pos_error = y_traj - y
        theta_traj = np.arctan2(y_dot_traj, x_dot_traj)
        heading_error = np.abs(self.find_angle_error(theta, theta_traj))
        location_error = np.sqrt(x_pos_error**2 + y_pos_error**2)
        #### Longitudinal Velocity ####
        # Desired Longitudinal Velocity
        x_dot_des =  x_dot_traj + x_pos_error*self._k_pos
        y_dot_des = y_dot_traj + y_pos_error*self._k_pos
        vel_vec_des = np.array([x_dot_des, y_dot_des])
        # vel_com = np.clip(np.linalg.norm(vel_vec_des), 0, self._max_vel) ##### temporary
        vel_traj = np.sqrt(x_dot_traj**2 + y_dot_traj**2)
        vel_min = np.min((self._turn_vel, vel_traj))
        vel_hat = np.array([np.cos(theta), np.sin(theta)])
        vel_com = np.max((np.dot(vel_vec_des,vel_hat), vel_min))
        #### Rudder Turn Rate #####
        # desired rudder turn rate
        theta_des = np.arctan2(y_dot_des, x_dot_des)
        theta_error = self.find_angle_error(theta, theta_des)
        theta_dot_des = theta_error*self._k_theta
        vel = np.sqrt(x_dot**2 + y_dot**2)
        delta_des = -np.arcsin(np.clip(theta_dot_des*(vel+self._c_b)/(self._c_r*np.arctan(vel**2)), -1, 1))
        delta_com = np.clip(delta_des, -self._max_delta, self._max_delta)
        delta_error = self.find_angle_error(delta, delta_com)
        delta_dot_des = delta_error*self._k_delta
        if location_error < self._location_fwd_tol and heading_error < self._heading_ffwd_tol:
            accel_vec_traj = np.array([x_ddot_traj, y_ddot_traj])
            vel_hat_traj = np.array([np.cos(theta_traj), np.sin(theta_traj)])
            vel_dot_traj = np.dot(accel_vec_traj, vel_hat_traj)
            theta_dot_traj = (x_dot_traj*y_ddot_traj - y_dot_traj*x_ddot_traj)/(x_dot_traj**2 + y_dot_traj**2)
            delta_traj = -np.arcsin(np.clip(theta_dot_traj*(vel_traj + self._c_b)/(self._c_r*np.arctan(vel_traj**2)),-1,1))
            term_1 = self._c_r*vel_dot_traj*np.sin(delta_traj)*np.arctan(vel_traj**2)/(self._c_b+vel_traj)**2
            term_2 = 2*self._c_r*vel_traj*vel_dot_traj*np.sin(delta_traj)/((vel_traj**4 + 1)*(self._c_b + vel_traj))
            theta_ddot_traj = ((x_dot_traj**2 + y_dot_traj**2)*(y_dddot_traj*x_dot_traj - x_dddot_traj*y_dot_traj) \
                + 2*(x_ddot_traj*y_dot_traj-x_dot_traj*y_ddot_traj)*(x_dot_traj*x_ddot_traj + y_dot_traj*y_ddot_traj)) \
                /(x_dot_traj**2 + y_dot_traj**2)**2
            term_3 = (self._c_b + vel_traj)/(self._c_r*np.cos(delta_traj)*np.arctan(vel_traj**2))
            delta_dot_ffwd = (term_1 - term_2 - theta_ddot_traj)*term_3
            delta_dot_com = delta_dot_ffwd + delta_dot_des
        else:
            delta_dot_com = delta_dot_des
        return vel_com, delta_dot_com
    
    def find_angle_error(self, angle, desired_angle):
        angle_error = self.find_turn_direction(angle, desired_angle) * self.get_closest_angle(angle, desired_angle)
        return angle_error 
    
    def find_turn_direction(self, angle, desired_angle):
        sign_direction =  np.sign(np.arctan2( np.sin(desired_angle - angle) , np.cos(desired_angle - angle)))
        return sign_direction
    
    def get_closest_angle(self, angle, angle_des):
        closest_angle = (np.pi - np.abs(np.abs(angle_des-angle) - np.pi) )
        return closest_angle