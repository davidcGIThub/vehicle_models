"""
Bicycle Trajectory Tracker Class
"""
from matplotlib.text import get_rotation
import numpy as np
from scipy.optimize import fsolve
from vehicle_simulator.vehicle_controllers.trajectory_tracker import TrajectoryTracker


class BicycleTrajectoryTracker(TrajectoryTracker):

    def __init__(self, 
                 k_pos = 1, 
                 k_vel = 1,
                 k_delta = 1,
                 max_vel = 7,
                 max_vel_dot = 10,
                 max_delta = np.pi/4,
                 location_fwd_tol = 1,
                 heading_ffwd_tol = 0.3,
                 lr = 0.5,
                 L = 1,
                 dt = 0.1,
                 turn_vel = 1):
        self._k_pos = k_pos
        self._k_vel = k_vel
        self._k_delta = k_delta
        self._max_vel = max_vel
        self._max_vel_dot = max_vel_dot
        self._max_delta = max_delta
        self._turn_vel = turn_vel
        self._lr = lr
        self._L = L
        self._dt = dt
        self._location_fwd_tol = location_fwd_tol
        self._heading_ffwd_tol = heading_ffwd_tol

    def mpc_control_accel_input(self, inputs, states, trajectory_states):
        #### Data Extraction ####
        # current states
        x = states[0,0]
        y = states[0,1]
        theta = states[0,2]
        x_dot = states[1,0]
        y_dot = states[1,1]
        theta_dot = states[1,2]
        # current inputs
        delta = inputs[0,1]
        # desired trajectory states
        x_traj = trajectory_states[0,0]
        y_traj = trajectory_states[0,1]
        x_dot_traj = trajectory_states[1,0]
        y_dot_traj = trajectory_states[1,1]
        x_ddot_traj = trajectory_states[2,0]
        y_ddot_traj = trajectory_states[2,1]
        #### longitudinal acceleration computation ####
        # vel_dot desired #
        x_error = x_traj - x
        y_error = y_traj - y
        x_dot_error = x_dot_traj - x_dot
        y_dot_error = y_dot_traj - y_dot
        x_ddot_des = x_error*self._k_pos + x_dot_error*self._k_vel
        y_ddot_des = y_error*self._k_pos + y_dot_error*self._k_vel
        accel_vec_des = np.array([x_ddot_des, y_ddot_des])
        chi = np.arctan2(y_dot, x_dot)
        vel_hat = np.array([np.cos(chi), np.sin(chi)])
        vel_dot_des = np.dot(accel_vec_des, vel_hat)
        # vel_dot feedforward #
        chi_traj = np.arctan2(y_dot_traj, x_dot_traj)
        heading_error = np.abs(self.find_angle_error(chi, chi_traj))
        location_error = np.sqrt(x_error**2 + y_error**2)
        accel_vec_traj = np.array([x_ddot_traj,y_ddot_traj])
        vel_hat_traj = np.array([np.cos(chi_traj), np.sin(chi_traj)])
        vel_dot_ffwd = np.dot(accel_vec_traj, vel_hat_traj)
        # vel dot command #
        if location_error < self._location_fwd_tol and heading_error < self._heading_ffwd_tol:
            vel_dot_com = vel_dot_ffwd + vel_dot_des
        else:
            vel_dot_com = vel_dot_des
        #### Wheel turn rate computation ####
        # desired delta dot
        x_dot_des = x_error*self._k_pos + x_dot_traj
        y_dot_des = y_error*self._k_pos + y_dot_traj
        chi_des = np.arctan2(y_dot_des, x_dot_des)
        beta_des = np.clip(self.find_angle_error(theta, chi_des), -np.pi/2, np.pi/2)
        delta_des = np.arctan2(self._L*np.tan(beta_des), self._lr)
        delta_com = np.clip(delta_des, -self._max_delta , self._max_delta) 
        delta_error = self.find_angle_error(delta,delta_com)
        delta_dot_des = delta_error*self._k_delta
        # feedforward delta dot
        beta_traj = np.clip(self.find_angle_error(theta, chi_traj),-np.pi/2,np.pi/2)
        delta_traj = np.arctan2(self._L*np.tan(beta_traj), self._lr)
        chi_dot_traj = (x_dot_traj*y_ddot_traj - y_dot_traj*x_ddot_traj)/(y_dot_traj**2 + x_dot_traj**2)
        beta_dot_traj = chi_dot_traj - theta_dot
        delta_dot_ffwd = beta_dot_traj*((self._lr**2)*np.sin(delta_traj)**2 + (self._L**2)*np.cos(delta_traj)**2)/(self._L*self._lr)
        if location_error < self._location_fwd_tol and heading_error < self._heading_ffwd_tol:
            delta_dot_com = delta_dot_ffwd + delta_dot_des
        else:
            delta_dot_com = delta_dot_des
        return vel_dot_com, delta_dot_com
    
    def mpc_control_velocity_input(self, inputs, states, trajectory_states):
        #### Data Extraction ####
        # current states
        x = states[0,0]
        y = states[0,1]
        theta = states[0,2]
        x_dot = states[1,0]
        y_dot = states[1,1]
        theta_dot = states[1,2]
        # current inputs
        delta = inputs[0,1]
        # desired trajectory states
        x_traj = trajectory_states[0,0]
        y_traj = trajectory_states[0,1]
        x_dot_traj = trajectory_states[1,0]
        y_dot_traj = trajectory_states[1,1]
        x_ddot_traj = trajectory_states[2,0]
        y_ddot_traj = trajectory_states[2,1]
        #### longitudinal acceleration computation ####
        # vel_dot desired #
        x_error = x_traj - x
        y_error = y_traj - y
        x_dot_des = x_error*self._k_pos 
        y_dot_des = y_error*self._k_pos
        vel_vec_des = np.array([x_dot_des, y_dot_des])
        chi = np.arctan2(y_dot, x_dot)
        vel_hat = np.array([np.cos(chi), np.sin(chi)])
        vel_des = np.dot(vel_vec_des, vel_hat)
        # vel_dot feedforward #
        chi = np.arctan2(y_dot, x_dot)
        chi_traj = np.arctan2(y_dot_traj, x_dot_traj)
        heading_error = np.abs(self.find_angle_error(chi, chi_traj))
        location_error = np.sqrt(x_error**2 + y_error**2)
        vel_ffwd = np.sqrt(y_dot_traj**2 + x_dot_traj**2)
        # vel dot command #
        if location_error < self._location_fwd_tol and heading_error < self._heading_ffwd_tol:
            vel_dot_com = vel_ffwd + vel_des
        else:
            vel_dot_com = vel_des
        #### Wheel turn rate computation ####
        # desired delta dot
        chi_des = np.arctan2(y_dot_des, x_dot_des)
        beta_des = np.clip(self.find_angle_error(theta, chi_des), -np.pi/2, np.pi/2)
        delta_des = np.arctan2(self._L*np.tan(beta_des), self._lr)
        delta_com = np.clip(delta_des, -self._max_delta , self._max_delta) 
        delta_error = self.find_angle_error(delta,delta_com)
        delta_dot_des = delta_error*self._k_delta
        # feedforward delta dot
        beta_traj = np.clip(self.find_angle_error(theta, chi_traj),-np.pi/2,np.pi/2)
        delta_traj = np.arctan2(self._L*np.tan(beta_traj), self._lr)
        chi_dot_traj = (x_dot_traj*y_ddot_traj - y_dot_traj*x_ddot_traj)/(y_dot_traj**2 + x_dot_traj**2)
        beta_dot_traj = chi_dot_traj - theta_dot
        delta_dot_ffwd = beta_dot_traj*((self._lr**2)*np.sin(delta_traj)**2 + (self._L**2)*np.cos(delta_traj)**2)/(self._L*self._lr)
        if location_error < self._location_fwd_tol and heading_error < self._heading_ffwd_tol:
            delta_dot_com = delta_dot_ffwd + delta_dot_des
        else:
            delta_dot_com = delta_dot_des
        return vel_dot_com, delta_dot_com
    
    def find_angle_error(self, angle, desired_angle):
        angle_error = self.find_turn_direction(angle, desired_angle) * self.get_closest_angle(angle, desired_angle)
        return angle_error 

    def find_turn_direction(self, angle, desired_angle):
        sign_direction =  np.sign(np.arctan2( np.sin(desired_angle - angle) , np.cos(desired_angle - angle)))
        return sign_direction
    
    def get_closest_angle(self, angle, angle_des):
        closest_angle = (np.pi - np.abs(np.abs(angle_des-angle) - np.pi) )
        return closest_angle


    

        



        

        
