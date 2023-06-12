"""
Bicycle Kinematic Controller Class
"""
from matplotlib.text import get_rotation
import numpy as np
from scipy.optimize import fsolve

class BicycleTrajectoryTracker:

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
                 dt = 0.1,):
        self._k_pos = k_pos
        self._k_vel = k_vel
        self._k_delta = k_delta
        self._max_vel = max_vel
        self._max_vel_dot = max_vel_dot
        self._max_delta = max_delta
        self._lr = lr
        self._L = L
        self._dt = dt
        self._location_fwd_tol = location_fwd_tol
        self._heading_ffwd_tol = heading_ffwd_tol

    def mpc_control_accel_input(self, states, trajectory_states):
        #### Data Extraction ####
        # current states
        x = states[0,0]
        y = states[0,1]
        theta = states[0,2]
        delta = states[0,3]
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
        # desired velocity
        x_pos_error = x_traj - x
        y_pos_error = y_traj - y
        x_vel_des = x_pos_error*self._k_pos + x_dot_traj
        y_vel_des = y_pos_error*self._k_pos + y_dot_traj
        vel_des = np.sqrt(x_vel_des**2 + y_vel_des**2)
        # desired velocity dot
        vel = np.sqrt(x_dot**2 + y_dot**2)
        vel_dot_des = (vel_des - vel)*self._k_vel
        vel_dot_com = vel_dot_des
        # feedforward tolerance error 
        chi_traj = np.arctan2(y_dot_traj, x_dot_traj)
        chi = np.arctan2(y_dot, x_dot)
        heading_error = np.abs(self.find_angle_error(chi, chi_traj))
        location_error = np.sqrt(x_pos_error**2 + y_pos_error**2)
        # velocity dot feedforward 
        if location_error < self._location_fwd_tol and heading_error < self._heading_ffwd_tol:
            vel_hat_traj = np.array([np.cos(chi_traj), np.sin(chi_traj)])
            accel_vec_traj = np.array([x_ddot_traj,y_ddot_traj])
            vel_dot_ffwd = np.dot(accel_vec_traj, vel_hat_traj)
            vel_dot_com = vel_dot_ffwd + vel_dot_des
            
        #### Wheel turn rate computation ####
        # desired wheel turn rate
        chi_des = np.arctan2(y_vel_des, x_vel_des)
        beta_des = np.clip(self.find_angle_error(theta, chi_des), -np.pi/2, np.pi/2)
        delta_des = np.arctan2(self._L*np.tan(beta_des), self._lr)
        delta_com = np.clip(delta_des, -self._max_delta , self._max_delta) 
        delta_error = self.find_angle_error(delta,delta_com)
        delta_dot_des = delta_error*self._k_delta
        delta_dot_com = delta_dot_des
        # feedforward for wheel turn rate
        if location_error < self._location_fwd_tol and heading_error < self._heading_ffwd_tol:
            beta_traj = np.clip(self.find_angle_error(theta, chi_traj),-np.pi/2,np.pi/2)
            delta_traj = np.arctan2(self._L*np.tan(beta_traj), self._lr)
            chi_dot_traj = (x_dot_traj*y_ddot_traj - y_dot_traj*x_ddot_traj)/(y_dot_traj**2 + x_dot_traj**2)
            beta_dot_traj = chi_dot_traj - theta_dot
            delta_dot_ffwd = beta_dot_traj*((self._lr**2)*np.sin(delta_traj)**2 + (self._L**2)*np.cos(delta_traj)**2)/(self._L*self._lr)
            delta_dot_com = delta_dot_ffwd + delta_dot_des
        return vel_dot_com, delta_dot_com
    
    def mpc_control_velocity_input(self, states, trajectory_states):
        # current states
        x = states.item(0,0)
        y = states.item(0,1)
        theta = states.item(0,2)
        delta = states.item(0,3)
        #desired trajectory states
        x_traj = trajectory_states.item(0,0)
        y_traj = trajectory_states.item(0,1)
        x_dot_traj = trajectory_states.item(1,0)
        y_dot_traj = trajectory_states.item(1,1)
        #velocity command computations
        x_pos_error = x_traj - x
        y_pos_error = y_traj - y
        x_vel_des = x_pos_error*self._k_pos + x_dot_traj
        y_vel_des = y_pos_error*self._k_pos + y_dot_traj
        vel_vec_des = np.array([x_vel_des,y_vel_des])
        vel_command = np.clip(np.linalg.norm(vel_vec_des), 0, self._max_vel)
        # wheel turn rate computations
        chi_des = np.arctan2(y_vel_des, x_vel_des)
        beta_des = np.clip(self.find_angle_error(theta, chi_des), -np.pi/2, np.pi/2)
        delta_des = np.clip(np.arctan2(self._L*np.tan(beta_des), self._lr), -self._max_delta , self._max_delta) 
        delta_error = self.find_angle_error(delta, delta_des)
        delta_dot_command = delta_error * self._k_delta
        return vel_command, delta_dot_command
    
    def find_angle_error(self, angle, desired_angle):
        angle_error = self.find_turn_direction(angle, desired_angle) * self.get_closest_angle(angle, desired_angle)
        return angle_error 

    def find_turn_direction(self, angle, desired_angle):
        sign_direction =  np.sign(np.arctan2( np.sin(desired_angle - angle) , np.cos(desired_angle - angle)))
        return sign_direction
    
    def get_closest_angle(self, angle, angle_des):
        closest_angle = (np.pi - np.abs(np.abs(angle_des-angle) - np.pi) )
        return closest_angle


    

        



        

        
