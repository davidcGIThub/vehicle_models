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
                 k_delta = 1,
                 vel_max = 7,
                 vel_turn = 0.1,
                 vel_dot_max = 10,
                 delta_max = np.pi/4,
                 lr = 0.5,
                 L = 1,
                 dt = 0.1):
        self._k_pos = k_pos
        self._k_vel = k_vel
        self._k_delta = k_delta
        self._vel_max = vel_max
        self._vel_turn = vel_turn
        self._vel_dot_max = vel_dot_max
        self._delta_max = delta_max
        self._lr = lr
        self._L = L
        self._dt = dt

    # def mpc_control_accel_input(self, states, trajectory_states):
    #     # current states
    #     x = states[0,0]
    #     y = states[0,1]
    #     theta = states[0,2]
    #     delta = states[0,3]
    #     x_dot = states[1,0]
    #     y_dot = states[1,1]
    #     theta_dot = states[1,2]
    #     #desired trajectory states
    #     x_traj = trajectory_states[0,0]
    #     y_traj = trajectory_states[0,1]
    #     x_dot_traj = trajectory_states[1,0]
    #     y_dot_traj = trajectory_states[1,1]
    #     x_ddot_traj = trajectory_states[2,0]
    #     y_ddot_traj = trajectory_states[2,1]
    #     # longitudinal acceleration computation
    #     x_pos_error = x_traj - x
    #     y_pos_error = y_traj - y
    #     x_vel_des = x_pos_error*self._k_pos + x_dot_traj
    #     y_vel_des = y_pos_error*self._k_pos + y_dot_traj
    #     x_vel_error = x_vel_des - x_dot
    #     y_vel_error = y_vel_des - y_dot
    #     x_accel_des = x_vel_error*self._k_vel + x_ddot_traj
    #     y_accel_des = y_vel_error*self._k_vel + y_ddot_traj
    #     vel_des = np.sqrt(x_vel_des**2 + y_vel_des**2)
    #     vel_com = np.clip(np.linalg.norm(vel_des), 0, self._vel_max)
    #     accel_vec_traj = np.array([x_ddot_traj,y_ddot_traj])
    #     chi_traj = np.arctan2(y_dot_traj, x_dot_traj)
    #     vel_hat_traj = np.array([np.sin(chi_traj), np.cos(chi_traj)]) * np.sqrt(x_dot_traj**2 + y_dot_traj**2)
    #     vel_dot_traj = np.dot(accel_vec_traj, vel_hat_traj)
    #     vel = np.sqrt(x_dot**2 + y_dot**2)
    #     vel_dot_com = (vel_com - vel)*self._k_vel + vel_dot_traj
    #     print("vel_dot_com: " , vel_dot_com)
    #     print("vel: " , vel)
    #     # Wheel turn rate command
    #     chi_des = np.arctan2(y_vel_des, x_vel_des)
    #     beta_des = np.clip(self.find_angle_error(theta, chi_des), -np.pi/2, np.pi/2)
    #     beta_dot_des = (x_vel_des*y_accel_des - y_vel_des*x_accel_des)/vel_des
    #     delta_des = np.clip(np.arctan2(self._L*np.tan(beta_des), self._lr), -self._delta_max , self._delta_max) 
    #     delta_dot_des = beta_dot_des*((self._lr**2)*np.sin(delta_des)**2 + (self._L**2)*np.cos(delta_des)**2)/(self._L*self._lr)
    #     delta_error = self.find_angle_error(delta,delta_des)
    #     delta_dot_com = delta_error*self._k_delta #+ delta_dot_des
    #     return vel_dot_com, delta_dot_com

    def mpc_control_accel_input(self, states, trajectory_states):
        # current states
        x = states.item(0,0)
        y = states.item(0,1)
        theta = states.item(0,2)
        delta = states.item(0,3)
        x_dot = states[1,0]
        y_dot = states[1,1]
        #desired trajectory states
        x_traj = trajectory_states.item(0,0)
        y_traj = trajectory_states.item(0,1)
        x_dot_traj = trajectory_states.item(1,0)
        y_dot_traj = trajectory_states.item(1,1)
        x_ddot_traj = trajectory_states[2,0]
        y_ddot_traj = trajectory_states[2,1]
        #velocity command computations
        x_pos_error = x_traj - x
        y_pos_error = y_traj - y
        x_vel_des = x_pos_error*self._k_pos + x_dot_traj
        y_vel_des = y_pos_error*self._k_pos + y_dot_traj
        vel_vec_des = np.array([x_vel_des,y_vel_des])
        vel_com = np.clip(np.linalg.norm(vel_vec_des), 0, self._vel_max)
        vel = np.sqrt(x_dot**2 + y_dot**2)
        accel_vec_traj = np.array([x_ddot_traj,y_ddot_traj])
        chi_traj = np.arctan2(y_dot_traj, x_dot_traj)
        vel_hat_traj = np.array([np.sin(chi_traj), np.cos(chi_traj)]) #* np.sqrt(x_dot_traj**2 + y_dot_traj**2)
        vel_dot_traj = np.dot(accel_vec_traj, vel_hat_traj)
        vel_dot_com = (vel_com - vel)*self._k_vel + vel_dot_traj
        # print("vel_dot_com: " , vel_dot_com)
        # wheel turn rate computations
        chi_des = np.arctan2(y_vel_des, x_vel_des)
        beta_des = np.clip(self.find_angle_error(theta, chi_des), -np.pi/2, np.pi/2)
        x_vel_error = x_vel_des - x_dot
        y_vel_error = y_vel_des - y_dot
        x_accel_des = x_vel_error*self._k_vel + x_ddot_traj
        y_accel_des = y_vel_error*self._k_vel + y_ddot_traj
        beta_dot_des = (x_vel_des*y_accel_des - y_vel_des*x_accel_des)/(x_vel_des**2 + y_vel_des**2)
        delta_des = np.clip(np.arctan2(self._L*np.tan(beta_des), self._lr), -self._delta_max , self._delta_max) 
        delta_error = self.find_angle_error(delta, delta_des)
        delta_dot_des = beta_dot_des*((self._lr**2)*np.sin(delta_des)**2 + (self._L**2)*np.cos(delta_des)**2)/(self._L*self._lr)
        delta_dot_com = delta_error * self._k_delta + delta_dot_des
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
        vel_command = np.clip(np.linalg.norm(vel_vec_des), 0, self._vel_max)
        # wheel turn rate computations
        chi_des = np.arctan2(y_vel_des, x_vel_des)
        beta_des = np.clip(self.find_angle_error(theta, chi_des), -np.pi/2, np.pi/2)
        delta_des = np.clip(np.arctan2(self._L*np.tan(beta_des), self._lr), -self._delta_max , self._delta_max) 
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


    

        



        

        
