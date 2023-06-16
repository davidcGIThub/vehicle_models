"""
Bicycle Model Class
"""
from matplotlib.text import get_rotation
import numpy as np
import matplotlib.pyplot as plt
from vehicle_simulator.vehicle_models.vehicle_model import VehicleModel

class BicycleModel(VehicleModel):

    def __init__(self, 
                 x = 0, 
                 y = 0, 
                 theta = 0, 
                 delta = 0, 
                 x_dot = 0, 
                 y_dot = 0, 
                 theta_dot = 0, 
                 delta_dot = 0, 
                 lr = 0.5, # rear wheel to COM length
                 L = 1, # body length
                 R = 0.2, # wheel radius
                 alpha = np.array([0.1,0.01,0.1,0.01]), # noise parameters
                 max_delta = np.pi/4,
                 max_vel = 5,
                 max_vel_dot = 5):
        self._x = x
        self._x_dot = x_dot
        self._x_ddot = 0
        self._y = y
        self._y_dot = y_dot
        self._y_ddot = 0
        self._theta = theta
        self._theta_dot = theta_dot
        self._theta_ddot = 0
        self._delta = delta
        self._delta_dot = delta_dot
        self._lr = lr
        self._L = L
        self._alpha1 = alpha[0]
        self._alpha2 = alpha[1]
        self._alpha3 = alpha[2]
        self._alpha4 = alpha[3]
        self._R = R # wheel radius
        self._vel = self.__calculate_velocity(self._x_dot, self._y_dot)
        self._vel_dot = 0
        self._max_delta = max_delta
        self._max_vel = max_vel
        self._max_vel_dot = max_vel_dot
        self._front_wheel_fig = plt.Polygon(self.get_front_wheel_points(),fc = 'k')
        self._back_wheel_fig = plt.Polygon(self.get_back_wheel_points(),fc = 'k')
        self._body_fig = plt.Polygon(self.get_body_points(),fc = 'g')
    
    def set_state(self,states: np.ndarray):
        self._x = states[0,0]
        self._y = states[0,1]
        self._theta = states[0,2]
        self._x_dot = states[1,0]
        self._y_dot = states[1,1]
        self._theta_dot = states[1,2]
        self._x_ddot = states[2,0]
        self._y_ddot = states[2,1]
        self._theta_ddot = states[2,2]

    def set_inputs(self, input_array: np.ndarray):
        ''' sets the current inputs '''
        self._vel = input_array[0,0]
        self._vel_dot = input_array[1,0]
        self._delta = input_array[0,1]
        self._delta_dot = input_array[1,1]
    
    def update_velocity_motion_model(self, velocity:float, wheel_turn_rate:float, dt:float):
        vel = velocity
        delta_dot = wheel_turn_rate # front wheel turning rate 
        vel_hat = np.clip( vel + (self._alpha1 * vel**2 + self._alpha4 * delta_dot**2) * np.random.randn(), 0, self._max_vel )
        delta_dot_hat = delta_dot + (self._alpha2 * vel**2 + self._alpha3 * delta_dot**2) * np.random.randn()
        if (delta_dot_hat > 0 and self._delta >= self._max_delta) or (delta_dot_hat < 0 and self._delta <= -self._max_delta):
            delta_dot_hat = 0
        delta = np.clip(self._delta + delta_dot_hat * dt, -self._max_delta, self._max_delta)
        # following functions should be called in the following order
        beta, beta_dot = self.__get_beta_states(delta, delta_dot_hat)
        vel_dot = (vel_hat - self.__calculate_velocity(self._x_dot, self._y_dot))/dt
        self.__update_inputs(vel_hat, vel_dot, delta, delta_dot_hat)
        self.__update_second_derivative_states(self._theta, self._theta_dot, delta, delta_dot_hat, \
                                               beta, beta_dot, vel, vel_dot)
        self.__update_derivative_states(vel_hat, self._theta, delta_dot_hat, beta, dt)
        self.__update_states(self._x_dot,self._y_dot, self._theta_dot,delta, dt)
        return vel_hat, delta_dot_hat

    def update_acceleration_motion_model(self, longitudinal_acceleration:float, 
                                         wheel_turn_rate:float, dt:float):
        vel_dot = longitudinal_acceleration
        delta_dot = wheel_turn_rate # front wheel turning rate 
        vel_dot_hat = np.clip(vel_dot + (self._alpha1 * vel_dot**2 + self._alpha4 * delta_dot**2) * np.random.randn(), -self._max_vel_dot, self._max_vel_dot)
        vel = np.clip(self.__calculate_velocity(self._x_dot, self._y_dot) + vel_dot_hat*dt , 0 , self._max_vel  )
        if (vel_dot_hat > 0 and vel >= self._max_vel) or (vel_dot_hat < 0 and vel <= 0):
            vel_dot_hat = 0
        delta_dot_hat = delta_dot + (self._alpha2 * vel_dot**2 + self._alpha3 * delta_dot**2) * np.random.randn()
        if (delta_dot_hat > 0 and self._delta >= self._max_delta) or (delta_dot_hat < 0 and self._delta <= -self._max_delta):
            delta_dot_hat = 0
        delta = np.clip(self._delta + delta_dot_hat * dt, -self._max_delta, self._max_delta)
        #following functions should be called in the following order
        beta, beta_dot = self.__get_beta_states(delta, delta_dot_hat)
        self.__update_inputs(vel, vel_dot_hat, delta, delta_dot_hat)
        self.__update_second_derivative_states(self._theta, self._theta_dot, delta, delta_dot_hat, \
                                               beta, beta_dot, vel, vel_dot_hat)
        self.__update_derivative_states(vel, self._theta, delta_dot_hat, beta, dt)
        self.__update_states(self._x_dot,self._y_dot, self._theta_dot,delta, dt)
        return vel_dot_hat, delta_dot_hat
    
    def get_vehicle_properties(self):
        return np.array([self._L, self._lr, self._R])
    
    def get_state(self):
        states = np.array([[self._x,self._y,self._theta,self._delta],
                    [self._x_dot,self._y_dot,self._theta_dot,self._delta_dot],
                    [self._x_ddot,self._y_ddot,self._theta_ddot,0]])
        return states
    
    def get_inputs(self):
        ''' Returns the current inputs '''
        return np.array([[self._vel     , self._delta],
                         [self._vel_dot , self._delta_dot]])
    
    def add_patches_to_axes(self, ax: plt.Axes):
        ax.add_patch(self._front_wheel_fig)
        ax.add_patch(self._back_wheel_fig)
        ax.add_patch(self._body_fig)

    def add_patches_to_tuple(self, patches: tuple):
        new_patches = patches + (self._body_fig, self._front_wheel_fig, self._back_wheel_fig)
        return new_patches

    def update_patches(self):
        self._front_wheel_fig.xy = self.get_front_wheel_points()
        self._back_wheel_fig.xy = self.get_back_wheel_points()
        self._body_fig.xy = self.get_body_points()
    
    def plot_vehicle_instance(self, ax: plt.Axes):
        front_wheel_fig = plt.Polygon(self.get_front_wheel_points(),fc = 'k',zorder=10)
        back_wheel_fig = plt.Polygon(self.get_back_wheel_points(),fc = 'k',zorder=10)
        body_fig = plt.Polygon(self.get_body_points(),fc = 'g',zorder=10)
        ax.add_patch(front_wheel_fig)
        ax.add_patch(back_wheel_fig)
        ax.add_patch(body_fig)

    def get_body_points(self):
        xy_body_frame = np.array([[-self._lr, self._L-self._lr, self._L-self._lr, -self._lr],
                                  [self._R/5, self._R/5, -self._R/5, -self._R/5]])
        body_points = self.__get_points(xy_body_frame)
        return body_points

    def get_back_wheel_points(self):
        xy_body_frame = np.array([[-self._lr-self._R, -self._lr+self._R, -self._lr+self._R, -self._lr-self._R],
                       [self._R/3,self._R/3,-self._R/3,-self._R/3]])
        backWheelPoints = self.__get_points(xy_body_frame)
        return backWheelPoints

    def get_front_wheel_points(self):
        xy_wheel_frame_straight = np.array([[-self._R, self._R, self._R, -self._R],
                                            [self._R/3, self._R/3, -self._R/3, -self._R/3]])
        wheel_rotation_matrix = self.__get_rotation_matrix(self._delta)
        xy_wheel_frame_rotated = np.dot(wheel_rotation_matrix , xy_wheel_frame_straight)
        xy_body_frame = xy_wheel_frame_rotated + np.array([[self._L - self._lr],[0]])
        frontWheelPoints = self.__get_points(xy_body_frame)
        return frontWheelPoints
    
    def get_center_of_mass_point(self):
        return np.array([self._x,self._y])
    
    def __get_points(self, xy:np.ndarray):
        rotation_matrix = self.__get_rotation_matrix(self._theta)
        xy = np.dot(rotation_matrix,xy)
        xy = xy + np.array([[self._x],[self._y]])
        return np.transpose(xy)
    
    def __calculate_velocity(self, x_dot: float, y_dot: float):
        return np.sqrt(x_dot**2 + y_dot**2)
    
    def __update_inputs(self, vel: float, vel_dot: float, delta: float, delta_dot: float):
        self._vel = vel
        self._vel_dot = vel_dot
        self._delta = delta
        self._delta_dot = delta_dot
    
    def __update_states(self,x_dot:float ,y_dot:float, theta_dot:float, delta:float, dt:float):
        self._x = self._x + x_dot * dt
        self._y = self._y + y_dot * dt
        self._theta = self.__wrap_angle(self._theta + theta_dot * dt)
        self._delta = delta

    def __update_derivative_states(self, vel:float, theta:float, delta_dot:float, beta:float, dt:float):
        self._x_dot = vel*np.cos(theta + beta)
        self._y_dot = vel*np.sin(theta + beta)
        self._delta = np.clip(self._delta + delta_dot * dt, -self._max_delta, self._max_delta)
        self._theta_dot = vel*np.cos(beta)*np.tan(self._delta)/self._L
    
    def __update_second_derivative_states(self, theta:float, theta_dot:float, delta:float,
                                           delta_dot:float, beta:float, beta_dot:float, vel:float, vel_dot:float):
        self._x_ddot = vel_dot*np.cos(theta + beta) - \
            (theta_dot+beta_dot)*vel*np.sin(beta+theta)
        self._y_ddot = vel_dot*np.sin(theta + beta) + \
            (theta_dot + beta_dot)*vel*np.cos(beta+theta)
        self._theta_ddot = vel_dot*np.cos(beta)*np.tan(delta) + \
            vel*(delta_dot*np.cos(beta)/np.cos(delta)**2 - beta_dot*np.sin(beta)*np.tan(delta))
    
    def __get_beta_states(self, delta:float, delta_dot:float):
        beta = np.arctan2(self._lr*np.tan(delta) , self._L)
        beta_dot = self._L*self._lr*delta_dot/ \
            ((self._lr**2)*np.sin(delta)**2 + (self._L**2)*np.cos(delta)**2)
        return beta, beta_dot

    def __get_rotation_matrix(self, theta:float):
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                    [np.sin(theta), np.cos(theta)]])
        return rotation_matrix
    
    def __wrap_angle(self,theta:float):
        return np.arctan2(np.sin(theta), np.cos(theta))



