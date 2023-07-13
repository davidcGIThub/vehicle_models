"""
Unicycle Model Class
"""
import numpy as np 
import matplotlib.pyplot as plt
from vehicle_simulator.vehicle_models.vehicle_model_2D import VehicleModel

class UnicycleModel(VehicleModel):

    def __init__(self, 
                 x = 0, 
                 y = 0, 
                 x_dot = 0,
                 y_dot = 0,
                 theta = np.pi/2.0, 
                 theta_dot = 0,
                 alpha = np.array([0.1,0.01,0.01,0.1]),
                 height = 0.5,
                 width = 0.25,
                 max_vel = 10,
                 max_vel_dot = 10,
                 max_theta_dot = 15,
                 max_theta_ddot = 15):
        self._x = x
        self._y = y
        self._theta = self.__wrap_angle(theta)
        self._x_dot = x_dot
        self._y_dot = y_dot
        self._theta_dot = theta_dot
        self._x_ddot = 0
        self._y_ddot = 0
        self._theta_ddot = 0
        self._vel = x_dot**2 + y_dot**2
        self._vel_dot = 0
        self._alpha1 = alpha[0]
        self._alpha2 = alpha[1]
        self._alpha3 = alpha[2]
        self._alpha4 = alpha[3]
        self._height = height
        self._width = width
        self._max_vel = max_vel
        self._max_vel_dot = max_vel_dot
        self._max_theta_dot = max_theta_dot
        self._max_theta_ddot = max_theta_ddot
        self._robot_fig = plt.Polygon(self.get_body_points(),fc = '0.25',zorder=5)

    def set_state(self, states: np.ndarray):
        self._x = states[0,0]
        self._y = states[0,1]
        self._theta = self.__wrap_angle(states[0,2])
        self._x_dot = states[1,0]
        self._y_dot = states[1,1]
        self._theta_dot = states[1,2]
        self._x_ddot = states[2,0]
        self._y_ddot = states[2,1]
        self._theta_ddot = states[2,2]

    def set_inputs(self, inputs: np.ndarray):
        '''sets the current inputs'''
        self._vel = inputs[0,0]
        self._vel_dot = inputs[1,0]
        self._theta = inputs[0,1]
        self._theta_dot = inputs[1,1]

    def update_velocity_motion_model(self, velocity, angular_rate, dt):
        vel = velocity
        theta_dot = angular_rate
        vel_hat = vel + (self._alpha1 * vel**2 + self._alpha2 * theta_dot**2) * np.random.randn()
        theta_dot_hat = np.clip(theta_dot + (self._alpha3 * vel**2 + self._alpha4 * theta_dot**2) * np.random.randn(), -self._max_theta_dot, self._max_theta_dot)
        vel_dot = (vel_hat - np.sqrt(self._x_dot**2 + self._y_dot**2))/dt
        self._x_ddot = vel_dot*np.cos(self._theta) - vel_hat*np.sin(self._theta)*theta_dot_hat
        self._y_ddot = vel_dot*np.sin(self._theta) + vel_hat*np.cos(self._theta)*theta_dot_hat
        self._theta_ddot = (theta_dot_hat - self._theta_dot)/dt
        self._x_dot = vel_hat * np.cos(self._theta)
        self._y_dot = vel_hat * np.sin(self._theta)
        self._theta_dot = theta_dot_hat
        self._x = self._x + self._x_dot * dt
        self._y = self._y + self._y_dot * dt
        self._theta = self.__wrap_angle(self._theta + self._theta_dot * dt)
        self.__update_inputs(vel_hat, vel_dot)

    def update_acceleration_motion_model(self, longitudinal_acceleration, angular_rate, dt):
        vel_dot = longitudinal_acceleration
        theta_dot = angular_rate
        vel_dot_hat = np.clip(vel_dot + (self._alpha1 * vel_dot**2 + self._alpha4 * theta_dot**2) * np.random.randn(), -self._max_vel_dot, self._max_vel_dot)
        vel = np.clip( np.sqrt(self._x_dot**2 + self._y_dot**2) + vel_dot_hat*dt , 0 , self._max_vel )
        if (vel_dot_hat > 0 and vel >= self._max_vel) or (vel_dot_hat < 0 and vel <= 0):
            vel_dot_hat = 0
        theta_dot_hat = np.clip(theta_dot + (self._alpha3 * vel_dot**2 + self._alpha4 * theta_dot**2) * np.random.randn(), -self._max_theta_dot, self._max_theta_dot)
        self._x_ddot = vel_dot_hat*np.cos(self._theta) - vel*np.sin(self._theta)*theta_dot
        self._y_ddot = vel_dot_hat*np.sin(self._theta) + vel*np.cos(self._theta)*theta_dot
        self._theta_ddot = (theta_dot_hat - self._theta_dot)/dt
        self._x_dot = vel * np.cos(self._theta)
        self._y_dot = vel * np.sin(self._theta)
        self._theta_dot = theta_dot_hat
        self._x = self._x + self._x_dot * dt
        self._y = self._y + self._y_dot * dt
        self._theta = self.__wrap_angle(self._theta + self._theta_dot * dt)
        self.__update_inputs(vel, vel_dot_hat)

    def get_vehicle_properties(self):
        return np.array([self._height, self._width])
    
    def get_state(self):
        return np.array([[self._x, self._y, self._theta],
                          [self._x_dot, self._y_dot, self._theta_dot],
                          [self._x_ddot, self._y_ddot, self._theta_ddot]])
    
    def get_inputs(self):
        ''' Returns the current inputs '''
        return np.array([[self._vel     , self._theta],
                         [self._vel_dot , self._theta_dot]])
    
    def add_patches_to_axes(self, ax: plt.Axes):
        ax.add_patch(self._robot_fig)

    def add_patches_to_tuple(self, patches: tuple):
        return patches + (self._robot_fig,)

    def update_patches(self):
        self._robot_fig.xy = self.get_body_points()

    def plot_vehicle_instance(self, ax: plt.Axes):
        robot_fig = plt.Polygon(self.get_body_points(),fc = '0.25',zorder=5)
        ax.add_patch(robot_fig)

    def get_body_points(self):
        R = self.__get_rotation_matrix(self._theta)
        xy = np.array([[-self._height, self._height, -self._height],
                       [self._width, 0, -self._width]])
        xy = np.dot(R,xy)
        xy = xy + np.array([[self._x],[self._y]])
        return np.transpose(xy)
    
    def get_center_of_mass_point(self):
        return np.array([self._x,self._y])

    def __get_rotation_matrix(self, theta):
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                    [np.sin(theta), np.cos(theta)]])
        return rotation_matrix

    def __wrap_angle(self,theta):
        return np.arctan2(np.sin(theta), np.cos(theta))
    
    def __update_inputs(self, vel: float, vel_dot: float):
        self._vel = vel
        self._vel_dot = vel_dot

