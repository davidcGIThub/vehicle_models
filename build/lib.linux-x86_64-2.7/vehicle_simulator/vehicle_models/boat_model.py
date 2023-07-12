"""
Boat Model Class
"""
import numpy as np
import matplotlib.pyplot as plt
import sys

# velocity motion model
import numpy as np 

class BoatModel:

    def __init__(self, 
                 x = 0, 
                 x_dot = 0,
                 x_ddot = 0,
                 y = 0, 
                 y_dot = 0,
                 y_ddot = 0,
                 theta = np.pi/2.0, 
                 theta_dot = 0,
                 theta_ddot = 0,
                 delta = 0,
                 delta_dot = 0,
                 delta_ddot = 0,
                 alpha = np.array([0.1,0.01,0.01,0.1]),
                 height = 0.5,
                 width = 0.25,
                 c_r = 1, #rudder constant
                 c_b = 0.01, #boat constant
                 max_delta = np.pi/2,
                 max_delta_dot = 3,
                 max_vel = 5,
                 max_vel_dot = 10):
        self._x = x
        self._x_dot = x_dot
        self._x_ddot = x_ddot
        self._y = y
        self._y_dot = y_dot
        self._y_ddot = y_ddot
        self._theta = self.__wrap_angle(theta)
        self._theta_dot = theta_dot
        self._theta_ddot = theta_ddot
        self._delta = delta
        self._delta_dot = delta_dot
        self._delta_ddot = delta_ddot
        self._vel = np.sqrt(self._x_dot**2 + self._y_dot**2)
        self._vel_dot = 0
        self._alpha1 = alpha[0]
        self._alpha2 = alpha[1]
        self._alpha3 = alpha[2]
        self._alpha4 = alpha[3]
        self._height = height
        self._width = width
        self._c_r = c_r
        self._c_b = c_b
        self._max_delta = max_delta
        self._max_delta_dot = max_delta_dot
        self._max_vel = max_vel
        self._max_vel_dot = max_vel_dot
        self._boat_fig = plt.Polygon(self.get_body_points(),fc = '0.25',zorder=5)
        self._rudder_fig = plt.Polygon(self.get_rudder_points(),fc = 'k',zorder=4)
    
    def set_state(self,states):
        self._x = states[0,0]
        self._y = states[0,1]
        self._theta = self.__wrap_angle(states[0,2])
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

    def update_acceleration_motion_model(self,longitudinal_acceleration, rudder_steering_turn_rate,dt):
        vel_dot = longitudinal_acceleration #acceleration
        if vel_dot > self._max_vel_dot:
            print("Exceeded vel dot: " , vel_dot)
        delta_dot = rudder_steering_turn_rate #rudder location
        vel_dot_hat = np.clip(vel_dot + (self._alpha1 * vel_dot**2 + self._alpha4 * delta_dot**2) * np.random.randn(), -self._max_vel_dot, self._max_vel_dot)
        vel = np.clip( np.sqrt(self._x_dot**2 + self._y_dot**2) + vel_dot_hat*dt , 0 , self._max_vel )
        if (vel_dot_hat > 0 and vel >= self._max_vel) or (vel_dot_hat < 0 and vel <= 0):
            vel_dot_hat = 0
        delta_dot_hat = np.clip(delta_dot + (self._alpha2 * vel_dot**2 + self._alpha3 * delta_dot**2) * np.random.randn(), -self._max_delta_dot, self._max_delta_dot)
        if (delta_dot_hat > 0 and self._delta >= self._max_delta) or (delta_dot_hat < 0 and self._delta <= -self._max_delta):
            delta_dot_hat = 0
        delta = np.clip(self._delta + delta_dot_hat*dt, -self._max_delta, self._max_delta)
        self._x_ddot = vel_dot*np.cos(self._theta) - vel*np.sin(self._theta)*self._theta_dot
        self._y_ddot = vel_dot*np.sin(self._theta) + vel*np.cos(self._theta)*self._theta_dot
        self._theta_ddot = self._c_r*vel_dot_hat*np.sin(delta)*np.arctan(vel**2)/(self._c_b+vel)**2 \
            - 2*self._c_r*vel*vel_dot*np.sin(delta)/((vel**4 + 1)*(self._c_b + vel)) \
            - self._c_r*delta_dot_hat*np.cos(delta)*np.arctan(vel**2)/(self._c_b+vel)
        self._x_dot = vel*np.cos(self._theta)
        self._y_dot = vel*np.sin(self._theta)
        self._theta_dot = self._c_r * np.sin(-delta) * np.arctan((vel**2))/(self._c_b + vel)
        self._x = self._x + self._x_dot*dt
        self._y = self._y + self._y_dot*dt
        self._theta = self.__wrap_angle(self._theta + self._theta_dot*dt)
        self.__update_inputs(vel, vel_dot_hat, delta, delta_dot_hat)


    def update_velocity_motion_model(self,velocity, rudder_steering_turn_rate, dt):
        vel = velocity #acceleration
        delta_dot = rudder_steering_turn_rate #rudder location
        vel_hat = vel + (self._alpha1 * vel**2 + self._alpha4 * delta_dot**2) * np.random.randn()
        vel_hat = np.clip(vel_hat, 0, self._max_vel)
        delta_dot_hat = np.clip(delta_dot + (self._alpha2 * vel**2 + self._alpha3 * delta_dot**2) * np.random.randn(), -self._max_delta_dot, self._max_delta_dot)
        if (delta_dot_hat > 0 and self._delta >= self._max_delta) or (delta_dot_hat < 0 and self._delta <= -self._max_delta):
            delta_dot_hat = 0
        delta = np.clip(self._delta + delta_dot_hat*dt, -self._max_delta, self._max_delta)
        theta_dot = self._c_r * np.sin(-delta) * np.arctan((vel_hat**2))/(self._c_b + vel_hat)
        vel_dot = (vel_hat - np.sqrt(self._x_dot**2 + self._y_dot**2))/dt
        self._x_ddot = vel_dot*np.cos(self._theta) - vel_hat*np.sin(self._theta)*theta_dot
        self._y_ddot = vel_dot*np.sin(self._theta) + vel_hat*np.cos(self._theta)*theta_dot
        self._theta_ddot = (theta_dot - self._theta_dot)/dt
        self._x_dot = vel_hat*np.cos(self._theta)
        self._y_dot = vel_hat*np.sin(self._theta)
        self._theta_dot = theta_dot
        self._delta_dot = delta_dot_hat
        self._x = self._x + self._x_dot*dt
        self._y = self._y + self._y_dot*dt
        self._theta = self.__wrap_angle(self._theta + self._theta_dot*dt)
        self.__update_inputs(vel_hat, vel_dot, delta, delta_dot_hat)

    def get_vehicle_properties(self):
        return np.array([self._height, self._width, 
                         self._c_r, self._c_b])

    def get_state(self):
        return np.array([[self._x, self._y, self._theta],
                          [self._x_dot, self._y_dot, self._theta_dot],
                          [self._x_ddot, self._y_ddot, self._theta_ddot]])
    
    def get_inputs(self):
        ''' Returns the current inputs '''
        return np.array([[self._vel     , self._delta],
                         [self._vel_dot , self._delta_dot]])
    
    def add_patches_to_axes(self, ax: plt.Axes):
        ax.add_patch(self._boat_fig)
        ax.add_patch(self._rudder_fig)

    def add_patches_to_tuple(self, patches: tuple):
        new_patches = patches + (self._boat_fig, self._rudder_fig)
        return new_patches

    def update_patches(self):
        self._boat_fig.xy = self.get_body_points()
        self._rudder_fig.xy = self.get_rudder_points()
    
    def plot_vehicle_instance(self, ax: plt.Axes):
        boat_fig = plt.Polygon(self.get_body_points(),fc = '0.25',zorder=5)
        rudder_fig = plt.Polygon(self.get_rudder_points(),fc = 'k',zorder=4)
        ax.add_patch(boat_fig)
        ax.add_patch(rudder_fig)

    def get_body_points(self):
        R = self.__get_rotation_matrix(self._theta)
        xy = np.array([[-self._height, self._height, -self._height],
                       [self._width, 0, -self._width]])
        theta = np.linspace(0,2*np.pi,50)
        x_points = np.cos(theta)*self._height
        y_points = np.sin(theta)*self._width
        xy = np.vstack((x_points,y_points))
        xy = np.dot(R,xy)
        xy = xy + np.array([[self._x],[self._y]])
        return np.transpose(xy)
    
    def get_rudder_points(self):
        xy_body_frame = np.array([[-self._height, 0, 0, -self._height],
                                  [self._width/5, self._width/5, -self._width/5, -self._width/5]])
        rudder_rotation = self.__get_rotation_matrix(self._delta)
        rudder_translation = np.array([[-self._height/5],[0]])
        rudder_points = np.dot(rudder_rotation , xy_body_frame) + rudder_translation
        body_rotation = self.__get_rotation_matrix(self._theta)
        body_translation = np.array([[self._x],[self._y]])
        rudder_points = np.dot(body_rotation, rudder_points) + body_translation
        return np.transpose(rudder_points)
    
    def get_center_of_mass_point(self):
        return np.array([self._x,self._y])

    def __get_rotation_matrix(self, theta):
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                    [np.sin(theta), np.cos(theta)]])
        return rotation_matrix

    def __wrap_angle(self,theta):
        return np.arctan2(np.sin(theta), np.cos(theta))
    
    def __update_inputs(self, vel: float, vel_dot: float, delta: float, delta_dot: float):
        self._vel = vel
        self._vel_dot = vel_dot
        self._delta = delta
        self._delta_dot = delta_dot