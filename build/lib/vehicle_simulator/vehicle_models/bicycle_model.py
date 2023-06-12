"""
Bicycle Model Class
"""
from matplotlib.text import get_rotation
import numpy as np
import matplotlib.pyplot as plt

class BicycleModel:

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
        self._R = R 
        self._max_delta = max_delta
        self._max_vel = max_vel
        self._max_vel_dot = max_vel_dot
    
    def setState(self,states):
        self._x = states[0,0]
        self._y = states[0,1]
        self._theta = states[0,2]
        self._delta = states[0,3]
        self._x_dot = states[1,0]
        self._y_dot = states[1,1]
        self._theta_dot = states[1,2]
        self._delta_dot = states[1,3]
        self._x_ddot = states[2,0]
        self._y_ddot = states[2,1]
        self._theta_ddot = states[2,2]
    
    def update_velocity_motion_model(self, velocity, wheel_turn_rate, dt):
        vel = velocity
        delta_dot = wheel_turn_rate # front wheel turning rate 
        vel_hat = np.clip( vel + (self._alpha1 * vel**2 + self._alpha4 * delta_dot**2) * np.random.randn(), 0, self._max_vel )
        delta_dot_hat = delta_dot + (self._alpha2 * vel**2 + self._alpha3 * delta_dot**2) * np.random.randn()
        if (delta_dot_hat > 0 and self._delta >= self._max_delta) or (delta_dot_hat < 0 and self._delta <= -self._max_delta):
            delta_dot_hat = 0
        delta = np.clip(self._delta + delta_dot_hat * dt, -self._max_delta, self._max_delta)
        # following functions should be called in the following order
        beta, beta_dot = self.__get_beta_states(delta, delta_dot_hat)
        vel_dot = (vel_hat - np.sqrt(self._x_dot**2 + self._y_dot**2))/dt
        self.__update_second_derivative_states(self._theta, self._theta_dot, delta, delta_dot_hat, \
                                               beta, beta_dot, vel, vel_dot)
        self.__update_derivative_states(vel_hat, self._theta, delta_dot_hat, beta, dt)
        self.__update_states(self._x_dot,self._y_dot, self._theta_dot,delta, dt)
        return vel_hat, delta_dot_hat


    def update_acceleration_motion_model(self, longitudinal_acceleration, wheel_turn_rate, dt):
        vel_dot = longitudinal_acceleration
        delta_dot = wheel_turn_rate # front wheel turning rate 
        vel_dot_hat = np.clip(vel_dot + (self._alpha1 * vel_dot**2 + self._alpha4 * delta_dot**2) * np.random.randn(), -self._max_vel_dot, self._max_vel_dot)
        vel = np.clip( np.sqrt(self._x_dot**2 + self._y_dot**2) + vel_dot_hat*dt , 0 , self._max_vel  )
        if (vel_dot_hat > 0 and vel >= self._max_vel) or (vel_dot_hat < 0 and vel <= 0):
            vel_dot_hat = 0
        delta_dot_hat = delta_dot + (self._alpha2 * vel_dot**2 + self._alpha3 * delta_dot**2) * np.random.randn()
        if (delta_dot_hat > 0 and self._delta >= self._max_delta) or (delta_dot_hat < 0 and self._delta <= -self._max_delta):
            delta_dot_hat = 0
        delta = np.clip(self._delta + delta_dot_hat * dt, -self._max_delta, self._max_delta)
        #following functions should be called in the following order
        beta, beta_dot = self.__get_beta_states(delta, delta_dot_hat)
        self.__update_second_derivative_states(self._theta, self._theta_dot, delta, delta_dot_hat, \
                                               beta, beta_dot, vel, vel_dot_hat)
        self.__update_derivative_states(vel, self._theta, delta_dot_hat, beta, dt)
        self.__update_states(self._x_dot,self._y_dot, self._theta_dot,delta, dt)
        return vel_dot_hat, delta_dot_hat
    
    def get_length(self):
        return self._L
    
    def get_angular_rate(self):
        return np.abs(self._theta_dot)
    
    def get_curvature(self):
        vel = np.sqrt(self._x_dot**2 + self._y_dot**2)
        return np.abs(self._theta_dot)/vel
    
    def get_centripetal_acceleration(self):
        vel = np.sqrt(self._x_dot**2 + self._y_dot**2)
        return np.abs(self._theta_dot)*vel
    
    def __update_states(self,x_dot,y_dot, theta_dot, delta, dt):
        self._x = self._x + x_dot * dt
        self._y = self._y + y_dot * dt
        self._theta = self.wrapAngle(self._theta + self._theta_dot * dt)
        self._delta = delta

    def __update_derivative_states(self, vel, theta, delta_dot, beta, dt):
        self._x_dot = vel*np.cos(theta + beta)
        self._y_dot = vel*np.sin(theta + beta)
        self._delta = np.clip(self._delta + delta_dot * dt, -self._max_delta, self._max_delta)
        self._theta_dot = vel*np.cos(beta)*np.tan(self._delta)/self._L
    
    def __update_second_derivative_states(self, theta, theta_dot, delta, delta_dot, beta, beta_dot, vel, vel_dot):
        self._x_ddot = vel_dot*np.cos(theta + beta) - \
            (theta_dot+beta_dot)*vel*np.sin(beta+theta)
        self._y_ddot = vel_dot*np.sin(theta + beta) + \
            (theta_dot + beta_dot)*vel*np.cos(beta+theta)
        self._theta_ddot = vel_dot*np.cos(beta)*np.tan(delta) + \
            vel*(delta_dot*np.cos(beta)/np.cos(delta)**2 - beta_dot*np.sin(beta)*np.tan(delta))
    
    def __get_beta_states(self, delta, delta_dot):
        beta = np.arctan2(self._lr*np.tan(delta) , self._L)
        beta_dot = self._L*self._lr*delta_dot/ \
            ((self._lr**2)*np.sin(delta)**2 + (self._L**2)*np.cos(delta)**2)
        return beta, beta_dot
    
    def get_velocity(self):
        velocity = np.sqrt(self._x_dot**2 + self._y_dot**2)
        return velocity
 

    def getState(self):
        states = np.array([[self._x,self._y,self._theta,self._delta],
                    [self._x_dot,self._y_dot,self._theta_dot,self._delta_dot],
                    [self._x_ddot,self._y_ddot,self._theta_ddot,0]])
        return states


    def getPoints(self,xy):
        rotation_matrix = self.getRotationMatrix(self._theta)
        xy = np.dot(rotation_matrix,xy)
        xy = xy + np.array([[self._x],[self._y]])
        return np.transpose(xy)

    def getBodyPoints(self):
        xy_body_frame = np.array([[-self._lr, self._L-self._lr, self._L-self._lr, -self._lr],
                                  [self._R/5, self._R/5, -self._R/5, -self._R/5]])
        body_points = self.getPoints(xy_body_frame)
        return body_points

    def getBackWheelPoints(self):
        xy_body_frame = np.array([[-self._lr-self._R, -self._lr+self._R, -self._lr+self._R, -self._lr-self._R],
                       [self._R/3,self._R/3,-self._R/3,-self._R/3]])
        backWheelPoints = self.getPoints(xy_body_frame)
        return backWheelPoints

    def getFrontWheelPoints(self):
        xy_wheel_frame_straight = np.array([[-self._R, self._R, self._R, -self._R],
                                            [self._R/3, self._R/3, -self._R/3, -self._R/3]])
        wheel_rotation_matrix = self.getRotationMatrix(self._delta)
        xy_wheel_frame_rotated = np.dot(wheel_rotation_matrix , xy_wheel_frame_straight)
        xy_body_frame = xy_wheel_frame_rotated + np.array([[self._L - self._lr],[0]])
        frontWheelPoints = self.getPoints(xy_body_frame)
        return frontWheelPoints

    def getRotationMatrix(self, theta):
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                    [np.sin(theta), np.cos(theta)]])
        return rotation_matrix
    
    def wrapAngle(self,theta):
        return np.arctan2(np.sin(theta), np.cos(theta))
    
    def plot_bike(self, ax):
        front_wheel_fig = plt.Polygon(self.getFrontWheelPoints(),fc = 'k',zorder=10)
        back_wheel_fig = plt.Polygon(self.getBackWheelPoints(),fc = 'k',zorder=10)
        body_fig = plt.Polygon(self.getBodyPoints(),fc = 'g',zorder=10)
        ax.add_patch(front_wheel_fig)
        ax.add_patch(back_wheel_fig)
        ax.add_patch(body_fig)



