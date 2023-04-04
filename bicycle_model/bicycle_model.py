"""
Bicycle Model Class
"""
from matplotlib.text import get_rotation
import numpy as np

class BicycleModel:

    def __init__(self, 
                 x = 2, 
                 y = 2, 
                 theta = np.pi/4.0, 
                 delta = 0,
                 lr = 0.5,
                 L = 1,
                 R = 0.2,
                 alpha = np.array([0.1,0.01,0.1,0.01]),
                 dt = 0.1,
                 delta_max = np.pi/4):
        self._x = x
        self._y = y
        self._theta = theta
        self._delta = delta
        self._lr = lr
        self._L = L
        self._alpha1 = alpha[0]
        self._alpha2 = alpha[1]
        self._alpha3 = alpha[2]
        self._alpha4 = alpha[3]
        self._R = R 
        self._dt = dt
        self._delta_max = delta_max
    
    def setState(self,x,y,theta,delta):
        self._x = x
        self._y = y
        self._theta = theta
        self._delta = delta

    def vel_motion_model(self,input):
        v = input[0] # velocity 
        phi = input[1] # front wheel turning rate

        v_hat = v + (self._alpha1 * v**2 + self._alpha4 * phi**2) * np.random.randn()
        phi_hat = phi + (self._alpha2 * v**2 + self._alpha3 * phi**2) * np.random.randn()
        self._delta = np.clip(self._delta + phi_hat * self._dt, -self._delta_max, self._delta_max)
        beta = np.arctan2(self._lr*np.tan(self._delta),self._L)
        self._theta = self._theta + v_hat*np.cos(beta)*np.tan(self._delta)/self._L * self._dt
        self._x = self._x + v_hat*np.cos(self._theta+beta) * self._dt
        self._y = self._y + v_hat*np.sin(self._theta+beta) * self._dt

    def getState(self):
        return np.array([self._x,self._y,self._theta,self._delta])

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

