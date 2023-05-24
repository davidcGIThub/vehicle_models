"""
Unicycle Model Class
"""
import numpy as np

# velocity motion model
import numpy as np 

class UnicycleModel:

    def __init__(self, 
                 x = 0, 
                 y = 0, 
                 theta = np.pi/2.0, 
                 alpha = np.array([0.1,0.01,0.01,0.1]),
                 height = 0.5,
                 width = 0.25,
                 max_vel = 10,
                 max_theta_dot = 15):
        self._x = x
        self._y = y
        self._theta = self.wrapAngle(theta)
        self._x_dot = 0
        self._y_dot = 0
        self._theta_dot = 0
        self._x_ddot = 0
        self._y_ddot = 0
        self._theta_ddot = 0
        self._alpha1 = alpha[0]
        self._alpha2 = alpha[1]
        self._alpha3 = alpha[2]
        self._alpha4 = alpha[3]
        self._height = height
        self._width = width
        self._max_vel = max_vel
        self._max_theta_dot = max_theta_dot

    def setState(self,states):
        self._x = states[0,0]
        self._y = states[0,1]
        self._theta = self.wrapAngle(states[0,2])
        self._x_dot = states[1,0]
        self._y_dot = states[1,1]
        self._theta_dot = states[1,2]
        self._x_ddot = states[2,0]
        self._y_ddot = states[2,1]
        self._theta_ddot = states[2,2]

    def getState(self):
        return np.array([[self._x, self._y, self._theta],
                          [self._x_dot, self._y_dot, self._theta_dot],
                          [self._x_ddot, self._y_ddot, self._theta_ddot]])

    def update_velocity_motion_model(self, velocity, angular_rate, dt):
        vel = velocity
        theta_dot = angular_rate
        vel_hat = vel + (self._alpha1 * vel**2 + self._alpha2 * theta_dot**2) * np.random.randn()
        theta_dot_hat = theta_dot + (self._alpha3 * vel**2 + self._alpha4 * theta_dot**2) * np.random.randn()
        vel_dot = (vel_hat - np.sqrt(self._x_dot**2 + self._y_dot**2))/dt
        self._x_ddot = vel_dot*np.cos(self._theta) - vel_hat*np.sin(self._theta)*theta_dot_hat
        self._y_ddot = vel_dot*np.sin(self._theta) + vel_hat*np.cos(self._theta)*theta_dot_hat
        self._theta_ddot = (theta_dot_hat - self._theta_dot)/dt
        self._x_dot = vel_hat * np.cos(self._theta)
        self._y_dot = vel_hat * np.sin(self._theta)
        self._theta_dot = theta_dot_hat
        self._x = self._x + self._x_dot * dt
        self._y = self._y + self._y_dot * dt
        self._theta = self.wrapAngle(self._theta + self._theta_dot * dt)

    def update_acceleration_motion_model(self, longitudinal_acceleration, angular_acceleration, dt):
        vel_dot = longitudinal_acceleration
        theta_ddot = angular_acceleration
        vel_dot_hat = vel_dot + (self._alpha1 * vel_dot**2 + self._alpha2 * theta_ddot**2) * np.random.randn()
        vel = np.clip( np.sqrt(self._x_dot**2 + self._y_dot**2) + vel_dot_hat*dt , 0 , self._max_vel )
        if (vel_dot_hat > 0 and vel >= self._max_vel) or (vel_dot_hat < 0 and vel <= 0):
            vel_dot_hat = 0
        theta_ddot_hat = theta_ddot + (self._alpha3 * vel_dot**2 + self._alpha4 * theta_ddot**2) * np.random.randn()
        theta_dot = np.clip(self._theta_dot + theta_ddot_hat*dt, -self._max_theta_dot, self._max_theta_dot)
        self._x_ddot = vel_dot_hat*np.cos(self._theta) - vel*np.sin(self._theta)*theta_dot
        self._y_ddot = vel_dot_hat*np.sin(self._theta) + vel*np.cos(self._theta)*theta_dot
        self._theta_ddot = theta_ddot_hat
        self._x_dot = vel * np.cos(self._theta)
        self._y_dot = vel * np.sin(self._theta)
        self._theta_dot = theta_dot
        self._x = self._x + self._x_dot * dt
        self._y = self._y + self._y_dot * dt
        self._theta = self.wrapAngle(self._theta + self._theta_dot * dt)

    def getPoints(self):
        R = self.getRotationMatrix(self._theta)
        xy = np.array([[-self._height, self._height, -self._height],
                       [self._width, 0, -self._width]])
        xy = np.dot(R,xy)
        xy = xy + np.array([[self._x],[self._y]])
        return np.transpose(xy)

    def getRotationMatrix(self, theta):
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                    [np.sin(theta), np.cos(theta)]])
        return rotation_matrix

    def wrapAngle(self,theta):
        return np.arctan2(np.sin(theta), np.cos(theta))
