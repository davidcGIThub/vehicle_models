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
                 dt = 0.1,
                 height = 1,
                 width = 0.5):
        self.x = x
        self.y = y
        self.theta = self.wrapAngle(theta)
        self.x_dot = 0
        self.y_dot = 0
        self.theta_dot = 0
        self.x_prev = x
        self.y_prev = y
        self.theta_prev = theta
        self.x_dot_prev = self.x_dot
        self.y_dot_prev = self.y_dot
        self.theta_dot_prev = self.theta_dot
        self.alpha1 = alpha[0]
        self.alpha2 = alpha[1]
        self.alpha3 = alpha[2]
        self.alpha4 = alpha[3]
        self.dt = dt
        self.height = height
        self.width = width
    
    def setState(self,x,y,theta,x_dot,y_dot,theta_dot):
        self.x = x
        self.y = y
        self.theta = self.wrapAngle(theta)
        self.x_dot = x_dot
        self.y_dot = y_dot
        self.theta_dot = theta_dot

    def setPreviousState(self, x_prev, y_prev, theta_prev,
                               x_dot_prev,y_dot_prev,theta_dot_prev):
        self.x_prev = x_prev
        self.y_prev = y_prev
        self.theta_prev = self.wrapAngle(theta_prev)
        self.x_dot_prev = x_dot_prev
        self.y_dot_prev = y_dot_prev
        self.theta_dot_prev = theta_dot_prev

    def velMotionModel(self,u):
        v = u[0]
        w = u[1]
        v_hat = v + (self.alpha1 * v**2 + self.alpha2 * w**2) * np.random.randn()
        w_hat = w + (self.alpha3 * v**2 + self.alpha4 * w**2) * np.random.randn()
        self.x_prev = self.x
        self.y_prev = self.y
        self.theta_prev = self.theta
        self.x_dot_prev = self.x_dot
        self.y_dot_prev = self.y_dot
        self.theta_dot_prev = self.theta_dot
        self.x_dot = v_hat * np.cos(self.theta)
        self.y_dot = v_hat * np.sin(self.theta)
        self.theta_dot = w_hat
        self.x = self.x + self.x_dot * self.dt
        self.y = self.y + self.y_dot * self.dt
        self.theta = self.wrapAngle(self.theta + self.theta_dot * self.dt)

    def getState(self):
        return np.array([self.x,self.y,self.theta, 
                         self.x_dot, self.y_dot, self.theta_dot])

    def getPreviousState(self):
        return np.array([self.x_prev,self.y_prev, self.theta_prev,
                         self.x_dot_prev, self.y_dot_prev, self.theta_dot_prev])

    def getPoints(self):
        R = self.getRotationMatrix(self.theta)
        xy = np.array([[-self.height, self.height, -self.height],
                       [self.width, 0, -self.width]])
        xy = np.dot(R,xy)
        xy = xy + np.array([[self.x],[self.y]])
        return np.transpose(xy)

    def getRotationMatrix(self, theta):
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                    [np.sin(theta), np.cos(theta)]])
        return rotation_matrix

    def wrapAngle(self,theta):
        return np.arctan2(np.sin(theta), np.cos(theta))

