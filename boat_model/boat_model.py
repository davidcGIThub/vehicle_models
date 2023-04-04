"""
Boat Model Class
"""
import numpy as np
import sys

# velocity motion model
import numpy as np 

class BoatModel:

    def __init__(self, 
                 x = 0, 
                 y = 0, 
                 vel = 0,
                 theta = np.pi/2.0, 
                 delta = 0,
                 alpha = np.array([0.1,0.01,0.01,0.1]),
                 dt = 0.1,
                 height = 1,
                 width = 0.5,
                 delta_max = np.pi/2,
                 a_max = 5,
                 v_max = 10):
        self.x = x
        self.y = y
        self.vel = vel
        self.theta = self.wrapAngle(theta)
        self.delta = delta
        self.x_dot = 0
        self.y_dot = 0
        self.theta_dot = 0
        self.x_prev = x
        self.y_prev = y
        self.vel_prev = vel
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
        self.delta_max = delta_max
        self.a_max = a_max
        self.v_max = v_max
    
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
        a = u[0] #acceleration
        delta = u[1] #rudder location
        a_hat = a + (self.alpha1 * a**2 + self.alpha4 * delta**2) * np.random.randn()
        a_hat = np.clip(a_hat, -self.a_max, self.a_max)
        delta_hat = delta + (self.alpha2 * a**2 + self.alpha3 * delta**2) * np.random.randn()
        self.delta = np.clip(delta_hat, -self.delta_max, self.delta_max)
        self.x_prev = self.x
        self.y_prev = self.y
        self.vel_prev = self.vel
        self.theta_prev = self.theta
        self.x_dot_prev = self.x_dot
        self.y_dot_prev = self.y_dot
        self.theta_dot_prev = self.theta_dot
        centripetal_acceleration = np.sin(-self.delta)
        self.vel = self.vel_prev + a_hat*self.dt
        self.vel = np.clip(self.vel, 0, self.v_max)
        if self.delta == 0:
            self.theta_dot = 0
        elif self.vel == 0:
            self.theta_dot = sys.float_info.max
        else:
            self.theta_dot = centripetal_acceleration / self.vel
        self.x_dot = self.vel * np.cos(self.theta)
        self.y_dot = self.vel * np.sin(self.theta)
        self.x = self.x + self.x_dot * self.dt
        self.y = self.y + self.y_dot * self.dt
        self.theta = self.wrapAngle(self.theta + self.theta_dot * self.dt)

    def getState(self):
        return np.array([self.x,self.y,self.theta, 
                         self.x_dot, self.y_dot, self.theta_dot])

    def getPreviousState(self):
        return np.array([self.x_prev,self.y_prev, self.theta_prev,
                         self.x_dot_prev, self.y_dot_prev, self.theta_dot_prev])

    def getBodyPoints(self):
        R = self.getRotationMatrix(self.theta)
        xy = np.array([[-self.height, self.height, -self.height],
                       [self.width, 0, -self.width]])
        theta = np.linspace(0,2*np.pi,50)
        x_points = np.cos(theta)*self.height
        y_points = np.sin(theta)*self.width
        xy = np.vstack((x_points,y_points))
        xy = np.dot(R,xy)
        xy = xy + np.array([[self.x],[self.y]])
        return np.transpose(xy)
    
    def getRudderPoints(self):
        xy_body_frame = np.array([[-self.height, 0, 0, -self.height],
                                  [self.width/5, self.width/5, -self.width/5, -self.width/5]])
        rudder_rotation = self.getRotationMatrix(self.delta)
        rudder_translation = np.array([[-self.height/5],[0]])
        rudder_points = np.dot(rudder_rotation , xy_body_frame) + rudder_translation
        body_rotation = self.getRotationMatrix(self.theta)
        body_translation = np.array([[self.x],[self.y]])
        rudder_points = np.dot(body_rotation, rudder_points) + body_translation
        return np.transpose(rudder_points)

    def getRotationMatrix(self, theta):
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                    [np.sin(theta), np.cos(theta)]])
        return rotation_matrix

    def wrapAngle(self,theta):
        return np.arctan2(np.sin(theta), np.cos(theta))