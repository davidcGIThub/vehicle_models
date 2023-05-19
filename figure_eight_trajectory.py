"""
This module contains code to evaluate figure eights trajectories
"""
from matplotlib import scale
import numpy as np 

class FigureEightTrajectoryGenerator:
    """
    This class contains code to evaluate  figure eight trajectories
    """

    def __init__(self, amplitude, frequency):
        '''
        Constructor for the BsplineMatrixEvaluation class, each column of
        control_points is a control point. Start time should be an integer.
        '''
        self._amplitude = amplitude
        self._frequency = frequency

    def evaluate_trajectory_at_time_t(self, t):
        x = self._amplitude*np.sin(t*self._frequency)
        y = self._amplitude*np.sin(t*self._frequency)*np.cos(t*self._frequency)
        return np.array([x,y])

    def evaluate_derivative_at_time_t(self, t):
        x_dot = self._amplitude*self._frequency*np.cos(t*self._frequency)
        y_dot = self._amplitude*self._frequency*(np.cos(t*self._frequency)**2 - np.sin(t*self._frequency)**2)
        return np.array([x_dot,y_dot])

    def evaluate_second_derivative_at_time_t(self, t):
        x_ddot = -self._amplitude*(self._frequency**2)*np.sin(t*self._frequency)
        y_ddot = -4*self._amplitude*(self._frequency**2)*np.sin(t*self._frequency)*np.cos(t*self._frequency)
        return np.array([x_ddot,y_ddot])

    def evaluate_trajectory_over_time_interval(self, time_array):
        position_array = np.zeros((len(time_array),2))
        for i in range(len(time_array)):
            xy = self.evaluate_trajectory_at_time_t(time_array[i])
            position_array[i,0] = xy[0]
            position_array[i,1] = xy[1]
        return position_array