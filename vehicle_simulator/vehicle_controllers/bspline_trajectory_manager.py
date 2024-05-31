import numpy as np
from vehicle_simulator.vehicle_controllers.bspline_evaluator import BsplineEvaluator

class SplineTrajectoryManager():
    def __init__(self, control_point_list: 'list[np.ndarray]', \
                 scale_factor_list: 'list[float]', start_time = 0, order=3):
        self._control_point_list = control_point_list
        self._scale_factor_list = scale_factor_list
        self._num_trajectories = len(control_point_list)
        self._start_time = start_time
        self._spline_eval = BsplineEvaluator(order)
        self._order = order
        self._end_time = self.get_end_time()

    def get_current_bspline(self, time:float):
        t = time
        if time < self._start_time:
            t = self._start_time
        elif time > self._end_time:
            t = self._end_time
        start_knot = self._start_time 
        control_points = self._control_point_list[0]
        scale_factor = self._scale_factor_list[0]
        for i in range(self._num_trajectories):
            control_points = self._control_point_list[i]
            scale_factor = self._scale_factor_list[i]
            num_intervals = self._spline_eval.count_number_of_control_points(control_points)
            end_knot = num_intervals*scale_factor + start_knot
            if t >= start_knot and t < end_knot:
                break
            elif t >= start_knot and t == self._end_time:
                break
            else:
                start_knot = end_knot
        return control_points, scale_factor, start_knot
    
    def get_control_point_list(self):
        return self._control_point_list
    
    def get_scale_factor_list(self):
        return self._scale_factor_list
    
    def get_start_time(self):
        return self._start_time

    def get_end_time(self):
        end_time = self._start_time
        for i in range(self._num_trajectories):
            control_points = self._control_point_list[i]
            scale_factor = self._scale_factor_list[i]
            num_control_points = self._spline_eval.count_number_of_control_points(control_points)
            num_intervals = num_control_points - self._order
            end_time += num_intervals*scale_factor + self._start_time
        return end_time
    
            
    def get_num_paths(self):
        return self._num_paths
