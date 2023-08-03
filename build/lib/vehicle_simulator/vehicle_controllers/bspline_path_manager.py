import numpy as np
from vehicle_simulator.vehicle_controllers.bspline_evaluator import BsplineEvaluator

class SplinePathManager():
    def __init__(self, control_point_list: 'list[np.ndarray]', order=3, tolerance = 50):
        self._control_point_list = control_point_list
        self._num_paths = len(control_point_list)
        self._path_index = 0
        self._spline_eval = BsplineEvaluator(order, 100, 100)
        self._tolerance = tolerance

    def get_current_path_control_points(self, position: np.ndarray):
        control_points = self._control_point_list[self._path_index]
        if self._path_index < self._num_paths-1:
            next_control_points = self._control_point_list[self._path_index+1]
            distance_to_end = self._spline_eval.get_distance_to_endpoint(control_points, position)
            distance_to_start_of_next = self._spline_eval.get_distance_to_startpoint(next_control_points, position)
            if distance_to_end < self._tolerance or distance_to_start_of_next < self._tolerance:
                self._path_index += 1
                control_points = next_control_points
        return control_points

    def get_num_paths(self):
        return self._num_paths
