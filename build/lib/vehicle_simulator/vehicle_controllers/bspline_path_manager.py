import numpy as np
from vehicle_simulator.vehicle_controllers.bspline_evaluator import BsplineEvaluator

class SplinePathManager():
    def __init__(self, control_point_list: 'list[np.ndarray]', order=3, tolerance = 1):
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
    
    def get_tracked_path_data(self, path_control_point_list:'list[np.ndarray]', pts_per_interval):
        num_paths = len(path_control_point_list)
        path_data_list = []
        tracked_path_data_list = []
        for i in range(num_paths):
            control_points = path_control_point_list[i]
            path_data = self._spline_eval.matrix_bspline_evaluation_for_dataset(control_points, pts_per_interval)
            path_data_list.append(path_data)
        for i in range(num_paths-1):
            path = path_data_list[i]
            next_path = path_data_list[i+1]
            start_point_next_path = next_path[:,0][:,None] 
            distances_to_start = np.linalg.norm(path - start_point_next_path,2,0)
            index_end = np.argmin(distances_to_start)
            tracked_path = path[:,0:index_end]
            tracked_path_data_list.append(tracked_path)
        tracked_path_data_list.append(path_data_list[-1])
        return tracked_path_data_list


    def get_num_paths(self):
        return self._num_paths
