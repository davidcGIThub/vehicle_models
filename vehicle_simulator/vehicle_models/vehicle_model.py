"""
Bicycle Model Class
"""
import numpy as np
import matplotlib.pyplot as plt

class VehicleModel:

    def __init__(self):
        pass
    
    def set_state(self,states: np.ndarray):
        pass

    def set_inputs(self, input_array: np.ndarray):
        pass
    
    def update_velocity_motion_model(self, motion_command:float, turn_command:float, dt:float):
        pass

    def update_acceleration_motion_model(self, motion_command:float, turn_command:float, dt:float):
        pass
    
    def get_vehicle_properties(self):
        return np.array([])
    
    def get_state(self):
        return np.array([])
    
    def get_inputs(self):
        return np.array([])
    
    def add_patches_to_axes(self, ax: plt.Axes):
        pass

    def add_patches_to_tuple(self, patches: tuple):
        return ()

    def update_patches(self):
        pass
    
    def plot_vehicle_instance(self, ax: plt.Axes):
        pass
    
    def get_center_of_mass_point(self):
        return np.array([])




