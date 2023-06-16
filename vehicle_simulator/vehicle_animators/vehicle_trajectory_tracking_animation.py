"""
Bicycle Trajectory Animation class
"""
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
from vehicle_simulator.vehicle_models.vehicle_model import VehicleModel
from vehicle_simulator.vehicle_controllers.trajectory_tracker import TrajectoryTracker
from dataclasses import dataclass
from time import sleep

@dataclass
class TrajectoryData:
    location_data: np.ndarray
    velocity_data: np.ndarray
    acceleration_data: np.ndarray
    jerk_data: np.ndarray
    time_data: np.ndarray
    curvature_data: np.ndarray = None
    angular_rate_data: np.ndarray = None
    centripetal_acceleration_data: np.ndarray = None
    longitudinal_acceleration_data: np.ndarray = None

    def __post_init__(self):
        location_dimension = np.shape(self.location_data)[0]
        velocity_dimension = np.shape(self.velocity_data)[0]
        acceleration_dimension = np.shape(self.acceleration_data)[0]
        jerk_dimension = np.shape(self.jerk_data)[0]
        location_length = np.shape(self.location_data)[1]
        velocity_length = np.shape(self.velocity_data)[1]
        acceleration_length = np.shape(self.acceleration_data)[1]
        jerk_length = np.shape(self.jerk_data)[1]
        time_length = len(self.time_data)
        if location_dimension != velocity_dimension or \
                velocity_dimension != acceleration_dimension or \
                acceleration_dimension != jerk_dimension:
            raise Exception("Trajectory data dimensions are not equal")
        if location_length != velocity_length or \
                velocity_length != acceleration_length or \
                acceleration_length != jerk_length or \
                jerk_length !=  time_length:
            raise Exception("Trajectory data lengths are not equal")
        cross_term_data = self.__calculate_cross_term_data()
        dot_product_data = self.__calculate_dot_product_term_data()
        velocity_magnitude_data = self.__calculate_velocity_magnitude_data()
        velocity_magnitude_data[velocity_magnitude_data < 8e-10] = 1
        self.centripetal_acceleration = cross_term_data/velocity_magnitude_data
        self.angular_rate_data = cross_term_data/velocity_magnitude_data**2
        self.curvature_data = cross_term_data/velocity_magnitude_data**3
        self.longitudinal_acceleration_data = dot_product_data/velocity_magnitude_data

    def __calculate_cross_term_data(self):
        cross_product_norm = np.abs(np.cross(self.velocity_data.T, self.acceleration_data.T).flatten())
        return cross_product_norm
    
    def __calculate_dot_product_term_data(self):
        dot_product_term = np.sum(self.acceleration_data*self.velocity_data,0)
        return dot_product_term
    
    def __calculate_velocity_magnitude_data(self):
        return np.linalg.norm(self.velocity_data,2,0)

class VehicleTrajectoryTrackingSimulator:

    def __init__(self, vehicle_model: VehicleModel, 
                 trajectory_tracker: TrajectoryTracker):
        self._vehicle_model = vehicle_model
        self._trajectory_tracker = trajectory_tracker

    def run_simulation(self, desired_trajectory_data: TrajectoryData, 
                       animate = True, plot=True):
        inputs_list, states_list, vehicle_trajectory_data = \
            self.collect_simulation_data(desired_trajectory_data)
        properties = self._vehicle_model.get_vehicle_properties()
        length = properties[0]
        if animate == True:
            self.animate_simulation(states_list, inputs_list, 
                                    vehicle_trajectory_data, desired_trajectory_data,
                                    margins=length)
        if plot == True:
            self.plot_simulation(states_list, inputs_list, 
                                vehicle_trajectory_data, 
                                desired_trajectory_data,
                                margins=length)
        return vehicle_trajectory_data
        
    def update_vehicle_state_and_inputs(self, inputs, state):
        self._vehicle_model.set_inputs(inputs)
        self._vehicle_model.set_state(state)

    def collect_simulation_data(self, desired_trajectory_data: TrajectoryData):
        #### extract path data ####
        location_data = desired_trajectory_data.location_data
        velocity_data = desired_trajectory_data.velocity_data
        acceleration_data = desired_trajectory_data.acceleration_data
        jerk_data = desired_trajectory_data.jerk_data
        time_data = desired_trajectory_data.time_data
        states_list = []
        inputs_list = []
        num_data_points = len(time_data)
        #### run simulation ####
        vehicle_location_data = location_data*0
        vehicle_velocity_data = velocity_data*0
        vehicle_acceleration_data = acceleration_data*0
        vehicle_jerk_data = jerk_data*0
        dt = time_data[1] - time_data[0]
        for i in range(num_data_points): 
            desired_states = np.vstack((location_data[:,i], velocity_data[:,i],
                                         acceleration_data[:,i],jerk_data[:,i]))
            inputs = self._vehicle_model.get_inputs()
            states = self._vehicle_model.get_state()
            motion_command, turn_command = self._trajectory_tracker.mpc_control_accel_input(inputs, states, desired_states)
            self._vehicle_model.update_acceleration_motion_model(motion_command, turn_command, dt)
            # save vehicle states
            inputs_list.append(inputs)
            states_list.append(states)
            # save vehicle calculated states
            vehicle_location_data[:,i] = states[0,0:2]
            vehicle_velocity_data[:,i] = states[1,0:2]
            vehicle_acceleration_data[:,i] = states[2,0:2]
            # vehicle_jerk_data = 
        vehicle_trajectory_data = TrajectoryData(vehicle_location_data, vehicle_velocity_data,
                                                 vehicle_acceleration_data, vehicle_jerk_data,
                                                 time_data)
        return inputs_list, states_list, vehicle_trajectory_data

    def animate_simulation(self, states_list: 'list[np.ndarray]', 
                           inputs_list: 'list[np.ndarray]', 
                           vehicle_trajectory_data: TrajectoryData, 
                           desired_trajectory_data: TrajectoryData, 
                           margins = 0, sleep_time = 0):
        time_data = desired_trajectory_data.time_data
        path_location_data = desired_trajectory_data.location_data
        vehicle_location_data = vehicle_trajectory_data.location_data
        x_limits = np.array([np.min(np.concatenate((vehicle_location_data[0,:], path_location_data[0,:]))) - margins, 
                             np.max(np.concatenate((vehicle_location_data[0,:], path_location_data[0,:]))) + margins])
        y_limits = np.array([np.min(np.concatenate((vehicle_location_data[1,:], path_location_data[1,:]))) - margins, 
                             np.max(np.concatenate((vehicle_location_data[1,:], path_location_data[1,:]))) + margins])
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                            xlim=(x_limits[0],x_limits[1]), ylim=(y_limits[0],y_limits[1]))
        ax.grid()
        center_of_mass = plt.Circle((vehicle_location_data[0,0], vehicle_location_data[1,0]), 
                                    radius=0.1, fc='none', ec="k", zorder=11)
        desired_position_fig = plt.Circle((0, 0), radius=0.1, fc='tab:blue', zorder=10)
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
        ax.plot(path_location_data[0,:],path_location_data[1,:])
        dt = time_data[1] - time_data[0]
        def init():
            #initialize animation
            self._vehicle_model.add_patches_to_axes(ax)
            time_text.set_text('')
            patches = (desired_position_fig, center_of_mass, time_text)
            all_patches = self._vehicle_model.add_patches_to_tuple(patches)
            ax.add_patch(desired_position_fig)
            ax.add_patch(center_of_mass)
            return all_patches
        def animate(i):
            # propogate robot motion
            t = time_data[i]
            self._vehicle_model.set_state(states_list[i])
            self._vehicle_model.set_inputs(inputs_list[i])
            self._vehicle_model.update_patches()
            desired_position_fig.center = (path_location_data[0,i], path_location_data[1,i])
            center_of_mass.center = self._vehicle_model.get_center_of_mass_point()
            time_text.set_text('time = %.1f' % t)
            sleep(sleep_time)
            patches = (desired_position_fig, center_of_mass, time_text)
            all_patches = self._vehicle_model.add_patches_to_tuple(patches)
            return all_patches
        animate(0)
        ani = animation.FuncAnimation(fig, animate, frames = np.size(time_data), 
                                        interval = dt*100, blit = True, 
                                        init_func = init, repeat = False)
        plt.show()


    def plot_simulation(self, states_list: 'list[np.ndarray]', 
                           inputs_list: 'list[np.ndarray]', 
                           vehicle_trajectory_data: TrajectoryData, 
                           desired_trajectory_data: TrajectoryData, 
                           margins = 0, vehicle_instances_per_plot = 5):
        path_location_data = desired_trajectory_data.location_data
        vehicle_location_data = vehicle_trajectory_data.location_data
        #### Path Plot ####
        fig = plt.figure()
        num_data_points = np.shape(path_location_data)[1]
        ax = fig.add_subplot(111, aspect='equal', autoscale_on=False)
        x_limits = np.array([np.min(np.concatenate((vehicle_location_data[0,:], path_location_data[0,:]))) - margins, 
                             np.max(np.concatenate((vehicle_location_data[0,:], path_location_data[0,:]))) + margins])
        y_limits = np.array([np.min(np.concatenate((vehicle_location_data[1,:], path_location_data[1,:]))) - margins, 
                             np.max(np.concatenate((vehicle_location_data[1,:], path_location_data[1,:]))) + margins])
        ax.set_ybound((y_limits[0],y_limits[1]))
        ax.set_xbound((x_limits[0],x_limits[1]))
        ax.plot(path_location_data[0,:],path_location_data[1,:], color = 'tab:blue', label = "path")
        ax.plot(vehicle_location_data[0,:],vehicle_location_data[1,:], linestyle="--",
            color = 'tab:red', label="true position")
        center_of_mass = plt.Circle((vehicle_location_data[0,0], vehicle_location_data[1,0]), 
                                    radius=0.1, fc='none', ec="k", zorder=10)
        for i in range(num_data_points):
            if i%int(num_data_points/vehicle_instances_per_plot) == 0:
                self._vehicle_model.set_state(states_list[i])
                self._vehicle_model.set_inputs(inputs_list[i])
                self._vehicle_model.plot_vehicle_instance(ax)
                center_of_mass = plt.Circle((vehicle_location_data[0,i], 
                                             vehicle_location_data[1,i]), 
                                            radius=0.1, fc='none', ec="k", zorder=10)
                ax.add_patch(center_of_mass)
        ax.set_xlabel("x position")
        ax.set_ylabel("y position")
        ax.legend()
        plt.show()

    def plot_simulation_dynamics(self, desired_trajectory_data: TrajectoryData, 
                                 vehicle_trajectory_data: TrajectoryData,
                                 max_velocity: float, max_acceleration: float,
                                 max_turn_value: float, turn_type: str):
        # data extraction
        path_location_data = desired_trajectory_data.location_data
        path_velocity_data = desired_trajectory_data.velocity_data
        path_acceleration_data = desired_trajectory_data.acceleration_data
        path_longitudinal_acceleration_data = desired_trajectory_data.longitudinal_acceleration_data
        path_time_data = desired_trajectory_data.time_data
        vehicle_location_data = vehicle_trajectory_data.location_data
        vehicle_velocity_data = vehicle_trajectory_data.location_data
        vehicle_longitudinal_acceleration_data = vehicle_trajectory_data.longitudinal_acceleration_data
        vehicle_time_data = vehicle_trajectory_data.time_data
        if turn_type == "curvature": 
            path_turn_data = desired_trajectory_data.curvature_data
            vehicle_turn_data = vehicle_trajectory_data.curvature_data
        elif turn_type == "angular_rate": 
            path_turn_data = desired_trajectory_data.angular_rate_data
            vehicle_turn_data = vehicle_trajectory_data.angular_rate_data
        elif turn_type == "centripetal_acceleration": 
            path_turn_data = desired_trajectory_data.centripetal_acceleration_data
            vehicle_turn_data = vehicle_trajectory_data.centripetal_acceleration_data
        # calculations
        position_error = np.linalg.norm((path_location_data - vehicle_location_data),2,0)
        path_velocity_magnitude_data = np.linalg.norm(path_velocity_data,2,0)
        path_acceleration_magnitude = np.linalg.norm(path_acceleration_data,2,0)
        path_long_accel_mag = np.abs(path_longitudinal_acceleration_data)
        vehicle_velocity_magnitude = np.linalg.norm(vehicle_velocity_data,2,0)
        vehicle_long_accel_mag = np.abs(vehicle_longitudinal_acceleration_data)
        fig, axs = plt.subplots(4,1)
        axs[0].plot(path_time_data,position_error, color = 'tab:red', label="tracking error")
        axs[0].plot(path_time_data,path_time_data*0, color = 'k')
        axs[0].set_ylabel("tracking error")
        axs[1].plot(path_time_data, path_time_data*0 + max_velocity, color='k', label="max vel", linestyle="--")
        axs[1].plot(path_time_data, path_velocity_magnitude_data, color = 'tab:blue', label= "path vel")
        axs[1].plot(vehicle_time_data, vehicle_velocity_magnitude, color = 'tab:red', label="vehicle vel")   
        axs[1].set_ylabel("velocity")
        axs[2].plot(path_time_data, path_time_data*0 + max_acceleration, color='k', label="max accel", linestyle="--")
        axs[2].plot(path_time_data, path_acceleration_magnitude,color='tab:cyan',label="path accel")
        # axs[2].plot(path_time_data, path_long_accel_mag,color='tab:blue',label="path long accel")
        axs[2].plot(vehicle_time_data, vehicle_long_accel_mag, color = 'tab:red', label = "vehicle long accel")
        axs[2].set_ylabel("acceleration")
        axs[3].plot(path_time_data,path_time_data*0 + max_turn_value, color='k', label="max " + turn_type, linestyle="--")
        if turn_type is not None:
            axs[3].plot(path_time_data,path_turn_data,color='tab:blue', label="path " + turn_type)
        axs[3].plot(vehicle_time_data,vehicle_turn_data,color='tab:red', label="vehicle " + turn_type)
        axs[3].set_ylabel(turn_type)
        axs[3].set_xlabel("time (sec)")
        axs[0].legend()
        axs[1].legend()
        axs[2].legend()
        axs[3].legend()
        plt.show()