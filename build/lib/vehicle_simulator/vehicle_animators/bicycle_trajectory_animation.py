"""
Bicycle Trajectory Animation class
"""
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
from vehicle_simulator.vehicle_models.bicycle_model import BicycleModel
from vehicle_simulator.vehicle_controllers.bicycle_trajectory_tracker import BicycleTrajectoryTracker
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
        self.longitudinal_acceleration_data: dot_product_data/velocity_magnitude_data

    def __calculate_cross_term_data(self):
        cross_product_norm = np.abs(np.cross(self.velocity_data.T, self.acceleration_data.T).flatten())
        return cross_product_norm
    
    def __calculate_dot_product_term_data(self):
        dot_product_term = np.sum(self.acceleration_data*self.velocity_data,0)
        return dot_product_term
    
    def __calculate_velocity_magnitude_data(self):
        return np.linalg.norm(self.velocity_data,2,0)

class VehicleSimulator:

    def __init__(self, vehicle_model, trajectory_tracker):
        self._vehicle_model = vehicle_model
        self._trajectory_tracker = trajectory_tracker

    def run_simulation(self, desired_trajectory_data: TrajectoryData):
        #### extract path data ####
        initial_state = self._vehicle_model.get_state()
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
        dt = time_data[1] - time_data[2]
        for i in range(num_data_points): 
            desired_states = np.vstack((location_data[:,i], velocity_data[:,i],
                                         acceleration_data[:,i],jerk_data[:,i]))
            motion_command, turn_command = self._trajectory_tracker.mpc_control_accel_input(states, desired_states)
            self._vehicle_model.update_acceleration_motion_model(motion_command, turn_command, dt)
            inputs = self._vehicle_model.getInputs()
            states = self._vehicle_model.getState()
            # save vehicle states
            inputs_list.append(inputs)
            states_list.append(states)
            # save vehicle calculated states
            vehicle_location_data = location_data*0
            vehicle_velocity_data = velocity_data*0
            vehicle_acceleration_data = acceleration_data*0
            vehicle_jerk_data = jerk_data*0
        vehicle_trajectory_data = TrajectoryData(vehicle_location_data, vehicle_velocity_data,
                                                 vehicle_acceleration_data, vehicle_jerk_data,
                                                 time_data)
        self._vehicle_model.zero_states_and_inputs()
        return inputs_list, states_list, vehicle_trajectory_data

    def animate_simulation(bicycle_model: BicycleModel, states_list: 'list[np.ndarray]', 
                           inputs_list: 'list[np.ndarray]', path_location_data: np.ndarray, 
                           time_data, margins = 0, sleep_time = 0):
        x_limits = np.array([np.min(path_location_data[0,:]) - margins, 
                             np.max(path_location_data[0,:]) + margins])
        y_limits = np.array([np.min(path_location_data[1,:]) - margins, 
                             np.max(path_location_data[1,:]) + margins])
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                            xlim=(x_limits[0],x_limits[1]), ylim=(y_limits[0],y_limits[1]))
        ax.grid()
        front_wheel_fig = plt.Polygon(bicycle_model.getFrontWheelPoints(),fc = 'k')
        back_wheel_fig = plt.Polygon(bicycle_model.getBackWheelPoints(),fc = 'k')
        body_fig = plt.Polygon(bicycle_model.getBodyPoints(),fc = 'g')
        center_of_mass = plt.Circle((0, 0), radius=0.1, fc=None, ec="k")
        desired_position_fig = plt.Circle((0, 0), radius=0.1, fc='tab:blue')
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
        ax.plot(path_location_data[0,:],path_location_data[1,:])
        dt = time_data[1] - time_data[0]
        def init():
            #initialize animation
            ax.add_patch(front_wheel_fig)
            ax.add_patch(back_wheel_fig)
            ax.add_patch(body_fig)
            ax.add_patch(desired_position_fig)
            ax.add_patch(center_of_mass)
            time_text.set_text('')
            return front_wheel_fig, back_wheel_fig, body_fig, desired_position_fig, \
                center_of_mass, time_text
        def animate(i):
            # propogate robot motion
            t = time_data[i]
            bicycle_model.setState(states_list[i])
            bicycle_model.setInputs(inputs_list[i])
            front_wheel_fig.xy = bicycle_model.getFrontWheelPoints()
            back_wheel_fig.xy = bicycle_model.getBackWheelPoints()
            body_fig.xy = bicycle_model.getBodyPoints()
            desired_position_fig.center = (path_location_data[0,i], path_location_data[1,i])
            center_of_mass.center = bicycle_model.getCenterOfMassPoint()
            time_text.set_text('time = %.1f' % t)
            sleep(sleep_time)
            return  front_wheel_fig, back_wheel_fig, body_fig, desired_position_fig, \
                    center_of_mass, time_text
        animate(0)
        ani = animation.FuncAnimation(fig, animate, frames = np.size(time_data), 
                                        interval = dt*100, blit = True, 
                                        init_func = init, repeat = False)
        plt.show()


    def plot_simulation(bicycle_model: BicycleModel, states_list: 'list[np.ndarray]', 
                           inputs_list: 'list[np.ndarray]', path_location_data: np.ndarray,
                           vehicle_instances_per_plot = 5, margins = None):
        #### Path Plot ####
        if margins == None:
            vehicle_length = bicycle_model.get_vehicle_properties()[0]
            margins = vehicle_length
        fig = plt.figure()
        vehicle_location_data = path_location_data*0
        num_data_points = np.shape(path_location_data)[1]
        x_limits_pos = np.array([np.min(vehicle_location_data[0,:])-margins, 
                                    np.max(vehicle_location_data[0,:])+margins])
        y_limits_pos = np.array([np.min(vehicle_location_data[1,:])-margins, 
                                    np.max(vehicle_location_data[1,:])+margins])
        ax = fig.add_subplot(111, aspect='equal', autoscale_on=False)
        ax.plot(path_location_data[0,:],path_location_data[1,:], color = 'tab:blue', label = "path")
        for i in range(num_data_points):
            if i%int(num_data_points/vehicle_instances_per_plot) == 0:
                bicycle_model.setState(states_list[i])
                bicycle_model.setInputs(inputs_list[i])
                bicycle_model.plot_bike(ax)
                vehicle_location_data[:,i] = states_list[i][0,0:2]
        ax.plot(vehicle_location_data[0,:],vehicle_location_data[1,:], linestyle="--",
            color = 'tab:red', label="true position")
        ax.set_ybound((y_limits_pos[0],y_limits_pos[1]))
        ax.set_xbound((x_limits_pos[0],x_limits_pos[1]))
        ax.legend()

    def plot_simulation_dynamics(desired_trajectory_data: TrajectoryData, 
                                 vehicle_trajectory_data: TrajectoryData,
                                 max_velocity: float, max_longitudinal_acceleration: float,
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
        vehicle_velocity_magnitude = np.linalg.norm(vehicle_velocity_data,2,0)
        fig, axs = plt.subplots(4,1)
        axs[0].plot(path_time_data,position_error, color = 'tab:red', label="error")
        axs[0].plot(path_time_data,path_time_data*0, color = 'k')
        axs[0].set_ylabel("position")
        axs[1].plot(path_time_data, path_time_data*0 + max_velocity, color='k', label="max", linestyle="--")
        axs[1].plot(path_time_data, path_velocity_magnitude_data, color = 'tab:blue', label= "path")
        axs[1].plot(vehicle_time_data, vehicle_velocity_magnitude, color = 'tab:green', label="vehicle")   
        axs[1].set_ylabel("velocity")
        axs[2].plot(path_time_data, path_time_data*0 + max_longitudinal_acceleration, color='k', label="max", linestyle="--")
        axs[2].plot(path_time_data, path_acceleration_data,color='tab:cyan',label="path total acceleration")
        axs[2].plot(path_time_data, path_longitudinal_acceleration_data,color='tab:blue',label="path")
        axs[2].plot(vehicle_time_data, vehicle_longitudinal_acceleration_data, color = 'tab:green', label = "vehicle")
        axs[2].set_ylabel("longitudinal \n acceleration")
        axs[3].plot(path_time_data,path_time_data*0 + max_turn_value, color='k', label="max", linestyle="--")
        if turn_type is not None:
            axs[3].plot(path_time_data,path_turn_data,color='tab:blue', label="path")
        axs[3].plot(vehicle_time_data,vehicle_turn_data,color='tab:green', label="vehicle")
        axs[3].set_ylabel(turn_type)
        axs[0].legend()
        axs[1].legend()
        plt.show()




                 
    # def animate_trajectory_following(self, bicycle_model: BicycleModel, traj_tracker: BicycleTrajectoryTracker, 
    #                                  trajectory_data: TrajectoryData, sleep_time: float = 0, plots: bool = True, animate: bool = True,
    #                                  turning_plot_type = "curvature", vehicle_time_instances_per_plot = 5, max_turn_value: float = 0):
    #     location_data = trajectory_data.location_data
    #     velocity_data = trajectory_data.velocity_data
    #     acceleration_data = trajectory_data.acceleration_data
    #     jerk_data = trajectory_data.jerk_data
    #     longitudinal_acceleration_data = trajectory_data.longitudinal_acceleration_data
    #     if turning_plot_type == "curvature": turn_data = trajectory_data.curvature_data
    #     elif turning_plot_type == "angular_rate": turn_data = trajectory_data.angular_rate_data
    #     elif turning_plot_type == "centripetal_acceleration": turn_data = trajectory_data.centripetal_acceleration_data
    #     time_data = trajectory_data.time_data
    #     vehicle_location_data = location_data*0
    #     vehicle_velocity_data = velocity_data*0
    #     # vehicle_turn_data = turn_data*0
    #     vehicle_turn_data = time_data*0
    #     vehicle_longitudinal_acceleration_data = time_data*0
    #     states_list = []
    #     dt = time_data[1] - time_data[0]
    #     num_data_points = len(time_data)

    #     #### run simulation ####
    #     for i in range(num_data_points):
    #         states = bicycle_model.getState() 
    #         states_list.append(states)
    #         desired_states = np.vstack((location_data[:,i], velocity_data[:,i],
    #                                      acceleration_data[:,i],jerk_data[:,i]))
    #         vel_dot, delta_dot = traj_tracker.mpc_control_accel_input(states, desired_states)
    #         v_dot_hat, delta_dot_hat = bicycle_model.update_acceleration_motion_model(vel_dot, delta_dot, dt)
    #         vehicle_location_data[:,i] = states[0,0:2]
    #         vehicle_velocity_data[:,i] = states[1,0:2]
    #         vehicle_longitudinal_acceleration_data[i] = v_dot_hat
    #         if turning_plot_type == "curvature": vehicle_turn_data[i] = bicycle_model.get_curvature()
    #         elif turning_plot_type == "angular_rate": vehicle_turn_data[i] = bicycle_model.get_angular_rate()
    #         elif turning_plot_type == "centripetal_acceleration": vehicle_turn_data[i] = bicycle_model.get_centripetal_acceleration()

    #     #### Animation ####
    #     if animate == True:
    #         x_limits = np.array([np.min(location_data [0,:]) - self._margins, np.max(location_data [0,:]) + self._margins])
    #         y_limits = np.array([np.min(location_data [1,:]) - self._margins, np.max(location_data [1,:]) + self._margins])
    #         fig = plt.figure()
    #         ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
    #                             xlim=(x_limits[0],x_limits[1]), ylim=(y_limits[0],y_limits[1]))
    #         ax.grid()
    #         front_wheel_fig = plt.Polygon(bicycle_model.getFrontWheelPoints(),fc = 'k')
    #         back_wheel_fig = plt.Polygon(bicycle_model.getBackWheelPoints(),fc = 'k')
    #         body_fig = plt.Polygon(bicycle_model.getBodyPoints(),fc = 'g')
    #         center_of_mass = plt.Circle((0, 0), radius=0.1, fc=None, ec="k")
    #         desired_position_fig = plt.Circle((0, 0), radius=0.1, fc='tab:blue')
    #         time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    #         ax.plot(location_data[0,:],location_data[1,:])

    #         def init():
    #             #initialize animation
    #             ax.add_patch(front_wheel_fig)
    #             ax.add_patch(back_wheel_fig)
    #             ax.add_patch(body_fig)
    #             ax.add_patch(desired_position_fig)
    #             ax.add_patch(center_of_mass)
    #             time_text.set_text('')
    #             return front_wheel_fig, back_wheel_fig, body_fig, desired_position_fig, center_of_mass, time_text

    #         def animate(i):
    #             # propogate robot motion
    #             t = time_data[i]
    #             bicycle_model.setState(states_list[i])
    #             front_wheel_fig.xy = bicycle_model.getFrontWheelPoints()
    #             back_wheel_fig.xy = bicycle_model.getBackWheelPoints()
    #             body_fig.xy = bicycle_model.getBodyPoints()
    #             desired_position_fig.center = (location_data[0,i], location_data[1,i])
    #             center_of_mass.center = bicycle_model.getCenterOfMassPoint()
    #             time_text.set_text('time = %.1f' % t)
    #             sleep(sleep_time)

    #             return  front_wheel_fig, back_wheel_fig, body_fig, desired_position_fig,center_of_mass, time_text
    #         animate(0)
    #         ani = animation.FuncAnimation(fig, animate, frames = np.size(time_data), 
    #                                     interval = dt*100, blit = True, init_func = init, repeat = False)
    #         plt.show()

    #     if plots == True:
    #         #### Path Plot ####
    #         fig_pos = plt.figure()
    #         bike_length = bicycle_model.get_length()
    #         x_limits_pos = np.array([np.min(vehicle_location_data[0,:])-bike_length, 
    #                                  np.max(vehicle_location_data[0,:])+bike_length])
    #         y_limits_pos = np.array([np.min(vehicle_location_data[1,:])-bike_length, 
    #                                  np.max(vehicle_location_data[1,:])+bike_length])
    #         ax_pos = fig_pos.add_subplot(111, aspect='equal', autoscale_on=False,
    #                         xlim=(x_limits_pos[0],x_limits_pos[1]), ylim=(y_limits_pos[0],y_limits_pos[1]))
    #         ax_pos.plot(location_data[0,:],location_data[1,:], color = 'tab:blue', label = "path")
    #         ax_pos.plot(vehicle_location_data[0,:],vehicle_location_data[1,:], linestyle="--", color = 'tab:red', label="true position")
    #         ax_pos.legend()
    #         for i in range(num_data_points):
    #             if i%int(num_data_points/vehicle_time_instances_per_plot) == 0:
    #                 bicycle_model.setState(states_list[i])
    #                 bicycle_model.plot_bike(ax_pos)
    #         #### !!! print the path length on this, as well as the time to complete trajectory
    #         plt.show()
    #         position_error = np.linalg.norm((location_data - vehicle_location_data),2,0)
    #         velocity_magnitude = np.linalg.norm(velocity_data,2,0)
    #         vehicle_velocity_magnitude = np.linalg.norm(vehicle_velocity_data,2,0)
    #         max_vel = traj_tracker._max_vel
    #         max_vehicle_long_accel = traj_tracker._max_vel_dot
    #         fig_analysis, axs = plt.subplots(4,1)
    #         axs[0].plot(time_data,position_error, color = 'tab:red', label="error")
    #         axs[0].plot(time_data,time_data*0, color = 'k')
    #         axs[0].set_ylabel("position")
    #         axs[1].plot(time_data,time_data*0 + max_vel, color='k', label="max", linestyle="--")
    #         axs[1].plot(time_data,velocity_magnitude, color = 'tab:blue', label= "path")
    #         axs[1].plot(time_data,vehicle_velocity_magnitude, color = 'tab:green', label="vehicle")   
    #         axs[1].set_ylabel("velocity")    
    #         axs[2].plot(time_data,time_data*0 + max_vehicle_long_accel, color='k', label="max", linestyle="--")
    #         axs[2].plot(time_data,longitudinal_acceleration_data,color='tab:blue',label="path")
    #         axs[2].plot(time_data,vehicle_longitudinal_acceleration_data, color = 'tab:green', label = "vehicle")
    #         axs[2].set_ylabel("longitudinal \n acceleration")
    #         axs[3].plot(time_data,time_data*0 + max_turn_value, color='k', label="max", linestyle="--")
    #         if turn_data is not None:
    #             axs[3].plot(time_data,turn_data,color='tab:blue', label="path")
    #         axs[3].plot(time_data,vehicle_turn_data,color='tab:green', label="vehicle")
    #         axs[3].set_ylabel(turning_plot_type)
    #         axs[0].legend()
    #         axs[1].legend()
    #         plt.show()



