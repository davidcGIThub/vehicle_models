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
    curvature_data: np.ndarray
    angular_rate_data: np.ndarray
    centripetal_acceleration_data: np.ndarray
    longitudinal_acceleration_data: np.ndarray
    time_data: np.ndarray

    def __post_init__(self):
        location_dimension = np.shape(self.location_data)[0]
        velocity_dimension = np.shape(self.velocity_data)[0]
        acceleration_dimension = np.shape(self.acceleration_data)[0]
        jerk_dimension = np.shape(self.jerk_data)[0]
        location_length = np.shape(self.location_data)[1]
        velocity_length = np.shape(self.velocity_data)[1]
        acceleration_length = np.shape(self.acceleration_data)[1]
        jerk_length = np.shape(self.jerk_data)[1]
        curvature_length = len(self.curvature_data)
        angular_rate_length = len(self.angular_rate_data)
        centripetal_acceleration_length = len(self.centripetal_acceleration_data)
        longitudinal_acceleration_length = len(self.longitudinal_acceleration_data)
        time_length = len(self.time_data)
        if location_dimension != velocity_dimension or \
                velocity_dimension != acceleration_dimension or \
                acceleration_dimension != jerk_dimension:
            raise Exception("Trajectory data dimensions are not equal")
        if location_length != velocity_length or \
                velocity_length != acceleration_length or \
                acceleration_length != jerk_length or \
                jerk_length != curvature_length or \
                curvature_length != angular_rate_length or \
                centripetal_acceleration_length != longitudinal_acceleration_length or \
                longitudinal_acceleration_length !=  time_length:
            raise Exception("Trajectory data lengths are not equal")

class BicycleTrajectoryAnimation:

    def __init__(self, margins = 5):
        self._margins = margins
                 
    def animate_trajectory_following(self, bicycle_model: BicycleModel, traj_tracker: BicycleTrajectoryTracker, 
                                     trajectory_data: TrajectoryData, sleep_time: float = 0, plots: bool = True, animate: bool = True,
                                     turning_plot_type = "curvature", vehicle_time_instances_per_plot = 5, max_turn_value: float = 0):
        location_data = trajectory_data.location_data
        velocity_data = trajectory_data.velocity_data
        acceleration_data = trajectory_data.acceleration_data
        jerk_data = trajectory_data.jerk_data
        longitudinal_acceleration_data = trajectory_data.longitudinal_acceleration_data
        if turning_plot_type == "curvature": turn_data = trajectory_data.curvature_data
        elif turning_plot_type == "angular_rate": turn_data = trajectory_data.angular_rate_data
        elif turning_plot_type == "centripetal_acceleration": turn_data = trajectory_data.centripetal_acceleration_data
        time_data = trajectory_data.time_data
        vehicle_location_data = location_data*0
        vehicle_velocity_data = velocity_data*0
        vehicle_turn_data = turn_data*0
        vehicle_longitudinal_acceleration_data = time_data*0
        states_list = []
        dt = time_data[1] - time_data[0]
        num_data_points = len(time_data)

        #### run simulation ####
        for i in range(num_data_points):
            states = bicycle_model.getState() 
            states_list.append(states)
            desired_states = np.vstack((location_data[:,i], velocity_data[:,i],
                                         acceleration_data[:,i],jerk_data[:,i]))
            vel_dot, delta_dot = traj_tracker.mpc_control_accel_input(states, desired_states)
            v_dot_hat, delta_dot_hat = bicycle_model.update_acceleration_motion_model(vel_dot, delta_dot, dt)
            vehicle_location_data[:,i] = states[0,0:2]
            vehicle_velocity_data[:,i] = states[1,0:2]
            vehicle_longitudinal_acceleration_data[i] = v_dot_hat
            if turning_plot_type == "curvature": vehicle_turn_data[i] = bicycle_model.get_curvature()
            elif turning_plot_type == "angular_rate": vehicle_turn_data[i] = bicycle_model.get_angular_rate()
            elif turning_plot_type == "centripetal_acceleration": vehicle_turn_data[i] = bicycle_model.get_centripetal_acceleration()

        #### Animation ####
        if animate == True:
            x_limits = np.array([np.min(location_data [0,:]) - self._margins, np.max(location_data [0,:]) + self._margins])
            y_limits = np.array([np.min(location_data [1,:]) - self._margins, np.max(location_data [1,:]) + self._margins])
            fig = plt.figure()
            ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                                xlim=(x_limits[0],x_limits[1]), ylim=(y_limits[0],y_limits[1]))
            ax.grid()
            front_wheel_fig = plt.Polygon(bicycle_model.getFrontWheelPoints(),fc = 'k')
            back_wheel_fig = plt.Polygon(bicycle_model.getBackWheelPoints(),fc = 'k')
            body_fig = plt.Polygon(bicycle_model.getBodyPoints(),fc = 'g')
            desired_position_fig = plt.Circle((0, 0), radius=0.1, fc='r')
            time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
            ax.plot(location_data[0,:],location_data[1,:])

            def init():
                #initialize animation
                ax.add_patch(front_wheel_fig)
                ax.add_patch(back_wheel_fig)
                ax.add_patch(body_fig)
                ax.add_patch(desired_position_fig)
                time_text.set_text('')
                return front_wheel_fig, back_wheel_fig, body_fig, desired_position_fig, time_text

            def animate(i):
                # propogate robot motion
                t = time_data[i]
                bicycle_model.setState(states_list[i])
                front_wheel_fig.xy = bicycle_model.getFrontWheelPoints()
                back_wheel_fig.xy = bicycle_model.getBackWheelPoints()
                body_fig.xy = bicycle_model.getBodyPoints()
                desired_position_fig.center = (location_data[0,i], location_data[1,i])
                time_text.set_text('time = %.1f' % t)
                sleep(sleep_time)

                return  front_wheel_fig, back_wheel_fig, body_fig,desired_position_fig, time_text
            animate(0)
            ani = animation.FuncAnimation(fig, animate, frames = np.size(time_data), 
                                        interval = dt*100, blit = True, init_func = init, repeat = False)
            plt.show()

        if plots == True:
            #### Path Plot ####
            fig_pos = plt.figure()
            bike_length = bicycle_model.get_length()
            x_limits_pos = np.array([np.min(vehicle_location_data[0,:])-bike_length, 
                                     np.max(vehicle_location_data[0,:])+bike_length])
            y_limits_pos = np.array([np.min(vehicle_location_data[1,:])-bike_length, 
                                     np.max(vehicle_location_data[1,:])+bike_length])
            ax_pos = fig_pos.add_subplot(111, aspect='equal', autoscale_on=False,
                            xlim=(x_limits_pos[0],x_limits_pos[1]), ylim=(y_limits_pos[0],y_limits_pos[1]))
            ax_pos.plot(location_data[0,:],location_data[1,:], color = 'tab:blue')
            ax_pos.plot(vehicle_location_data[0,:],vehicle_location_data[1,:], linestyle="--", color = 'tab:red')
            for i in range(num_data_points):
                if i%int(num_data_points/vehicle_time_instances_per_plot) == 0:
                    print("here")
                    bicycle_model.setState(states_list[i])
                    bicycle_model.plot_bike(ax_pos)
            plt.show()
            position_error = np.linalg.norm((location_data - vehicle_location_data),2,0)
            velocity_magnitude = np.linalg.norm(velocity_data,2,0)
            vehicle_velocity_magnitude = np.linalg.norm(vehicle_velocity_data,2,0)
            max_vel = traj_tracker._max_vel
            max_vehicle_long_accel = traj_tracker._max_vel_dot
            fig_analysis, axs = plt.subplots(4,1)
            axs[0].plot(time_data,position_error, color = 'tab:red', label="error")
            axs[0].plot(time_data,time_data*0, color = 'k')
            axs[0].set_ylabel("position")
            axs[1].plot(time_data,time_data*0 + max_vel, color='k', label="max", linestyle="--")
            axs[1].plot(time_data,velocity_magnitude, color = 'tab:blue', label= "path")
            axs[1].plot(time_data,vehicle_velocity_magnitude, color = 'tab:green', label="vehicle")   
            axs[1].set_ylabel("velocity")    
            axs[2].plot(time_data,time_data*0 + max_vehicle_long_accel, color='k', label="max", linestyle="--")
            axs[2].plot(time_data,longitudinal_acceleration_data,color='tab:blue',label="path")
            axs[2].plot(time_data,vehicle_longitudinal_acceleration_data, color = 'tab:green', label = "vehicle")
            axs[2].set_ylabel("longitudinal \n acceleration")
            axs[3].plot(time_data,time_data*0 + max_turn_value, color='k', label="max", linestyle="--")
            axs[3].plot(time_data,turn_data,color='tab:blue', label="path")
            axs[3].plot(time_data,vehicle_turn_data,color='tab:green', label="vehicle")
            axs[3].set_ylabel(turning_plot_type)
            axs[0].legend()
            axs[1].legend()
            plt.show()



