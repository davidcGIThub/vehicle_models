"""
Bicycle Trajectory Animation class
"""
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
from vehicle_simulator.vehicle_models.fixed_wing_model import FixedWingModel
from vehicle_simulator.vehicle_controllers.fixed_wing_autopilot import FixedWingAutopilot
from vehicle_simulator.vehicle_controllers.fixed_wing_trajectory_tracker import FixedWingTrajectoryTracker
from vehicle_simulator.vehicle_controllers.bspline_evaluator import BsplineEvaluator
from vehicle_simulator.vehicle_controllers.bspline_trajectory_manager import SplineTrajectoryManager
# import time
 

import sys
from dataclasses import dataclass


@dataclass
class TrajectoryData:
    position_data: np.ndarray
    velocity_data: np.ndarray
    acceleration_data: np.ndarray
    curvature_data: np.ndarray
    slope_data: np.ndarray
    time_data: np.ndarray

class FixedWingTrajectoryTrackingSimulator:

    def __init__(self,
                 plane_model: FixedWingModel, 
                 plane_autopilot: FixedWingAutopilot,
                 trajectory_tracker: FixedWingTrajectoryTracker,
                 trajectory_manager: SplineTrajectoryManager):
        pass
        self._plane_model = plane_model
        self._plane_autopilot = plane_autopilot
        self._trajectory_tracker = trajectory_tracker
        self._trajectory_manager = trajectory_manager
        self._order = self._trajectory_tracker.get_order()
        self._spline_eval = BsplineEvaluator(self._order)

    def run_simulation(self,  waypoints=np.array([]), dt: float = 0.01, frame_width = 15,
                       animate = True, plot = True, instances_per_plot=10, graphic_scale=1):
        states_list, vehicle_trajectory_data, trajectory_data, \
            = self.collect_simulation_data(dt)
        if animate == True:
            self.animate_simulation(states_list, trajectory_data, waypoints=waypoints, 
                           frame_width=frame_width, dt=dt)
        if plot == True:
            self.plot_simulation(states_list, vehicle_trajectory_data, trajectory_data, \
                                 waypoints, instances_per_plot, graphic_scale=graphic_scale)
        return vehicle_trajectory_data, trajectory_data

    def collect_simulation_data(self, dt: float = 0.01):
        #### Initilize Data ####
        # extract path data #
        control_point_list = self._trajectory_manager.get_control_point_list()
        scale_factor_list = self._trajectory_manager.get_scale_factor_list()
        run_time = self.get_trajectory_run_time(control_point_list, scale_factor_list)
        time_data = np.linspace(0,run_time, int(run_time/dt+1))
        num_data_points = len(time_data)
        vehicle_position_data = np.zeros((3,num_data_points))
        vehicle_velocity_data = np.zeros((3,num_data_points))
        vehicle_acceleration_data = np.zeros((3,num_data_points))
        trajectory_position_data = np.zeros((3,num_data_points))
        trajectory_velocity_data = np.zeros((3,num_data_points))
        trajectory_acceleration_data = np.zeros((3,num_data_points))
        wind = np.array([0,0,0,0,0,0])
        states_list = []
        start_time = self._trajectory_manager.get_start_time()
        t = start_time
        for i in range(num_data_points):
            state = self._plane_model.get_state()
            position = np.array([state.item(0), state.item(1), state.item(2)])
            velocity = self._plane_model.get_inertial_velocity()
            acceleration = self._plane_model.get_inertial_acceleration()
            control_points, scale_factor, start_knot = self._trajectory_manager.get_current_bspline(t)
            cmds = self._trajectory_tracker.get_commands_from_bspline(control_points, scale_factor, start_knot, state, t)
            delta = self._plane_autopilot.get_commands(cmds, state, wind, dt)
            self._plane_model.update(delta, wind, dt)
            states_list.append(state)
            vehicle_position_data[:,i] = position
            vehicle_velocity_data[:,i] = velocity
            vehicle_acceleration_data[:,i] = acceleration               
            trajectory_position_data[:,i] = self._spline_eval.get_position_vector_from_spline(t, start_knot, control_points, scale_factor).flatten()
            trajectory_velocity_data[:,i] = self._spline_eval.get_velocity_vector_from_spline(t, start_knot, control_points, scale_factor).flatten()
            trajectory_acceleration_data[:,i] = self._spline_eval.get_acceleration_vector_from_spline(t, start_knot, control_points, scale_factor).flatten()
            t += dt
        vehicle_curvature_data = self.__calculate_curvature_data(vehicle_velocity_data, vehicle_acceleration_data)
        vehicle_slope_data = self.__calculate_inclination_data(vehicle_velocity_data)
        trajectory_curvature_data = self.__calculate_curvature_data(trajectory_velocity_data, trajectory_acceleration_data)
        trajectory_slope_data = self.__calculate_inclination_data(trajectory_velocity_data)
        vehicle_data = TrajectoryData(vehicle_position_data, vehicle_velocity_data, \
                                                 vehicle_acceleration_data, vehicle_curvature_data, \
                                                 vehicle_slope_data, time_data)
        trajectory_data = TrajectoryData(trajectory_position_data, trajectory_velocity_data, \
                                             trajectory_acceleration_data, trajectory_curvature_data, \
                                             trajectory_slope_data,  time_data)
        return states_list, vehicle_data, trajectory_data


    def animate_simulation(self, states_list: 'list[np.ndarray]', 
                           trajectory_data: TrajectoryData,
                           waypoints: np.ndarray = np.array([]),
                           frame_width:float = 15, dt: float = 0.01):
        fig = plt.figure("Animation")
        ax = plt.axes(projection='3d')
        fig.add_axes(ax)
        trajectory_point, = ax.plot([],[],[],lw=.5,color="tab:blue", marker = 'o')
        trajectory_position = trajectory_data.position_data
        ax.plot(trajectory_position[0,:], trajectory_position[1,:],trajectory_position[2,:], alpha=0.8, color="tab:blue")
        self._plane_model.reset_graphic_axes(ax)
        if waypoints.size != 0:
            ax.scatter(waypoints[0,:], waypoints[1,:],waypoints[2,:], marker="o", color="tab:green")
        def update_line(num, plane_model: FixedWingModel):
            state = states_list[num]
            plane_model.set_state(state)
            plane_model.update_graphics()
            x = state.item(0)
            y = state.item(1)
            z = state.item(2)
            ax.set_xlim3d([x-frame_width/2, x+frame_width/2])
            ax.set_ylim3d([y-frame_width/2, y+frame_width/2])
            ax.set_zlim3d([z-frame_width/2, z+frame_width/2])
            trajectory_point.set_xdata(trajectory_position[0,num])
            trajectory_point.set_ydata(trajectory_position[1,num])
            trajectory_point.set_3d_properties(trajectory_position[2,num])
        # Setting the axes properties
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Fixed Wing Trajectory Tracking')
        ax.view_init(elev=190, azim=45)
        # Creating the Animation object
        delayBetweenFrames_ms = dt*1000 # seconds between frames
        line_ani = animation.FuncAnimation(fig, update_line, fargs=[self._plane_model] , interval=delayBetweenFrames_ms, blit=False)
        plt.show()


    def plot_simulation(self, 
                           states_list: np.ndarray,
                           vehicle_data: TrajectoryData, 
                           trajectory_data: TrajectoryData,
                           waypoints: np.ndarray = np.array([]),
                           instances_per_plot = 10, graphic_scale = 10):
        trajectory_location_data = trajectory_data.position_data
        vehicle_location_data = vehicle_data.position_data
        num_frames = np.shape(vehicle_location_data)[1]
        steps = int(num_frames/instances_per_plot)
        self._plane_model.scale_plane_graphic(graphic_scale)
        fig = plt.figure("Animation Plot")
        ax = plt.axes(projection='3d')
        fig.add_axes(ax)
        ax.plot(trajectory_location_data[0,:], trajectory_location_data[1,:],trajectory_location_data[2,:], alpha=0.8, color="tab:blue", label = "trajectory")
        ax.plot(vehicle_location_data[0,:], vehicle_location_data[1,:], vehicle_location_data[2,:], linestyle="dashed", color="0.5", label = "vehicle path" )
        if waypoints.size != 0:
            ax.plot(waypoints[0,:], waypoints[1,:],waypoints[2,:], marker="o", linestyle='None', color="tab:green", markersize=8, alpha=0.65, label = "waypoints")
        for i in range(num_frames):
            if i%steps == 0:
                state = states_list[i]
                self._plane_model.set_state(state)
                self._plane_model.update_graphics()
                self._plane_model.plot_plane(ax)
                ax.scatter([trajectory_location_data[0,i]],
                        [trajectory_location_data[1,i]],
                        [trajectory_location_data[2,i]],lw=.5,color="tab:blue")
        # Setting the axes properties
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        max_x = np.max(np.concatenate((vehicle_location_data[0,:], trajectory_location_data[0,:])))
        min_x = np.min(np.concatenate((vehicle_location_data[0,:], trajectory_location_data[0,:])))
        max_y = np.max(np.concatenate((vehicle_location_data[1,:], trajectory_location_data[1,:])))
        min_y = np.min(np.concatenate((vehicle_location_data[1,:], trajectory_location_data[1,:])))
        max_z = np.max(np.concatenate((vehicle_location_data[2,:], trajectory_location_data[2,:])))
        min_z = np.min(np.concatenate((vehicle_location_data[2,:], trajectory_location_data[2,:])))
        ax.set_xlim3d([min_x, max_x])
        ax.set_ylim3d([min_y, max_y])
        ax.set_zlim3d([min_z, max_z])
        ax.set_title('Fixed Wing Trajectory Tracking')
        ax.view_init(elev=190, azim=45)
        ax.legend()
        self.__set_axes_equal(ax) ## TODO fix axes size
        plt.show()

    # def plot_simulation_analytics(self, vehicle_path_data: PathData, tracked_path_data: PathData, 
    #             max_curvature: float, max_incline_angle: float, closest_distances_to_obstacles:np.ndarray = np.empty(0)):
    #     time_data = vehicle_path_data.time_data
    #     path_curvature = tracked_path_data.curvature_data
    #     vehicle_curvature = vehicle_path_data.curvature_data
    #     path_incline = tracked_path_data.inclination_data
    #     vehicle_incline = vehicle_path_data.inclination_data
    #     tracking_error = np.linalg.norm(vehicle_path_data.location_data - tracked_path_data.location_data,2, 0)
    #     fig, axs = plt.subplots(4,1)
    #     axs[0].plot(time_data,tracking_error, color = 'tab:red', label="tracking\n error")
    #     axs[0].plot(time_data,tracking_error*0, color = 'k')
    #     axs[0].set_ylabel("Tracking Error \n (m)")
    #     axs[0].set_xlabel("Time (sec)")

    #     axs[1].plot(time_data, path_curvature*0 + max_curvature, color='k', label="max")
    #     axs[1].plot(time_data, path_curvature, color = 'tab:blue', label= "path")
    #     axs[1].plot(time_data, vehicle_curvature, color = 'tab:olive', label= "vehicle",linestyle="--")   
    #     axs[1].set_ylabel("Curvature")
    #     axs[1].set_xlabel("Time (sec)")
    #     if max_incline_angle != None:
    #         axs[2].plot(time_data, path_incline*0 + np.degrees(max_incline_angle), color='k', label="bounds")
    #         axs[2].plot(time_data, path_incline*0 - np.degrees(max_incline_angle), color='k')
    #     # axs[2].plot(path_time_data, path_acceleration_magnitude,color='tab:cyan',label="des accel")
    #     axs[2].plot(time_data, np.degrees(path_incline),color='tab:blue',label="path")
    #     axs[2].plot(time_data, np.degrees(vehicle_incline), color = 'tab:olive', label =  "vehicle",linestyle="--")
    #     axs[2].set_ylabel("Slope Angle (deg)")
    #     axs[2].set_xlabel("Time (sec)")
    #     velocity = (vehicle_path_data.location_data[:,1:] - vehicle_path_data.location_data[:,0:-1]) / (time_data[1:] - time_data[0:-1])
    #     velocity_mag = np.linalg.norm(velocity,2,0)
    #     axs[3].plot(time_data[0:-1], velocity_mag, color = 'tab:olive', label =  "vehicle",linestyle="--")
    #     axs[3].plot(time_data[0:-1],time_data[0:-1]*0 + 20, color='tab:blue', label =  'desired')
    #     axs[3].set_ylabel("Velocity (m/s)")
    #     axs[3].set_xlabel("time (sec)")
    #     axs[0].legend(loc='upper left')
    #     axs[1].legend(loc='lower left')
    #     axs[2].legend(loc='lower left')
    #     axs[3].legend(loc='upper left')
    #     # axs[3].legend(loc='lower left')
    #     # axs[0].tick_params(labelbottom = False, bottom = False)
    #     # axs[1].tick_params(labelbottom = False, bottom = False)
    #     # axs[2].tick_params(labelbottom = False, bottom = False)
    #     plt.show()

    def __calculate_curvature_data(self, velocity_data, acceleration_data):
            cross_product_norm = np.linalg.norm(np.transpose(np.cross(velocity_data.T, acceleration_data.T)),2,0)
            velocity_magnitude_data = np.linalg.norm(velocity_data,2,0)
            velocity_magnitude_data[velocity_magnitude_data < 8e-10] = 1
            curvature_data = cross_product_norm/velocity_magnitude_data**3
            return curvature_data
    
    def __calculate_inclination_data(self, velocity_data):
        vertical_velocity = velocity_data[2,:]
        horizontal_velocity = np.linalg.norm(velocity_data[0:2,:],2,0)
        vertical_velocity[horizontal_velocity == 0] = sys.float_info.max
        horizontal_velocity[horizontal_velocity == 0] = 1
        inclination_data = np.arctan2(vertical_velocity, horizontal_velocity)
        return inclination_data

    def __set_axes_equal(self, ax):
        """
        Make axes of 3D plot have equal scale so that spheres appear as spheres,
        cubes as cubes, etc.
        Input
        ax: a matplotlib axis, e.g., as output from plt.gca().
        """
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)

        # The plot bounding box is a sphere in the sense of the infinity
        # norm, hence I call half the max range the plot radius.
        plot_radius = 0.5*max([x_range, y_range, z_range])

        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

    def get_trajectory_run_time(self, control_point_list: 'list[np.ndarray]', 
                                scale_factor_list: 'list[float]'):
        num_trajectories = len(scale_factor_list)
        run_time = 0
        for i in range(num_trajectories):
            control_points = control_point_list[i]
            scale_factor = scale_factor_list[i]
            num_intervals = self.count_num_intervals(control_points)
            run_time += num_intervals*scale_factor
        return run_time

    def count_num_intervals(self, control_points):
        num_intervals = np.shape(control_points)[1] - self._order
        return num_intervals
