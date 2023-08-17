"""
Bicycle Trajectory Animation class
"""
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
from vehicle_simulator.vehicle_models.fixed_wing_model import FixedWingModel
from vehicle_simulator.vehicle_models.fixed_wing_parameters import FixedWingParameters
from vehicle_simulator.vehicle_controllers.fixed_wing_autopilot import FixedWingControlParameters, FixedWingAutopilot
from vehicle_simulator.vehicle_controllers.fixed_wing_path_follower import FixedWingSplinePathFollower
from vehicle_simulator.vehicle_controllers.bspline_evaluator import BsplineEvaluator
from vehicle_simulator.vehicle_simulators.spatial_violations import get_box_violations_from_spline, get_obstacle_violations, \
    Obstacle, plot_3D_obstacles
from vehicle_simulator.vehicle_controllers.bspline_path_manager import SplinePathManager
from time import sleep
import sys
from dataclasses import dataclass

@dataclass
class PathData:
    location_data: np.ndarray
    curvature_data: np.ndarray
    inclination_data: np.ndarray
    time_data: np.ndarray

class FixedWingPathFollowingSimulator:

    def __init__(self,
                 plane_model: FixedWingModel, 
                 plane_autopilot: FixedWingAutopilot,
                 path_follower: FixedWingSplinePathFollower,
                 path_manager: SplinePathManager):
        self._plane_model = plane_model
        self._plane_autopilot = plane_autopilot
        self._path_follower = path_follower
        self._path_manager = path_manager
        self._order = self._path_follower.get_order()
        self._spline_eval = BsplineEvaluator(self._order)

    def run_simulation(self, path_control_point_list: 'list[np.ndarray]', desired_speed, 
                       obstacle_list:'list[Obstacle]' = [], sfc_list:list = [],
                       intervals_per_sfc: np.ndarray = np.array([]), waypoints=np.array([]),
                       dt: float = 0.01, run_time: float= 30, frame_width = 15,
                       animate = True, plot = True, instances_per_plot=10):

        states_list, vehicle_path_data, path_data_list, tracked_path_data, \
            closest_distances_to_obstacles, closest_distances_to_sfc_walls \
            = self.collect_simulation_data(path_control_point_list, desired_speed,
                                           obstacle_list, sfc_list, intervals_per_sfc,
                                           dt, run_time)
        if animate == True:
            self.animate_simulation(states_list, 
                           path_data_list, tracked_path_data, 
                           obstacle_list, sfc_list, waypoints=waypoints, 
                           frame_width=frame_width, dt=dt)
        if plot == True:
            self.plot_simulation(states_list, vehicle_path_data, path_data_list, tracked_path_data, 
                                 obstacle_list, sfc_list, waypoints, instances_per_plot, dt)
        return vehicle_path_data, tracked_path_data, closest_distances_to_obstacles, closest_distances_to_sfc_walls

    def collect_simulation_data(self, path_control_point_list: 'list[np.ndarray]', 
                                desired_speed: float = 30, obstacle_list:'list[Obstacle]' = [], 
                                sfc_list:list = [], intervals_per_sfc:list = [],
                                dt: float = 0.01, run_time: float= 30):
        #### Initilize Data ####
        # extract path data #
        time_data = np.linspace(0,run_time, int(run_time/dt+1))
        num_data_points = len(time_data)
        vehicle_location_data = np.zeros((3,num_data_points))
        vehicle_velocity_data = np.zeros((3,num_data_points))
        vehicle_acceleration_data = np.zeros((3,num_data_points))
        tracked_location_data = np.zeros((3,num_data_points))
        tracked_velocity_data = np.zeros((3,num_data_points))
        tracked_acceleration_data = np.zeros((3,num_data_points))
        wind = np.array([0,0,0,0,0,0])
        states_list = []
        path_data_list = []
        # path_data = self._spline_eval.matrix_bspline_evaluation_for_dataset(path_control_points, 1000)
        for j in range(len(path_control_point_list)):
            path_control_points = path_control_point_list[j]
            path_data = self._spline_eval.matrix_bspline_evaluation_for_dataset(path_control_points, 1000)
            path_data_list.append(path_data)
        for i in range(num_data_points):
            state = self._plane_model.get_state()
            position = np.array([state.item(0), state.item(1), state.item(2)])
            velocity = self._plane_model.get_inertial_velocity()
            acceleration = self._plane_model.get_inertial_acceleration()
            path_control_points = self._path_manager.get_current_path_control_points(position)
            cmds = self._path_follower.get_commands(path_control_points, position, desired_speed)
            delta = self._plane_autopilot.get_commands(cmds, state, wind, dt)
            self._plane_model.update(delta, wind, dt)
            scale_factor = 1
            closest_point, closest_velocity_vector, closest_acceleration_vector = self._spline_eval.get_closest_point_and_derivatives(path_control_points, scale_factor, position)
            states_list.append(state)
            tracked_location_data[:,i] = closest_point.flatten()
            tracked_velocity_data[:,i] = closest_velocity_vector.flatten()
            tracked_acceleration_data[:,i] = closest_acceleration_vector.flatten()
            vehicle_location_data[:,i] = position
            vehicle_velocity_data[:,i] = velocity
            vehicle_acceleration_data[:,i] = acceleration
        # tracked_path_data = self._path_manager.get_tracked_path_data(path_control_point_list, num_data_points)
        vehicle_curvature_data = self.__calculate_curvature_data(vehicle_velocity_data, vehicle_acceleration_data)
        vehicle_incline_data = self.__calculate_inclination_data(vehicle_velocity_data)
        tracked_curvature_data = self.__calculate_curvature_data(tracked_velocity_data, tracked_acceleration_data)
        tracked_incline_data = self.__calculate_inclination_data(tracked_velocity_data)
        vehicle_path_data = PathData(vehicle_location_data, vehicle_curvature_data, vehicle_incline_data, time_data)
        tracked_path_data = PathData(tracked_location_data, tracked_curvature_data, tracked_incline_data, time_data)
        closest_distances_to_obstacles = get_obstacle_violations(obstacle_list, tracked_location_data)
        closest_distances_to_sfc_walls = get_box_violations_from_spline(sfc_list, intervals_per_sfc, path_control_points, self._path_follower.get_order())
        return states_list, vehicle_path_data, path_data_list, tracked_path_data, \
                closest_distances_to_obstacles, closest_distances_to_sfc_walls

    def animate_simulation(self, states_list: 'list[np.ndarray]', 
                           path_data_list: 'list[np.ndarray]',
                           tracked_path_data: PathData,
                           obstacle_list:list = [], sfc_list:list = [],
                           waypoints: np.ndarray = np.array([]),
                           frame_width:float = 15, dt: float = 0.01):
        closest_path_point_data = tracked_path_data.location_data
        fig = plt.figure("Animation")
        ax = plt.axes(projection='3d')
        print(type(ax))
        fig.add_axes(ax)
        closest_point, = ax.plot([],[],[],lw=.5,color="tab:blue", marker = 'o')
        for i in range(len(path_data_list)):
            path_data = path_data_list[i]
            ax.plot(path_data[0,:], path_data[1,:],path_data[2,:], alpha=0.8, color="tab:blue")
        self._plane_model.reset_graphic_axes(ax)
        if waypoints.size != 0:
            ax.scatter(waypoints[0,:], waypoints[1,:],waypoints[2,:], marker="o", color="tab:green")
        plot_3D_obstacles(obstacle_list, ax)
        for j in range(len(sfc_list)):
            sfc_points = sfc_list[j]
            ax.plot(sfc_points[0,:], sfc_points[1,:],sfc_points[2,:], alpha=0.5)
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
            closest_point.set_xdata(closest_path_point_data[0,num])
            closest_point.set_ydata(closest_path_point_data[1,num])
            closest_point.set_3d_properties(closest_path_point_data[2,num])
        # Setting the axes properties
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Fixed Wing Path Following')
        ax.view_init(elev=190, azim=45)
        # Creating the Animation object
        delayBetweenFrames_ms = dt*1000 # seconds between frames
        line_ani = animation.FuncAnimation(fig, update_line, fargs=[self._plane_model] , interval=delayBetweenFrames_ms, blit=False)
        plt.show()

    def plot_simulation(self, 
                           states_list: np.ndarray,
                           vehicle_path_data: PathData, 
                           path_data_list: 'list[np.ndarray]',
                           tracked_path_data: PathData,
                           obstacle_list:list = [], sfc_list:list = [],
                           waypoints: np.ndarray = np.array([]),
                           instances_per_plot = 10, dt: float = 0.01, graphic_scale = 10):
        closest_path_point_data = tracked_path_data.location_data
        vehicle_location_data = vehicle_path_data.location_data
        num_frames = np.shape(vehicle_location_data)[1]
        steps = int(num_frames/instances_per_plot)
        self._plane_model.scale_plane_graphic(graphic_scale)
        fig = plt.figure("Animation Plot")
        ax = plt.axes(projection='3d')
        fig.add_axes(ax)
        for i in range(len(path_data_list)):
            path_data = path_data_list[i]
            ax.plot(path_data[0,:], path_data[1,:],path_data[2,:], alpha=0.8, color="tab:blue")
        ax.plot(vehicle_location_data[0,:], vehicle_location_data[1,:], vehicle_location_data[2,0], linestyle="dashed", color="0.5" )
        if waypoints.size != 0:
            ax.plot(waypoints[0,:], waypoints[1,:],waypoints[2,:], marker="o", linestyle='None', color="tab:green", markersize=8, alpha=0.65)
        plot_3D_obstacles(obstacle_list, ax)
        for j in range(len(sfc_list)):
            sfc_points = sfc_list[j]
            ax.plot(sfc_points[0,:], sfc_points[1,:],sfc_points[2,:], alpha=0.5)
        for i in range(num_frames):
            if i%steps == 0:
                state = states_list[i]
                print("i: " , i)
                print("state: " , state)
                self._plane_model.set_state(state)
                self._plane_model.update_graphics()
                self._plane_model.plot_plane(ax)
                ax.plot([closest_path_point_data[0,i]],
                        [closest_path_point_data[1,i]],
                        [closest_path_point_data[2,i]],lw=.5,color="tab:blue", marker = 'o', fillstyle='none')
        # Setting the axes properties
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        max_x = np.max(np.concatenate((vehicle_location_data[0,:], path_data[0,:])))
        min_x = np.min(np.concatenate((vehicle_location_data[0,:], path_data[0,:])))
        max_y = np.max(np.concatenate((vehicle_location_data[1,:], path_data[0,:])))
        min_y = np.min(np.concatenate((vehicle_location_data[1,:], path_data[0,:])))
        max_z = np.max(np.concatenate((vehicle_location_data[2,:], path_data[0,:])))
        min_z = np.min(np.concatenate((vehicle_location_data[2,:], path_data[0,:])))
        ax.set_xlim3d([min_x, max_x])
        ax.set_ylim3d([min_y, max_y])
        ax.set_zlim3d([min_z, max_z])
        ax.set_title('Fixed Wing Path Following')
        ax.view_init(elev=190, azim=45)
        plt.show()

    def plot_simulation_analytics(self, vehicle_path_data: PathData, tracked_path_data: PathData, 
                max_curvature: float, max_incline: float):
        time_data = vehicle_path_data.time_data
        path_curvature = tracked_path_data.curvature_data
        vehicle_curvature = vehicle_path_data.curvature_data
        path_incline = tracked_path_data.inclination_data
        vehicle_incline = vehicle_path_data.inclination_data
        tracking_error = np.linalg.norm(vehicle_path_data.location_data - tracked_path_data.location_data,2, 0)
        fig, axs = plt.subplots(4,1)
        axs[0].plot(time_data,tracking_error, color = 'tab:red', label="tracking\n error")
        axs[0].plot(time_data,tracking_error*0, color = 'k')
        axs[0].set_ylabel("tracking error \n (m)")
        axs[0].set_xlabel("time (sec)")

        axs[1].plot(time_data, path_curvature*0 + max_curvature, color='k', label="max")
        axs[1].plot(time_data, path_curvature, color = 'tab:blue', label= "path")
        axs[1].plot(time_data, vehicle_curvature, color = 'tab:olive', label= "vehicle",linestyle="--")   
        axs[1].set_ylabel("curvature")
        axs[1].set_xlabel("time (sec)")

        axs[2].plot(time_data, path_incline*0 + max_incline, color='k', label="max")
        # axs[2].plot(path_time_data, path_acceleration_magnitude,color='tab:cyan',label="des accel")
        axs[2].plot(time_data, path_incline,color='tab:blue',label="path")
        axs[2].plot(time_data, vehicle_incline, color = 'tab:olive', label =  "vehicle",linestyle="--")
        axs[2].set_ylabel("incline")
        axs[2].set_xlabel("time (sec)")


        axs[0].legend(loc='upper left')
        axs[1].legend(loc='lower left')
        axs[2].legend(loc='lower left')
        axs[3].legend(loc='lower left')
        # axs[0].tick_params(labelbottom = False, bottom = False)
        # axs[1].tick_params(labelbottom = False, bottom = False)
        # axs[2].tick_params(labelbottom = False, bottom = False)
        plt.show()


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
        inclination_data = vertical_velocity / horizontal_velocity
        return inclination_data
    