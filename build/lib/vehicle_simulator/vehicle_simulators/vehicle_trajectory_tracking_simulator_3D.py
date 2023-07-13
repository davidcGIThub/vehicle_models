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
from vehicle_simulator.vehicle_simulators.spatial_violations import get_box_violations_from_spline, get_obstacle_violations
from time import sleep

class FixedWingPathFollowingSimulator:

    def __init__(self,
                 plane_model: FixedWingModel, 
                 plane_autopilot: FixedWingAutopilot,
                 path_follower: FixedWingSplinePathFollower,):
        self._plane_model = plane_model
        self._plane_autopilot = plane_autopilot
        self._path_follower = path_follower

    def run_simulation(self, path_control_points, desired_speed, 
                       obstacle_list:list = [], sfc_list:list = [], 
                       dt: float = 0.01, run_time: float= 30,
                       animate = True, plot=True, sleep_time = 0):
        states_list, vehicle_location_data, closest_path_point_data, closest_distances_to_obstacles, closest_distances_to_sfc_walls \
              = self.collect_simulation_data(path_control_points, desired_speed, 
                                             obstacle_list, sfc_list, dt, run_time)
        # if animate == True:
        #     self.animate_simulation(states_list, inputs_list, 
        #                             desired_trajectory_data, true_trajectory_data,
        #                             margins=length, sleep_time = sleep_time)
        # if plot == True:
        #     self.plot_simulation(states_list, inputs_list,  
        #                         desired_trajectory_data,
        #                         true_trajectory_data,
        #                         margins=length)
        tracking_error = np.linalg.norm(vehicle_location_data - closest_path_point_data, 2, 0)
        return states_list, tracking_error, closest_distances_to_obstacles, closest_distances_to_sfc_walls

    def collect_simulation_data(self, path_control_points: np.ndarray, 
                                desired_speed: float = 30, obstacle_list:list = [], 
                                sfc_list:list = [], intervals_per_sfc:list = [],
                                dt: float = 0.01, run_time: float= 30):
        #### Initilize Data ####
        # extract path data #
        time_data = np.linspace(0,run_time, int(run_time/dt+1))
        num_data_points = len(time_data)
        vehicle_location_data = np.zeros(3,num_data_points)
        closest_path_point_data = np.zeros(3,num_data_points)
        wind = np.array([0,0,0,0,0,0])
        states_list = []
        order = self._path_follower.get_order()
        spline_eval = BsplineEvaluator(order)
        path_data = spline_eval.matrix_bspline_evaluation_for_dataset(path_control_points,order,1000)
        for i in range(num_data_points):
            state = self._plane_model.get_state()
            position = np.array([state.item(0), state.item(1), state.item(2)])
            cmds = self._path_follower.get_commands(path_control_points, position, desired_speed)
            delta = self._plane_autopilot.get_commands(cmds, state, wind, dt)
            self._plane_model.update(delta, wind, dt)
            closest_point, t = spline_eval.get_closest_point_and_t(path_control_points, 1, position)
            states_list.append(state)
            closest_path_point_data[:,i] = closest_point
            vehicle_location_data[:,i] = position
        closest_distances_to_obstacles = get_obstacle_violations(obstacle_list, path_data)
        closest_distances_to_sfc_walls = get_box_violations_from_spline(sfc_list, intervals_per_sfc, path_control_points, self._path_follower.get_order())
        return states_list, vehicle_location_data, closest_path_point_data, \
                closest_distances_to_obstacles, closest_distances_to_sfc_walls

    def animate_simulation(self, states_list: 'list[np.ndarray]', 
                           vehicle_location_data: np.ndarray, 
                           path_control_points: np.ndarray,
                           margins:float = 0, sleep_time:float = 0,
                           frame_width:float = 30):
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        fig.add_axes(ax)
        center_of_mass, = self._plane_model.get_ax().plot([],[],[],lw=.5,color="tab:blue", marker = 'o')
        def update_line(num, plane_model: FixedWingModel):
            state = states_list[num]
            plane_model.set_state(states_list[state])
            plane_model.update_graphics()
            x = state.item(0)
            y = state.item(1)
            z = state.item(2)
            ax.set_xlim3d([x-frame_width/2, x+frame_width/2])
            ax.set_ylim3d([y-frame_width/2, y+frame_width/2])
            ax.set_zlim3d([z-frame_width/2, z+frame_width/2])
            center_of_mass.set_xdata(vehicle_location_data[0,num])
            center_of_mass.set_ydata(vehicle_location_data[1,num])
            center_of_mass.set_3d_properties(vehicle_location_data[2,num])

        # Setting the axes properties
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Fixed Wing Path Following')
        ax.view_init(elev=190, azim=45)
        # ax.plot(vehicle_location_data[0,:], vehicle_location_data[1,:], vehicle_location_data[2,:], color="tab:yellow")
        ax.plot(sfc_points[0,:], sfc_points[1,:],sfc_points[2,:], alpha=0.5)
        ax.plot_surface(obstacle_x_data, obstacle_y_data, obstacle_z_data, color="r")
        ax.scatter(waypoints[0,:], waypoints[1,:],waypoints[2,:], marker="o")
        # Creating the Animation object
        delayBetweenFrames_ms = 50
        dt = delayBetweenFrames_ms / 1000 #seconds between frames
        line_ani = animation.FuncAnimation(fig, update_line, fargs=[plane_model,autopilot,path_follower, dt] , interval=delayBetweenFrames_ms, blit=False)

        plt.show()


    def plot_simulation(self, states_list: 'list[np.ndarray]', 
                           inputs_list: 'list[np.ndarray]',  
                           desired_trajectory_data: TrajectoryData,
                           true_trajectory_data: TrajectoryData, 
                           margins = 0, vehicle_instances_per_plot = 5):
        path_location_data = desired_trajectory_data.location_data
        true_location_data = true_trajectory_data.location_data
        #### Path Plot ####
        fig = plt.figure()
        num_data_points = np.shape(path_location_data)[1]
        ax = fig.add_subplot(111, aspect='equal', autoscale_on=False)
        x_limits = np.array([np.min(np.concatenate((true_location_data[0,:], path_location_data[0,:]))) - margins, 
                             np.max(np.concatenate((true_location_data[0,:], path_location_data[0,:]))) + margins])
        y_limits = np.array([np.min(np.concatenate((true_location_data[1,:], path_location_data[1,:]))) - margins, 
                             np.max(np.concatenate((true_location_data[1,:], path_location_data[1,:]))) + margins])
        ax.set_ybound((y_limits[0],y_limits[1]))
        ax.set_xbound((x_limits[0],x_limits[1]))
        ax.plot(path_location_data[0,:],path_location_data[1,:], color = 'tab:blue', label = "path")
        ax.plot(true_location_data[0,:],true_location_data[1,:], linestyle="--",
            color = 'tab:olive', label="true position")
        center_of_mass = plt.Circle((true_location_data[0,0], true_location_data[1,0]), 
                                    radius=0.1, fc='tab:olive', ec="none", zorder=10)
        path_point = plt.Circle((path_location_data[0,0], path_location_data[1,0]), 
                                    radius=0.1, fc='none', ec="tab:blue", zorder=11)
        path_length = 0
        distance_travelled = 0
        for i in range(num_data_points):
            if i != 0: 
                path_length += np.linalg.norm(path_location_data[:,i] - path_location_data[:,i-1])
                distance_travelled +=  np.linalg.norm(true_location_data[:,i] - true_location_data[:,i-1]) 
            if i%int(num_data_points/vehicle_instances_per_plot) == 0:
                self._vehicle_model.set_state(states_list[i])
                self._vehicle_model.set_inputs(inputs_list[i])
                self._vehicle_model.plot_vehicle_instance(ax)
                center_of_mass = plt.Circle((true_location_data[0,i], 
                                             true_location_data[1,i]), 
                                            radius=0.1, fc='tab:olive', ec="none", zorder=11)
                path_point = plt.Circle((path_location_data[0,i], path_location_data[1,i]), 
                                    radius=0.1, fc='none', ec="tab:blue", zorder=11)
                ax.add_patch(center_of_mass)
                ax.add_patch(path_point)
        time_to_traverse = desired_trajectory_data.time_data[-1] - desired_trajectory_data.time_data[0]
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        y_height = y_limits[1] - y_limits[0]
        y_step = y_height/15 
        ax.text(x_limits[0]+1,y_limits[0]+3*y_step,"Path Length: " + str(np.round(path_length,2)) + " m")
        ax.text(x_limits[0]+1,y_limits[0]+2*y_step,"Distance Travelled: "  + str(np.round(distance_travelled,2)) + " m")
        ax.text(x_limits[0]+1,y_limits[0]+1*y_step,"Traversal Time: " + str(np.round(time_to_traverse,2)) + " sec")
        ax.legend(loc="upper right")
        plt.show()

    def plot_simulation_dynamics(self, vehicle_motion_data: VehicleMotionData,
                                 desired_trajectory_data: TrajectoryData, 
                                 true_trajectory_data: TrajectoryData,
                                 max_velocity: float, max_acceleration: float,
                                 max_turn_value: float, turn_type: str,
                                 vehicle_type: str):
        # data extraction
        path_location_data = desired_trajectory_data.location_data
        path_velocity_data = desired_trajectory_data.velocity_data
        path_acceleration_data = desired_trajectory_data.acceleration_data
        path_longitudinal_acceleration_data = desired_trajectory_data.longitudinal_acceleration_data
        path_time_data = desired_trajectory_data.time_data
        true_location_data = true_trajectory_data.location_data
        true_velocity_data = true_trajectory_data.velocity_data
        true_longitudinal_acceleration_data = true_trajectory_data.longitudinal_acceleration_data
        true_time_data = true_trajectory_data.time_data
        vehicle_velocity_data = vehicle_motion_data.velocity_magnitude_data
        vehicle_velocity_data[vehicle_velocity_data <= 0] = 1
        if turn_type == "curvature": 
            path_turn_data = desired_trajectory_data.curvature_data
            true_turn_data = true_trajectory_data.curvature_data
            vehicle_turn_data = vehicle_motion_data.heading_angular_rate_data/vehicle_velocity_data
            turn_label = "curvature"
            turn_legend_label = "curv"
        elif turn_type == "angular_rate": 
            path_turn_data = desired_trajectory_data.angular_rate_data
            true_turn_data = true_trajectory_data.angular_rate_data
            vehicle_turn_data = vehicle_motion_data.heading_angular_rate_data
            turn_label = "angular rate \n (rad/sec)"
            turn_legend_label = "ang rate"
        elif turn_type == "centripetal_acceleration": 
            path_turn_data = desired_trajectory_data.centripetal_acceleration_data
            true_turn_data = true_trajectory_data.centripetal_acceleration_data
            vehicle_turn_data = vehicle_motion_data.heading_angular_rate_data*vehicle_velocity_data
            turn_label = "centripetal \n acceleration \n (m/s^2)"
            turn_legend_label = "centr accel"
        vehicle_turn_data = np.abs(vehicle_turn_data)
        # calculations
        position_error = np.linalg.norm((path_location_data - true_location_data),2,0)
        path_velocity_magnitude_data = np.linalg.norm(path_velocity_data,2,0)
        path_acceleration_magnitude = np.linalg.norm(path_acceleration_data,2,0)
        path_long_accel_mag = np.abs(path_longitudinal_acceleration_data)
        true_velocity_magnitude = np.linalg.norm(true_velocity_data,2,0)
        true_long_accel_mag = np.abs(true_longitudinal_acceleration_data)
        fig, axs = plt.subplots(4,1)
        axs[0].plot(path_time_data,position_error, color = 'tab:red', label="tracking\n error")
        axs[0].plot(path_time_data,path_time_data*0, color = 'k')
        axs[0].set_ylabel("tracking error \n (m)")
        axs[1].plot(path_time_data, path_time_data*0 + max_velocity, color='k', label="max")
        axs[1].plot(path_time_data, path_velocity_magnitude_data, color = 'tab:blue', label= "des")
        axs[1].plot(true_time_data, true_velocity_magnitude, color = 'tab:olive', label= "true",linestyle="--")   
        axs[1].set_ylabel("velocity \n (m/s)")
        axs[2].plot(path_time_data, path_time_data*0 + max_acceleration, color='k', label="max")
        # axs[2].plot(path_time_data, path_acceleration_magnitude,color='tab:cyan',label="des accel")
        axs[2].plot(path_time_data, path_long_accel_mag,color='tab:blue',label="des")
        axs[2].plot(true_time_data, true_long_accel_mag, color = 'tab:olive', label =  "true",linestyle="--")
        axs[2].set_ylabel("longitudinal \n acceleration \n (m/s^2)")
        axs[3].plot(path_time_data,path_time_data*0 + max_turn_value, color='k', label="max")
        axs[3].plot(path_time_data,path_turn_data,color='tab:blue', label="des")
        if vehicle_type == "bike":
            axs[3].plot(true_time_data,vehicle_turn_data,color='tab:olive', label= "body")
        else:
            axs[3].plot(true_time_data,true_turn_data,color='tab:olive', label= "true",linestyle="--")

        axs[3].set_ylabel(turn_label)
        axs[3].set_xlabel("time (sec)")
        axs[0].legend(loc='upper left')
        axs[1].legend(loc='lower left')
        axs[2].legend(loc='lower left')
        axs[3].legend(loc='lower left')
        axs[0].tick_params(labelbottom = False, bottom = False)
        axs[1].tick_params(labelbottom = False, bottom = False)
        axs[2].tick_params(labelbottom = False, bottom = False)
        plt.show()

    def run_simulation_real_time(self, desired_trajectory_data: TrajectoryData, 
                                    sleep_time = 0, margins = 0):
        #### extract path data ####
        location_data = desired_trajectory_data.location_data
        velocity_data = desired_trajectory_data.velocity_data
        acceleration_data = desired_trajectory_data.acceleration_data
        jerk_data = desired_trajectory_data.jerk_data
        time_data = desired_trajectory_data.time_data
        #### run simulation ####
        vehicle_location_data = location_data*0
        dt = time_data[1] - time_data[0]
        x_limits = np.array([np.min(np.concatenate((vehicle_location_data[0,:], location_data[0,:]))) - margins, 
                             np.max(np.concatenate((vehicle_location_data[0,:], location_data[0,:]))) + margins])
        y_limits = np.array([np.min(np.concatenate((vehicle_location_data[1,:], location_data[1,:]))) - margins, 
                             np.max(np.concatenate((vehicle_location_data[1,:], location_data[1,:]))) + margins])
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                            xlim=(x_limits[0],x_limits[1]), ylim=(y_limits[0],y_limits[1]))
        ax.grid()
        desired_position_fig = plt.Circle((0, 0), radius=0.1, fc='none', ec="tab:blue", zorder=10)
        vehicle_states = self._vehicle_model.get_state()
        center_of_mass = plt.Circle((vehicle_states[0,0], vehicle_states[0,1]), 
                                    radius=0.1, fc='tab:olive', ec="none", zorder=10)
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
        ax.plot(location_data[0,:],location_data[1,:])
        dt = time_data[1] - time_data[0]
        def init():
            #initialize animation
            self._vehicle_model.add_patches_to_axes(ax)
            time_text.set_text('')
            patches = (desired_position_fig,center_of_mass, time_text)
            all_patches = self._vehicle_model.add_patches_to_tuple(patches)
            ax.add_patch(desired_position_fig)
            ax.add_patch(center_of_mass)
            return all_patches
        def animate(i):
            # propogate robot motion
            t = time_data[i]
            desired_states = np.vstack((location_data[:,i], velocity_data[:,i],
                                         acceleration_data[:,i],jerk_data[:,i]))
            inputs = self._vehicle_model.get_inputs()
            states = self._vehicle_model.get_state()
            motion_command, turn_command = self._trajectory_tracker.mpc_control_accel_input(inputs, states, desired_states)
            self._vehicle_model.update_acceleration_motion_model(motion_command, turn_command, dt)
            self._vehicle_model.update_patches()
            desired_position_fig.center = (location_data[0,i], location_data[1,i])
            center_of_mass.center = (states[0,0], states[0,1])
            time_text.set_text('time = %.1f' % t)
            sleep(sleep_time)
            patches = (desired_position_fig,center_of_mass, time_text)
            all_patches = self._vehicle_model.add_patches_to_tuple(patches)
            return all_patches
        animate(0)
        ani = animation.FuncAnimation(fig, animate, frames = np.size(time_data), 
                                        interval = dt*100, blit = True, 
                                        init_func = init, repeat = False)
        plt.show()
