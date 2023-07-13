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
                 path_follower: FixedWingSplinePathFollower):
        self._plane_model = plane_model
        self._plane_autopilot = plane_autopilot
        self._path_follower = path_follower
        self._order = self._path_follower.get_order()
        self._spline_eval = BsplineEvaluator(self._order)

    def run_simulation(self, path_control_points, desired_speed, 
                       obstacle_list:list = [], sfc_list:list = [],
                       intervals_per_sfc: np.ndarray = np.array([]), 
                       dt: float = 0.01, run_time: float= 30,
                       animate = True, plot=True, sleep_time = 0):
        states_list, vehicle_location_data, closest_path_point_data, closest_distances_to_obstacles, closest_distances_to_sfc_walls \
              = self.collect_simulation_data(path_control_points, desired_speed,
                                obstacle_list, sfc_list, intervals_per_sfc,
                                dt, run_time)
        if animate == True:
            self.animate_simulation(self, states_list, vehicle_location_data, 
                           path_control_points, obstacle_list, sfc_list,
                           margins=30, sleep_time=0, waypoints=np.array([]), 
                           frame_width=30, dt=0.01)
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
        path_data = self._spline_eval.matrix_bspline_evaluation_for_dataset(path_control_points,self._order,1000)
        for i in range(num_data_points):
            state = self._plane_model.get_state()
            position = np.array([state.item(0), state.item(1), state.item(2)])
            cmds = self._path_follower.get_commands(path_control_points, position, desired_speed)
            delta = self._plane_autopilot.get_commands(cmds, state, wind, dt)
            self._plane_model.update(delta, wind, dt)
            closest_point, t = self._spline_eval.get_closest_point_and_t(path_control_points, 1, position)
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
                           obstacle_list:list = [], sfc_list:list = [],
                           margins:float = 0, sleep_time:float = 0,
                           waypoints: np.ndarray = np.array([]),
                           frame_width:float = 30, dt: float = 0.01):
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        fig.add_axes(ax)
        path_data = self._spline_eval.matrix_bspline_evaluation_for_dataset(path_control_points, self._order, 1000)
        center_of_mass, = ax.plot([],[],[],lw=.5,color="tab:blue", marker = 'o')
        ax.plot(path_data[0,:], path_data[1,:],path_data[2,:], alpha=0.5)
        ax.scatter(waypoints[0,:], waypoints[1,:],waypoints[2,:], marker="o")
        def update_line(num, plane_model: FixedWingModel):
            state = states_list[num]
            plane_model.set_state(states_list[state])
            plane_model.plot_plane(ax)
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
        # Creating the Animation object
        delayBetweenFrames_ms = dt*1000 # seconds between frames
        line_ani = animation.FuncAnimation(fig, update_line, fargs=[self._plane_model] , interval=delayBetweenFrames_ms, blit=False)
        plt.show()


    # def plot_simulation()

    # def plot_simulation_analytics()