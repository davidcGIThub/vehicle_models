
"""
A simple example of an animated plot... In 3D!
"""
import numpy as np
import matplotlib.pyplot as plt
from vehicle_simulator.vehicle_models.fixed_wing_model import FixedWingModel
from vehicle_simulator.vehicle_models.fixed_wing_parameters import FixedWingParameters
from vehicle_simulator.vehicle_controllers.fixed_wing_autopilot import FixedWingControlParameters, FixedWingAutopilot
from vehicle_simulator.vehicle_controllers.fixed_wing_path_follower import FixedWingSplinePathFollower
from vehicle_simulator.vehicle_controllers.bspline_path_manager import SplinePathManager
from vehicle_simulator.vehicle_controllers.bspline_evaluator import BsplineEvaluator
from vehicle_simulator.vehicle_simulators.fixed_wing_path_follower_simulator import FixedWingPathFollowingSimulator
from vehicle_simulator.vehicle_simulators.spatial_violations import Obstacle
from vehicle_simulator.vehicle_models.helper_functions import euler_to_quaternion



order = 3
desired_speed = 16
run_time = 300
gravity = 9.8
max_roll = np.radians(25)
desired_airspeed = 20
max_pitch = np.radians(15)
max_curvature = gravity*np.tan(max_roll)/(desired_airspeed**2)

max_incline_angle = max_pitch
max_incline = np.tan(max_incline_angle)
max_incline = 0.2

obstacle_radius = 50
obstacle_center = np.array([[250],[250],[150]])
obstacle = Obstacle(center=obstacle_center, radius=obstacle_radius, height=0)
obstacle_list = [obstacle] 

control_points_0 = np.load("control_points_0.npy")
control_points_1 = np.load("control_points_1.npy")
control_points_2 = np.load("control_points_2.npy")
control_points_3 = np.load("control_points_3.npy")
control_points_4 = np.load("control_points_4.npy")
control_points_5 = np.load("control_points_5.npy")
waypoints = np.load("waypoints.npy")
# obstacle_1 = Obstacle(center=np.array([[25,],[25],[-20]]), radius = 20)
# obstacle_2 = Obstacle(center=np.array([[100,],[25],[-20]]), radius = 20)
# obstacle_list = [obstacle_1, obstacle_2]
bspline_eval = BsplineEvaluator(order)
start_direction = bspline_eval.get_velocity_vector(0, control_points_0[:,0:4], 1)
position_array_0 = bspline_eval.matrix_bspline_evaluation_for_dataset(control_points_0, 1000)
control_point_list = [control_points_0, control_points_1, control_points_2,control_points_3,
                      control_points_4,control_points_5]

fixed_wing_parameters = FixedWingParameters()
control_parameters = FixedWingControlParameters()
# Attaching 3D axis to the figure
fig = plt.figure()
ax = plt.axes(projection='3d')
fig.add_axes(ax)
north = position_array_0[0,0]
east = position_array_0[1,0]
down = position_array_0[2,0]
quat = euler_to_quaternion(0,0,np.pi)
u = 25
v = 0
w = 0
e0 = quat.item(0)
e1 = quat.item(1)
e2 = quat.item(2)
e3 = quat.item(3)
p = 0
q = 0
r = 0
wingspan = 2
fuselage_length = 2
state0 = np.array([north, east, down,  u, v, w,
                      e0,   e1,   e2, e3, p, q, r])

plane_model = FixedWingModel(ax, fixed_wing_parameters,
                  wingspan = wingspan, fuselage_length = fuselage_length,
                    state = state0)
autopilot = FixedWingAutopilot(control_parameters)
path_follower = FixedWingSplinePathFollower(order, distance_p_gain = 6, distance_i_gain = 0.05, distance_d_gain = 3.5,
                                            path_direction_gain = 60, feedforward_gain = 600, feedforward_distance = 3, 
                                            start_position = np.array([north,east,down]))
path_manager = SplinePathManager(control_point_list)

wing_sim = FixedWingPathFollowingSimulator(plane_model, autopilot, path_follower, path_manager)
vehicle_path_data, tracked_path_data, closest_distances_to_obstacles, closest_distances_to_sfc_walls \
    = wing_sim.run_simulation(control_point_list, desired_speed, dt=0.1, run_time=run_time, graphic_scale=20, waypoints =waypoints,
                              obstacle_list = obstacle_list)

wing_sim.plot_simulation_analytics(vehicle_path_data, tracked_path_data,
                max_curvature, max_incline_angle=max_incline, closest_distances_to_obstacles=closest_distances_to_obstacles)