
"""
A simple example of an animated plot... In 3D!
"""
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
from vehicle_simulator.vehicle_models.fixed_wing_model import FixedWingModel
from vehicle_simulator.vehicle_models.fixed_wing_parameters import FixedWingParameters
from vehicle_simulator.vehicle_controllers.fixed_wing_autopilot import FixedWingControlParameters, FixedWingAutopilot
from vehicle_simulator.vehicle_controllers.fixed_wing_path_follower import FixedWingSplinePathFollower
from vehicle_simulator.vehicle_controllers.bspline_path_manager import SplinePathManager
from vehicle_simulator.vehicle_controllers.bspline_evaluator import BsplineEvaluator
from vehicle_simulator.vehicle_simulators.fixed_wing_path_follower_simulator import FixedWingPathFollowingSimulator
from vehicle_simulator.vehicle_simulators.spatial_violations import Obstacle

from time import sleep


order = 3
desired_speed = 21.375
control_points_1 = np.array([[-1.07492420e+02, -2.83848959e-21,  1.07492420e+02,  1.82460572e+02,
                              1.77879263e+02,  1.06184682e+02,  9.32950387e-01, -1.09916484e+02],
                            [-3.92419879e-19,  1.23142785e-19, -1.28381816e-19,  8.45441767e+01,
                              2.13257604e+02,  3.28469421e+02,  3.60765290e+02,  3.28469421e+02],
                            [-2.00000000e+01, -2.00000000e+01, -2.00000000e+01, -1.95152704e+01,
                              -1.87773538e+01, -1.81218640e+01, -1.79390680e+01, -1.81218640e+01]])

control_points_2 = np.array([[101.64932239,    0.85365222, -105.06393129, -203.66106584, -227.31437978,
                            -141.72545411,   -1.51708532,  147.79379538],
                          [ 330.29945962,  359.85027019,  330.29945962,  396.60900548 , 524.74031082,
                            670.13664262,  714.93167869,  670.13664262],
                          [ -18.11150593,  -17.94424703,  -18.11150593,  -19.14083048,  -20.50137237,
                            -21.76106676,  -22.11946662,  -21.76106676]])
# obstacle_1 = Obstacle(center=np.array([[25,],[25],[-20]]), radius = 20)
# obstacle_2 = Obstacle(center=np.array([[100,],[25],[-20]]), radius = 20)
# obstacle_list = [obstacle_1, obstacle_2]
bspline_eval = BsplineEvaluator(order)
position_array_1 = bspline_eval.matrix_bspline_evaluation_for_dataset(control_points_1, 1000)
# print("position_array_1: " , np.shape(position_array_1))
position_array_2 = bspline_eval.matrix_bspline_evaluation_for_dataset(control_points_2, 1000)
control_point_list = [control_points_1, control_points_2]
max_curvature = 0.01
max_incline = 0.01
fixed_wing_parameters = FixedWingParameters()
control_parameters = FixedWingControlParameters()
# Attaching 3D axis to the figure
fig = plt.figure()
ax = plt.axes(projection='3d')
fig.add_axes(ax)
north = 0
east = 0
down = -20
u = desired_speed
v = 0
w = 0
e0 = 1
e1 = 0
e2 = 0
e3 = 0
p = 0
q = 0
r = 0
wingspan = 3
fuselage_length = 3
state0 = np.array([north, east, down,  u, v, w,
                      e0,   e1,   e2, e3, p, q, r])
plane_model = FixedWingModel(ax, fixed_wing_parameters,
                  wingspan = wingspan, fuselage_length = fuselage_length,
                    state = state0)
autopilot = FixedWingAutopilot(control_parameters)
path_follower = FixedWingSplinePathFollower(order, distance_gain=4, path_gain=3, feedforward_gain=2, feedforward_distance=5)
path_manager = SplinePathManager(control_point_list)

wing_sim = FixedWingPathFollowingSimulator(plane_model, autopilot, path_follower, path_manager)
vehicle_path_data, tracked_path_data, closest_distances_to_obstacles, closest_distances_to_sfc_walls \
    = wing_sim.run_simulation(control_point_list, desired_speed, dt=0.1, run_time=48, graphic_scale=20)

wing_sim.plot_simulation_analytics(vehicle_path_data, tracked_path_data,
                max_curvature, max_incline_angle=None, closest_distances_to_obstacles=closest_distances_to_obstacles)