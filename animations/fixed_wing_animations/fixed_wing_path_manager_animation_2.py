
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

from time import sleep


order = 3
desired_speed = 15
control_points_1 =  np.array([[-7.23418482e+01,  2.99798671e-21,  7.23418482e+01,  1.13010579e+02,
   1.05179340e+02,  6.28997713e+01,  8.24636404e-01, -6.61983169e+01],
 [ 5.56169859e-22,  1.61043950e-21, -4.34816644e-21,  8.67548531e+01,
   2.17591255e+02,  3.29143647e+02,  3.60428177e+02,  3.29143647e+02],
 [-2.00000000e+01, -2.00000000e+01, -2.00000000e+01, -1.95117695e+01,
  -1.87644762e+01, -1.81203927e+01, -1.79398036e+01, -1.81203927e+01]])

control_points_2 = np.array([[  66.55742562,    0.92623392,  -70.26236129, -134.56164128, -143.59918351,
   -88.64354515,   -0.81038716,   91.88509378],
 [ 326.5740854,   361.7129573,   326.5740854 ,  384.77681413,  519.55742394,
   669.45418878,  715.27290561,  669.45418878],
 [ -18.13522543, -17.93238729,  -18.13522543,  -19.06544087,  -20.63603497,
   -21.83535411,  -22.08232294,  -21.83535411]])
bspline_eval = BsplineEvaluator(order)
position_array_1 = bspline_eval.matrix_bspline_evaluation_for_dataset(control_points_1, 1000)
# print("position_array_1: " , np.shape(position_array_1))
position_array_2 = bspline_eval.matrix_bspline_evaluation_for_dataset(control_points_2, 1000)
control_point_list = [control_points_1, control_points_2]
max_curvature = 2
max_incline = 1
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
state0 = np.array([north, east, down, u, v, w,
                   e0,  e1,  e2,  e3, p, q, r])
plane_model = FixedWingModel(ax, fixed_wing_parameters,
                  wingspan = wingspan, fuselage_length = fuselage_length,
                    state = state0)
autopilot = FixedWingAutopilot(control_parameters)
path_follower = FixedWingSplinePathFollower(order, distance_gain=2, path_gain=2, feedforward_gain=1)
path_manager = SplinePathManager(control_point_list)

wing_sim = FixedWingPathFollowingSimulator(plane_model, autopilot, path_follower, path_manager)
vehicle_path_data, tracked_path_data, closest_distances_to_obstacles, closest_distances_to_sfc_walls \
    = wing_sim.run_simulation(control_point_list, desired_speed, dt=0.1, run_time=65)

wing_sim.plot_simulation_analytics(vehicle_path_data, tracked_path_data, 
                max_curvature, max_incline)