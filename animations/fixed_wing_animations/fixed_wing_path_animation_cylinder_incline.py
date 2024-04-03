
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
desired_speed = 20
run_time = 37.4
gravity = 9.8
max_roll = np.radians(25)
desired_airspeed = 20
max_pitch = np.radians(15)
max_curvature = gravity*np.tan(max_roll)/(desired_airspeed**2)

max_incline_angle = max_pitch
max_incline = np.tan(max_incline_angle)

control_points = np.array([
    [ -65.00226093,   -2.57838108,   75.31578524,  136.03337131,  236.59760875, 381.35082625,  513.02041039,  566.56753221],
 [ -10.6803,      -16.40017867,   76.28101469,  227.92915389,  364.15485439, 480.94752858,  509.52623571,  480.94752858],
 [-106.4248479,   -96.78757605, -106.4248479,  -132.74180927, -163.73702832, -194.73303362, -202.63348319 ,-194.73303362]])

control_point_list = [control_points]
fixed_wing_parameters = FixedWingParameters()
control_parameters = FixedWingControlParameters()
# Attaching 3D axis to the figure
fig = plt.figure()
ax = plt.axes(projection='3d')
fig.add_axes(ax)
north = 0
east = 0
down = -100
u = desired_speed
v = 0
w = 0

e0 = 0.9617692
e1 = 0
e2 = 0
e3 = 0.27386128
p = 0
q = 0
r = 0
wingspan = 1.5
fuselage_length = 1.5
state0 = np.array([north, east, down,  u, v, w,
                      e0,   e1,   e2, e3, p, q, r])
plane_model = FixedWingModel(ax, fixed_wing_parameters,
                  wingspan = wingspan, fuselage_length = fuselage_length,
                    state = state0)
autopilot = FixedWingAutopilot(control_parameters)
path_follower = FixedWingSplinePathFollower(order, distance_gain=5, path_direction_gain=60, feedforward_gain=500, feedforward_distance=3, integrator_gain=0.1)
path_manager = SplinePathManager(control_point_list)

obstacle = Obstacle(np.array([250,250,0]), 141.42/2, 300)
# obstacle_list = [obstacle]
obstacle_list = []
wing_sim = FixedWingPathFollowingSimulator(plane_model, autopilot, path_follower, path_manager)
vehicle_path_data, tracked_path_data, closest_distances_to_obstacles, closest_distances_to_sfc_walls \
    = wing_sim.run_simulation(control_point_list, desired_speed, dt=0.1, 
                              run_time=run_time, graphic_scale=20, obstacle_list = obstacle_list,obstacle_type="cylinder")

wing_sim.plot_simulation_analytics(vehicle_path_data, tracked_path_data,
                max_curvature, max_incline_angle, closest_distances_to_obstacles)