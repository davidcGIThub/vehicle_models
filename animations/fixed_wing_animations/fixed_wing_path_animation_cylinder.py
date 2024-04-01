
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
run_time = 39.6
gravity = 9.8
max_roll = np.radians(25)
desired_airspeed = 20
max_pitch = np.radians(15)
max_curvature = gravity*np.tan(max_roll)/(desired_airspeed**2)

max_incline_angle = max_pitch
max_incline = np.tan(max_incline_angle)

control_points = np.array([
    [-94.24438422,   5.64360774,  71.66995326,  65.00296967, 172.88335329, 349.94638496, 522.94202478 , 558.28551591],
    [-16.90078032, -17.25572767,  85.92369099, 270.23051531, 416.70937845, 487.4740207,  506.26298965, 487.4740207],
    [-100,                -100,        -100,         -100,        -100,         -100,         -100,       -100]])

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
u = desired_airspeed
v = 0
w = 0

e0 = 0.9617692
e1 = 0
e2 = 0
e3 = 0.27386128
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
path_follower = FixedWingSplinePathFollower(order, distance_gain=5, path_direction_gain=60, feedforward_gain=500, feedforward_distance=3, integrator_gain=0.1)
path_manager = SplinePathManager(control_point_list)

obstacle = Obstacle(np.array([250,250,0]), 282.84/2, 300)
# obstacle_list = [obstacle]
obstacle_list = []

wing_sim = FixedWingPathFollowingSimulator(plane_model, autopilot, path_follower, path_manager)
vehicle_path_data, tracked_path_data, closest_distances_to_obstacles, closest_distances_to_sfc_walls \
    = wing_sim.run_simulation(control_point_list, desired_airspeed, dt=0.1, 
                              run_time=run_time, graphic_scale=20, obstacle_list =obstacle_list,obstacle_type="cylinder")

wing_sim.plot_simulation_analytics(vehicle_path_data, tracked_path_data,
                max_curvature, max_incline_angle, closest_distances_to_obstacles)