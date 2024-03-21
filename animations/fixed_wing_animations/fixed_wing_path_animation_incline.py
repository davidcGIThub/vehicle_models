
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
desired_speed = 25
run_time = 45
# control_points_1 = np.array([[-1.52861597e+02,  1.09044181e-01,  1.52425420e+02,  2.97438455e+02,
#    2.97634662e+02,  1.52506282e+02,  1.98254829e-02, -1.52585584e+02],
#  [-5.82620036e+00,  2.91310018e+00, -5.82620036e+00,  8.69934173e+01,
#    2.63074638e+02,  3.55752540e+02,  3.47123730e+02,  3.55752540e+02],
#  [-7.76838787e+01,  8.84193936e+00, -7.76838787e+01, -1.66018163e+02,
#   -2.53921240e+02, -3.42137105e+02, -4.28931447e+02, -3.42137105e+02]])
control_points_1 = np.array([[-1.92718378e+02, -2.12381699e+00,  2.01213646e+02 , 4.35400071e+02,
   4.53600650e+02,  2.31293646e+02,  2.35320807e-01, -2.32234929e+02],
 [-2.67206060e+01,  1.33603030e+01, -2.67206060e+01,  3.52323025e+01,
   2.75741582e+02,  3.72160695e+02,  3.38919653e+02,  3.72160695e+02],
 [-7.84626885e+01,  9.23134424e+00, -7.84626885e+01, -1.66158330e+02,
  -2.53844195e+02, -3.41540557e+02, -4.29229722e+02, -3.41540557e+02]])
gravity = 9.8
max_roll = np.radians(30)
desired_airspeed = 25
max_pitch = np.radians(24)
max_curvature = gravity*np.tan(max_roll)/(desired_airspeed**2)
max_incline_angle = max_pitch
max_incline = np.tan(max_incline_angle)

control_point_list = [control_points_1]
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
path_follower = FixedWingSplinePathFollower(order, distance_gain=2, path_gain=2, feedforward_gain=4, feedforward_distance=3)
path_manager = SplinePathManager(control_point_list)

wing_sim = FixedWingPathFollowingSimulator(plane_model, autopilot, path_follower, path_manager)
vehicle_path_data, tracked_path_data, closest_distances_to_obstacles, closest_distances_to_sfc_walls \
    = wing_sim.run_simulation(control_point_list, desired_speed, dt=0.1, run_time=run_time, graphic_scale=20)

wing_sim.plot_simulation_analytics(vehicle_path_data, tracked_path_data,
                max_curvature, max_incline_angle, closest_distances_to_obstacles)