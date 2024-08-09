
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
from vehicle_simulator.vehicle_models.helper_functions import euler_to_quaternion
from time import sleep



order = 3
run_time = 64
gravity = 9.8
max_roll = np.radians(25)
desired_airspeed = 27.5
max_pitch = np.radians(15)
max_curvature = gravity*np.tan(max_roll)/(desired_airspeed**2)

max_incline_angle = max_pitch
max_incline = np.tan(max_incline_angle)

# obstacle_list = []

control_points = np.array([[ 770.97554794,  600.15676313,  428.39739953,  383.62744434,  501.58016088,
   693.56792611,  839.91077866,  829.0830777,   813.79780047 , 770.75819338,
   600.26544037,  428.18004515],
 [-154.78908565 , -72.60545718, -154.78908565 ,-341.38959488, -491.90995326,
  -493.11767462 ,-349.12348753, -100.32523817,  159.4482591,   345.26423442,
   427.36788279 , 345.26423442],
                 [300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300]])

# control_points = np.array([
#     [-94.24438422,   5.64360774,  71.66995326,  65.00296967, 172.88335329, 349.94638496, 522.94202478 , 558.28551591],
#     [-16.90078032, -17.25572767,  85.92369099, 270.23051531, 416.70937845,  487.4740207,  506.26298965, 487.4740207 ],
#     [-100,                 -100,         -100,         -100,         -100,         -100,          -100,       -100  ]])

# control_points = np.array([
#     [-94.24438422,   5.64360774,  71.66995326,  65.00296967, 172.88335329, 349.94638496, 522.94202478 , 558.28551591],
#     [-94.24438422,   5.64360774,  71.66995326,  65.00296967, 172.88335329, 349.94638496, 522.94202478 , 558.28551591 ],
#     [-100,                 -100,         -100,         -100,         -100,         -100,          -100,       -100  ]])

control_point_list = [control_points]
fixed_wing_parameters = FixedWingParameters()
control_parameters = FixedWingControlParameters()
# Attaching 3D axis to the figure
# ax = plt.axes(projection='3d')
# plt.show()
scale_factor = 1

bspline_eval = BsplineEvaluator(order)
start_velocity = bspline_eval.get_velocity_vector(0, control_points[:,0:4], scale_factor)
start_position = bspline_eval.get_position_vector(0, control_points[:,0:4], scale_factor)

north = start_position.item(0)
east = start_position.item(1)
down = start_position.item(2)
quat = euler_to_quaternion(0,0,np.pi)
u = 26
v = 0
w = 0
e0 = quat.item(0)
e1 = quat.item(1)
e2 = quat.item(2)
e3 = quat.item(3)
p = 0
q = 0
r = 0
wingspan = 3
fuselage_length = 3
state0 = np.array([north, east, down,  u, v, w,
                      e0,   e1,   e2, e3, p, q, r])
plane_model = FixedWingModel(vehicle_parameters = fixed_wing_parameters,
                  wingspan = wingspan, fuselage_length = fuselage_length, state = state0)
autopilot = FixedWingAutopilot(control_parameters)
path_follower = FixedWingSplinePathFollower(order, distance_p_gain = 6, distance_i_gain = 0.05, distance_d_gain = 3.5,
                                            path_direction_gain = 60, feedforward_gain = 600, feedforward_distance = 3, 
                                            start_position = np.array([north,east,down]))
    
path_manager = SplinePathManager(control_point_list)

wing_sim = FixedWingPathFollowingSimulator(plane_model, autopilot, path_follower, path_manager)

vehicle_path_data, tracked_path_data, closest_distances_to_obstacles, closest_distances_to_sfc_walls \
    = wing_sim.run_simulation(control_point_list, desired_airspeed, dt=0.01, 
                              run_time=run_time, graphic_scale=20)

wing_sim.plot_simulation_analytics(vehicle_path_data, tracked_path_data,
                max_curvature, max_incline_angle, closest_distances_to_obstacles)
