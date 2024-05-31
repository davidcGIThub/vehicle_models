
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
run_time = 36.5
gravity = 9.8
max_roll = np.radians(25)
desired_airspeed = 20
max_pitch = np.radians(15)
max_curvature = gravity*np.tan(max_roll)/(desired_airspeed**2)

max_incline_angle = max_pitch
max_incline = np.tan(max_incline_angle)

obstacle_data = np.load('obstacles.npy')

obstacle_list = []
num_obstacles = np.shape(obstacle_data)[1]
for i in range(num_obstacles):
    center_ = np.array([obstacle_data[0,i], obstacle_data[1,i], 0])
    height_ = obstacle_data[2,i]
    radius_ = obstacle_data[3,i]
    obstacle_ = Obstacle(center= center_, radius=radius_, height=height_)
    obstacle_list.append(obstacle_)
# obstacle_list = []

control_points = np.array([[-39.40236971, -18.37747273, 112.91226063, 214.92273374, 304.45155231,
  395.12175263, 475.5594883,  512.22025585, 475.5594883 ],
 [-33.95807155,  -6.61999668 , 60.43805826, 135.33386488, 214.65073076,
  293.49811072, 386.80754004, 514.45502228, 555.37237084],
  [-100, -100, -100, -100, -100, -100, -100, -100, -100]])

control_point_list = [control_points]
fixed_wing_parameters = FixedWingParameters()
control_parameters = FixedWingControlParameters()
# Attaching 3D axis to the figure
# ax = plt.axes(projection='3d')
# plt.show()

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
plane_model = FixedWingModel(vehicle_parameters = fixed_wing_parameters,
                  wingspan = wingspan, fuselage_length = fuselage_length, state = state0)
autopilot = FixedWingAutopilot(control_parameters)
path_follower = FixedWingSplinePathFollower(order, distance_p_gain = 6, distance_i_gain = 0.05, distance_d_gain = 3.5,
                                            path_direction_gain = 60, feedforward_gain = 600, feedforward_distance = 3, 
                                            start_position = np.array([north,east,down]))
    
path_manager = SplinePathManager(control_point_list)

wing_sim = FixedWingPathFollowingSimulator(plane_model, autopilot, path_follower, path_manager)

vehicle_path_data, tracked_path_data, closest_distances_to_obstacles, closest_distances_to_sfc_walls \
    = wing_sim.run_simulation(control_point_list, desired_airspeed, dt=0.1, 
                              run_time=run_time, graphic_scale=20, obstacle_list = obstacle_list, obstacle_type="building")

wing_sim.plot_simulation_analytics(vehicle_path_data, tracked_path_data,
                max_curvature, max_incline_angle, closest_distances_to_obstacles)
