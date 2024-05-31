
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
run_time = 30.5
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
obstacle_list = []

control_points = np.array([[ -79.50277224,    3.9407126,    63.73992184,   56.56083014,  -18.58082395,
  -148.69068352 -204.36757345 -185.82158397 -100.38247264  -12.64852547]
 [ -33.19478476,   -5.59606966,   55.57906341,  107.76989604 , 127.82599768
    97.32091198,   28.60834649,  -30.94174684,  -59.52912658 , -30.94174684]
 [-276.794472,   -311.602764,   -276.794472 ,  -198.39364613, -149.33533766
  -130.92747221,  -96.19054002,  -21.156749  ,   10.5783745,   -21.156749  ]])

control_point_list = [control_points]
fixed_wing_parameters = FixedWingParameters()
control_parameters = FixedWingControlParameters()
# Attaching 3D axis to the figure
# ax = plt.axes(projection='3d')
# plt.show()

north = 0
east = 0
down = -200
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
                              run_time=run_time,instances_per_plot=7, graphic_scale=20, obstacle_list = obstacle_list, obstacle_type="building")

wing_sim.plot_simulation_analytics(vehicle_path_data, tracked_path_data,
                max_curvature, max_incline_angle, closest_distances_to_obstacles)
