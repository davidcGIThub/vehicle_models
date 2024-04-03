
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
from vehicle_simulator.vehicle_simulators.fixed_wing_path_follower_simulator import FixedWingPathFollowingSimulator, PathData
from vehicle_simulator.vehicle_simulators.spatial_violations import Obstacle

from time import sleep
states_arr = np.load('vehicle_states.npy')
path_curvature_arr = np.load('path_curvature.npy')
path_incline_arr = np.load('path_incline.npy')
path_location_arr = np.load('path_location.npy').transpose()
path_perpindicular_arr = np.load('path_perpindicular.npy').transpose()
path_tangent_arr = np.load('path_tangent.npy').transpose()
time_arr = np.load('time_arr.npy')
parameters = np.load('parameters.npy')
obstacle_data = np.load('obstacle_data.npy')
states_list = []
num_states = np.shape(states_arr)[0]
vehicle_location_data = np.zeros((3,num_states))
vehicle_curvature_data = np.zeros(num_states)
vehicle_incline_data = np.zeros(num_states)
closest_location_data = np.zeros((3,num_states))
closest_curvature_data = np.zeros(num_states)
closest_incline_data = np.zeros(num_states)


num_points = len(path_curvature_arr)
indices = np.linspace(0,num_points-1,10,dtype=int)
spread_locations = path_location_arr[:,indices]
spread_tangents = path_tangent_arr[:,indices]
spread_perpindiculars = path_perpindicular_arr[:,indices]
ax = plt.figure().add_subplot(projection='3d')
ax.plot(path_location_arr[0,:]  ,path_location_arr[1,:],path_location_arr[2,:])
ax.scatter(spread_locations[0,:],spread_locations[1,:] ,spread_locations[2,:])
ax.quiver(spread_locations[0,:] ,spread_locations[1,:] ,spread_locations[2,:],
            spread_tangents[0,:],spread_tangents[1,:]  ,spread_tangents[2,:],
                length=100, normalize=True, color = 'g')
ax.quiver(spread_locations[0,:],spread_locations[1,:],spread_locations[2,:],
            spread_perpindiculars[0,:],spread_perpindiculars[1,:],spread_perpindiculars[2,:],
                length=100, normalize=True,color="r")
plt.show()



order = 3

gravity = 9.8
max_roll = np.radians(25)
desired_airspeed = 20
max_pitch = np.radians(15)
max_curvature = gravity*np.tan(max_roll)/(desired_airspeed**2)
max_incline_angle = max_pitch
max_incline = np.tan(max_incline_angle)

# obstacle_1 = Obstacle(center=np.array([[25,],[25],[-20]]), radius = 20)
# obstacle_2 = Obstacle(center=np.array([[100,],[25],[-20]]), radius = 20)
obstacle = Obstacle(np.array([250,250,0]), 141.42/2, 300)
obstacle_list = [obstacle]
obstacle_list = []


fixed_wing_parameters = FixedWingParameters()
control_parameters = FixedWingControlParameters()
# Attaching 3D axis to the figure

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
                  wingspan = wingspan, fuselage_length = fuselage_length,
                    state = state0)

autopilot = FixedWingAutopilot(control_parameters)

path_follower = FixedWingSplinePathFollower(order, distance_p_gain = 6, distance_i_gain = 0.05, distance_d_gain = 3.5,
                                            path_direction_gain = 60, feedforward_gain = 600, feedforward_distance = 3, 
                                            start_position = np.array([north,east,down]))

control_points_1 = np.array([[-1.92718378e+02, -2.12381699e+00,  2.01213646e+02 , 4.35400071e+02],
                             [-2.67206060e+01,  1.33603030e+01, -2.67206060e+01,  3.52323025e+01],
                             [-7.84626885e+01,  9.23134424e+00, -7.84626885e+01, -1.66158330e+02]])

control_point_list = [control_points_1]

path_manager = SplinePathManager(control_point_list)

wing_sim = FixedWingPathFollowingSimulator(plane_model, autopilot, path_follower, path_manager)

vehicle_path_data, tracked_path_data, closest_distances_to_obstacles, closest_distances_to_sfc_walls \
     = wing_sim.run_simulation_dubins(path_position_data = path_location_arr, path_tangent_data = path_tangent_arr,
                               path_perpindicular_data = path_perpindicular_arr, 
                               path_curvature_data = path_curvature_arr, desired_speed = desired_airspeed, 
                               obstacle_list = obstacle_list, obstacle_type="cylinder", graphic_scale=20)

wing_sim.plot_simulation_analytics(vehicle_path_data, tracked_path_data,
                                    max_curvature,    max_incline_angle, closest_distances_to_obstacles)