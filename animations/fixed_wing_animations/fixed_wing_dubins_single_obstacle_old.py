
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

def quaternion_to_rotation(quaternion):
    """
    converts a quaternion attitude to a rotation matrix
    """
    e0 = quaternion.item(0)
    e1 = quaternion.item(1)
    e2 = quaternion.item(2)
    e3 = quaternion.item(3)

    R = np.array([[e1 ** 2.0 + e0 ** 2.0 - e2 ** 2.0 - e3 ** 2.0, 2.0 * (e1 * e2 - e3 * e0), 2.0 * (e1 * e3 + e2 * e0)],
                  [2.0 * (e1 * e2 + e3 * e0), e2 ** 2.0 + e0 ** 2.0 - e1 ** 2.0 - e3 ** 2.0, 2.0 * (e2 * e3 - e1 * e0)],
                  [2.0 * (e1 * e3 - e2 * e0), 2.0 * (e2 * e3 + e1 * e0), e3 ** 2.0 + e0 ** 2.0 - e1 ** 2.0 - e2 ** 2.0]])
    R = R/np.linalg.det(R)

    return R

from time import sleep
states_arr = np.load('vehicle_states.npy')
path_curvature_arr = np.load('path_curvature.npy')
path_incline_arr = np.load('path_incline.npy')
path_location_arr = np.load('path_location.npy').transpose()
path_perpindicular_arr = np.load('path_perpindicular.npy')
path_tangent_arr = np.load('path_tangent_arr.npy')
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

for i in range(num_states):
    # vehicle states
    state = states_arr[i,:]
    states_list.append(state)
    vehicle_location_data[:,i][:,None] = np.array([[state.item(0)],
                                           [state.item(1)],
                                           [state.item(2)]])
    R = quaternion_to_rotation(state[6:10]) # rotation from body to world frame
    velocity = R @ np.array([[state[3]],[state[4]],[state[5]]])
    if i == num_states - 1:
        vehicle_curvature_data[i] = vehicle_curvature_data[i-1]
    else:
        next_state = states_arr[i+1,:]
        next_velocity = quaternion_to_rotation(next_state[6:10]) @ np.array([[next_state[3]],[next_state[4]],[next_state[5]]])
        acceleration = (next_velocity - velocity) / (time_arr.item(i+1)-time_arr.item(i))
        vehicle_curvature_data[i] = np.linalg.norm(np.cross(velocity.flatten(),acceleration.flatten()))/np.linalg.norm(velocity)**3
    vehicle_rise = velocity.item(2)
    vehicle_run = np.sqrt(velocity.item(0)**2 + velocity.item(1)**2)
    vehicle_incline_data[i] = -np.arctan(vehicle_rise/vehicle_run)
for i in range(num_states):
    #closest path data
    distances = np.linalg.norm(vehicle_location_data[:,i][:,None] - path_location_arr , 2 , 0)
    index = np.argmin(distances)
    closest_location_data[:,i] = path_location_arr[:,index]
    closest_curvature_data[i] = path_curvature_arr[index]
    closest_incline_data[i] = path_incline_arr[index]

closest_time_data = time_arr
vehicle_path_data = PathData(vehicle_location_data, vehicle_curvature_data.flatten(), vehicle_incline_data.flatten(), time_arr.flatten())
path_time = np.linspace(0,time_arr[-1],len(time_arr.flatten()))
# path_data = PathData(path_location_arr, path_curvature_arr, path_incline_arr, path_time)
path_data_list = [path_location_arr]
closest_path_data = PathData(closest_location_data, closest_curvature_data, closest_incline_data, closest_time_data)
obstacle_type = "cylinder" 
if obstacle_data.item(0) == 1:
    obstacle_type = "rect_prism"
obstacle = Obstacle(center=np.array([obstacle_data.item(1) ,obstacle_data.item(2), 0]),radius= obstacle_data.item(3)/2 ,height= obstacle_data.item(4))
obstacle_list = [obstacle]
closest_distance_to_obstacle = np.min(np.linalg.norm(vehicle_location_data[0:2,:] - np.array([[obstacle_data.item(1)] ,[obstacle_data.item(2)]]),2,0)) - obstacle_data.item(3)/2
    # def set_state(self,state):
    #     self._north = state.item(0)  # initial north position
    #     self._east = state.item(1)  # initial east position
    #     self._down = state.item(2)  # initial down position
    #     self._u = state.item(3)  # initial velocity along body x-axis
    #     self._v = state.item(4)  # initial velocity along body y-axis
    #     self._w = state.item(5)  # initial velocity along body z-axis
    #     self._e0 = state.item(6)
    #     self._e1 = state.item(7)
    #     self._e2 = state.item(8)
    #     self._e3 = state.item(9)
    #     self._p = state.item(10)  # initial roll rate
    #     self._q = state.item(11)  # initial pitch rate
    #     self._r = state.item(12)


order = 3
desired_speed = parameters.item(0)

# obstacle_1 = Obstacle(center=np.array([[25,],[25],[-20]]), radius = 20)
# obstacle_2 = Obstacle(center=np.array([[100,],[25],[-20]]), radius = 20)
# obstacle_list = [obstacle_1, obstacle_2]

max_curvature = parameters.item(1)
max_incline = np.arctan(parameters.item(2))
fixed_wing_parameters = FixedWingParameters()
control_parameters = FixedWingControlParameters()
# Attaching 3D axis to the figure
fig = plt.figure()
ax = plt.axes(projection='3d')
fig.add_axes(ax)
north = 0
east = 0
down = -20

u = 17
v = 10.535653752852738
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
control_points_1 = np.array([[-1.92718378e+02, -2.12381699e+00,  2.01213646e+02 , 4.35400071e+02],
                             [-2.67206060e+01,  1.33603030e+01, -2.67206060e+01,  3.52323025e+01],
                             [-7.84626885e+01,  9.23134424e+00, -7.84626885e+01, -1.66158330e+02]])
control_point_list = [control_points_1]
path_manager = SplinePathManager(control_point_list)

wing_sim = FixedWingPathFollowingSimulator(plane_model, autopilot, path_follower, path_manager)

wing_sim.run_simulation_dubins(path_position_data, path_tangent_data, path_perpindicular_data, 
                               path_curvature_data, desired_speed, obstacle_list)

# wing_sim.plot_simulation(states_list, vehicle_path_data, path_data_list, closest_path_data, obstacle_list=obstacle_list, obstacle_type=obstacle_type)
# wing_sim.plot_simulation_analytics(vehicle_path_data, closest_path_data, max_curvature, max_incline,closest_distances_to_obstacles=np.array([closest_distance_to_obstacle]))
