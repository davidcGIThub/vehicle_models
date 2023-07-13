
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
from vehicle_simulator.vehicle_controllers.bspline_evaluator import BsplineEvaluator
from vehicle_simulator.vehicle_simulators.fixed_wing_path_follower_simulator import FixedWingPathFollowingSimulator
from time import sleep


order = 3
control_points = np.array([[  5.12944029,  -2.56472015,   5.12944029,  33.72586905,  68.41308514,
   94.37956304, 102.81021848,  94.37956304],
   [100.25649204,  99.87175398, 100.25649204, 101.68637201, 103.4207176,
  104.71897844, 105.14051078, 104.71897844],
   [-18.26855358,   0.70966115,  15.429909,    22.01121802,  21.01919592,
   13.01947556,   0.26964944, -14.09807331]])*np.array([[1/13],[1/10],[1/7]]) + np.array([[-1.4],[-10],[-3]])
bspline_eval = BsplineEvaluator(order)
position_array = bspline_eval.matrix_bspline_evaluation_for_dataset(control_points, order, 1000)
waypoints = position_array[:,np.array([0,500,1000])]
sfc_points = np.array([[ 8,  -2,  -2,   8,   8,  -2,  -2,   8,   8,   8,   8,   8,  -2,  -2, -2,  -2],
 [-1,  -1,   1,   1,   1,   1,  -1,  -1,  -1,   1,   1,  -1,  -1,  -1, 1,   1. ],
 [-1.5, -1.5, -1.5, -1.5,  1.5,  1.5,  1.5,  1.5, -1.5, -1.5,  1.5,  1.5,  1.5, -1.5, -1.5,  1.5]])*1
theta = 90*np.pi/180
R = np.array([[np.cos(theta), 0, np.sin(theta)],
              [0,           1,               0],
              [-np.sin(theta),0, np.cos(theta)]])
sfc_points_2 = np.dot(R,sfc_points)
sfc_list = [sfc_points, sfc_points_2]

obstacle_1 = np.array([2,4,1,4])
obstacle_2 = np.array([8,1,-3,2])
obstacle_list = [obstacle_1, obstacle_2]

fixed_wing_parameters = FixedWingParameters()
control_parameters = FixedWingControlParameters()
# Attaching 3D axis to the figure
fig_2 = plt.figure()
ax = plt.axes(projection='3d')
plane_model = FixedWingModel(ax)
autopilot = FixedWingAutopilot(control_parameters)
path_follower = FixedWingSplinePathFollower(order, distance_gain=5)

plane_sim = FixedWingPathFollowingSimulator(plane_model, autopilot, path_follower)
desired_speed = 30
dt = 0.01
run_time = 20
sleep_time=0
animate = True
plot= False
intervals_per_sfc = [2,3]
plane_sim.run_simulation(control_points, desired_speed, 
                       obstacle_list, sfc_list,
                       intervals_per_sfc, dt, run_time,
                       animate, plot, sleep_time)