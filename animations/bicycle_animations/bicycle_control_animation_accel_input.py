#homework 7
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
from vehicle_simulator.vehicle_models.bicycle_model import BicycleModel
from vehicle_simulator.vehicle_controllers.bicycle_trajectory_tracker import BicycleTrajectoryTracker
from vehicle_simulator.vehicle_simulators.vehicle_trajectory_tracking_simulator import VehicleTrajectoryTrackingSimulator, TrajectoryData
from bsplinegenerator.bsplines import BsplineEvaluation
import os
import time
from time import sleep

# control_points = np.array([[-5.1092889,  -6.44535555, -5.1092889,  -1.64036059,  1.64045914,  5.10907334,
#    6.44546333,  5.10907334],
#  [-3.9985993,   0.45304904,  2.18640314,  0.8680498,  -0.86623853, -2.18542364,
#   -0.45353879,  3.99957881]])
# scale_factor = 0.3

control_points = np.array([[-4.73449447, -6.63275277, -4.73449447, -1.24883457,  1.24861455,  4.73303911,
   6.63348044,  4.73303911],
 [-3.5377917,   0.2226169,   2.64732412,  1.37367557, -1.37128066, -2.64831555,
  -0.22212118,  3.53680028]])
# scale_factor = 1.2370231646042702
scale_factor = 1

sec = 90
start_time = 0
bspline_gen = BsplineEvaluation(control_points, 3,start_time,scale_factor)
num_data_points = 100
location_data, time_data = bspline_gen.get_spline_data(num_data_points)
velocity_data, time_data = bspline_gen.get_spline_derivative_data(num_data_points,1)
acceleration_data, time_data = bspline_gen.get_spline_derivative_data(num_data_points,2)
jerk_data, time_data = bspline_gen.get_spline_derivative_data(num_data_points,3)
curvature_data, time_data = bspline_gen.get_spline_curvature_data(num_data_points)
angular_rate_data, time_data = bspline_gen.get_angular_rate_data(num_data_points)
centripetal_acceleration_data, time_data = bspline_gen.get_centripetal_acceleration_data(num_data_points)
longitudinal_acceleration_data, time_data = bspline_gen.get_longitudinal_acceleration_data(num_data_points)

start_direction = velocity_data[:,0]/np.linalg.norm(velocity_data[:,0],2,0)
start_point = location_data[:,0]
start_vel = velocity_data[:,0]
start_heading = np.arctan2(start_direction[1], start_direction[0])

# Bicycle Model
dt = time_data[1]
L = 1
l_r = 0.5
R = 0.2
max_vel = 10
max_delta = np.pi/4
max_beta = np.arctan2(l_r*np.tan(max_delta), L)
print("max_beta: " , max_beta)
max_curvature = np.tan(max_delta)*np.cos(max_beta)/L
print("max_curvature: " , max_curvature)
max_vel_dot = 10

bike = BicycleModel(x = start_point[0], 
                    y = start_point[1],
                    # theta = start_heading,
                    theta = 0,
                    delta = 0,
                    x_dot = start_vel[0], 
                    y_dot = start_vel[1], 
                    theta_dot = 0, 
                    delta_dot = 0,
                    lr = l_r,
                    L = L,
                    R = R,
                    alpha = np.array([0.1,0.01,0.1,0.01]),
                    # alpha = np.array([0,0,0,0]),
                    max_delta = max_delta,
                    max_vel = max_vel,
                    max_vel_dot = max_vel_dot)
controller = BicycleTrajectoryTracker(k_pos = 5, 
                                        k_vel = 5,
                                        k_delta = 5,
                                        max_vel_dot = max_vel_dot,
                                        max_vel = max_vel,
                                        max_delta = max_delta,
                                        lr = l_r,
                                        L = L)
bike_traj_sim = VehicleTrajectoryTrackingSimulator(bike, controller)
des_traj_data = TrajectoryData(location_data, velocity_data, acceleration_data, 
                           jerk_data, time_data)
vehicle_traj_data, vehicle_motion_data = bike_traj_sim.run_simulation(des_traj_data)
bike_traj_sim.plot_simulation_dynamics(vehicle_motion_data, des_traj_data, vehicle_traj_data, max_vel,
                                       max_vel_dot, max_curvature, "curvature", "bike")