#homework 7
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
from vehicle_simulator.vehicle_models.unicycle_model import UnicycleModel
from vehicle_simulator.vehicle_controllers.unicycle_trajectory_tracker import UnicycleTrajectoryTracker
from vehicle_simulator.vehicle_simulators.vehicle_trajectory_tracking_simulator import VehicleTrajectoryTrackingSimulator, TrajectoryData
from bsplinegenerator.bsplines import BsplineEvaluation
import os
import time
from time import sleep

control_points = np.array([[-5.1092889,  -6.44535555, -5.1092889,  -1.64036059,  1.64045914,  5.10907334,
   6.44546333,  5.10907334],
 [-3.9985993,   0.45304904,  2.18640314,  0.8680498,  -0.86623853, -2.18542364,
  -0.45353879,  3.99957881]])
scale_factor = 0.3

# control_points = np.array([[-4.73449447, -6.63275277, -4.73449447, -1.24883457,  1.24861455,  4.73303911,
#    6.63348044,  4.73303911],
#  [-3.5377917,   0.2226169,   2.64732412,  1.37367557, -1.37128066, -2.64831555,
#   -0.22212118,  3.53680028]])
# scale_factor = 1.2370231646042702

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
max_vel = 5
max_vel_dot = 5
max_theta_dot = 5

unicycle = UnicycleModel(x = location_data[0,0], 
                         y = location_data[1,0],
                         theta = np.arctan2(velocity_data[1,0],velocity_data[0,0]),
                        #  theta = -np.pi/2,
                         x_dot = velocity_data[0,0],
                         y_dot = velocity_data[1,0],
                        #  alpha = np.array([0.1,0.01,0.01,0.1]),
                         alpha = np.array([0,0,0,0]),
                         max_vel = max_vel,
                         max_vel_dot = max_vel_dot,
                         max_theta_dot = max_theta_dot)

controller = UnicycleTrajectoryTracker(k_pos = 10, 
                                       k_vel = 10,
                                       k_theta = 10,
                                       location_fwd_tol = 2,
                                       heading_ffwd_tol = 0.3,
                                       max_vel = max_vel,
                                       max_vel_dot = max_vel_dot,
                                       max_theta_dot = max_theta_dot)

unicycle_traj_sim = VehicleTrajectoryTrackingSimulator(unicycle, controller)
des_traj_data = TrajectoryData(location_data, velocity_data, acceleration_data, 
                           jerk_data, time_data)
vehicle_traj_data = unicycle_traj_sim.run_simulation(des_traj_data)
unicycle_traj_sim.plot_simulation_dynamics(des_traj_data, vehicle_traj_data, max_vel, 
                                       max_vel_dot, max_theta_dot, "angular_rate")