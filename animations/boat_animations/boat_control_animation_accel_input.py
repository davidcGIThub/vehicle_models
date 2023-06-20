#homework 7
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
from vehicle_simulator.vehicle_models.boat_model import BoatModel
from vehicle_simulator.vehicle_controllers.boat_trajectory_tracker import BoatTrajectoryTracker
from vehicle_simulator.vehicle_simulators.vehicle_trajectory_tracking_simulator import VehicleTrajectoryTrackingSimulator, TrajectoryData
from bsplinegenerator.bsplines import BsplineEvaluation
import os
import time
from time import sleep

control_points = np.array([[-5.1092889,  -6.44535555, -5.1092889,  -1.64036059,  1.64045914,  5.10907334,
   6.44546333,  5.10907334],
 [-3.9985993,   0.45304904,  2.18640314,  0.8680498,  -0.86623853, -2.18542364,
  -0.45353879,  3.99957881]])
scale_factor = 1

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

# Boat Model
dt = time_data[1]
c_r = 50
c_b = 0.01
max_vel = 10
max_vel_dot = 20
max_delta = np.pi/3
max_delta_dot = 10
max_centripetal_acceleration = c_r*np.sin(max_delta)*np.pi/2
print("max cent accel: " , max_centripetal_acceleration)

start_direction = velocity_data[:,0]/np.linalg.norm(velocity_data[:,0],2,0)
start_point = location_data[:,0]
start_vel = velocity_data[:,0]
start_heading = np.arctan2(start_direction[1], start_direction[0])
start_accel = acceleration_data[:,0]
x_dot_s = start_vel[0]
y_dot_s = start_vel[1]
x_ddot_s = start_accel[0]
y_ddot_s = start_accel[1]
vel_start_mag = np.linalg.norm(start_vel)
theta_dot_start = (x_dot_s*y_ddot_s - y_dot_s*x_ddot_s)/(x_dot_s**2 + y_dot_s**2)
delta_start = -np.arcsin(np.clip(theta_dot_start*(vel_start_mag+c_b)/(c_r*np.arctan2(vel_start_mag**2,1)), -1, 1))
dir_angle = np.arctan2(start_direction[1], start_direction[0])

boat = BoatModel(x = start_point[0], 
                 y = start_point[1], 
                 theta = dir_angle, 
                #  theta = -np.pi/2,
                #  delta = 0,
                 delta = delta_start,
                #  x_dot = 0,
                #  y_dot = 0,
                 x_dot = start_vel[0],
                 y_dot = start_vel[1],
                 x_ddot = start_accel[0],
                 y_ddot = start_accel[1],
                 alpha = np.array([0,0,0,0]),
                #  alpha = np.array([0.1,0.01,0.01,0.1]),
                 height = 0.8,
                 width = 0.4,
                 c_r = c_r, #rudder constant
                 c_b = c_b, #boat constant
                 max_delta = max_delta,
                 max_delta_dot = max_delta_dot,
                 max_vel = max_vel,
                 max_vel_dot = max_vel_dot)

controller = BoatTrajectoryTracker(c_r = c_r,
                                    c_b = c_b,
                                    k_pos = 10, 
                                    k_vel = 10,
                                    k_theta = 50,
                                    k_delta = 50,
                                    max_vel = max_vel,
                                    max_vel_dot = max_vel_dot,
                                    max_delta = max_delta,
                                    turn_vel = 0.5,
                                    location_fwd_tol = 2,
                                    heading_ffwd_tol = 0.3)

bike_traj_sim = VehicleTrajectoryTrackingSimulator(boat, controller)
des_traj_data = TrajectoryData(location_data, velocity_data, acceleration_data, 
                           jerk_data, time_data)
# vehicle_traj_data = bike_traj_sim.run_simulation_real_time(des_traj_data, sleep_time=0.1,margins=5)
vehicle_traj_data, vehicle_motion_data = bike_traj_sim.run_simulation(des_traj_data, sleep_time=0)
bike_traj_sim.plot_simulation_dynamics(vehicle_motion_data, des_traj_data, vehicle_traj_data, max_vel, 
                                       max_vel_dot, max_centripetal_acceleration, "centripetal_acceleration", "boat")