#homework 7
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
from vehicle_simulator.vehicle_models.unicycle_model import UnicycleModel
from vehicle_simulator.vehicle_controllers.unicycle_trajectory_tracker import UnicycleTrajectoryTracker
from bsplinegenerator.bsplines import BsplineEvaluation
import os
from time import sleep


# Trajectory
control_points = np.array([[-6.99995484, -7.00002258, -6.99995484, -6.99994635, -6.99996772, -6.99999218, -7.00000391 -6.99999218],
                             [-6.75065227, -7.12821241, -6.73649808, -5.64749128, -4.03283927, -2.09741205, -0.01285767,  2.14884274]])
scale_factor =  0.7077091316298347

# control_points = np.array([[-4.73449447, -6.63275277, -4.73449447, -1.24883457,  1.24861455,  4.73303911,
#    6.63348044,  4.73303911],
#  [-3.5377917,   0.2226169,   2.64732412,  1.37367557, -1.37128066, -2.64831555,
#   -0.22212118,  3.53680028]])
# scale_factor = 1.2370231646042702
# scale_factor = 0.5

x_limits = [np.min(control_points[0,:])-5, np.max(control_points[0,:])+5]
y_limits = [np.min(control_points[1,:])-5, np.max(control_points[1,:])+5]

bspline_gen = BsplineEvaluation(control_points, 3, 0, scale_factor= scale_factor)

global path, velocity_data, acceleration_data
num_data_points = 100
path, time_data = bspline_gen.get_spline_data(num_data_points)
velocity_data, time_data = bspline_gen.get_spline_derivative_data(num_data_points,1)
acceleration_data, time_data = bspline_gen.get_spline_derivative_data(num_data_points,2)
jerk_data, time_data = bspline_gen.get_spline_derivative_data(num_data_points, 3)
end_time = bspline_gen.get_end_time()
num_intervals = bspline_gen.get_num_intervals()
print("end time: " , end_time)
# time_array = np.linspace(0,end_time,num_data_points*num_intervals)
time_array = time_data
dt = time_array[1]


max_vel = 10
max_vel_dot = 5
max_theta_dot = 5
unicycle = UnicycleModel(x = path[0,0], 
                         y = path[1,0],
                         theta = np.arctan2(velocity_data[1,0],velocity_data[0,0]),
                        #  theta = -np.pi/2,
                         x_dot = velocity_data[0,0],
                         y_dot = velocity_data[1,0],
                        #  alpha = np.array([0.1,0.01,0.01,0.1]),
                         alpha = np.array([0,0,0,0]),
                         max_vel = max_vel,
                         max_theta_dot = max_theta_dot)
controller = UnicycleTrajectoryTracker(k_pos = 5, 
                                       k_vel = 8,
                                       k_theta = 8,
                                       k_theta_dot = 5, 
                                       location_fwd_tol = 2,
                                       heading_ffwd_tol = 0.3,
                                       max_vel = max_vel,
                                       max_vel_dot = max_vel_dot,
                                       max_theta_dot = max_theta_dot)
fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                     xlim=(x_limits), ylim=(y_limits))
ax.grid()
robot_fig = plt.Polygon(unicycle.getPoints(),fc = 'g')
desired_position_fig = plt.Circle((0, 0), radius=0.1, fc='r')
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
ax.plot(path[0,:],path[1,:])
states_array = unicycle.getState()

def init():
    #initialize animation
    ax.add_patch(robot_fig)
    ax.add_patch(desired_position_fig)
    time_text.set_text('')
    return robot_fig,desired_position_fig, time_text

def animate(i):
    global unicycle, controller, traj_gen, states_array
    # propogate robot motion
    # x_d = 10
    # y_d = 10
    # states_desired = np.array([x_d,y_d])
    # sleep(0.02)
    t = time_array[i]
    position = path[:,i]
    velocity = velocity_data[:,i]
    acceleration = acceleration_data[:,i]
    jerk = jerk_data[:,i]
    desired_trajectory = np.vstack((position, velocity, acceleration, jerk))
    states = unicycle.getState()
    # if i > 0:
    #     states_array = np.vstack((states_array,states))
    a_c, omega_dot = controller.mpc_control_accel_input(states, desired_trajectory)
    unicycle.update_acceleration_motion_model(a_c, omega_dot, dt)
    robot_fig.xy = unicycle.getPoints()
    # update time
    # time_text.set_text('time = %.1f' % time_array[i])
    # time_text.set_text('omega_dot_c = %.1f' % omega_dot_c)
    desired_position_fig.center = (position[0],position[1])
    return robot_fig,desired_position_fig, time_text

from time import time
animate(0)

ani = animation.FuncAnimation(fig, animate, frames = np.size(time_array), 
                            interval = dt, blit = True, init_func = init, repeat = False)
plt.show()
# plt.show(block=False)
# plt.pause(5)

file_name = os.getcwd() + "/unicycle_animation.gif"
writergif = animation.PillowWriter(fps=30) 
ani.save(file_name, writer=writergif)

# file_name = os.getcwd() + "/unicycle_animation.gif"
# ani.save(file_name, writer='imagemagick', fps=60)

# plt.figure(1)
# plt.plot(time_array,states_array[:,0],label="robot state")
# plt.plot(time_array,path[:,0],label="desired state")
# plt.xlabel("Time")
# plt.ylabel("X")
# plt.legend()
# plt.show()