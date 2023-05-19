#homework 7
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
from unicycle_model import UnicycleModel
from unicycle_trajectory_tracker import UnicycleTrajectoryTracker
from bsplinegenerator.bsplines import BsplineEvaluation
import os

x_limits = 10
y_limits = 10

v_max = 5

# Trajectory
control_points = np.array([[-0.78239366,  0.53552146,  1.95280528,  3.24396037,  3.98445455,  4.32363038, 5.09089489,  6.46946519,  7.98779535,  9.2222135 ],
 [ 0.94721576,  1.17503746,  0.94370588,  1.56019985,  2.83357583,  5.06946717, 6.48835075,  7.13807965,  6.93096018,  7.13807965]])+ 1
scale_factor = 0.2
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
time_array = np.linspace(0,end_time,num_data_points*num_intervals)
dt = time_array[1]


unicycle = UnicycleModel(x = path[0,0], 
                         y = path[1,0],
                         theta = np.arctan2(velocity_data[1,0],velocity_data[0,0]),
                         alpha = np.array([0.1,0.01,0.01,0.1]),
                         max_vel = 5)
controller = UnicycleTrajectoryTracker(k_pos = 3, 
                                       k_vel = 2,
                                       k_accel = 1,
                                       k_theta = 2,
                                       k_theta_dot = 1, 
                                       max_vel = 5,
                                       max_vel_dot = 5,
                                       max_theta_dot = 3)
fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                     xlim=(-x_limits,x_limits), ylim=(-y_limits,y_limits))
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
    t = time_array[i]
    position = path[:,i]
    velocity = velocity_data[:,i]
    acceleration = acceleration_data[:,i]
    jerk = jerk_data[:,i]
    desired_trajectory = np.vstack((position, velocity, acceleration, jerk))
    states = unicycle.getState()
    if i > 0:
        states_array = np.vstack((states_array,states))
    v_c, omega_dot_c = controller.mpc_control_vel_input(states, desired_trajectory)
    unicycle.update_velocity_motion_model(v_c, omega_dot_c, dt)
    robot_fig.xy = unicycle.getPoints()
    # update time
    # time_text.set_text('time = %.1f' % time_array[i])
    time_text.set_text('omega_dot_c = %.1f' % omega_dot_c)
    desired_position_fig.center = (position[0],position[1])
    return robot_fig,desired_position_fig, time_text

from time import time
animate(0)

ani = animation.FuncAnimation(fig, animate, frames = np.size(time_array), 
                            interval = dt*100, blit = True, init_func = init, repeat = False)
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
