#homework 7
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
from unicycle_model import UnicycleModel
from unicycle_kinematic_trajectory_tracker import UnicycleKinematicTrajectoryTracker
from bsplinegenerator.bsplines import BsplineEvaluation
import os

x_limits = 10
y_limits = 10
sec = 90
time_array = np.linspace(0,sec,int(sec/0.1+1))
dt = time_array[1]
v_max = 5

# Trajectory
control_points = np.array([[-0.78239366,  0.53552146,  1.95280528,  3.24396037,  3.98445455,  4.32363038, 5.09089489,  6.46946519,  7.98779535,  9.2222135 ],
 [ 0.94721576,  1.17503746,  0.94370588,  1.56019985,  2.83357583,  5.06946717, 6.48835075,  7.13807965,  6.93096018,  7.13807965]])+ 1
bspline_gen = BsplineEvaluation(control_points, 3,0,1)
global path, velocity_data, acceleration_data
num_data_points = 100
path, time_data = bspline_gen.get_spline_data(num_data_points)
velocity_data, time_data = bspline_gen.get_spline_derivative_data(num_data_points,1)
acceleration_data, time_data = bspline_gen.get_spline_derivative_data(num_data_points,2)


unicycle = UnicycleModel(x = 0, 
                         y = 0,
                         theta = np.pi/4,
                         alpha = np.array([0.1,0.01,0.01,0.1]),
                         dt = dt)
controller = UnicycleKinematicTrajectoryTracker(dt = dt,
                                                kp_p = 3, 
                                                kp_i = 0,
                                                kv_p = 1,
                                                ktheta_p = 1,
                                                v_max = 15,
                                                omega_max = np.pi/4,
                                                tolerance = 0.005)
fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                     xlim=(-x_limits,x_limits), ylim=(-y_limits,y_limits))
ax.grid()
robot_fig = plt.Polygon(unicycle.getPoints(),fc = 'g')
desired_position_fig = plt.Circle((0, 0), radius=0.1, fc='r')
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
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
    states_desired = np.array([position[0],position[1],None,velocity[0],velocity[1],None])
    states = unicycle.getState()
    if i > 0:
        states_array = np.vstack((states_array,states))
    v_c, omega_c = controller.trajectory_tracker(states, states_desired)
    input = np.array([v_c, omega_c])
    unicycle.velMotionModel(input)
    robot_fig.xy = unicycle.getPoints()
    # update time
    # time_text.set_text('time = %.1f' % time_array[i])
    time_text.set_text('omega_c = %.1f' % omega_c)
    plt.plot(path[:,0],path[:,1])
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

plt.figure(1)
plt.plot(time_array,states_array[:,0],label="robot state")
plt.plot(time_array,path[:,0],label="desired state")
plt.xlabel("Time")
plt.ylabel("X")
plt.legend()
plt.show()
