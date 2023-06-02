#homework 7
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
from boat_model import BoatModel
from boat_trajectory_tracker import BoatTrajectoryTracker
from bsplinegenerator.bsplines import BsplineEvaluation
import os
from time import sleep

x_limits = 10
y_limits = 10
sec = 90

# Trajectory
control_points = np.array([[-4.73449447, -6.63275277, -4.73449447, -1.24883457,  1.24861455,  4.73303911,
   6.63348044,  4.73303911],
 [-3.5377917,   0.2226169,   2.64732412,  1.37367557, -1.37128066, -2.64831555,
  -0.22212118,  3.53680028]])
scale_factor = 1.2370231646042702
bspline_gen = BsplineEvaluation(control_points, 3,0,scale_factor)

x_limits = [np.min(control_points[0,:])-5, np.max(control_points[0,:])+5]
y_limits = [np.min(control_points[1,:])-5, np.max(control_points[1,:])+5]

global path, velocity_data, acceleration_data
num_data_points = 100
path, time_data = bspline_gen.get_spline_data(num_data_points)
velocity_data, time_data = bspline_gen.get_spline_derivative_data(num_data_points,1)
acceleration_data, time_data = bspline_gen.get_spline_derivative_data(num_data_points,2)
jerk_data, time_data = bspline_gen.get_spline_derivative_data(num_data_points, 3)

# spline_at_knot_points, knot_points = bspline_gen.get_spline_at_knot_points()
# bezier_control_points = bspline_gen.get_bezier_control_points()
start_direction = velocity_data[:,0]/np.linalg.norm(velocity_data[:,0],2,0)
start_point = path[:,0]
start_vel = velocity_data[:,0]


# Boat Model
dt = time_data[1]
delta_max = np.pi/4
dir_angle = np.arctan2(start_direction[1], start_direction[0])

c_r = 100
c_b = 0.01
max_vel = 10
max_vel_dot = 3
max_delta = np.pi/2
max_delta_dot = 20

boat = BoatModel(x = start_point[0], 
                 y = start_point[1], 
                 theta = dir_angle, 
                 delta = 0,
                 x_dot = start_vel[0],
                 y_dot = start_vel[1],
                 alpha = np.array([0.1,0.01,0.01,0.1]),
                 height = 1,
                 width = 0.5,
                 c_r = c_r, #rudder constant
                 c_b = c_b, #boat constant
                 max_delta = max_delta,
                 max_delta_dot = max_delta_dot,
                 max_vel = max_vel,
                 max_vel_dot = max_vel_dot)

## plotting

fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                     xlim=(x_limits), ylim=(y_limits))
ax.grid()
boat_fig = plt.Polygon(boat.getBodyPoints(),fc = 'g')
rudder_fig = plt.Polygon(boat.getRudderPoints(),fc = 'k')
desired_position_fig = plt.Circle((0, 0), radius=0.1, fc='r')

time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
ax.plot(path[0,:],path[1,:])

controller = BoatTrajectoryTracker(c_r = 2/np.pi,
                                    c_b = 0.01,
                                    k_pos = 6, 
                                    k_vel = 10,
                                    k_theta = 3,
                                    k_delta = 2,
                                    max_vel = max_vel,
                                    max_vel_dot = max_vel_dot,
                                    max_delta = max_delta,
                                    max_delta_dot = max_delta_dot,
                                    turn_vel = 0.5,
                                    location_fwd_tol = 2,
                                    heading_ffwd_tol = 0.3)

def init():
    #initialize animation
    ax.add_patch(boat_fig)
    ax.add_patch(rudder_fig)
    ax.add_patch(desired_position_fig)
    time_text.set_text('')
    return boat_fig, rudder_fig, desired_position_fig, time_text

def animate(i):
    global boat, controller, path, velocity_data, acceleration_data
    # propogate robot motion
    states = boat.getState() 
    t = time_data[i]
    position = path[:,i]
    velocity = velocity_data[:,i]
    print(np.sqrt(states[1,0]**2 + states[1,1]**2))
    sleep(0.01)
    acceleration = acceleration_data[:,i]
    position = path[:,i]
    velocity = velocity_data[:,i]
    acceleration = acceleration_data[:,i]
    jerk = jerk_data[:,i]
    desired_trajectory = np.vstack((position, velocity, acceleration, jerk))
    vel_c, delta_dot_c = controller.mpc_control_vel_input(states, desired_trajectory)
    boat.update_velocity_motion_model(vel_c, delta_dot_c, dt)
    # sleep(0.01)
    boat_fig.xy = boat.getBodyPoints()
    rudder_fig.xy = boat.getRudderPoints()
    desired_position_fig.center = (position[0],position[1])
    # update time
    time_text.set_text('time = %.1f' % t)

    return  boat_fig, rudder_fig,desired_position_fig, time_text

from time import time
animate(0)

ani = animation.FuncAnimation(fig, animate, frames = np.size(time_data), 
                            interval = dt*100, blit = True, init_func = init, repeat = False)

plt.show()

# file_name = os.getcwd() + "/bike_animation.gif"
# writergif = animation.PillowWriter(fps=30) 
# ani.save(file_name, writer=writergif)

# file_name = os.getcwd() + "/bike_animation.gif"
# ani.save(file_name, writer='imagemagick', fps=60)