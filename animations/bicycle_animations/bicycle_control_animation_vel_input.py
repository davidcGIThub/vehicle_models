#homework 7
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
from vehicle_simulator.vehicle_models.bicycle_model import BicycleModel
from vehicle_simulator.vehicle_controllers.bicycle_trajectory_tracker import BicycleTrajectoryTracker
from bsplinegenerator.bsplines import BsplineEvaluation
import os
import time
from time import sleep


# Trajectory
# control_points = np.array([[-0.78239366,  0.53552146,  1.95280528,  3.24396037,  3.98445455,  4.32363038, 5.09089489,  6.46946519,  7.98779535,  9.2222135 ],
#  [ 0.94721576,  1.17503746,  0.94370588,  1.56019985,  2.83357583,  5.06946717, 6.48835075,  7.13807965,  6.93096018,  7.13807965]])
# control_points = np.array([[-5.1092889,  -6.44535555, -5.1092889,  -1.64036059,  1.64045914,  5.10907334,
#    6.44546333,  5.10907334],
#  [-3.9985993,   0.45304904,  2.18640314,  0.8680498,  -0.86623853, -2.18542364,
#   -0.45353879,  3.99957881]])
# scale_factor = 1.2370004896215194
control_points = np.array([[-4.73449447, -6.63275277, -4.73449447, -1.24883457,  1.24861455,  4.73303911,
   6.63348044,  4.73303911],
 [-3.5377917,   0.2226169,   2.64732412,  1.37367557, -1.37128066, -2.64831555,
  -0.22212118,  3.53680028]])
scale_factor = 5
margin = 2
x_limits = np.array([np.min(control_points[0,:]) - margin, np.max(control_points[0,:]) + margin])
y_limits = np.array([np.min(control_points[1,:]) - margin, np.max(control_points[1,:]) + margin])
start_time = 0
bspline_gen = BsplineEvaluation(control_points, 3,start_time,scale_factor)

global path, velocity_data, acceleration_data, true_x_position, true_y_position, true_velocity, prev_states
num_data_points = 100
path, time_data = bspline_gen.get_spline_data(num_data_points)
velocity_data, time_data = bspline_gen.get_spline_derivative_data(num_data_points,1)
acceleration_data, time_data = bspline_gen.get_spline_derivative_data(num_data_points,2)
true_x_position = time_data*0
true_y_position = time_data*0
true_velocity = time_data*0

# spline_at_knot_points, knot_points = bspline_gen.get_spline_at_knot_points()
# bezier_control_points = bspline_gen.get_bezier_control_points()
start_direction = velocity_data[:,0]/np.linalg.norm(velocity_data[:,0],2,0)
start_point = path[:,0]


# Bicycle Model
dt = time_data[1]
L = 1
lr = 0.5
R = 0.2
v_max = 5
delta_max = np.pi/6
max_curvature = np.tan(delta_max)/L
print("max_curvature: " , max_curvature)
dir = np.arctan2(start_direction[1], start_direction[0])

bike = BicycleModel(x = start_point[0], 
                    y = start_point[1],
                    theta = dir,
                    delta = 0,
                    lr = lr,
                    L = L,
                    R = R,
                    alpha = np.array([0,0,0,0]),
                    # alpha = np.array([0.1,0.01,0.1,0.01]),
                    max_delta = delta_max,
                    max_vel = v_max)

## plotting

def plot_sfc(ax,sfc):
    ax.add_patch(Rectangle((sfc.min_bounds[0], sfc.min_bounds[1]),\
        sfc.max_bounds[0]-sfc.min_bounds[0], sfc.max_bounds[1]-sfc.min_bounds[1], edgecolor = 'tab:green', fill=False))

fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                     xlim=(x_limits[0],x_limits[1]), ylim=(y_limits[0],y_limits[1]))
ax.grid()
front_wheel_fig = plt.Polygon(bike.get_front_wheel_points(),fc = 'k')
back_wheel_fig = plt.Polygon(bike.get_back_wheel_points(),fc = 'k')
body_fig = plt.Polygon(bike.get_body_points(),fc = 'g')
desired_position_fig = plt.Circle((0, 0), radius=0.1, fc='r')

time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
ax.plot(path[0,:],path[1,:])



controller = BicycleTrajectoryTracker(k_pos = 3, 
                                      k_vel = 3,
                                      k_delta = 2,
                                      max_vel = v_max,
                                      max_delta = delta_max,
                                      lr = lr,
                                      L = L)

prev_states = bike.get_state()

def init():
    #initialize animation
    ax.add_patch(front_wheel_fig)
    ax.add_patch(back_wheel_fig)
    ax.add_patch(body_fig)
    ax.add_patch(desired_position_fig)
    time_text.set_text('')
    return front_wheel_fig, back_wheel_fig, body_fig, desired_position_fig, time_text

def animate(i):
    global bike, controller, path, velocity_data, acceleration_data, prev_states
    # propogate robot motion
    states = bike.get_state() 
    inputs = bike.get_inputs()
    true_x_position[i] = states[0,0]
    true_y_position[i] = states[0,1]
    t = time_data[i]
    position = path[:,i]
    velocity = velocity_data[:,i]

    acceleration = acceleration_data[:,i]
    des_states = np.array([[position[0], position[1]],
                          [velocity[0], velocity[1]],
                          [acceleration[0], acceleration[1]]])
    # v_c1, phi_c1 = controller.mpc_control_velocity_input(states, prev_states, des_states, dt)
    v_c1, phi_c1 = controller.mpc_control_velocity_input(inputs, states, des_states)
    v_hat, phi_hat = bike.update_velocity_motion_model(v_c1,phi_c1,dt)
    true_velocity[i] = v_hat
    front_wheel_fig.xy = bike.get_front_wheel_points()
    back_wheel_fig.xy = bike.get_back_wheel_points()
    body_fig.xy = bike.get_body_points()
    desired_position_fig.center = (position[0],position[1])
    prev_states = states
    # update time
    time_text.set_text('time = %.1f' % t)
    # sleep(0.1)

    return  front_wheel_fig, back_wheel_fig, body_fig,desired_position_fig, time_text

from time import time
animate(0)

ani = animation.FuncAnimation(fig, animate, frames = np.size(time_data), 
                            interval = dt, blit = True, init_func = init, repeat = False)

plt.show()

x_error = true_x_position - path[0,:]
y_error = true_y_position - path[1,:]
position_error = np.sqrt(x_error**2 + y_error**2)

# plt.figure()
# plt.plot(time_data, position_error)
# plt.title("Position Error")
# plt.show()

# velocity_error = true_velocity - np.linalg.norm(velocity_data,2,0)
# plt.figure()
# plt.plot(time_data, velocity_error)
# plt.title("Velocity Error")
# plt.show()

# file_name = os.getcwd() + "/bike_animation.gif"
# writergif = animation.PillowWriter(fps=30) 
# ani.save(file_name, writer=writergif)

# file_name = os.getcwd() + "/bike_animation.gif"
# ani.save(file_name, writer='imagemagick', fps=60)