#homework 7
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
from boat_model import BoatModel
from boat_kinematic_controller import BoatKinematicController
from bsplinegenerator.bsplines import BsplineEvaluation
import os
import time

x_limits = 10
y_limits = 10
sec = 90

# Trajectory
control_points = np.array([[-0.78239366,  0.53552146,  1.95280528,  3.24396037,  3.98445455,  4.32363038, 5.09089489,  6.46946519,  7.98779535,  9.2222135 ],
 [ 0.94721576,  1.17503746,  0.94370588,  1.56019985,  2.83357583,  5.06946717, 6.48835075,  7.13807965,  6.93096018,  7.13807965]])+ 1
bspline_gen = BsplineEvaluation(control_points, 3,0,1)
global path, velocity_data, acceleration_data
num_data_points = 100
path, time_data = bspline_gen.get_spline_data(num_data_points)
velocity_data, time_data = bspline_gen.get_spline_derivative_data(num_data_points,1)
acceleration_data, time_data = bspline_gen.get_spline_derivative_data(num_data_points,2)

# spline_at_knot_points, knot_points = bspline_gen.get_spline_at_knot_points()
# bezier_control_points = bspline_gen.get_bezier_control_points()
start_direction = velocity_data[:,0]/np.linalg.norm(velocity_data[:,0],2,0)
start_point = path[:,0]


# Boat Model
dt = time_data[1]
delta_max = np.pi/4
dir = np.arctan2(start_direction[1], start_direction[0])

v_max = 1
boat = BoatModel(x = start_point[0], 
                 y = start_point[1], 
                 vel = 0.2,
                 theta = dir, 
                 delta = 0,
                 alpha = np.array([0.0,0.0,0.0,0.0]),
                 dt = 0.1,
                 height = 0.5,
                 width = 0.2,
                 delta_max = delta_max,
                 a_max = 5,
                 v_max = v_max
                 )

## plotting

fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                     xlim=(0,x_limits), ylim=(0,y_limits))
ax.grid()
boat_fig = plt.Polygon(boat.getBodyPoints(),fc = 'g')
rudder_fig = plt.Polygon(boat.getRudderPoints(),fc = 'k')
desired_position_fig = plt.Circle((0, 0), radius=0.1, fc='r')

time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
ax.plot(path[0,:],path[1,:])

controller = BoatKinematicController(kpos_xy = 1, 
                                    kvel_xy = 1,
                                    k_ang_theta = 0.5,
                                    k_ang_rate_theta = 1, 
                                    kpos_delta = 1,
                                    a_max = 5,
                                    v_max = v_max,
                                    delta_max = np.pi/2,
                                    v_coast=0.1)

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
    acceleration = acceleration_data[:,i]
    # x_des_states = np.array([position[0], velocity[0], acceleration[0]])
    # y_des_states = np.array([position[1], velocity[1], acceleration[1]])
    # v_c1, phi_c1 = controller.pd_control(states[0], states[1], states[2], states[3],x_des_states,y_des_states)
    # v_c1, phi_c1 = controller.p_control(states[0], states[1], states[2], states[3], position[0], position[1],.01)
    # self.x,self.y,self.theta, 
    #                      self.x_dot, self.y_dot
    # self, x, y, x_dot, theta, y_dot, x_des, y_des
    accel_c, delta_c = controller.pos_control(states[0], states[1], states[2], states[3],states[4], position[0], position[1], tolerance = 0.1)
    # input = np.array([v_c[i], phi_c[i]])
    input = np.array([accel_c, delta_c])
    boat.velMotionModel(input)
    boat_fig.xy = boat.getBodyPoints()
    rudder_fig.xy = boat.getRudderPoints()
    desired_position_fig.center = (position[0],position[1])
    plt.plot(path[:,0],path[:,1])
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