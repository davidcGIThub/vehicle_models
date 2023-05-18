#homework 7
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
from vehicle_simulator.vehicle_models.bicycle_model import BicycleModel
import os

x_limits = 5
y_limits = 5
sec = 45
resolution = 1000
time_array = np.linspace(0,sec,resolution)
dt = time_array[1]
L = 1
lr = 0.5
R = 0.2
v_max = 5
delta_max = np.pi/4
bike = BicycleModel(x = 0, 
                    y = 0,
                    theta = np.pi/4,
                    delta = 0,
                    x_dot = 0, 
                    y_dot = 0, 
                    theta_dot = 0, 
                    delta_dot = 0,
                    lr = lr,
                    L = L,
                    R = R,
                    alpha = np.array([0.1,0.01,0.1,0.01]),
                    delta_max = delta_max,
                    vel_max=5)



fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                     xlim=(-x_limits,x_limits), ylim=(-y_limits,y_limits))
ax.grid()
front_wheel_fig = plt.Polygon(bike.getFrontWheelPoints(),fc = 'k')
back_wheel_fig = plt.Polygon(bike.getBackWheelPoints(),fc = 'k')
body_fig = plt.Polygon(bike.getBodyPoints(),fc = 'g')

time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

global a_c, delta_dot_c, a_x_array, a_y_array, v_x_array, v_y_array 
a_c = (0.5*np.cos(.5*time_array))/3
# a_c = (0.5 + 0.5*np.cos(.5*time_array))/3
a_x_array = 0*time_array
a_y_array = 0*time_array
v_x_array = 0*time_array
v_y_array = 0*time_array
delta_dot_c = np.cos(0.1*time_array)

def init():
    #initialize animation
    ax.add_patch(front_wheel_fig)
    ax.add_patch(back_wheel_fig)
    ax.add_patch(body_fig)
    time_text.set_text('')
    return front_wheel_fig, back_wheel_fig, body_fig, time_text

def animate(i):
    global bike, controller, traj_gen
    # propogate robot motion
    # x_d = 5
    # y_d = 5
    states = bike.getState() 
    a_x_array[i] = states[2,0]
    a_y_array[i] = states[2,1]
    v_x_array[i] = states[1,0]
    v_y_array[i] = states[1,1]
    t = time_array[i]
    bike.update_acceleration_motion_model(a_c[i],delta_dot_c[i],dt)
    # bike.update_velocity_motion_model(a_c[i],delta_dot_c[i],dt)
    front_wheel_fig.xy = bike.getFrontWheelPoints()
    back_wheel_fig.xy = bike.getBackWheelPoints()
    body_fig.xy = bike.getBodyPoints()
    
    # update time
    time_text.set_text('time = %.1f' % time_array[i])

    return  front_wheel_fig, back_wheel_fig, body_fig, time_text

from time import time
animate(0)

ani = animation.FuncAnimation(fig, animate, frames = np.size(time_array), 
                            interval = dt*100, blit = True, init_func = init, repeat = False)

plt.show()

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
ax1.plot(time_array, v_x_array)
ax1.set_title("X velocity")
ax2.plot(time_array, v_y_array)
ax2.set_title("Y velocity")
a_x_array_discrete = (v_x_array[1:] - v_x_array[0:-1])/dt
a_y_array_discrete = (v_y_array[1:] - v_y_array[0:-1])/dt
ax3.plot(time_array, a_x_array)
ax3.plot(time_array[1:], a_x_array_discrete)
ax3.set_title("X acceleration")
ax4.plot(time_array, a_y_array)
ax4.plot(time_array[1:], a_y_array_discrete)
ax4.set_title("Y acceleration")
plt.show()

# file_name = os.getcwd() + "/bike_animation.gif"
# writergif = animation.PillowWriter(fps=30) 
# ani.save(file_name, writer=writergif)

file_name = os.getcwd() + "/bike_animation.gif"
ani.save(file_name, writer='imagemagick', fps=60)