#homework 7
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
from vehicle_simulator.vehicle_models.unicycle_model import UnicycleModel
import os

x_limits = 5
y_limits = 5
sec = 90
time_array = np.linspace(0,sec,int(sec/0.1+1))
dt = time_array[1]
v_max = 5
unicycle = UnicycleModel(x = 0, 
                    y = 0,
                    theta = np.pi/4,
                    alpha = np.array([0.1,0.01,0.1,0.01]),
                    max_vel=5,
                    max_theta_dot=0.5)

fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                     xlim=(-x_limits,x_limits), ylim=(-y_limits,y_limits))
ax.grid()
robot_fig = plt.Polygon(unicycle.getPoints(),fc = 'g')
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

global a_c, theta_dot_c               
a_c = (0.5*np.cos(.5*time_array))/3
theta_dot_c = np.cos(0.1*time_array)

def init():
    #initialize animation
    ax.add_patch(robot_fig)
    time_text.set_text('')
    return robot_fig, time_text

def animate(i):
    global unicycle
    # propogate robot motion
    # x_d = 5
    # y_d = 5
    states = unicycle.getState() 
    t = time_array[i]
    unicycle.update_acceleration_motion_model(a_c[i],theta_dot_c[i],dt)
    robot_fig.xy = unicycle.getPoints()
    
    # update time
    time_text.set_text('time = %.1f' % time_array[i])

    return robot_fig, time_text

from time import time
animate(0)

ani = animation.FuncAnimation(fig, animate, frames = np.size(time_array), 
                            interval = dt*100, blit = True, init_func = init, repeat = False)

plt.show()

# file_name = os.getcwd() + "/bike_animation.gif"
# writergif = animation.PillowWriter(fps=30) 
# ani.save(file_name, writer=writergif)

file_name = os.getcwd() + "/unicycle_animation.gif"
ani.save(file_name, writer='imagemagick', fps=60)