#homework 7
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
from boat_model import BoatModel
import os

x_limits = 30
y_limits = 30
sec = 90
time_array = np.linspace(0,sec,1000)
dt = time_array[1]
max_delta = np.pi/2
boat = BoatModel(x = 0, 
                 y = 0, 
                 theta = np.pi/2.0, 
                 delta = 0,
                 alpha = np.array([0.0,0.0,0.0,0.0]),
                 height = 1,
                 width = 0.5,
                 c_r = 1,
                 max_delta = max_delta,
                 max_accel = 5,
                 max_vel = 5
                 )


fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                     xlim=(-x_limits,x_limits), ylim=(-y_limits,y_limits))
ax.grid()
boat_fig = plt.Polygon(boat.getBodyPoints(),fc = 'g')
rudder_fig = plt.Polygon(boat.getRudderPoints(),fc = 'k')
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

global v_c, delta_dot_c          
a_c = np.cos(0.5*time_array)/5
# v_c = time_array*0 + 1
delta_dot_c = np.cos(0.2*time_array)*0.3
# delta_c = time_array*0

def init():
    #initialize animation
    ax.add_patch(boat_fig)
    ax.add_patch(rudder_fig)
    time_text.set_text('')
    return boat_fig, rudder_fig, time_text

def animate(i):
    global boat
    # propogate robot motion
    # x_d = 5
    # y_d = 5
    states = boat.getState() 
    # t = time_array[i]
    delta_dot_com = delta_dot_c[i]
    # delta_dot_com = 0
    boat.update_acceleration_motion_model(a_c[i], delta_dot_com,dt)
    boat_fig.xy = boat.getBodyPoints()
    rudder_fig.xy = boat.getRudderPoints()
    
    # update time
    time_text.set_text('time = %.1f' % time_array[i])

    return boat_fig, rudder_fig, time_text

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