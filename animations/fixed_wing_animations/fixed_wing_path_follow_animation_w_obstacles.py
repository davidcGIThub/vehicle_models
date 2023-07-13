
"""
A simple example of an animated plot... In 3D!
"""
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
from vehicle_simulator.vehicle_models.fixed_wing_model import FixedWingModel
from vehicle_simulator.vehicle_models.fixed_wing_parameters import FixedWingParameters
from vehicle_simulator.vehicle_controllers.fixed_wing_autopilot import FixedWingControlParameters, FixedWingAutopilot
from vehicle_simulator.vehicle_controllers.fixed_wing_path_follower import FixedWingSplinePathFollower
from vehicle_simulator.vehicle_controllers.bspline_evaluator import BsplineEvaluator
from time import sleep


order = 3
control_points = np.array([[-18.26855358,   0.70966115,  15.429909,    22.01121802,  21.01919592,
   13.01947556,   0.26964944, -14.09807331],
 [  5.12944029,  -2.56472015,   5.12944029,  33.72586905,  68.41308514,
   94.37956304, 102.81021848,  94.37956304],
 [100.25649204,  99.87175398, 100.25649204, 101.68637201, 103.4207176,
  104.71897844, 105.14051078, 104.71897844]]) * np.array([[15],[10],[0]])
bspline_eval = BsplineEvaluator(order)
position_array = bspline_eval.matrix_bspline_evaluation_for_dataset(control_points, order, 1000)
waypoints = position_array[:,np.array([0,500,1000])]
sfc_points = np.array([[ 8,  -2,  -2,   8,   8,  -2,  -2,   8,   8,   8,   8,   8,  -2,  -2, -2,  -2],
 [-1,  -1,   1,   1,   1,   1,  -1,  -1,  -1,   1,   1,  -1,  -1,  -1, 1,   1. ],
 [-1.5, -1.5, -1.5, -1.5,  1.5,  1.5,  1.5,  1.5, -1.5, -1.5,  1.5,  1.5,  1.5, -1.5, -1.5,  1.5]])*10

obstacle_x_data = np.array([[15, 15.34202014, 15.64278761, 15.8660254,  15.98480775, 15.98480775, 15.8660254,  15.64278761, 15.34202014, 15],
                            [15,15.32348855, 15.6079596,  15.81910176, 15.93144815, 15.93144815, 15.81910176, 15.6079596,  15.32348855, 15]])*10 - 150
obstacle_y_data = np.array([[-10, -10, -10, -10, -10, -10, -10, -10, -10, -10 ],
                            [-10, -9.88894624, -9.7912872, -9.71880201, -9.68023345,  -9.68023345,  -9.71880201,  -9.7912872,   -9.88894624, -10]])*10 + 100
obstacle_z_data = np.array([[7, 6.93969262, 6.76604444, 6.5, 6.17364818, 5.82635182, 5.5, 5.23395556, 5.06030738, 5],
                             [7,6.93969262, 6.76604444, 6.5, 6.17364818, 5.82635182, 5.5, 5.23395556, 5.06030738, 5]])*10 - 60
 
fixed_wing_parameters = FixedWingParameters()
control_parameters = FixedWingControlParameters()
# Attaching 3D axis to the figure
fig = plt.figure()
ax = plt.axes(projection='3d')
fig.add_axes(ax)
north = 0
east = 0
down = 0
u = 25
v = 0
w = 0
e0 = 1
e1 = 0
e2 = 0
e3 = 0
p = 0
q = 0
r = 0
wingspan = 3
fuselage_length = 3
state0 = np.array([north, east, down, u, v, w,
                   e0,  e1,  e2,  e3, p, q, r])
plane_model = FixedWingModel(ax, fixed_wing_parameters,
                  wingspan = wingspan, fuselage_length = fuselage_length,
                    state = state0)
autopilot = FixedWingAutopilot(control_parameters)
path_follower = FixedWingSplinePathFollower(order, distance_gain=5)

# define forces
def update_line(num, plane_model: FixedWingModel, autopilot: FixedWingAutopilot,
                path_follower: FixedWingSplinePathFollower,  dt):
    frame_width = 25
    state = plane_model.get_state()
    wind = np.array([0,0,0,0,0,0])
    desired_airspeed = 25
    position = np.array([state.item(0), state.item(1), state.item(2)])
    cmds = path_follower.get_commands(control_points, position, desired_airspeed)
    delta = autopilot.get_commands(cmds, state, wind, dt)
    plane_model.update(delta, wind, dt)
    x = state.item(0)
    y = state.item(1)
    z = state.item(2)
    ax.set_xlim3d([x-frame_width/2, x+frame_width/2])
    ax.set_ylim3d([y-frame_width/2, y+frame_width/2])
    ax.set_zlim3d([z-frame_width/2, z+frame_width/2])

# Setting the axes properties
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Plane Model Test')
ax.view_init(elev=190, azim=45)
ax.plot(position_array[0,:], position_array[1,:], position_array[2,:], color="tab:blue")
ax.plot(sfc_points[0,:], sfc_points[1,:],sfc_points[2,:], alpha=0.5)
ax.plot_surface(obstacle_x_data, obstacle_y_data, obstacle_z_data, color="r")
ax.scatter(waypoints[0,:], waypoints[1,:],waypoints[2,:], marker="o")
# Creating the Animation object
delayBetweenFrames_ms = 50
dt = delayBetweenFrames_ms / 1000 #seconds between frames
line_ani = animation.FuncAnimation(fig, update_line, fargs=[plane_model,autopilot,path_follower, dt] , interval=delayBetweenFrames_ms, blit=False)

plt.show()