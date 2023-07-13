import numpy as np
import matplotlib.pyplot as plt
from vehicle_simulator.vehicle_simulators.spatial_violations import get_box_violation_distance, get_greatest_box_violation_distance, \
    get_box_violations_from_spline
from vehicle_simulator.vehicle_controllers.bspline_evaluator import BsplineEvaluator

scale = 1

sfc_points = np.array([[ 8,  -2,  -2,   8,   8,  -2,  -2,   8,   8,   8,   8,   8,  -2,  -2, -2,  -2],
 [-1,  -1,   1,   1,   1,   1,  -1,  -1,  -1,   1,   1,  -1,  -1,  -1, 1,   1. ],
 [-1.5, -1.5, -1.5, -1.5,  1.5,  1.5,  1.5,  1.5, -1.5, -1.5,  1.5,  1.5,  1.5, -1.5, -1.5,  1.5]])*scale
theta = 90*np.pi/180
# R = np.array([[1,0,0],[0,np.cos(theta), -np.sin(theta)],[0,np.sin(theta), np.cos(theta)]])
point_1 = np.array([4, 0,0  ])*scale
point_2 = np.array([8, 0,1.5])*scale
point_3 = np.array([0,-5,10 ])*scale
points = np.concatenate((point_1[:,None], point_2[:,None], point_3[:,None]), 1)

print("box violation 1: " , get_box_violation_distance(sfc_points, point_1) )
print("box violation 2: " , get_box_violation_distance(sfc_points, point_2) )
print("box violation 3: " , get_box_violation_distance(sfc_points, point_3) )
print("greatest box violation: " , get_greatest_box_violation_distance(sfc_points, points) )

fig = plt.figure()
ax = plt.axes(projection='3d')
fig.add_axes(ax)
ax.plot(sfc_points[0,:], sfc_points[1,:], sfc_points[2,:])
ax.scatter(point_1[0], point_1[1],point_1[2])
ax.scatter(point_2[0], point_2[1],point_2[2])
ax.scatter(point_3[0], point_3[1],point_3[2])
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.show()

### Fig 2 ###
order = 3
control_points = np.array([[  5.12944029,  -2.56472015,   5.12944029,  33.72586905,  68.41308514,
   94.37956304, 102.81021848,  94.37956304],
   [100.25649204,  99.87175398, 100.25649204, 101.68637201, 103.4207176,
  104.71897844, 105.14051078, 104.71897844],
   [-18.26855358,   0.70966115,  15.429909,    22.01121802,  21.01919592,
   13.01947556,   0.26964944, -14.09807331]])*np.array([[1/13],[1/10],[1/7]]) + np.array([[-1.4],[-10],[-3]])

spline_eval = BsplineEvaluator(order)
num_intervals = np.shape(control_points)[1] - order
points_per_interval = 1000
path = spline_eval.matrix_bspline_evaluation_for_dataset(control_points, order, points_per_interval)
interval_endpoints = path[:,np.linspace(0,points_per_interval*num_intervals,num_intervals+1).astype(int)]

R = np.array([[np.cos(theta), 0, np.sin(theta)],
              [0,           1,               0],
              [-np.sin(theta),0, np.cos(theta)]])
sfc_points_2 = np.dot(R,sfc_points)
sfc_list = [sfc_points_2, sfc_points]
intervals_per_sfc = [2,3]

violations = get_box_violations_from_spline(sfc_list, intervals_per_sfc, control_points, order)
print(("violations: " , violations))

fig = plt.figure()
ax = plt.axes(projection='3d')
fig.add_axes(ax)
ax.plot(sfc_points[0,:], sfc_points[1,:], sfc_points[2,:])
ax.plot(sfc_points_2[0,:], sfc_points_2[1,:], sfc_points_2[2,:])
ax.plot(path[0,:],path[1,:],path[2,:])
ax.scatter(interval_endpoints[0,:],interval_endpoints[1,:],interval_endpoints[2,:])
ax.scatter(control_points[0,:],control_points[1,:],control_points[2,:])
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.show()