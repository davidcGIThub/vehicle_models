import numpy as np
import sys
from vehicle_simulator.vehicle_controllers.bspline_evaluator import BsplineEvaluator
from dataclasses import dataclass

def get_box_violations_from_spline(box_list,intervals_per_box, control_points, order):
    spline_eval = BsplineEvaluator(order)
    points_per_interval = 1000
    path_data = spline_eval.matrix_bspline_evaluation_for_dataset(control_points, points_per_interval)
    closest_distances_to_sfc_walls = np.zeros(len(box_list))
    for j in range(len(box_list)):
        box_points = box_list[j]
        num_prev_intervals = np.sum(intervals_per_box[0:j])
        num_intervals = intervals_per_box[j]
        start_index = int(num_prev_intervals*points_per_interval)
        end_index = int(num_prev_intervals*points_per_interval + num_intervals*points_per_interval)
        interval_data = path_data[:,start_index:end_index]
        distance_sfc = get_greatest_box_violation_distance(box_points, interval_data)
        closest_distances_to_sfc_walls[j] = distance_sfc
    return closest_distances_to_sfc_walls

def get_greatest_box_violation_distance(box_points, points):
    num_box_points = np.shape(box_points)[1]
    num_points = np.shape(points)[1]
    center_point = get_box_center(box_points)
    min_distance = sys.float_info.max
    for j in range(num_points):
        point = points[:,j]
        for i in range(num_box_points - 2):
            A = box_points[:,i  ]
            B = box_points[:,i+1]
            C = box_points[:,i+2]
            distance = get_distance_to_wall(point, center_point, A,B,C)
            if distance < min_distance:
                min_distance = distance
    return min_distance

def get_box_violation_distance(box_points, point):
    num_points = np.shape(box_points)[1]
    center_point = get_box_center(box_points)
    min_distance = sys.float_info.max
    for i in range(num_points - 2):
        A = box_points[:,i  ]
        B = box_points[:,i+1]
        C = box_points[:,i+2]
        distance = get_distance_to_wall(point, center_point, A,B,C)
        if distance < min_distance:
            min_distance = distance
    return min_distance

def get_box_center(box_points):
    unique_points = np.unique(box_points, axis=1)
    center = np.mean(unique_points,axis=1)
    return center

def get_distance_to_wall(point, box_center, A,B,C):
    distance = get_distance_to_plane(point, A,B,C)
    side = check_if_points_on_same_side_of_plane(point, box_center, A, B, C)
    return distance*side

def get_distance_to_plane(point, A,B,C):
    normal_vec = get_normal_vector(A,B,C)
    a = normal_vec.item(0)
    b = normal_vec.item(1)
    c = normal_vec.item(2)
    x0 = point.item(0)
    y0 = point.item(1)
    z0 = point.item(2)
    d = -(a*A.item(0) + b*A.item(1) + c*A.item(2))
    distance = (a*x0 + b*y0 + c*z0 + d)/np.sqrt(a**2 + b**2 + c**2)
    return np.abs(distance)

def check_if_points_on_same_side_of_plane(point1, point2, A, B, C):
    normal_vec = get_normal_vector(A,B,C)
    value = 0
    if np.dot(normal_vec, point1 - A) * np.dot(normal_vec, point2 - A) > 0:
        value = 1
    else:
        value = -1
    return value 

def get_normal_vector(A,B,C):
    normal_vec = np.cross((B-A), (C-A))
    return normal_vec


def get_obstacle_violations(obstacle_list: 'list[Obstacle]', location_data):
    closest_distances_to_obstacles = np.zeros(len(obstacle_list))
    print("obstacle list in func: " , obstacle_list)
    for i in range(len(obstacle_list)):
        obstacle = obstacle_list[i]
        obstacle_center = obstacle.center.flatten()[:,None]
        distance_obstacle = np.min(np.linalg.norm(location_data - obstacle_center,2,0) - obstacle.radius)
        closest_distances_to_obstacles[i] = distance_obstacle
    return closest_distances_to_obstacles
        
@dataclass
class Obstacle:
    center: np.ndarray
    radius: np.double
    height: np.double

def plot_3D_obstacle(obstacle: Obstacle, ax):
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = obstacle.radius * np.cos(u)*np.sin(v) + obstacle.center.item(0)
    y = obstacle.radius * np.sin(u)*np.sin(v) + obstacle.center.item(1)
    z = obstacle.radius * np.cos(v) + obstacle.center.item(2)
    ax.plot_surface(x, y, z, color="r")

def plot_3D_obstacles(obstacles: list, ax):
    for i in range(len(obstacles)):
        plot_3D_obstacle(obstacles[i], ax)

def plot_cylinder(obstacle: Obstacle, ax):
    z = np.linspace(0, obstacle.height, 50)
    theta = np.linspace(0, 2*np.pi, 50)
    theta_grid, z_grid=np.meshgrid(theta, z)
    x_grid = obstacle.radius*np.cos(theta_grid) + obstacle.center.item(0)
    y_grid = obstacle.radius*np.sin(theta_grid) + obstacle.center.item(1)
    ax.plot_surface(x_grid,y_grid,z_grid, color="r")

def plot_cylinders(obstacles: list, ax):
    for i in range(len(obstacles)):
        plot_cylinder(obstacles[i], ax)









def data_for_cylinder_along_z(center_x,center_y,radius,height_z):
    z = np.linspace(0, height_z, 50)
    theta = np.linspace(0, 2*np.pi, 50)
    theta_grid, z_grid=np.meshgrid(theta, z)
    x_grid = radius*np.cos(theta_grid) + center_x
    y_grid = radius*np.sin(theta_grid) + center_y
    return x_grid,y_grid,z_grid

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

Xc,Yc,Zc = data_for_cylinder_along_z(0.2,0.2,0.05,0.1)
ax.plot_surface(Xc, Yc, Zc, alpha=0.5)