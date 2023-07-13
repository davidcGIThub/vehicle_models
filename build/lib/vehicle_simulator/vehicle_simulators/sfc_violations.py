import numpy as np
import sys

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
