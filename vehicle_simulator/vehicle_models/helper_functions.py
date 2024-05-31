import numpy as np

def rotation_matrix_between_vectors(v1_, v2_):
    # Normalize input vectors
    v1 = v1_.flatten() / np.linalg.norm(v1_.flatten())
    v2 = v2_.flatten() / np.linalg.norm(v2_.flatten())

    # Calculate rotation axis (cross product)
    rotation_axis = np.cross(v1.flatten(), v2.flatten())
    
    # Calculate rotation angle (dot product)
    rotation_angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
    
    # Normalize rotation axis
    rotation_axis /= np.linalg.norm(rotation_axis)
    
    # Construct rotation matrix
    c = np.cos(rotation_angle)
    s = np.sin(rotation_angle)
    t = 1 - c
    
    rotation_matrix = np.array([[t * rotation_axis[0]**2 + c, t * rotation_axis[0] * rotation_axis[1] - s * rotation_axis[2], t * rotation_axis[0] * rotation_axis[2] + s * rotation_axis[1]],
                                [t * rotation_axis[0] * rotation_axis[1] + s * rotation_axis[2], t * rotation_axis[1]**2 + c, t * rotation_axis[1] * rotation_axis[2] - s * rotation_axis[0]],
                                [t * rotation_axis[0] * rotation_axis[2] - s * rotation_axis[1], t * rotation_axis[1] * rotation_axis[2] + s * rotation_axis[0], t * rotation_axis[2]**2 + c]])
    return rotation_matrix

def rotation_to_quaternion(R):
    """
    converts a rotation matrix to a unit quaternion
    """
    r11 = R[0][0]
    r12 = R[0][1]
    r13 = R[0][2]
    r21 = R[1][0]
    r22 = R[1][1]
    r23 = R[1][2]
    r31 = R[2][0]
    r32 = R[2][1]
    r33 = R[2][2]

    tmp=r11+r22+r33
    if tmp>0:
        e0 = 0.5*np.sqrt(1+tmp)
    else:
        e0 = 0.5*np.sqrt(((r12-r21)**2+(r13-r31)**2+(r23-r32)**2)/(3-tmp))

    tmp=r11-r22-r33
    if tmp>0:
        e1 = 0.5*np.sqrt(1+tmp)
    else:
        e1 = 0.5*np.sqrt(((r12+r21)**2+(r13+r31)**2+(r23-r32)**2)/(3-tmp))

    tmp=-r11+r22-r33
    if tmp>0:
        e2 = 0.5*np.sqrt(1+tmp)
    else:
        e2 = 0.5*np.sqrt(((r12+r21)**2+(r13+r31)**2+(r23+r32)**2)/(3-tmp))

    tmp=-r11+-22+r33
    if tmp>0:
        e3 = 0.5*np.sqrt(1+tmp)
    else:
        e3 = 0.5*np.sqrt(((r12-r21)**2+(r13+r31)**2+(r23+r32)**2)/(3-tmp))

    return np.array([[e0], [e1], [e2], [e3]])


def euler_to_quaternion(phi, theta, psi):
    """
    Converts an euler angle attitude to a quaternian attitude
    :param euler: Euler angle attitude in a np.matrix(phi, theta, psi)
    :return: Quaternian attitude in np.array(e0, e1, e2, e3)
    """

    e0 = np.cos(psi/2.0) * np.cos(theta/2.0) * np.cos(phi/2.0) + np.sin(psi/2.0) * np.sin(theta/2.0) * np.sin(phi/2.0)
    e1 = np.cos(psi/2.0) * np.cos(theta/2.0) * np.sin(phi/2.0) - np.sin(psi/2.0) * np.sin(theta/2.0) * np.cos(phi/2.0)
    e2 = np.cos(psi/2.0) * np.sin(theta/2.0) * np.cos(phi/2.0) + np.sin(psi/2.0) * np.cos(theta/2.0) * np.sin(phi/2.0)
    e3 = np.sin(psi/2.0) * np.cos(theta/2.0) * np.cos(phi/2.0) - np.cos(psi/2.0) * np.sin(theta/2.0) * np.sin(phi/2.0)

    return np.array([[e0],[e1],[e2],[e3]])