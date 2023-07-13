import numpy as np



class VehicleModel3D:
    def __init__(self, ax):
        self.translation = np.array([[0],[0],[0]])
        self.rotation = np.eye(3)
        
    def update(self):
        pass

    def get_state(self):
        return np.array([])
    
    def set_state(self,state):
        pass
    
    def _update_dynamics(self):
        pass

    def _Quaternion2Euler(self, quaternion):
        """
        converts a quaternion attitude to an euler angle attitude
        :param quaternion: the quaternion to be converted to euler angles in a np.matrix
        :return: the euler angle equivalent (phi, theta, psi) in a np.array
        """
        e0 = quaternion.item(0)
        e1 = quaternion.item(1)
        e2 = quaternion.item(2)
        e3 = quaternion.item(3)
        phi = np.arctan2(2.0 * (e0 * e1 + e2 * e3), e0**2.0 + e3**2.0 - e1**2.0 - e2**2.0)
        theta = np.arcsin(2.0 * (e0 * e2 - e1 * e3))
        psi = np.arctan2(2.0 * (e0 * e3 + e1 * e2), e0**2.0 + e1**2.0 - e2**2.0 - e3**2.0)
        return phi, theta, psi

    def _Quaternion2Rotation(self, quaternion):
        """
        converts a quaternion attitude to a rotation matrix
        """
        e0 = quaternion.item(0)
        e1 = quaternion.item(1)
        e2 = quaternion.item(2)
        e3 = quaternion.item(3)
        R = np.array([[e1 ** 2.0 + e0 ** 2.0 - e2 ** 2.0 - e3 ** 2.0, 2.0 * (e1 * e2 - e3 * e0), 2.0 * (e1 * e3 + e2 * e0)],
                    [2.0 * (e1 * e2 + e3 * e0), e2 ** 2.0 + e0 ** 2.0 - e1 ** 2.0 - e3 ** 2.0, 2.0 * (e2 * e3 - e1 * e0)],
                    [2.0 * (e1 * e3 - e2 * e0), 2.0 * (e2 * e3 + e1 * e0), e3 ** 2.0 + e0 ** 2.0 - e1 ** 2.0 - e2 ** 2.0]])
        R = R/np.linalg.det(R)
        return R

    #### Graphic Functions ####
    def _update_graphics(self):
        pass

    def _draw(self):
        pass