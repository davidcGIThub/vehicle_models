"""
autopilot block for mavsim_python
    - Beard & McLain, PUP, 2012
    - Last Update:
        2/6/2019 - RWB
"""
import sys
import numpy as np
from dataclasses import dataclass


@dataclass
class FixedWingControlParameters:
    roll_kp: float =  0.7640372086731854
    roll_kd: float = -0.0648579802288258
    course_kp: float = 2.54841997961264
    course_ki: float = 0.63710499490316
    yaw_damper_p_wo: float = 0.45
    yaw_damper_kr: float = 0.2
    pitch_kp: float = -3.4628718287546185
    pitch_kd: float = -0.44071472422622815
    altitude_kp: float = 0.07196972780521166
    altitude_ki: float = 0.03598486390260583
    airspeed_throttle_kp: float = 0.7243048339964638
    airspeed_throttle_ki: float = 0.27413830411465756

class FixedWingAutopilot:
    def __init__(self, control_parameters: FixedWingControlParameters = FixedWingControlParameters()):
        # instantiate lateral-directional controllers
        self._AP = control_parameters
        self.roll_from_aileron = PDControlWithRate(
                        kp=self._AP.roll_kp,
                        kd=self._AP.roll_kd,
                        limit=np.radians(45))
        self.course_from_roll = PIControl(
                        kp=self._AP.course_kp,
                        ki=self._AP.course_ki,
                        limit=np.radians(30))
        self.yaw_damper = TFControl(
                        k=self._AP.yaw_damper_kr,
                        n0=0.0,
                        n1=1.0,
                        d0=self._AP.yaw_damper_p_wo,
                        d1=1)

        # instantiate longitudinal controllers
        self.pitch_from_elevator = PDControlWithRate(
                        kp=self._AP.pitch_kp,
                        kd=self._AP.pitch_kd,
                        limit=np.radians(45))
        self.altitude_rate_from_pitch = PIControl(
                        kp=self._AP.altitude_kp,
                        ki=self._AP.altitude_ki,
                        limit=np.radians(30))
        self.airspeed_from_throttle = PIControl(
                        kp=self._AP.airspeed_throttle_kp,
                        ki=self._AP.airspeed_throttle_ki,
                        limit=1.0)

    def get_commands(self, cmd, state, wind, dt):
        # get commmands
        course_command = cmd.item(0)
        climb_rate_command = cmd.item(1)
        airspeed_command = cmd.item(2)
        phi_feedforward = cmd.item(3)
        # get states
        phi, theta, psi = self.__Quaternion2Euler(state[6:10])
        R = self.__Quaternion2Rotation(state[6:10])
        pdot = R @ state[3:6]
        p = state.item(10)
        q = state.item(11)
        r = state.item(12)
        chi = np.arctan2(pdot.item(1), pdot.item(0))
        wind_body_frame = R.T @ wind[0:3] + wind[3:6] # rotate steady state wind to body frame and add gust
        v_air = state[3:6] - wind_body_frame # velocity vector relative to the airmass
        Va = np.sqrt(v_air.item(0)**2 + v_air.item(1)**2 + v_air.item(2)**2)
        # lateral autopilot
        chi_c = self.__wrap(course_command, chi)
        phi_c = self.__saturate(phi_feedforward + self.course_from_roll.update(chi_c, chi, dt), -np.radians(30), np.radians(30))
        delta_a = self.roll_from_aileron.update(phi_c, phi, p)
        delta_r = self.yaw_damper.update(r, dt)
        # longitudinal autopilot
        theta_c = self.altitude_rate_from_pitch.update(climb_rate_command, -pdot.item(2), dt)
        delta_e = self.pitch_from_elevator.update(theta_c, theta, q)
        delta_t = self.airspeed_from_throttle.update(airspeed_command, Va, dt)
        delta_t = self.__saturate(delta_t, 0.0, 1.0)
        # construct control outputs and commanded states
        delta = np.array([delta_t, delta_e, delta_a, delta_r])
        return delta

    def __saturate(self, input, low_limit, up_limit):
        if input <= low_limit:
            output = low_limit
        elif input >= up_limit:
            output = up_limit
        else:
            output = input
        return output
    
    def __wrap(self, chi_1, chi_2):
        while chi_1 - chi_2 > np.pi:
            chi_1 = chi_1 - 2.0 * np.pi
        while chi_1 - chi_2 < -np.pi:
            chi_1 = chi_1 + 2.0 * np.pi
        return chi_1

    def __Quaternion2Euler(self, quaternion):
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
    
    def __Quaternion2Rotation(self, quaternion):
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

class PIControl:
    def __init__(self, kp=0.0, ki=0.0, limit=1.0):
        self.kp = kp
        self.ki = ki
        self.limit = limit
        self.integrator = 0.0
        self.error_delay_1 = 0.0

    def update(self, y_ref, y, ts):

        # compute the error
        error = y_ref - y
        # update the integrator using trapazoidal rule
        self.integrator = self.integrator \
                          + (ts/2) * (error + self.error_delay_1)
        # PI control
        u = self.kp * error \
            + self.ki * self.integrator
        # saturate PI control at limit
        u_sat = self._saturate(u)
        # integral anti-windup
        #   adjust integrator to keep u out of saturation
        if np.abs(self.ki) > 0.0001:
            self.integrator = self.integrator \
                              + (ts / self.ki) * (u_sat - u)
        # update the delayed variables
        self.error_delay_1 = error
        return u_sat

    def _saturate(self, u):
        # saturate u at +- self.limit
        if u >= self.limit:
            u_sat = self.limit
        elif u <= -self.limit:
            u_sat = -self.limit
        else:
            u_sat = u
        return u_sat
    
class PDControlWithRate:
    # PD control with rate information
    # u = kp*(yref-y) - kd*ydot
    def __init__(self, kp=0.0, kd=0.0, limit=1.0):
        self.kp = kp
        self.kd = kd
        self.limit = limit

    def update(self, y_ref, y, ydot):
        u = self.kp * (y_ref - y)  - self.kd * ydot
        # saturate PID control at limit
        u_sat = self._saturate(u)
        return u_sat

    def _saturate(self, u):
        # saturate u at +- self.limit
        if u >= self.limit:
            u_sat = self.limit
        elif u <= -self.limit:
            u_sat = -self.limit
        else:
            u_sat = u
        return u_sat

class TFControl:
    def __init__(self, k=0.0, n0=0.0, n1=0.0, d0=0.0, d1=0.0, limit=1.0):
        self.k = k
        self.n0 = n0
        self.n1 = n1
        self.d0 = d0
        self.d1 = d1
        self.limit = limit
        self.y = 0.0
        self.u = 0.0
        self.y_delay_1 = 0.0
        self.u_delay_1 = 0.0

    def update(self, y, ts):
        # calculate transfer function output (u) using difference equation
        self.b0 = - self.k * (2.0 * self.n1 - ts * self.n0) / (2.0 * self.d1 + ts * self.d0)
        self.b1 = self.k * (2.0 * self.n1 + ts * self.n0) / (2.0 * self.d1 + ts * self.d0)
        self.a0 = (2.0 * self.d1 - ts * self.d0) / (2.0 * self.d1 + ts * self.d0)
        u = self.a0 * self.u_delay_1 + self.b1 * y + self.b0 * self.y_delay_1
        # saturate transfer function output at limit
        u_sat = self._saturate(u)
        # update the delayed variables
        self.y_delay_1 = y
        self.u_delay_1 = u_sat
        return u_sat

    def _saturate(self, u):
        # saturate u at +- self.limit
        if u >= self.limit:
            u_sat = self.limit
        elif u <= -self.limit:
            u_sat = -self.limit
        else:
            u_sat = u
        return u_sat