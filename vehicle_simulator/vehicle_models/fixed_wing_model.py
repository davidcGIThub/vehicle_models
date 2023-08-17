import numpy as np
from vehicle_simulator.vehicle_models.fixed_wing_parameters import FixedWingParameters
from vehicle_simulator.vehicle_models.vehicle_model_3D import VehicleModel3D
import matplotlib.pyplot as plt

class FixedWingModel(VehicleModel3D):
    def __init__(self, ax=None, vehicle_parameters: FixedWingParameters = FixedWingParameters(),
                  wingspan = 3, fuselage_length = 3,
                    state = np.array([0,0,-50,25,0,0,1,0,0,0,0,0,0])):
        self._north = state.item(0)  # initial north position
        self._east = state.item(1)  # initial east position
        self._down = state.item(2)  # initial down position
        self._u = state.item(3)  # initial velocity along body x-axis
        self._v = state.item(4)  # initial velocity along body y-axis
        self._w = state.item(5)  # initial velocity along body z-axis
        self._e0 = state.item(6)
        self._e1 = state.item(7)
        self._e2 = state.item(8)
        self._e3 = state.item(9)
        self._p = state.item(10)  # initial roll rate
        self._q = state.item(11)  # initial pitch rate
        self._r = state.item(12)  # initial yaw rate
        self._u_dot = 0
        self._v_dot = 0
        self._w_dot = 0
        self._p_dot = 0
        self._q_dot = 0
        self._r_dot = 0
        #model parameters
        self._MAV = vehicle_parameters
        #Graphics
        self._fuselage_length = fuselage_length
        self._wingspan = wingspan
        self._fuselage_color = 'k'
        self._wings_color = '0.5'
        self.translation = np.array([[self._north],[self._east],[self._down]])
        quaternion = np.array([self._e0, self._e1, self._e2, self._e3])
        self.rotation = self._Quaternion2Rotation(quaternion)
        self.ax = ax
        if self.ax is not None:
            self.fuselage, = ax.plot([], [], [], lw=2, color=self._fuselage_color)
            self.wings, = ax.plot([], [], [], lw=2, color=self._wings_color)
            self.tail, = ax.plot([], [], [], lw=2, color=self._wings_color)
            self.rudder, = ax.plot([], [], [], lw=2, color=self._fuselage_color)

    def scale_plane_graphic(self, scale_factor):
        self._fuselage_length = self._fuselage_length*scale_factor
        self._wingspan = self._wingspan*scale_factor
        
    def update(self, delta, wind, dt):
        self._update_dynamics(delta, wind, dt)
        self.update_graphics()

    def reset_graphic_axes(self, ax):
        self.ax = ax
        self.fuselage, = ax.plot([], [], [], lw=2, color=self._fuselage_color)
        self.wings, = ax.plot([], [], [], lw=2, color=self._wings_color)
        self.tail, = ax.plot([], [], [], lw=2, color=self._wings_color)
        self.rudder, = ax.plot([], [], [], lw=2, color=self._fuselage_color)

    def get_state(self):
        state = np.array([self._north, self._east, self._down,
                    self._u, self._v, self._w,
                    self._e0, self._e1, self._e2, self._e3, 
                    self._p, self._q, self._r])
        return state
    
    def get_inertial_velocity(self):
        quat = np.array([self._e0, self._e1, self._e2, self._e3])
        body_vel = np.array([self._u, self._v, self._w])
        pos_dot = self._Quaternion2Rotation(quat) @ body_vel
        return pos_dot

    def get_inertial_acceleration(self):
        quat = np.array([self._e0, self._e1, self._e2, self._e3])
        body_accel = np.array([self._u_dot, self._v_dot, self._w_dot])
        accel = self._Quaternion2Rotation(quat) @ body_accel
        return accel

    def set_acceleration_states(self, derivatives):
        self._u_dot = derivatives.item(3)
        self._v_dot = derivatives.item(4)
        self._w_dot = derivatives.item(5)
        self._p_dot = derivatives.item(10)
        self._q_dot = derivatives.item(11)
        self._r_dot = derivatives.item(12)

    def set_state(self,state):
        self._north = state.item(0)  # initial north position
        self._east = state.item(1)  # initial east position
        self._down = state.item(2)  # initial down position
        self._u = state.item(3)  # initial velocity along body x-axis
        self._v = state.item(4)  # initial velocity along body y-axis
        self._w = state.item(5)  # initial velocity along body z-axis
        self._e0 = state.item(6)
        self._e1 = state.item(7)
        self._e2 = state.item(8)
        self._e3 = state.item(9)
        self._p = state.item(10)  # initial roll rate
        self._q = state.item(11)  # initial pitch rate
        self._r = state.item(12) 

    def get_ax(self):
        return self.ax
    
    def _update_dynamics(self, delta, wind, dt):
        '''
            Integrate the differential equations defining dynamics, update sensors
            delta = (delta_a, delta_e, delta_r, delta_t) are the control inputs
            wind is the wind vector in inertial coordinates
            Ts is the time step between function calls.
        '''
        # get forces and moments acting on rigid bod
        state = self.get_state()
        forces_moments = self._forces_moments(delta, state, wind)
        # Integrate ODE using Runge-Kutta RK4 algorithm
        time_step = dt
        k1 = self._derivatives(state, forces_moments)
        k2 = self._derivatives(state + time_step/2.*k1, forces_moments)
        k3 = self._derivatives(state + time_step/2.*k2, forces_moments)
        k4 = self._derivatives(state + time_step*k3, forces_moments)
        derivatives = (k1 + 2*k2 + 2*k3 + k4)/6
        new_state = state + time_step*derivatives
        self.set_acceleration_states(derivatives)
        # normalize the quaternion
        e0 = new_state.item(6)
        e1 = new_state.item(7)
        e2 = new_state.item(8)
        e3 = new_state.item(9)
        normE = np.sqrt(e0**2+e1**2+e2**2+e3**2)
        new_state[6] = new_state.item(6)/normE
        new_state[7] = new_state.item(7)/normE
        new_state[8] = new_state.item(8)/normE
        new_state[9] = new_state.item(9)/normE
        # update the airspeed, angle of attack, and side slip angles using new state
        self.set_state(new_state)

    def _derivatives(self, state, forces_moments):
        """
        for the dynamics xdot = f(x, u), returns f(x, u)
        """
        # extract the states
        # north = state.item(0)
        # east = state.item(1)
        # down = state.item(2)
        u = state.item(3)
        v = state.item(4)
        w = state.item(5)
        e0 = state.item(6)
        e1 = state.item(7)
        e2 = state.item(8)
        e3 = state.item(9)
        p = state.item(10)
        q = state.item(11)
        r = state.item(12)
        #   extract forces/moments
        fx = forces_moments.item(0)
        fy = forces_moments.item(1)
        fz = forces_moments.item(2)
        l = forces_moments.item(3)
        m = forces_moments.item(4)
        n = forces_moments.item(5)
        # position kinematics
        pos_dot = self._Quaternion2Rotation(state[6:10]) @ state[3:6]
        north_dot = pos_dot.item(0)
        east_dot = pos_dot.item(1)
        down_dot = pos_dot.item(2)
        # position dynamics
        u_dot = r*v - q*w + fx/self._MAV.mass
        v_dot = p*w - r*u + fy/self._MAV.mass
        w_dot = q*u - p*v + fz/self._MAV.mass
        # rotational kinematics
        e0_dot = 0.5 * (-p*e1 - q*e2 - r*e3)
        e1_dot = 0.5 * (p*e0 + r*e2 - q*e3)
        e2_dot = 0.5 * (q*e0 - r*e1 + p*e3)
        e3_dot = 0.5 * (r*e0 + q*e1 -p*e2)
        # rotatonal dynamics
        p_dot = self._MAV.gamma1*p*q - self._MAV.gamma2*q*r + self._MAV.gamma3*l + self._MAV.gamma4*n
        q_dot = self._MAV.gamma5*p*r - self._MAV.gamma6*(p**2-r**2) + m/self._MAV.Jy
        r_dot = self._MAV.gamma7*p*q - self._MAV.gamma1*q*r + self._MAV.gamma4*l + self._MAV.gamma8*n
        # collect the derivative of the states
        x_dot = np.array([north_dot, east_dot, down_dot, u_dot, v_dot, w_dot,
                           e0_dot, e1_dot, e2_dot, e3_dot, p_dot, q_dot, r_dot])
        return x_dot

    def __compute_wind_velocity_data(self, state, wind=np.zeros((6,1))):
        steady_state = wind[0:3]
        gust = wind[3:6]
        # convert wind vector from world to body frame
        R = self._Quaternion2Rotation(state[6:10]) # passive rotation from body to world frame
        wind_body_frame = R.T @ steady_state  # rotate steady state wind to body frame
        wind_body_frame += gust  # add the gust
        self._wind = R @ wind_body_frame  # wind in the world frame
        # velocity vector relative to the airmass
        v_air = state[3:6] - wind_body_frame
        ur = v_air.item(0)
        vr = v_air.item(1)
        wr = v_air.item(2)
        # compute airspeed
        Va = np.sqrt(ur**2 + vr**2 + wr**2)
        # compute angle of attack
        if ur == 0:
            alpha = np.sign(wr)*np.pi/2.
        else:
            alpha = np.arctan(wr/ur)
        # compute sideslip angle
        tmp = np.sqrt(ur**2 + wr**2)
        if tmp == 0:
            beta = np.sign(vr)*np.pi/2.
        else:
            beta = np.arcsin(vr/tmp)
        return Va, alpha, beta

    def _forces_moments(self, delta, state, wind):
        """
        return the forces on the UAV based on the state, wind, and control surfaces
        :param delta: np.matrix(delta_a, delta_e, delta_r, delta_t)
        :return: Forces and Moments on the UAV np.matrix(Fx, Fy, Fz, Ml, Mn, Mm)
        """
        throttle = delta.item(0)  # throttle command
        elevator = delta.item(1)  # elevator command
        aileron = delta.item(2)  # aileron command
        rudder = delta.item(3)  # rudder command
        Va, alpha, beta = self.__compute_wind_velocity_data(state, wind)
        p = state.item(10)
        q = state.item(11)
        r = state.item(12)
        # compute gravitational forces
        R = self._Quaternion2Rotation(state[6:10]) # rotation from body to world frame
        f_g = R.T @ np.array([[0.], [0.], [self._MAV.mass * self._MAV.gravity]])
        fx = f_g.item(0)
        fy = f_g.item(1)
        fz = f_g.item(2)
        # intermediate variables
        qbar = 0.5 * self._MAV.rho * Va**2
        ca = np.cos(alpha)
        sa = np.sin(alpha)
        p_nondim = p * self._MAV.b / (2 * Va)  # nondimensionalize p
        q_nondim = q * self._MAV.c / (2 * Va)  # nondimensionalize q
        r_nondim = r * self._MAV.b / (2 * Va)  # nondimensionalize r
        # compute Lift and Drag coefficients
        tmp1 = np.exp(-self._MAV.M * (alpha - self._MAV.alpha0))
        tmp2 = np.exp(self._MAV.M * (alpha + self._MAV.alpha0))
        sigma = (1 + tmp1 + tmp2) / ((1 + tmp1) * (1 + tmp2))
        CL = (1 - sigma) * (self._MAV.C_L_0 + self._MAV.C_L_alpha * alpha) \
             + sigma * 2 * np.sign(alpha) * sa**2 * ca
        CD = self._MAV.C_D_p + ((self._MAV.C_L_0 + self._MAV.C_L_alpha * alpha)**2)/(np.pi * self._MAV.e * self._MAV.AR)
        # compute Lift and Drag Forces
        F_lift = qbar * self._MAV.S_wing * (
                CL
                + self._MAV.C_L_q * q_nondim
                + self._MAV.C_L_delta_e * elevator
        )
        F_drag = qbar * self._MAV.S_wing * (
                CD
                + self._MAV.C_D_q * q_nondim
                + self._MAV.C_D_delta_e * elevator
        )
        # compute longitudinal forces in body frame
        fx = fx - ca * F_drag + sa * F_lift
        fz = fz - sa * F_drag - ca * F_lift
        # compute lateral forces in body frame
        fy += qbar * self._MAV.S_wing * (
                self._MAV.C_Y_0
                + self._MAV.C_Y_beta * beta
                + self._MAV.C_Y_p * p_nondim
                + self._MAV.C_Y_r * r_nondim
                + self._MAV.C_Y_delta_a * aileron
                + self._MAV.C_Y_delta_r * rudder
        )
        # compute logitudinal torque in body frame
        My = qbar * self._MAV.S_wing * self._MAV.c * (
                self._MAV.C_m_0
                + self._MAV.C_m_alpha * alpha
                + self._MAV.C_m_q * q_nondim
                + self._MAV.C_m_delta_e * elevator
        )
        # compute lateral torques in body frame
        Mx = qbar * self._MAV.S_wing * self._MAV.b * (
                self._MAV.C_ell_0
                + self._MAV.C_ell_beta * beta
                + self._MAV.C_ell_p * p_nondim
                + self._MAV.C_ell_r * r_nondim
                + self._MAV.C_ell_delta_a * aileron
                + self._MAV.C_ell_delta_r * rudder
        )
        Mz = qbar * self._MAV.S_wing * self._MAV.b * (
                self._MAV.C_n_0 + self._MAV.C_n_beta * beta
                + self._MAV.C_n_p * p_nondim
                + self._MAV.C_n_r * r_nondim
                + self._MAV.C_n_delta_a * aileron
                + self._MAV.C_n_delta_r * rudder
        )
        thrust_prop, torque_prop = self._motor_thrust_torque(Va, throttle)
        fx += thrust_prop
        Mx += -torque_prop
        return np.array([[fx, fy, fz, Mx, My, Mz]]).T

    def _motor_thrust_torque(self, Va, delta_t):
        # compute thrust and torque due to propeller
        # map delta_t throttle command(0 to 1) into motor input voltage
        v_in = self._MAV.V_max * delta_t
        # Quadratic formula to solve for motor speed
        a = self._MAV.C_Q0 * self._MAV.rho * np.power(self._MAV.D_prop, 5) \
            / ((2.*np.pi)**2)
        b = (self._MAV.C_Q1 * self._MAV.rho * np.power(self._MAV.D_prop, 4)
             / (2.*np.pi)) * Va + self._MAV.KQ * self._MAV.KV / self._MAV.R_motor
        c = self._MAV.C_Q2 * self._MAV.rho * np.power(self._MAV.D_prop, 3) \
            * Va**2 - (self._MAV.KQ / self._MAV.R_motor) * v_in + self._MAV.KQ * self._MAV.i0
        # Angular speed of propeller
        omega_p = (-b + np.sqrt(b**2 - 4*a*c)) / (2.*a)
        # compute advance ratio
        J_p = 2 * np.pi * Va / (omega_p * self._MAV.D_prop)
        # compute non-dimensionalized coefficients of thrust and torque
        C_T = self._MAV.C_T2 * J_p**2 + self._MAV.C_T1 * J_p + self._MAV.C_T0
        C_Q = self._MAV.C_Q2 * J_p**2 + self._MAV.C_Q1 * J_p + self._MAV.C_Q0
        # compute propeller thrust and torque
        n = omega_p / (2 * np.pi)
        thrust_prop = self._MAV.rho * n**2 * np.power(self._MAV.D_prop, 4) * C_T
        torque_prop = self._MAV.rho * n**2 * np.power(self._MAV.D_prop, 5) * C_Q
        return thrust_prop, torque_prop

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
    def plot_plane(self, ax=None):
        if self.ax is not None or ax is not None:
            if ax is None:
                ax = self.ax
            fuselage, = ax.plot([], [], [], lw=2, color=self._fuselage_color)
            wings, = ax.plot([], [], [], lw=2, color=self._wings_color)
            tail, = ax.plot([], [], [], lw=2, color=self._wings_color)
            rudder, = ax.plot([], [], [], lw=2, color=self._fuselage_color)
            self._draw_fuselage(fuselage)
            self._draw_wings(wings)
            self._draw_tail(tail)
            self._draw_rudder(rudder)
            print("plotting: ")

    def update_graphics(self):
        self.translation = np.array([[self._north],[self._east],[self._down]])
        quaternion = np.array([self._e0, self._e1, self._e2, self._e3])
        self.rotation = self._Quaternion2Rotation(quaternion)
        if self.ax is not None:
            self._draw()

    def _draw(self):
        self._draw_fuselage(self.fuselage)
        self._draw_wings(self.wings)
        self._draw_tail(self.tail)
        self._draw_rudder(self.rudder)

    def _draw_fuselage(self, fuselage):
        back_point = np.array([[-self._fuselage_length/2],[0],[0]])
        front_point = np.array([[self._fuselage_length/2],[0],[0]])
        back_point = np.dot(self.rotation, back_point) + self.translation
        front_point = np.dot(self.rotation, front_point) + self.translation
        fuselage.set_data(np.array([back_point.item(0),front_point.item(0)]) , np.array([back_point.item(1),front_point.item(1)]))
        fuselage.set_3d_properties(np.array([back_point.item(2),front_point.item(2)]))

    def _draw_wings(self, wings):
        chord = self._fuselage_length/4
        left_back = np.array([[0],[-self._wingspan/2],[0]])
        left_front = np.array([[chord],[-self._wingspan/2],[0]])
        right_front = np.array([[chord],[self._wingspan/2],[0]])
        right_back = np.array([[0],[self._wingspan/2],[0]])
        left_back = np.dot(self.rotation, left_back) + self.translation
        left_front = np.dot(self.rotation, left_front) + self.translation
        right_front = np.dot(self.rotation, right_front) + self.translation
        right_back = np.dot(self.rotation, right_back) + self.translation
        wings.set_data(np.array([left_back.item(0),left_front.item(0),right_front.item(0),right_back.item(0),left_back.item(0)]),
                            np.array([left_back.item(1),left_front.item(1),right_front.item(1),right_back.item(1),left_back.item(1)]))
        wings.set_3d_properties(np.array([left_back.item(2),left_front.item(2),right_front.item(2),right_back.item(2),left_back.item(2)]))

    def _draw_tail(self, tail):
        tail_span = self._wingspan/2
        tail_chord = self._fuselage_length/5
        left_back = np.array([[-self._fuselage_length/2],[-tail_span/2],[0]])
        left_front = np.array([[-self._fuselage_length/2+tail_chord],[-tail_span/2],[0]])
        right_front = np.array([[-self._fuselage_length/2+tail_chord],[tail_span/2],[0]])
        right_back = np.array([[-self._fuselage_length/2],[tail_span/2],[0]])
        left_back = np.dot(self.rotation, left_back) + self.translation
        left_front = np.dot(self.rotation, left_front) + self.translation
        right_front = np.dot(self.rotation, right_front) + self.translation
        right_back = np.dot(self.rotation, right_back) + self.translation
        tail.set_data(np.array([left_back.item(0),left_front.item(0),right_front.item(0),right_back.item(0),left_back.item(0)]),
                            np.array([left_back.item(1),left_front.item(1),right_front.item(1),right_back.item(1),left_back.item(1)]))
        tail.set_3d_properties(np.array([left_back.item(2),left_front.item(2),right_front.item(2),right_back.item(2),left_back.item(2)]))

    def _draw_rudder(self, rudder):
        rudder_chord = self._fuselage_length/5
        rudder_height = self._fuselage_length/5
        bottom_back = np.array([[-self._fuselage_length/2],[0],[0]])
        bottom_front = np.array([[-self._fuselage_length/2+rudder_chord],[0],[0]])
        top = np.array([[-self._fuselage_length/2],[0],[-rudder_height]])
        bottom_back = np.dot(self.rotation, bottom_back) + self.translation
        bottom_front = np.dot(self.rotation, bottom_front) + self.translation
        top = np.dot(self.rotation, top) + self.translation
        rudder.set_data(np.array([bottom_back.item(0),bottom_front.item(0),top.item(0),bottom_back.item(0)]),
                           np.array([bottom_back.item(1),bottom_front.item(1),top.item(1),bottom_back.item(1)]))
        rudder.set_3d_properties(np.array([bottom_back.item(2),bottom_front.item(2),top.item(2),bottom_back.item(2)]))

