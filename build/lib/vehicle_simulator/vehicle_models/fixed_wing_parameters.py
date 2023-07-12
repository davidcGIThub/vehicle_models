import numpy as np
from dataclasses import dataclass


@dataclass
class FixedWingParameters:
    ######################################################################################
                    #   Physical Parameters
    ######################################################################################
    mass: float = 11. #kg
    Jx: float = 0.8244 #kg m^2
    Jy: float = 1.135
    Jz: float = 1.759
    Jxz: float = 0.1204
    S_wing: float = 0.55
    b: float = 2.8956
    c: float = 0.18994
    S_prop: float = 0.2027
    rho: float = 1.2682
    e: float = 0.9
    AR: float = (b**2) / S_wing
    gravity: float = 9.81

    ######################################################################################
                    #   Longitudinal Coefficients
    ######################################################################################
    C_L_0: float = 0.23
    C_D_0: float = 0.043
    C_m_0: float = 0.0135
    C_L_alpha: float = 5.61
    C_D_alpha: float = 0.03
    C_m_alpha: float = -2.74
    C_L_q: float = 7.95
    C_D_q: float = 0.0
    C_m_q: float = -38.21
    C_L_delta_e: float = 0.13
    C_D_delta_e: float = 0.0135
    C_m_delta_e: float = -0.99
    M: float = 50.0
    alpha0: float = 0.47
    epsilon: float = 0.16
    C_D_p: float = 0.0


    ######################################################################################
                    #   Lateral Coefficients
    ######################################################################################
    C_Y_0: float = 0.0
    C_ell_0: float = 0.0
    C_n_0: float = 0.0
    C_Y_beta: float = -0.98
    C_ell_beta: float = -0.13
    C_n_beta: float = 0.073
    C_Y_p: float = 0.0
    C_ell_p: float = -0.51
    C_n_p: float = 0.069
    C_Y_r: float = 0.0
    C_ell_r: float = 0.25
    C_n_r: float = -0.095
    C_Y_delta_a: float = 0.075
    C_ell_delta_a: float = 0.17
    C_n_delta_a: float = -0.011
    C_Y_delta_r: float = 0.19
    C_ell_delta_r: float = 0.0024
    C_n_delta_r: float = -0.069

    ######################################################################################
                    #   Propeller thrust / torque parameters (see addendum by McLain)
    ######################################################################################
    # Prop parameters
    D_prop: float = 20*(0.0254)     # prop diameter in m
    # Motor parameters
    KV_rpm_per_volt: float = 145.                            # Motor speed constant from datasheet in RPM/V
    KV: float = (1. / KV_rpm_per_volt) * 60. / (2. * np.pi)  # Back-emf constant, KV in V-s/rad
    KQ: float = KV                                           # Motor torque constant, KQ in N-m/A
    R_motor: float = 0.042              # ohms
    i0: float = 1.5                     # no-load (zero-torque) current (A)
    # Inputs
    ncells: float = 12.
    V_max: float = 3.7 * ncells  # max voltage for specified number of battery cells
    # Coeffiecients from prop_data fit
    C_Q2: float = -0.01664
    C_Q1: float = 0.004970
    C_Q0: float = 0.005230
    C_T2: float = -0.1079
    C_T1: float = -0.06044
    C_T0: float = 0.09357

    ######################################################################################
                    #   Calculation Variables
    ######################################################################################
    #   gamma parameters pulled from page 36 (dynamics)
    gamma: float = Jx * Jz - (Jxz**2)
    gamma1: float = (Jxz * (Jx - Jy + Jz)) / gamma
    gamma2: float = (Jz * (Jz - Jy) + (Jxz**2)) / gamma
    gamma3: float = Jz / gamma
    gamma4: float = Jxz / gamma
    gamma5: float = (Jz - Jx) / Jy
    gamma6: float = Jxz / Jy
    gamma7: float = ((Jx - Jy) * Jx + (Jxz**2)) / gamma
    gamma8: float = Jx / gamma

    #   C values defines on pag 62
    C_p_0: float         = gamma3 * C_ell_0      + gamma4 * C_n_0
    C_p_beta: float      = gamma3 * C_ell_beta   + gamma4 * C_n_beta
    C_p_p: float         = gamma3 * C_ell_p      + gamma4 * C_n_p
    C_p_r: float         = gamma3 * C_ell_r      + gamma4 * C_n_r
    C_p_delta_a: float    = gamma3 * C_ell_delta_a + gamma4 * C_n_delta_a
    C_p_delta_r: float    = gamma3 * C_ell_delta_r + gamma4 * C_n_delta_r
    C_r_0: float        = gamma4 * C_ell_0      + gamma8 * C_n_0
    C_r_beta: float      = gamma4 * C_ell_beta   + gamma8 * C_n_beta
    C_r_p: float         = gamma4 * C_ell_p      + gamma8 * C_n_p
    C_r_r: float         = gamma4 * C_ell_r      + gamma8 * C_n_r
    C_r_delta_a: float    = gamma4 * C_ell_delta_a + gamma8 * C_n_delta_a
    C_r_delta_r: float    = gamma4 * C_ell_delta_r + gamma8 * C_n_delta_r
