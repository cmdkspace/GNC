# rocket_ode (Master ODE), all forces/torques 

# NED frame is assumed to be inertial here

import numpy as np
from core.atmosphere import air_density, gravity_force, G0
from core.quaternion import rotation_matrix, quat_kinematics

# Convention: rotation out of the plane is +ve
# LIMIT: |delta| < 5deg = 0.0873 rad
def thrust_body(T, delta_p, delta_y):
    return T * np.array([
        -np.sin(delta_y), 
        np.sin(delta_p),
        np.cos(delta_p) * np.cos(delta_y)
    ], dtype=float)

def aero_drag(v, Aref, CD, h):
    rho = air_density(h)
    v_sq = np.dot(v, v)
    if v_sq < 1e-10:
        return np.zeros(3, dtype=float)
    v_mag = np.sqrt(v_sq)
    return -0.5 * rho * CD * Aref * v_mag * v # A vector is returned in NED
    # v elocity should later be depended on wind gusts, but for now we will ignore that, since it is usually much smaller than the velocity of the rocket itself, especially at high altitudes where the air density is low. We can also ignore the variation of CD with Mach number, since it is usually not very significant for our purposes, and it would require us to implement a more complex aerodynamic model, which we want to avoid for now.
# ===================================================================
# for some fuck's sake, it is very counter intutive to me
def tvc_torque(T, delta_p, delta_y, r_c2tvc):
    Tb = thrust_body(T, delta_p, delta_y)
    # handles both scalar and vector values
    if np.isscalar(r_c2tvc):
        r_eng = np.array([0.0, 0.0, -r_c2tvc])
    else:
        r_eng = r_c2tvc
    return np.cross(r_eng, Tb)
# ===================================================================

def thrust_NED(T, delta_p, delta_y, q):
    Tb = thrust_body(T, delta_p, delta_y)
    return rotation_matrix(q) @ Tb

def rot_dynamics_euler(omega, I_body, tau):
    wx, wy, wz = omega
    Ixx, Iyy, Izz = I_body #tuple(Ixx, Iyy, Izz)    # ASSUMPTION: I is caliberated as a diagonal matrix
    wx_dot = ((Iyy - Izz) * wy * wz + tau[0]) / Ixx
    wy_dot = ((Izz - Ixx) * wz * wx + tau[1]) / Iyy
    wz_dot = ((Ixx - Iyy) * wx * wy + tau[2]) / Izz
    return np.array([wx_dot, wy_dot, wz_dot], dtype=float)

def mass_rate(T, Isp):
    return -T/(Isp * G0)  

# prototype without params
def rocket_ode(T, X, u, params):
    r = X[0:3]
    v = X[3:6]
    q = X[6:10]
    omega = X[10:13]
    m = X[13]

    del_p, del_y = u

    Isp, I_body, Aref, CD, r_c2tvc = (params['Isp'], params['I_body'], params['Aref'], params['CD'], params['r_c2tvc'])

    if m > params['m_dry']:
        T = params['T_max']
    else:
        T = 0.0

    h = r[2]

    F_thrust = thrust_NED(T, del_p, del_y, q)
    F_g = gravity_force(m)
    F_aero = aero_drag(v, Aref, CD, h)
    F_total = F_thrust + F_g + F_aero

    tau_tvc = tvc_torque(T, del_p, del_y, r_c2tvc)
    # for later versions, we can add aero torque as well, but for now we will ignore it. We can also ignore the torque due to gravity, since it is usually much smaller than the TVC torque as well.
    # tau_aero = np.cross(r, F_aero) # ASSUMPTION: the aerodynamic force is applied at the center of pressure, which is assumed to be at the center of mass. This is a simplification, but it should be fine for our purposes.
    # tau_total = tau_tvc + tau_aero

    # derivatives
    r_dot = v
    v_dot = F_total / m
    q_dot = quat_kinematics(q, omega)
    # omega_dot = rot_dynamics_euler(omega, I_body, tau_total)
    omega_dot = rot_dynamics_euler(omega, I_body, tau_tvc)
    m_dot = mass_rate(T, Isp)

    return np.concatenate([r_dot, v_dot, q_dot, omega_dot, [m_dot]])