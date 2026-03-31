# rocket_ode (Master ODE), all forces/torques 

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
# ===================================================================
# for some fuck's sake, it is very counter intutive to me
def tvc_torque(T, delta_p, delta_y, r_c2tvc):
    Tb = thrust_body(T, delta_p, delta_y)
    r_eng = np.array([0, 0, -r_c2tvc])
    return np.cross(r_eng, Tb)
# ===================================================================