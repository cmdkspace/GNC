# compute_lqr, linearization, apply_lqr

import numpy as np
from scipy.linalg import solve_continuous_are

def lqr_gain(A, B, Q, R):
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ B.T @ P
    return K

def lqr_controller_build(params, gains):

    Ixx, Iyy, Izz = params['I_body']
    l = params['r_c2tvc']
    T = params['T_max']

    A = np.array([
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ])

    B = np.array([
        [0, 0],
        [0, 0],
        [T * l / Ixx, 0],
        [0, T * l / Iyy]
    ])

    Q = gains['Q']
    R = gains['R']

    K = lqr_gain(A, B, Q, R)

    return K

def lqr_controller(t, X, q_ref, K, params, gains):

    # Extract quaternion
    q = X[6:10]

    # Small-angle approx error (only x-axis rotation used)
    theta_error = np.array([
        2 * (q[1] - q_ref[1]),
        2 * (q[2] - q_ref[2])
    ])

    omega = X[10:12]

    x = np.hstack((theta_error, omega))

    u = -K @ x

    # Saturation
    delta_max = gains['delta_max']
    u = np.clip(u, -delta_max, delta_max)

    return u