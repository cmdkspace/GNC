# Rotation_matrix, quat_multiply, quat_conjugate, quat_normalize, quat_kinematics, quaternion_error

import numpy as np

def rotation_matrix(q):
    q0, q1, q2, q3 = q
    return np.array([
        [1-2*(q2**2+q3**2), 2*(q1*q2-q0*q3), 2*(q1*q3+q0*q2)],
        [2*(q1*q2+q0*q3), 1-2*(q1**2+q3**2), 2*(q2*q3-q0*q1)],
        [2*(q1*q3-q0*q2), 2*(q2*q3+q0*q1), 1-2*(q1**2+q2**2)]], dtype=float)

def quat_multiply(p, q):
    p0, p1, p2, p3 = p
    q0, q1, q2, q3 = q
    return np.array([
        p0*q0 - p1*q1 - p2*q2 - p3*q3,
        p0*q1 + p1*q0 + p2*q3 - p3*q2,
        p0*q2 - p1*q3 + p2*q0 + p3*q1,
        p0*q3 + p1*q2 - p2*q1 + p3*q0
    ], dtype=float)

def quat_conjugate(q):
    q0, q1, q2, q3 = q
    return np.array([q0, -q1, -q2, -q3], dtype=float)

def quat_normalize(q):
    n = np.linalg.norm(q)
    if n < 1e-12:
        raise ValueError("Cannot normalize a zero quaternion")
    return q / n

# quaternion kinematics
# omega(w) = 0 -w^T
#            w  -[w]x

def quat_kinematics(q, omega):
    q0, q1, q2, q3 = q
    wx, wy, wz = omega   # rotation about X, Y, Z (body frame)

    return 0.5 * np.array([
        -wx*q1 - wy*q2 - wz*q3,
         wx*q0 + wz*q2 - wy*q3,
         wy*q0 - wz*q1 + wx*q3,
         wz*q0 + wy*q1 - wx*q2
    ], dtype=float)

def quaternion_error(q, q_ref):
    # q_e = q_ref ^ {-1} x q
    q_ref_conj = quat_conjugate(q_ref)  # since q_ref is aunit quaternion, its inverse is its conjugate
    q_e = quat_multiply(q_ref_conj, q)
    if q_e[0] < 0:  # take shortest rotaion path
        q_e = -q_e
    return q_e  # q_e[1 : 4] is the error vector