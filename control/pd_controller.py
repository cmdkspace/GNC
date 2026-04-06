# pd_controller, quaternion PD

import numpy as np
from core.quaternion import quaternion_error

def pd_controller(x, q_ref, gains):
    q = x[6:10]
    omega = x[10:13]

    Kp = gains['Kp']
    Kd = gains['Kd']
    delta_max = gains['delta_max']

    qe = quaternion_error(q, q_ref)
    qev = qe[1:4]

    delta_p = -(Kp * qev[0] + Kd * omega[0])  # pitch
    delta_y = -(Kp * qev[1] + Kd * omega[1])  # yaw

    delta_p = np.clip(delta_p, -delta_max, delta_max)
    delta_y = np.clip(delta_y, -delta_max, delta_max)

    return np.array([delta_p, delta_y])