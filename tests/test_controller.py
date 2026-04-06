# zero-error -> zero command, convergence

import numpy as np
from control.pd_controller import pd_controller
from core.quaternion import quaternion_error

def test_quaternion_error_identity():
    q = np.array([1.0, 0.0, 0.0, 0.0])
    q_ref = np.array([1.0, 0.0, 0.0, 0.0])

    qe = quaternion_error(q, q_ref)

    assert np.allclose(qe, np.array([1.0, 0.0, 0.0, 0.0]), atol=1e-8)


def test_zero_error():
    x = np.zeros(14)
    x[6] = 1.0  # identity quaternion

    q_ref = np.array([1.0, 0.0, 0.0, 0.0])
    gains = {'Kp':1.0, 'Kd':0.1, 'delta_max':0.087}

    u = pd_controller(x, q_ref, gains)

    assert np.allclose(u, np.zeros(2), atol=1e-8)


def test_pitch_error():
    theta = 0.1

    x = np.zeros(14)
    x[6:10] = np.array([1.0, theta/2, 0.0, 0.0])  # pitch error

    q_ref = np.array([1.0, 0.0, 0.0, 0.0])
    gains = {'Kp':1.0, 'Kd':0.0, 'delta_max':1.0}

    u = pd_controller(x, q_ref, gains)

    assert u[0] < 0          # pitch correction opposes error
    assert abs(u[1]) < 1e-6  # no yaw command


def test_yaw_error():
    theta = 0.1

    x = np.zeros(14)
    x[6:10] = np.array([1.0, 0.0, theta/2, 0.0])  # yaw error

    q_ref = np.array([1.0, 0.0, 0.0, 0.0])
    gains = {'Kp':1.0, 'Kd':0.0, 'delta_max':1.0}

    u = pd_controller(x, q_ref, gains)

    assert u[1] < 0          # yaw correction opposes error
    assert abs(u[0]) < 1e-6  # no pitch command


def test_damping():
    x = np.zeros(14)
    x[6] = 1.0
    x[10] = 0.5  # pitch angular velocity

    q_ref = np.array([1.0, 0.0, 0.0, 0.0])
    gains = {'Kp':0.0, 'Kd':1.0, 'delta_max':1.0}

    u = pd_controller(x, q_ref, gains)

    assert u[0] < 0  # should oppose angular velocity


def test_saturation():
    theta = 1.0  # large error

    x = np.zeros(14)
    x[6:10] = np.array([1.0, theta/2, 0.0, 0.0])

    q_ref = np.array([1.0, 0.0, 0.0, 0.0])
    gains = {'Kp':10.0, 'Kd':0.0, 'delta_max':0.087}

    u = pd_controller(x, q_ref, gains)

    assert abs(u[0]) <= 0.087