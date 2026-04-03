import numpy as np
from core.integrator import rk4
from core.rocket_model import rocket_ode


def base_params():
    return {
        'T_max': 0.0,
        'Isp': 300.0,
        'I_body': (100.0, 100.0, 100.0),
        'CD': 0.0,
        'Aref': 1.0,
        'r_c2tvc': np.array([0.0, 0.0, -1.0]),
        'm_dry': 50.0
    }


def test_constant_acceleration_motion():
    """
    With only gravity → motion should follow:
    z = 0.5 * g * t^2
    """
    x = np.zeros(14)
    x[6] = 1.0
    x[13] = 100.0

    params = base_params()
    u = np.zeros(2)

    dt = 0.01
    t = 0.0

    steps = 1000  # 10 seconds

    for _ in range(steps):
        x = rk4(rocket_ode, t, x, dt, u, params)
        t += dt

    z_sim = x[2]
    t_total = steps * dt

    z_expected = -0.5 * 9.80665 * t_total**2

    assert np.isclose(z_sim, z_expected, rtol=1e-2)


def test_velocity_growth_under_gravity():
    """
    v = g * t
    """
    x = np.zeros(14)
    x[6] = 1.0
    x[13] = 100.0

    params = base_params()
    u = np.zeros(2)

    dt = 0.01
    t = 0.0
    steps = 1000

    for _ in range(steps):
        x = rk4(rocket_ode, t, x, dt, u, params)
        t += dt

    vz_sim = x[5]
    t_total = steps * dt

    vz_expected = -9.80665 * t_total

    assert np.isclose(vz_sim, vz_expected, rtol=1e-2)


def test_quaternion_norm_preserved():
    """
    Quaternion must remain unit length after integration
    """
    x = np.zeros(14)
    x[6] = 1.0
    x[13] = 100.0

    params = base_params()
    u = np.zeros(2)

    dt = 0.01
    t = 0.0

    for _ in range(500):
        x = rk4(rocket_ode, t, x, dt, u, params)
        t += dt

    q = x[6:10]

    assert np.isclose(np.linalg.norm(q), 1.0, atol=1e-6)