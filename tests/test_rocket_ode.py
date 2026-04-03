import numpy as np
from core.rocket_model import rocket_ode

def base_params():
    return {
        'T_max': 0.0,
        'Isp': 300.0,
        'I_body': (100.0, 100.0, 100.0),
        'CD': 0.5,
        'Aref': 1.0,
        'r_c2tvc': 1.0,
        'm_dry': 50.0
    }

def base_state():
    x = np.zeros(14)
    x[6] = 1.0      # identity quaternion
    x[13] = 100.0   # mass
    return x

# ---------- TESTS ----------

def test_position_derivative_equals_velocity():
    """
    r_dot = v must ALWAYS hold
    """
    x = base_state()
    x[3:6] = np.array([10.0, -5.0, 2.0])

    u = np.zeros(2)
    params = base_params()

    x_dot = rocket_ode(0.0, x, u, params)

    assert np.allclose(x_dot[0:3], x[3:6])


def test_gravity_only_when_velocity_zero():
    """
    If v=0 → drag=0 automatically
    If thrust=0 → only gravity acts
    This must hold regardless of CD/Aref
    """
    x = base_state()

    u = np.zeros(2)
    params = base_params()

    # make drag "large" to ensure it doesn't sneak in
    params['CD'] = 100.0
    params['Aref'] = 10.0

    x_dot = rocket_ode(0.0, x, u, params)

    acc = x_dot[3:6]

    assert np.allclose(acc, np.array([0.0, 0.0, -9.80665]), atol=1e-6)


def test_drag_opposes_velocity():
    """
    Drag must always oppose velocity (dissipative force)
    """
    x = base_state()
    x[3:6] = np.array([50.0, -20.0, 10.0])

    u = np.zeros(2)
    params = base_params()

    x_dot = rocket_ode(0.0, x, u, params)

    acc = x_dot[3:6]

    # drag component must oppose velocity → dot product negative
    assert np.dot(acc, x[3:6]) < 0


def test_drag_scales_quadratically():
    """
    Doubling velocity → ~4x drag magnitude
    """
    x1 = base_state()
    x1[3:6] = np.array([10.0, 0.0, 0.0])

    x2 = base_state()
    x2[3:6] = np.array([20.0, 0.0, 0.0])

    u = np.zeros(2)
    params = base_params()

    a1 = rocket_ode(0.0, x1, u, params)[3:6]
    a2 = rocket_ode(0.0, x2, u, params)[3:6]

    # remove gravity contribution (only compare drag effect)
    g = np.array([0.0, 0.0, -9.80665])
    d1 = a1 - g
    d2 = a2 - g

    assert np.isclose(np.linalg.norm(d2), 4 * np.linalg.norm(d1), rtol=0.2)


def test_mass_decreases_with_thrust():
    """
    With thrust → fuel must burn → m_dot < 0
    """
    x = base_state()

    u = np.zeros(2)
    params = base_params()
    params['T_max'] = 1000.0

    x_dot = rocket_ode(0.0, x, u, params)

    assert x_dot[13] < 0


def test_no_thrust_no_mass_change():
    """
    No thrust → no fuel burn
    """
    x = base_state()

    u = np.zeros(2)
    params = base_params()

    x_dot = rocket_ode(0.0, x, u, params)

    assert np.isclose(x_dot[13], 0.0)


def test_zero_tvc_no_angular_acceleration():
    """
    No gimbal → no torque → no angular acceleration
    """
    x = base_state()

    u = np.array([0.0, 0.0])
    params = base_params()

    x_dot = rocket_ode(0.0, x, u, params)

    omega_dot = x_dot[10:13]

    assert np.allclose(omega_dot, np.zeros(3), atol=1e-8)


def test_quaternion_derivative_zero_when_no_rotation():
    """
    If angular velocity = 0 → quaternion should not change
    """
    x = base_state()

    u = np.zeros(2)
    params = base_params()

    x_dot = rocket_ode(0.0, x, u, params)

    q_dot = x_dot[6:10]

    assert np.allclose(q_dot, np.zeros(4), atol=1e-8)