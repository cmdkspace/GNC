# gravity, drag, thrust rotation

import numpy as np
from core.atmosphere import gravity_force, air_density, G0
from core.rocket_model import aero_drag

# Gravity tests
def test_gravity_direction_and_magnitude():
    m = np.random.uniform(1.0, 1000.0) 
    g = gravity_force(m)
    # direction
    assert np.allclose(g[:2], [0.0, 0.0])
    # z must be negative
    assert g[2] < 0.0
    # magnitude must be m*G0
    assert np.isclose(np.linalg.norm(g), m * G0)
    # finally, check the actual value of g
    assert np.allclose(g, np.array([0., 0., -m * 9.80665], dtype=float))

# Atmosphere tests 
def test_air_density_decay():
    heights = np.linspace(0, 50000, 100)
    densities = [air_density(h) for h in heights]
    assert all(densities[i] >= densities[i+1] for i in range(len(densities)-1))

    rho0 = air_density(0.0)
    assert np.isclose(rho0, 1.225, atol=1e-3)

def test_density_positive():
    for h in np.linspace(0, 50000, 100):
        assert air_density(h) > 0.0

# Drag Tests

def test_aero_drag_opposes_velocity():
    v = np.array([10.0, -5.0, 3.0])
    drag = aero_drag(v, Aref=1.0, CD=0.5, h=0.0)
    assert np.dot(drag, v) < 0.0

def test_drag_zero_when_velocity_zero():
    drag = aero_drag(np.zeros(3), Aref=1.0, CD=0.5, h=0.0)
    assert np.allclose(drag, np.zeros(3))

def test_drag_scales_with_velocity_squared():
    v1 = np.array([10.0, 0.0, 0.0])
    v2 = 2 * v1
    d1 = aero_drag(v1, Aref=1.0, CD=0.5, h=0.0)
    d2 = aero_drag(v2, Aref=1.0, CD=0.5, h=0.0)
    assert np.isclose(np.linalg.norm(d2), 4 * np.linalg.norm(d1), rtol=0.1)