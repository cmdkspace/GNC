import numpy as np
from core.integrator import simulate

G0 = 9.80665

def base_params():
    return {
        'T_max': 1000.0,   # constant thrust
        'Isp': 300.0,
        'I_body': (100.0, 100.0, 100.0),
        'CD': 0.0,         # no drag
        'Aref': 1.0,
        'r_c2tvc': np.array([0.0, 0.0, -1.0]),
        'm_dry': 50.0
    }

def base_state(m0):
    x = np.zeros(14)
    x[6] = 1.0      # identity quaternion
    x[13] = m0
    return x

def zero_control(t, x):
    return np.zeros(2)

# ---------- TEST 1: Tsiolkovsky Delta-V ----------
def test_tsiolkovsky_delta_v():
    # With gravity present:
    # Δv_sim + gravity_loss ≈ Δv_theoretical

    params = base_params()
    m0 = 100.0
    x0 = base_state(m0)
    dt = 0.01
    t_span = (0.0, 20.0)
    t_arr, x_arr = simulate(x0, zero_control, t_span, dt, params)
    # Simulated delta-v (Z direction)
    dv_sim = x_arr[-1, 5] - x_arr[0, 5]
    t_total = t_arr[-1] - t_arr[0]
    g_loss = 9.80665 * t_total #gravity losses

    # Compensated delta-v
    dv_corrected = dv_sim + g_loss
    m_final = x_arr[-1, 13]
    dv_theo = params['Isp'] * 9.80665 * np.log(m0 / m_final)
    assert abs(dv_corrected - dv_theo) / dv_theo < 0.05
    
# ---------- TEST 2: Mass depletion ----------
def test_mass_monotonic_and_rate():
    params = base_params()
    m0 = 100.0
    x0 = base_state(m0)
    dt = 0.01
    t_span = (0.0, 5.0) # sim duration in sec is 5 sec, which is enough to see a significant mass depletion without burning all the fuel (since we have T_max=1000 N and Isp=300 s, we will burn about 16.7 kg in 5 seconds, which is a good test case)
    t_arr, x_arr = simulate(x0, zero_control, t_span, dt, params)
    m_arr = x_arr[:, 13]

    # --- 1. Monotonic decrease ---
    assert np.all(np.diff(m_arr) <= 1e-10) 
    # --- 2. Rate check ---
    dm_expected = -params['T_max'] / (params['Isp'] * G0)
    dm_actual = np.mean(np.diff(m_arr) / dt)
    assert abs(dm_actual - dm_expected) / abs(dm_expected) < 0.01