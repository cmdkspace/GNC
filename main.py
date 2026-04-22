#run all three cases, saves plots

import numpy as np
from core.integrator import simulate
from control.pd_controller import pd_controller
from optimization.optimizer import run_optimizer
from optimization.cost_function import initial_state_build, control_law_build
from analysis.plots import plot_altComp
from control.lqr import lqr_controller, lqr_controller_build


params  = {
    'T_max': 15000.0,
    'Isp': 260.0,
    'I_body': (2000.0, 2000.0, 200.0),
    'CD': 0.4,
    'Aref': np.pi * (0.5 / 2)**2,
    'r_c2tvc': 2.0,
    'm_dry': 50.0,
    'm0': 100.0
}

sim_params = {
    't_span': (0.0, 60.0),
    'dt': 0.01,
    'h_target': 5000.0,
    'w1': 10.0,     # tuned
    'w2': 0.01
}

gains = {
    'Kp' : 0.6,
    'Kd' : 0.3,
    'delta_max' : 0.087
}

lqr_gains = {
    'Q': np.diag([10, 10, 1, 1]),
    'R': np.diag([1, 1]),
    'delta_max': 0.087
}

K = lqr_controller_build(params, lqr_gains)

# baseline
def run_baseline():
    X0 = np.zeros(14)
    X0[6] = 1.0 # no rotation
    X0[13] = params['m0'] # initial mass[kg]
    
    def u_zero(t, x):
        return np.zeros(2)
    
    t_arr, X_arr = simulate(X0, u_zero, sim_params['t_span'], sim_params['dt'], params)

    print("Baseline max altitude:", np.max(X_arr[:, 2]))

    return t_arr, X_arr

# PD Controlled (non - optimal)
def run_pd():

    X0 = initial_state_build([0, 0, 0, 0], params)

    def u_func(t, X):
        q_ref = np.array([1.0, 0.0, 0.0, 0.0])
        return pd_controller(t, X, q_ref, gains)
    
    t_arr, X_arr = simulate(
        X0,
        u_func,
        sim_params['t_span'],
        sim_params['dt'],
        params
    )

    print("PD max altitude:", np.max(X_arr[:, 2]))

    return t_arr, X_arr

# LQR Controlled
def run_lqr():

    X0 = np.zeros(14)
    X0[6] = 1.0
    X0[13] = params['m0']

    def u_func(t, X):
        q_ref = np.array([1.0, 0.0, 0.0, 0.0])
        return lqr_controller(t, X, q_ref, K, params, lqr_gains)

    t_arr, X_arr = simulate(
        X0,
        u_func,
        sim_params['t_span'],
        sim_params['dt'],
        params
    )

    print("LQR max altitude:", np.max(X_arr[:, 2]))

    return t_arr, X_arr

# optimized

def run_optimized():
    
    burn_time = (params['m0'] - params['m_dry']) / (
        params['T_max'] / (params['Isp'] * 9.81)
    )

    theta0 = np.array([0.05, 6.0, 2.0, 2.0])
    
    bounds = [
        (-0.08, 0.08),     # θ_max
        (1.0, burn_time),  # cutoff
        (0.5, 5.0),        # t_turn
        (0.5, 5.0)         # t_ramp
    ]

    theta_opt, J_opt, _ = run_optimizer(theta0, bounds, params, sim_params, gains, None)

    print("theta_opt: ", theta_opt)
    print("cost: ", J_opt)

    X0 = initial_state_build(theta_opt, params)
    u_func = control_law_build(gains, theta_opt)

    params_local = dict(params)
    params_local['t_cutoff'] = theta_opt[1]

    t_arr, X_arr = simulate(
        X0,
        u_func,
        sim_params['t_span'],
        sim_params['dt'],
        params_local
    )

    print("Optimized max altitude:", np.max(X_arr[:, 2]))

    return t_arr, X_arr

# main
def main():
    t_base, X_base = run_baseline()
    t_pd, X_pd = run_pd()
    t_lqr, X_lqr = run_lqr()
    t_opt, X_opt = run_optimized()

    plot_altComp(
        t_base, X_base,
        t_pd, X_pd,
        t_lqr, X_lqr,
        t_opt, X_opt
    )

if __name__ == "__main__": 
    main()