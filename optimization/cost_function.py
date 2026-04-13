# altitude_cost, control_effort, combined ...

# In later versions, include gravity turn timing and pitch program as well for optimization

import numpy as np
from control.pd_controller import pd_controller
from core.integrator import simulate

# init state
def initial_state_build(theta, params):
    # pitch = theta[0]
    pitch0 = 0.0 #launch vertical

    X0 = np.zeros(14)
    # X0[6] = np.cos(pitch/2)  # identity quaternion with pitch error
    # X0[7] = np.sin(pitch/2) # small pitch error, about x axis
    X0[6] = np.cos(pitch0/2)  # identity quaternion with pitch error
    X0[7] = np.sin(pitch0/2) # small pitch error, about x axis
    X0[13] = params['m0']
    return X0

# pitch profile(linear ramp)
def pitch_profile(t, theta):    #modifiable
    theta_max = theta[0]
    t_turn = theta[2]
    t_ramp = theta[3]

    if t < t_turn: return 0.0
    elif t < t_turn + t_ramp: return theta_max * (t - t_turn)/t_ramp
    else: return theta_max

def control_law_build(gains, theta):
    t_cutoff = theta[1]
    def control(t, X):
        if t > t_cutoff:
            return np.zeros(2)
        
        pitch_ref = pitch_profile(t, theta)

        q_ref = np.array([
            np.cos(pitch_ref / 2),
            np.sin(pitch_ref / 2), 
            0.0,
            0.0
        ])

        return pd_controller(t, X, q_ref, gains)
    return control

def cost_function(theta, params, sim_params, gains, q_ref_unused = None):
    
    params_local = dict(params)
    params_local['t_cutoff'] = theta[1]

    X0 = initial_state_build(theta, params)
    u_func = control_law_build(gains, theta)

    t_arr, X_arr = simulate(X0, u_func, sim_params['t_span'], sim_params['dt'], params_local)

    # Final altitude
    # h_final = X_arr[-1, 2] # for minimizing final altitude penalty
    h_final = np.max(X_arr[:, 2]) #for minimizing max altitude

    # Control effort
    u_arr = np.array([u_func(t, X_arr[i]) for i, t in enumerate(t_arr)])
    ctrl_cost  = np.trapezoid(u_arr[:, 0]**2 + u_arr[:, 1]**2, t_arr)

    # Total Cost
    J = sim_params['w1'] * (sim_params['h_target'] - h_final)**2 + sim_params['w2'] * ctrl_cost

    return J