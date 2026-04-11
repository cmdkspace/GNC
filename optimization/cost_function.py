# altitude_cost, control_effort, combined ...

# In later versions, include gravity turn timing and pitch program as well for optimization

import numpy as np
from control.pd_controller import pd_controller
from core.integrator import simulate

# init state
def initial_state_build(theta, params):
    pitch = theta[0]
    X0 = np.zeros(14)
    X0[6] = np.cos(pitch/2)  # identity quaternion with pitch error
    X0[7] = np.sin(pitch/2) # small pitch error, about x axis
    X0[13] = params['m0']
    return X0

def control_law_build(q_ref, gains, theta):
    t_cutoff = theta[1]
    def control(t, X):
        if t > t_cutoff:
            return np.zeros(2)
        
        return pd_controller(t, X, q_ref, gains)
    return control

def cost_function(theta, params, sim_params, gains, q_ref):
    
    params_local = dict(params)
    params_local['t_cutoff'] = theta[1]

    X0 = initial_state_build(theta, params)
    u_func = control_law_build(q_ref, gains, theta)

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