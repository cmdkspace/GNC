# optimized trajectory
# quasi linerized

import numpy as np
import matplotlib.pyplot as plt
from optimization.optimizer import run_optimizer
from optimization.cost_function import initial_state_build, control_law_build
from core.integrator import simulate

params = {
    'T_max' : 15000.0, 
    'Isp' : 260.0, 
    'I_body' : (2000.0, 2000.0, 200.0), 
    'CD' : 0.4, 
    'Aref' : np.pi * (0.5 / 2) ** 2, 
    'r_c2tvc' : 2.0, 
    'm_dry' : 50.0, 
    'm0' : 100.0
}

sim_params = {
    't_span' : (0.0, 60.0), 
    'dt' : 0.01, 
    'h_target' : 5000.0, 
    'w1' : 1.0, 
    'w2' : 0.01
}

q_ref = np.array([1.0, 0.0, 0.0, 0.0])

gains = {
    'Kp' : 0.6, 
    'Kd' : 0.3, 
    'delta_max' : 0.087
}

# initial guess
theta0 = np.array([0.05]) #small pitch
bounds = [(-0.2, 0.2)]

theta_opt, J_opt, result = run_optimizer(theta0, bounds, params, sim_params, gains, q_ref)

print("Optimal theta:", theta_opt)
print("Optimal cost:", J_opt)

# simulate
X0_opt = initial_state_build(theta_opt, params)
u_func = control_law_build(q_ref, gains)

t_arr, X_arr = simulate(
    X0_opt,
    u_func,
    sim_params['t_span'],
    sim_params['dt'],
    params
)

print("Max altitude:", np.max(X_arr[:, 2]))
print("Final altitude:", X_arr[-1, 2])

# plots
plt.figure()
plt.plot(t_arr, X_arr[:,2])
plt.xlabel("Time")
plt.ylabel("Altitude")
plt.title("Optimized Trajectory")
plt.grid()
plt.show()