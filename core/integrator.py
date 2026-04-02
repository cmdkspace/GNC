# RK4_step, simulate
import numpy as np
from core.quaternion import quat_normalize
from core.rocket_model import rocket_ode

def rk4(f, t, X, dt, u, params):
    # f is the function that computes the derivative of x, given t, x, u, and params and it will be of the form X_dot = f(t, x, u, params)
    # rocket_ode will be the function that computes the derivative of x, given t, x, u, and params and it will be of the form X_dot = rocket_ode(t, x, u, params) and in it the x will be 14 dimensional vector containing r, v, q, omega, m. SO, now we have 14 stare vectors, a control variable,a time, and params will be having: T, Isp, I_body, Aref, CD, r_c2tvc
    k1 = f(t, X, u, params)
    k2 = f(t + dt/2, X + dt/2 * k1, u, params)
    k3 = f(t + dt/2, X + dt/2 * k2, u, params)
    k4 = f(t + dt, X + dt * k3, u, params)
    X_new = X + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
    # NORMALIZATION OF QUATERNION AFTER EVERY STEP TO AVOID DRIFT
    X_new[6:10] = quat_normalize(X_new[6:10])
    return X_new

def simulate(X0, u_func, t_span, dt, params):
    t0, tf = t_span; t = t0; X = X0.copy()
    t_hist, X_hist = [t], [X.copy()]
    p = dict(params) # make a copy of params to avoid modifying the original one
    while t < tf:
        u = u_func(t, X) 
        if X[13] <= p['m_dry']: 
            X[13] = p['m_dry']
        X = rk4(rocket_ode, t, X, dt, u, p)
        t += dt
        if X[2] < 0 and t > 1: break
        t_hist.append(t); X_hist.append(X.copy())
    return np.array(t_hist), np.array(X_hist)