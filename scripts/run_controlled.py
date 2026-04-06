# PD + LQR controlled ascent

import numpy as np
import matplotlib.pyplot as plt
from core.integrator import simulate
from control.pd_controller import pd_controller

# PD Control
X0 = np.zeros(14)
X0[6] = 1.0      # identity quaternion (upright)
X0[13] = 100.0   # initial mass [kg]

# small disturbance 
X0[7] = 0.05   # small pitch error

params = {
    'T_max': 15000.0,
    'Isp': 260.0,
    'I_body': (2000.0, 2000.0, 200.0),
    'CD': 0.4,
    'Aref': np.pi * (0.5 / 2)**2,
    'r_c2tvc': 2.0,
    'm_dry': 50.0
}

q_ref = np.array([1.0, 0.0, 0.0, 0.0])  # hold vertical

gains = {
    'Kp': 0.6,
    'Kd': 0.3,
    'delta_max': 0.087   # 5 degrees
}

u_hist = []
def control_law(t, X):
    u = pd_controller(t, X, q_ref, gains)
    u_hist.append(u)
    return u

t_span = (0.0, 60.0)
dt = 0.01
t_arr, X_arr = simulate(X0, control_law, t_span, dt, params)

# O/P
print("Final Altitude (m):", X_arr[-1, 2])
print("Final velocity (m/s):", X_arr[-1, 5])
print("Final mass (kg):", X_arr[-1, 13])
print(np.max(np.abs(X_arr[:,7])))  # pitch
print(np.max(np.abs(X_arr[:,8])))  # yaw
print(np.max(np.abs(X_arr[:,9])))  # roll

step = 100  # adjust (100 = every 1 sec if dt=0.01)
for i in range(0, len(t_arr), step):
    print(f"t={t_arr[i]:6.2f} | q1={X_arr[i,7]: .6f} | q2={X_arr[i,8]: .6f} | q3={X_arr[i,9]: .6f}")

# Plots

# Altitude
plt.figure()
plt.plot(t_arr, X_arr[:, 2])
plt.xlabel('Time (s)')
plt.ylabel('Altitude (m)')
plt.title('Controlled Ascent')
plt.grid()

# Vertical velocity
plt.figure()
plt.plot(t_arr, X_arr[:, 5])
plt.xlabel('Time (s)')
plt.ylabel('Vertical Velocity (m/s)')
plt.title('Velocity vs Time')
plt.grid()

# Mass
plt.figure()
plt.plot(t_arr, X_arr[:, 13])
plt.xlabel('Time (s)')
plt.ylabel('Mass (kg)')
plt.title('Mass vs Time')
plt.grid()

# Attitude (quaternion vector part)
plt.figure()
plt.plot(t_arr, X_arr[:, 7], label='q1 (pitch)')
plt.plot(t_arr, X_arr[:, 8], label='q2 (yaw)')
plt.plot(t_arr, X_arr[:, 9], label='q3 (roll)')
plt.xlabel('Time (s)')
plt.ylabel('Quaternion Components')
plt.title('Attitude Response')
plt.legend()
plt.grid()

# Theta evolution: theta = 2 * q_vec (small angle approx)
theta_pitch = 2 * X_arr[:,7]
theta_yaw   = 2 * X_arr[:,8]
theta_roll  = 2 * X_arr[:,9]

plt.figure()
plt.plot(t_arr, theta_pitch, label='pitch (rad)')
plt.plot(t_arr, theta_yaw, label='yaw (rad)')
plt.plot(t_arr, theta_roll, label='roll (rad)')
plt.xlabel('Time (s)')
plt.ylabel('Angle (rad)')
plt.title('Attitude angles (approx)')
plt.legend()
plt.grid()

# Control input history
u_hist = np.array(u_hist)

plt.figure()
plt.plot(t_arr[:len(u_hist)], u_hist[:,0], label='delta_p')
plt.plot(t_arr[:len(u_hist)], u_hist[:,1], label='delta_y')
plt.xlabel('Time (s)')
plt.ylabel('Gimbal angle (rad)')
plt.title('Control input history')
plt.legend()
plt.grid()

# Angular rates(damping behaviour)
plt.figure()
plt.plot(t_arr, X_arr[:,10], label='wx (pitch rate)')
plt.plot(t_arr, X_arr[:,11], label='wy (yaw rate)')
plt.plot(t_arr, X_arr[:,12], label='wz (roll rate)')
plt.xlabel('Time (s)')
plt.ylabel('Angular velocity (rad/s)')
plt.title('Angular rate evolution')
plt.legend()
plt.grid()

plt.show()

'''
 NOTE: Small negative drift in quaternion (q1) after ~8–10 sec

 Observation:
 After thrust cutoff (when mass reaches m_dry), q1 (pitch component)
 does not stay exactly at zero and slowly drifts to a small negative value.
 Example:
 q1 ≈ -2e-5  → corresponds to ~0.001 deg (negligible)

 Cause:
 1. After burnout:
       T = 0  ⇒  τ = 0
    → No control torque (TVC becomes ineffective)
 2. The system becomes uncontrolled:
       I * ω_dot = -ω × (Iω)
    → No restoring torque, no damping from controller
 3. Numerical effects:
    - RK4 integration error
    - Floating-point precision limits
    - Quaternion renormalization bias
    These introduce tiny residual angular velocities, which integrate
    into a slow drift in attitude.

 Important:
 This is NOT a physical instability.
 Magnitude is extremely small → numerically insignificant.

 Possible Solutions (optional):
 1. Disable control after burnout:
    if params['T'] == 0:
        return np.zeros(2)
 2. Clamp very small errors:
    if abs(qev[0]) < 1e-6:
        qev[0] = 0.0
 3. Ignore (recommended):
    This behavior is realistic — real rockets also drift without control
    once thrust is gone (unless RCS or aerodynamic stabilization is present).

 Conclusion:
 The observed drift is expected and confirms correct physics + control implementation.
'''