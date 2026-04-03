# open-loop vertical ascent
import numpy as np
import matplotlib.pyplot as plt
from core.integrator import simulate


#Control (no input)
def zero_control(t, X):
    return np.zeros(2)

# Initial state
X0 = np.zeros(14)
X0[6] = 1.0 # no rotation
X0[13] = 100.0 # initial mass[kg]

# Params
params = {
    'T_max': 15000.0,
    'Isp': 260.0,
    'I_body': (2000.0, 2000.0, 200.0),
    'CD': 0.4,
    'Aref': np.pi * (0.5 / 2)**2,   # from diameter = 0.5 m
    'r_c2tvc': 2.0,                 # scalar (your current setup)
    'm_dry': 50.0
}

# Simulate
t_span = (0.0, 60.0)
dt = 0.01
t_arr, X_arr = simulate(X0, zero_control, t_span, dt, params)

# O/P (sanity check)
print("Final Altitude (m):", X_arr[-1, 2])
print("Final velocity (m/s):", X_arr[-1, 5])
print("Final mass (kg):", X_arr[-1, 13])

# Plots(debugging level)
# altitude
plt.figure()
plt.plot(t_arr, X_arr[:, 2])
plt.xlabel('Time (s)')
plt.ylabel('Altitude (m)')
plt.title('Baseline Ascent')
plt.grid()

# velocity
plt.figure()
plt.plot(t_arr, X_arr[:, 5])
plt.xlabel('Time (s)')
plt.ylabel('Vertical Velocity (m/s)')
plt.title('Velocity v/s time')
plt.grid()

# mass
plt.figure()
plt.plot(t_arr, X_arr[:, 13])
plt.xlabel('Time (s)')
plt.ylabel('Mass (kg)')
plt.title('Mass v/s time')
plt.grid()

plt.show()