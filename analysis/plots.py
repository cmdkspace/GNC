# plots_trajectory, plot_attitude, plot_control, plot_cost_history, compare_cases

import numpy as np
import matplotlib.pyplot as plt

# ALTITUDE COMP
def plot_altComp(t_base, X_base, t_pd, X_pd, t_opt, X_opt):
    plt.figure()

    plt.plot(t_base, X_base[:, 2], label = "Baseline")
    plt.plot(t_pd, X_pd[:, 2], label = "PD Controlled")
    plt.plot(t_opt, X_opt[:, 2], label = "Optimized")

    plt.xlabel("Time (s)")
    plt.ylabel("Alt (m)")
    plt.title("Altitude Comparison")
    plt.legend()
    plt.grid()

    plt.show()

# Pitch v/s Time
def plot_pitch(t_arr, X_arr, label =""):

    # pitch from quaternioni
    q = X_arr[:, 6:10]
    pitch = 2 * np.arctan2(q[:, 1], q[:, 0]) #about x axis
    
    plt.figure()
    plt.plot(t_arr, pitch, label=label)

    plt.xlabel("Time (s)")
    plt.ylabel("Pitch (rad)")
    plt.title("Pitch vs Time")
    plt.grid()
    if label:
        plt.legend()

    plt.show()

# Control
def plot_control(t_arr, u_arr, label=""):

    plt.figure()

    plt.plot(t_arr, u_arr[:, 0], label="u1")
    plt.plot(t_arr, u_arr[:, 1], label="u2")

    plt.xlabel("Time (s)")
    plt.ylabel("Control Input")
    plt.title("Control Inputs vs Time")
    plt.grid()
    plt.legend()

    plt.show()