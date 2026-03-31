# air_density, US std atm
import numpy as np

RHO0 = 1.225 # kg/m^3
Hs = 8500.0 # m
G0 = 9.80665 # m/s^2 comes with upto ~3% error, under 100km altitude, but we can ignore that for now, since we are not doing precision landing or something like that. We can also ignore the variation of G0 with altitude, since it is very small compared to the variation of air density with altitude.

def air_density(h):
    return RHO0 * np.exp(-h / Hs)

def gravity_force(m):
    return np.array([0., 0., -m * G0], dtype=float) #this in in NED, and Z is positive upwards, so gravity is negative in Z direction