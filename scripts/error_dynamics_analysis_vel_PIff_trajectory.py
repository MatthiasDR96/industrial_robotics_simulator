import math
import matplotlib.pyplot as plt
import numpy as np

'''This script analyzes the error dynamics of a PI + feedforward controlled trajectory (moving reference point)'''

# Desired joint position
t = np.arange(0, 2, 0.001)
theta_d = np.sin(t)

# Error PI
theta_error_pi_0 = 1
kp_u = 11
ki_u = 144
kp_c = 30
ki_c = (kp_c/2)**2
wd = math.sqrt(ki_u-(kp_u**2/4))

# Error dynamics
theta_error_pi_underdamped = (theta_error_pi_0 * np.cos(wd * t) + (kp_u/math.sqrt(4*ki_u-kp_u**2)) * np.sin(
    wd * t)) * np.exp(-(kp_u/2) * t)
theta_error_pi_critical = (theta_error_pi_0 + theta_error_pi_0 * (kp_c/2) * t) * np.exp(-(kp_c/2) * t)

# Plot desired position
plt.figure(1)
plt.plot(t, theta_d, label="Desired trajectory")
plt.xlabel("Time (s)")
plt.ylabel("Position")

# Plot real position
plt.plot(t, theta_d - theta_error_pi_underdamped, label="PI underdamped")
plt.plot(t, theta_d - theta_error_pi_critical, label="PI critical damped")
plt.legend()

# Plot desired error
plt.figure(2)
zero_error = np.repeat(0, len(t))
plt.plot(t, zero_error, label="Desired error")
plt.xlabel("Time (s)")
plt.ylabel("Error")

# Plot error dynamics of PI control
plt.plot(t, theta_error_pi_underdamped, label="PI underdamped")
plt.plot(t, theta_error_pi_critical, label="PI critical damped")
plt.legend()
plt.show()
