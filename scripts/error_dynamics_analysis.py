import math

import matplotlib.pyplot as plt
import numpy as np

'''This script compares the error dynamics of a P and a PI controlled trajectory (moving reference point)'''

# Desired joint position
t = np.arange(0, 1, 0.001)
theta_d = t

# Error P
theta_error_p_0 = -0.2
kp = 20.0
theta_error_p = (1 / kp) + (theta_error_p_0 - (1 / kp)) * np.exp(-kp * t)

# Error PI
theta_error_pi_0 = -0.2
ki = 250
theta_error_pi = (-0.2 * np.cos(math.sqrt(ki - 100) * t) + ((10 / math.sqrt(ki)) / math.sqrt(1 - (100 / ki))) * np.sin(
    math.sqrt(ki - 100) * t)) * np.exp(-10 * t)

# Plot desired position
plt.figure(1)
plt.plot(t, theta_d, label="Desired trajectory")
plt.xlabel("Time (s)")
plt.ylabel("Position")

# Plot real position P
plt.plot(t, theta_d - theta_error_p, label="P controlled trajectory")

# Plot real position PI
plt.plot(t, theta_d - theta_error_pi, label="PI controlled trajectory")
plt.legend()

# Plot desired error
plt.figure(2)
zero_error = np.repeat(0, len(t))
plt.plot(t, zero_error, label="Desired error")
plt.xlabel("Time (s)")
plt.ylabel("Error")

# Plot error dynamics of P control
plt.plot(t, theta_error_p, label="P controlled error")

# Plot error dynamics of PI control
plt.plot(t, theta_error_pi, label="PI controlled error")
plt.legend()
plt.show()
