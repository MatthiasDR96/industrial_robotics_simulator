import matplotlib.pyplot as plt
import numpy as np

'''This script analyzes the error dynamics of a P controlled setpoint'''

# Desired joint position
t = np.arange(0, 1, 0.001)
theta_d = np.repeat(1, len(t))  # Constant one

# Params
theta_error_p_0 = 1
kp = 20.0

# Error dynamics
theta_error_p = theta_error_p_0 * np.exp(-kp * t)

# Plot position
plt.figure(1)
plt.plot(t, theta_d, label="Desired trajectory")
plt.plot(t, theta_d - theta_error_p, label="P controlled trajectory")
plt.xlabel("Time (s)")
plt.ylabel("Position")
plt.legend()

# Plot error
plt.figure(2)
zero_error = np.repeat(0, len(t))
plt.plot(t, zero_error, label="Desired error")
plt.plot(t, theta_error_p, label="P controlled error")
plt.xlabel("Time (s)")
plt.ylabel("Error")
plt.legend()
plt.show()