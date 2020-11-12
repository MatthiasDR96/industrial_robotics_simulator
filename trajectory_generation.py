import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Params
q0 = np.array([[0], [0], [0]])
q1 = np.array([[1], [2], [3]])
T = 4
f = 10

# Time vector
t = np.linspace(0, T, f*T)

# Position interpolation
X = q0 + (q1 - q0) * ((3 / T ** 2) * t ** 2 - (2 / T ** 3) * t ** 3)
Xdot = ((6 / T ** 2) * t - (6 / T ** 3) * t ** 2) * (q1 - q0)
Xdotdot = ((6 / T ** 2) - (12 / T ** 3) * t) * (q1 - q0)

# Plot trajectory
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(X[0], X[1], X[2], '.')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('Position trajectory')

# Plot Position
plt.figure()
plt.subplot(1, 3, 1)
plt.plot(t, X[0], '.', label='x')
plt.plot(t, X[1], '.', label='y')
plt.plot(t, X[2], '.', label='z')
plt.title("Position")
plt.xlabel('t (s)')
plt.ylabel('x, y, and z (m)')
plt.legend()

# Plot Velocity
plt.subplot(1, 3, 2)
plt.plot(t, Xdot[0], '.', label='x')
plt.plot(t, Xdot[1], '.', label='y')
plt.plot(t, Xdot[2], '.', label='z')
plt.title("Velocity")
plt.xlabel('t (s)')
plt.ylabel('xdot, ydot, and zdot (m/s)')
plt.legend()

# Plot Acceleration
plt.subplot(1, 3, 3)
plt.plot(t, Xdotdot[0], '.', label='x')
plt.plot(t, Xdotdot[1], '.', label='y')
plt.plot(t, Xdotdot[2], '.', label='z')
plt.title("Acceleration")
plt.xlabel('t (s)')
plt.ylabel('xdotdot, ydotdot, and zdotdot (m/s^2)')
plt.legend()
plt.tight_layout()

plt.show()