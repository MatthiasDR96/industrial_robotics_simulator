# Imports
import numpy as np
import matplotlib.pyplot as plt
from industrial_robotics_simulator.Interpolator import *

# Params
q1 = np.array([0, 0, 0])
q2 = np.array([1, 2, 3])
T = 4
k_traj = 100
t = np.linspace(0, 4, 100)

# Interpolate
traj, dtraj, ddtraj, dddtraj = interpolate_cubic(q1, q2, k_traj, T)
traj, dtraj, ddtraj, dddtraj = interpolate_quintic(q1, q2, k_traj, T)
traj, dtraj, ddtraj, dddtraj = interpolate_septic(q1, q2, k_traj, T)
traj, dtraj, ddtraj, dddtraj = interpolate_nonic(q1, q2, k_traj, T)

print(traj[0])

# Plot Position
plt.figure()
plt.subplot(1, 4, 1)
plt.plot(t, traj[0])
plt.plot(t, traj[1])
plt.plot(t, traj[2])
plt.subplot(1, 4, 2)
plt.plot(t, dtraj[0])
plt.plot(t, dtraj[1])
plt.plot(t, dtraj[2])
plt.subplot(1, 4, 3)
plt.plot(t, ddtraj[0])
plt.plot(t, ddtraj[1])
plt.plot(t, ddtraj[2])
plt.subplot(1, 4, 4)
plt.plot(t, dddtraj[0])
plt.plot(t, dddtraj[1])
plt.plot(t, dddtraj[2])
plt.title("Position")
plt.xlabel('t (s)')
plt.ylabel('q (rad or m)')
plt.show()