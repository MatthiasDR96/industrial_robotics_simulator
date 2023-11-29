import matplotlib.pyplot as plt
import numpy as np

from industrial_robotics_simulator.arms import *

'''This script calculates the inverse kinematics of a robot model '''

# Create robot
robot = TwoDofArm()

# Print robot information
print(robot)

# Cartesian space configuration
xy = [0, 1.5]

# Inverse kinematics
q_init = np.array([0.0, 0.0])
q = robot.inverse_kinematics(xy, q_init)
print("Joint position:")
print("\tq1: " + str(q[0]))
print("\tq2: " + str(q[1]))

# Plot robot
robot.plot(q)
plt.show()
