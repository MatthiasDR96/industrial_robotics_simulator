from math import *

import matplotlib.pyplot as plt

from src.arms import *

'''This script calculates the forward kinematics of a robot model '''

# Create robot
robot = TwoDofArm()

# Print robot information
print(robot)

# Joint space configuration
q = [pi / 4, pi / 4]

# Forward kinematics
fk = robot.forward_kinematics(q)
print("Cartesian position:")
print("\tx: " + str(fk[0][0]))
print("\ty: " + str(fk[1][0]))

# Plot robot
robot.plot(q)
plt.show()
