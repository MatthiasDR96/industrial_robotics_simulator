import math

from src.Simulator import *
from src.arms import *

'''This script illustrates the dynamic behavior of a robot model without control inputs '''

# Create robot
q_init = np.array([[-math.pi / 10]])
robot = OneDofArm()
robot.set_q_init(q_init)

# Create simulator without controller
sim = Simulator(robot)

# Simulate dynamics
sim.simulate()

# OR Step (10 times) through the dynamics
# dt = 0.1
# joint_torque = np.zeros((2, 1))
# for i in range(10):
# robot.step(joint_torque, dt)
# robot.plot()
# plt.show()
