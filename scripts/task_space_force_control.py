import math

from src.Simulator import *
from src.arms import *
from src.controllers.task_space_force_controller import *

""" This script uses force control and simulates contact with a wall at x=1.1m."""

# Create robot
q_init = np.array([[math.pi / 2], [-math.pi / 2]])
robot = TwoDofArm()
robot.set_q_init(q_init)

# Create controller
controller = Control(robot)

# Create desired force in task space (3N in positive x-direction)
f_des = np.array([[3.0], [0.0], [0.0], [0.0], [0.0], [0.0]])
controller.set_force_target(f_des)

# Run animation
joint_of_interest = 1
sim = Simulator(robot, controller, joint_of_interest)
sim.simulate()
