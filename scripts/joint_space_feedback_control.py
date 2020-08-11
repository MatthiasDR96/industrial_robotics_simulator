from src.Simulator import *
from src.arms import *
from src.controllers.joint_space_feedback_controller import *

""" This script uses feedback motion control to move to a target."""

# Create robot
q_init = np.array([[-math.pi / 2], [0.0]])
robot = TwoDofArm()
robot.set_q_init(q_init)

# Create controller
controller = Control(robot)

# Create target pose in joint space
q_des = np.array([[-math.pi / 4], [0.0]])
controller.set_joint_space_target(q_des)

# Run animation
joint_of_interest = 1
sim = Simulator(robot, controller, joint_of_interest)
sim.simulate()
