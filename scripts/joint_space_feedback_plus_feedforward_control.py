from industrial_robotics_simulator.Simulator import *
from industrial_robotics_simulator.arms import *
from industrial_robotics_simulator.controllers.joint_space_feedback_plus_feedforward_controller import *

""" This script uses feedback plus feedforward motion control to move to a target."""

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
