from src.Simulator import *
from src.arms import *
from src.controllers.joint_space_impedance_controller import *

""" This script uses joint impedance control to move to a target under an external force."""

# Create robot
q_init = np.array([[math.pi / 2], [-math.pi / 2]])
robot = TwoDofArm()
robot.set_q_init(q_init)

# Create controller
controller = Control(robot)

# Create target pose in joint space
q_des = np.array([[math.pi / 2], [-math.pi / 2]])
controller.set_joint_space_target(q_des)

# Create external force in joint space (20Nm about joint 1)
frames = 100  # Amount of animation iterations before restart
torque_joint1 = np.repeat(20, frames)
torque_joint1[0:20] = 0  # Start torque at frame 20
torque_joint1[50:-1] = 0  # End torque at frame 50
torque_joint2 = np.repeat(0, frames)
torque = np.vstack((torque_joint1, torque_joint2))
controller.set_external_force_function(torque)

# Run animation
joint_of_interest = 1
sim = Simulator(robot, controller, joint_of_interest)
sim.simulate()
