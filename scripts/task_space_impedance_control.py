from src.Simulator import *
from src.arms import *
from src.controllers.task_space_impedance_controller import *

""" This script uses task impedance control to move to a target under an external force."""

# Create robot
q_init = np.array([[math.pi / 2], [-math.pi / 2]])
robot = TwoDofArm()
robot.set_q_init(q_init)

# Create controller
controller = Control(robot)

# Create target pose in cartesian space (X and Y)
x_des = np.array([[1.0], [1.0]])
controller.set_task_space_target(x_des)

# Create external force in task space (20N in positive x-direction)
frames = 100  # Amount of animation iterations before restart
force_x = np.repeat(20, frames)
force_x[0:20] = 0  # Start force at frame 20
force_x[50:-1] = 0  # End force at frame 50
force_y = np.repeat(0, frames)
force = np.vstack((force_x, force_y))
controller.set_external_force_function(force)

# Run animation
joint_of_interest = 1
sim = Simulator(robot, controller, joint_of_interest)
sim.simulate()
