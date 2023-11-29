from industrial_robotics_simulator.Simulator import *
from industrial_robotics_simulator.arms import *
from industrial_robotics_simulator.controllers.task_space_motion_controller import *

""" This script uses task space motion control to move to a target."""

# Create robot
q_init = np.array([[math.pi / 2], [0.0]])
robot = TwoDofArm()
robot.set_q_init(q_init)

# Create controller
controller = Control(robot)

# Create target pose in cartesian space (X and Y)
x_des = np.array([[1.0], [1.0]])
controller.set_task_space_target(x_des)

# Run animation
sim = Simulator(robot, controller)
sim.simulate()
