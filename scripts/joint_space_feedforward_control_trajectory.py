from src.Interpolator import *
from src.Simulator import *
from src.arms import *
from src.controllers.joint_space_feedforward_controller import *

""" This script uses feedforward motion control to track a trajectory."""

# Create robot
q_init = np.array([[3 * math.pi / 4], [0.0]])
robot = TwoDofArm()
robot.set_q_init(q_init)

# Create controller
controller = Control(robot)

# Create joint space trajectory
p1 = np.reshape(q_init, (robot.DOF,))
p2 = np.array([math.pi / 4, 0.0])
k_traj = 100  # Needs to be equal to the amount of animation frames = 100 (Set in Simulator class)
t = 5  # Needs to be equal to the simulation time of 100 frames
q_des, dq_des, ddq_des, dddq_des = interpolate_cubic(p1, p2, k_traj, t)
controller.set_joint_space_trajectory(q_des, dq_des, ddq_des)

# Run animation
joint_of_interest = 1
sim = Simulator(robot, controller, joint_of_interest)
sim.simulate()
