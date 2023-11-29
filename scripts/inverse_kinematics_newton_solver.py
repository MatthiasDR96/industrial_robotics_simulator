from math import *

import matplotlib.pyplot as plt
import numpy as np

from industrial_robotics_simulator.arms.TwoDofArm import TwoDofArm

'''This script illustrates an inverse kinematic solver '''


def debug():
    print("\nIteration " + str(i))
    print("\tJacobian:")
    print("\t" + str(jac))
    print("\tJacobian inverse:")
    print("\t" + str(jac_inv))
    print("\tDelta:")
    print("\t" + str(delta))
    print("\tCorrection:")
    print("\t" + str(correction))
    print("\tNew config:")
    print("\t" + str(np.degrees(q)))


if __name__ == "__main__":

    # Create arm
    arm = TwoDofArm()

    # Create figure
    fig = plt.figure()
    ax1 = fig.add_subplot(111, aspect='equal', autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
    line1, = ax1.plot([], [], 'o-', lw=2)
    time_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes)
    ax1.grid()

    # Init joint config q1, q2
    q_init = np.resize(np.array([[0], [pi / 4]]), (2, 1))

    # Desired cartesian config, x, y
    xd = np.resize(np.array([[0], [1]]), (2, 1))

    # Tuning params
    learning_rate = 0.5
    max_iterations = 100
    position_tolerance = 1e-2

    # Plot initial position
    line1.set_data(arm.position(q_init))
    plt.draw()
    plt.pause(2)

    # Activate interactive plot
    plt.ion()

    # Loop
    q = q_init
    for i in range(max_iterations):

        # Calculate Jacobian
        jac11 = -sin(q[0]) - sin(q[0] + q[1])
        jac12 = -sin(q[0] + q[1])
        jac21 = cos(q[0]) + cos(q[0] + q[1])
        jac22 = cos(q[0] + q[1])
        jac = np.resize([[jac11, jac12], [jac21, jac22]], (2, 2))

        # Calculate Jacobian inverse
        jac_inv = np.linalg.pinv(jac)

        # Calculate Cartesian error
        delta = xd - arm.forward_kinematics(q)

        # Project to joint error
        correction = np.dot(jac_inv, delta)

        # Update joint config
        q = q + learning_rate * correction

        # Debug
        debug()

        # Get new arm position
        position = arm.position(q)

        # Plot
        line1.set_data(position)
        plt.draw()
        plt.pause(2)

        # Stop criterium
        if np.linalg.norm(arm.forward_kinematics(q) - xd) < position_tolerance:
            break

    plt.show()
    # Check correctness
    print("\nAmount of iterations:" + str(i + 1))
    print("\nForward kinematics:")
    print(np.round(arm.forward_kinematics(q), 4))
    print("\nError:")
    print(np.round(arm.forward_kinematics(q) - xd, 4))
