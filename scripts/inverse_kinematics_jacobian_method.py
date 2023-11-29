from math import *

import matplotlib.pyplot as plt
from industrial_robotics_simulator.plot import *
import sympy as sp
from matplotlib.pyplot import *
from mpl_toolkits.mplot3d import Axes3D

'''This script calculates the position Jacobian for general open chains using the symbolic package 'sympy' '''


def plot(q__, ax):

    # Plot world frame
    plot_frame_t(np.identity(4), ax, 'w')

    # Value to substitute
    subst = [(q[0], q__[0][0]), (q[1], q__[1][0]), (q[2], q__[2][0]), (q[3], q__[3][0])]

    # Plot joint frames and links
    T_ = []
    T_.append(T01.subs(subst))
    T_.append(T02.subs(subst))
    T_.append(T03.subs(subst))
    T_.append(T04.subs(subst))
    T__ = []
    T__.append(T01.subs(subst))
    T__.append(T12.subs(subst))
    T__.append(T23.subs(subst))
    T__.append(T34.subs(subst))
    plot_frame_t(np.array(T_[0]), ax, 'j' + str(1))
    plot_transf_p(np.identity(4), np.array(T_[0]), ax)
    for i in range(1, 4):
        plot_frame_t(np.array(T_[i]), ax, 'j' + str(i + 1))
        plot_transf_p(np.array(T_[i - 1]), np.array(T__[i]), ax)

    set_axes_equal(ax)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_zlim3d(0, 1)


def calculate_jacobian(name):

    # Dofs
    dofs = 4

    # Calculate transformation matrix
    T = calculate_transform(name)

    # Orientation part of the Jacobian (compensating for angular velocity)
    kz = sp.Matrix([0, 0, 1])  # Screw vector in local joint frame for revolute joints
    kp = sp.Matrix([0, 0, 0])  # Screw vector in local joint frame for prismatic joints
    J_orientation = [
        calculate_transform('joint1')[:3, :3] * kz,  # joint 1 orientation
        calculate_transform('joint2')[:3, :3] * kz,  # joint 2 orientation
        calculate_transform('joint3')[:3, :3] * kp,  # joint 3 orientation
        calculate_transform('joint4')[:3, :3] * kz]  # joint 4 orientation

    # Compute symbolic forward kinematics
    x = sp.Matrix([0, 0, 0, 1])
    Tx = T * x

    # Init Jacobian
    J = []

    # Calculate linear Jacobian
    for ii in range(dofs):
        J.append([])
        dxdq = sp.simplify(Tx[0].diff(q[ii]))
        dydq = sp.simplify(Tx[1].diff(q[ii]))
        dzdq = sp.simplify(Tx[2].diff(q[ii]))
        J[ii].append(dxdq)  # dx/dq[ii]
        J[ii].append(dydq)  # dy/dq[ii]
        J[ii].append(dzdq)  # dz/dq[ii]

    # Calculate angular Jacobian
    end_point = name.strip('link').strip('joint')
    if end_point != 'EE':
        end_point = min(int(end_point), dofs)
        # add on the orientation information up to the last joint
        for ii in range(end_point):
            J[ii] = sp.Matrix(list(J[ii]) + list(J_orientation[ii]))
        # fill in the rest of the joints orientation info with 0
        for ii in range(end_point, dofs):
            J[ii] = J[ii] + [0, 0, 0]
    J = sp.Matrix(np.reshape(J, (dofs, 6))).T
    return J


def calculate_transform(name):

    # Transformation matrices of the COM in each link frame
    T0COM1 = sp.simplify(sp.Matrix([[-sp.cos(q[0]), sp.sin(q[0]), 0, -0.5 * 0.2 * sp.cos(q[0])],
                                    [-sp.sin(q[0]), -sp.cos(q[0]), 0, -0.5 * 0.2 * sp.sin(q[0])],
                                    [0, 0, 1, 0.3],
                                    [0, 0, 0, 1]]))

    T1COM2 = sp.simplify(sp.Matrix([[sp.sin(q[1]), sp.cos(q[1]), 0, 0.5 * 0.25 * sp.sin(q[1])],
                                    [-sp.cos(q[1]), sp.sin(q[1]), 0, -0.5 * 0.25 * sp.cos(q[1])],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]]))

    T2COM3 = sp.simplify(sp.Matrix([[1, 0, 0, 0],
                                    [0, 1, 0, 0],
                                    [0, 0, 1, q[2] - 0.5 * 0.1],
                                    [0, 0, 0, 1]]))

    # Transformation matrices from base to each joint
    if name == 'joint1':
        T = sp.simplify(T01)
    elif name == 'link1':
        T = sp.simplify(T0COM1)
    elif name == 'joint2':
        T = sp.simplify(T01 * T12)
    elif name == 'link2':
        T = sp.simplify(T01 * T1COM2)
    elif name == 'joint3':
        T = sp.simplify(T01 * T12 * T23)
    elif name == 'link3':
        T = sp.simplify(T01 * T12 * T2COM3)
    elif name == 'joint4' or name == 'link4' or name == 'EE':
        T = sp.simplify(T01 * T12 * T23 * T34)
    else:
        raise Exception('Invalid transformation name: %s' % name)
    return T


if __name__ == "__main__":

    # Symbolic variables
    q = [sp.Symbol('q1'), sp.Symbol('q2'), sp.Symbol('q3'), sp.Symbol('q4')]

    # Homogeneous transformation matrices
    T01 = sp.Matrix([[-sp.cos(q[0]), sp.sin(q[0]), 0, -0.2 * sp.cos(q[0])],
                     [-sp.sin(q[0]), -sp.cos(q[0]), 0, -0.2 * sp.sin(q[0])],
                     [0, 0, 1, 0.3],
                     [0, 0, 0, 1]])

    T12 = sp.Matrix([[sp.sin(q[1]), sp.cos(q[1]), 0, 0.25 * sp.sin(q[1])],
                     [-sp.cos(q[1]), sp.sin(q[1]), 0, -0.25 * sp.cos(q[1])],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])

    T23 = sp.Matrix([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, q[2] - 0.1],
                     [0, 0, 0, 1]])

    T34 = sp.Matrix([[-sp.sin(q[3]), sp.cos(q[3]), 0, 0],
                     [sp.cos(q[3]), sp.sin(q[3]), 0, 0],
                     [0, 0, -1, 0],
                     [0, 0, 0, 1]])
    T02 = T01 * T12
    T03 = T01 * T12 * T23
    T04 = T01 * T12 * T23 * T34

    # Compute symbolic forward kinematics
    x = sp.Matrix([0, 0, 0, 1])
    Tx = calculate_transform('EE') * x
    print("\nForward kinematics positions:")
    print("X-coordinate: " + str(Tx[0]))
    print("Y-coordinate: " + str(Tx[1]))
    print("Z-coordinate: " + str(Tx[2]))

    # Compute symbolic Jacobian from base to the end-effector
    J0COM4 = calculate_jacobian('link4')
    print("\nTotal symbolic Jacobian matrix from base to COM4: \n" + str(J0COM4))

    # Plot inverse kinematics iterative process
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Params
    iterations = 5
    gain = 0.2

    # Curret position
    x = np.mat([-0.2, 0.25, 0.2])

    # Desired positon
    xd = np.mat([0.0, 0.4, 0.1])

    # Iterate
    q_init = np.reshape([0, 0, 0, 0], (4, 1))
    q_ = q_init
    plt.ion()
    for i in range(iterations):

        # Position error
        delta_x = xd - x

        # Compute Jacobian
        J_ = np.array(J0COM4.subs(
            [(q[0], q_[0][0]), (q[1], q_[1][0]), (q[2], q_[2][0]), (q[3], q_[3][0])]), dtype='float')

        # Compute Jacobian inverse
        J_inv_ = np.linalg.pinv(J_)

        # Compute correction
        corr = gain * J_inv_[:, 0:3] * np.reshape(delta_x, (3, 1))

        # Update position
        q_ = q_ + corr

        # Plot
        plot(q_, ax)
        plt.draw()
        plt.pause(1)
    plt.show()

    # Analytical inverse kinematics
    print("\nAnalytic inverse kinematics solution:")
    theta_2 = -acos((0.4 ** 2 - 0.2 ** 2 - 0.25 ** 2) / (2 * 0.2 * 0.25))
    theta_1 = atan2(0.4, 0.0) - atan2(0.2 * sin(theta_2), (0.2 + 0.25 * cos(theta_2)))
    theta_3 = 0.1 - 0.3 + 0.1
    print("Theta 1: " + str(theta_1))
    print("Theta 2: " + str(theta_2))
    print("Theta 3: " + str(theta_3))

    # Check correctness by computing forward kinematics
    print("\nNumeric inverse kinematics solution:")
    print("Theta 1: " + str(q_[0]))
    print("Theta 2: " + str(q_[1]))
    print("Theta 3: " + str(q_[2]))
