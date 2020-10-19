from math import *

import matplotlib.pyplot as plt
import scipy.optimize
import sympy as sp
from matplotlib.pyplot import *
from mpl_toolkits.mplot3d import Axes3D

'''This script calculates the position Jacobian for general open chains using the symbolic package 'sympy' '''


def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = x_limits[1] - x_limits[0]
    x_mean = np.mean(x_limits)
    y_range = y_limits[1] - y_limits[0]
    y_mean = np.mean(y_limits)
    z_range = z_limits[1] - z_limits[0]
    z_mean = np.mean(z_limits)
    plot_radius = 0.5 * max([x_range, y_range, z_range])
    ax.set_xlim3d([x_mean - plot_radius, x_mean + plot_radius])
    ax.set_ylim3d([y_mean - plot_radius, y_mean + plot_radius])
    ax.set_zlim3d([z_mean - plot_radius, z_mean + plot_radius])


def plot_frame_t(t, ax, text=''):
    axis_length = 0.05
    r = t[0:3, 0:3]
    x = t[0][3]
    y = t[1][3]
    z = t[2][3]
    pose_ix = np.dot(r, np.array([axis_length, 0, 0]))
    pose_iy = np.dot(r, np.array([0, axis_length, 0]))
    pose_iz = np.dot(r, np.array([0, 0, axis_length]))
    ax.plot([x, x + pose_ix[0]], [y, y + pose_ix[1]], [z, z + pose_ix[2]], 'r', linewidth=2)
    ax.plot([x, x + pose_iy[0]], [y, y + pose_iy[1]], [z, z + pose_iy[2]], 'g', linewidth=2)
    ax.plot([x, x + pose_iz[0]], [y, y + pose_iz[1]], [z, z + pose_iz[2]], 'b', linewidth=2)
    pose_t = np.dot(r, np.array([0.3 * axis_length, 0.3 * axis_length, 0.3 * axis_length]))
    ax.text(x + pose_t[0], y + pose_t[1], z + pose_t[2], text, fontsize=11)


def plot_transf_p(tref, ttransf, ax):
    t = np.dot(tref, ttransf)
    x = t[0][3]
    y = t[1][3]
    z = t[2][3]
    x0 = tref[0][3]
    y0 = tref[1][3]
    z0 = tref[2][3]
    ax.plot([x0, x], [y0, y], [z0, z], 'k:', linewidth=1)


def plot(q__, ax):
    # Plot world frame
    plot_frame_t(np.identity(4), ax, 'w')

    # Value to substitute
    subst = [(q[0], q__[0][0]), (q[1], q__[1][0]), (q[2], q__[2][0]), (q[3], q__[3][0]), (l[0], 0.3), (l[1], 0.2),
             (l[2], 0.25),
             (l[3], 0.1)]

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
    TOCOM1_ = T0COM1.subs(subst)
    TOCOM2_ = T0COM2.subs(subst)
    TOCOM3_ = T0COM3.subs(subst)
    plot_frame_t(np.array(T_[0]), ax, 'j' + str(1))
    plot_frame_t(np.array(TOCOM1_), ax, 'COM' + str(1))
    plot_frame_t(np.array(TOCOM2_), ax, 'COM' + str(2))
    plot_frame_t(np.array(TOCOM3_), ax, 'COM' + str(3))
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
    # Calculate transformation matrix
    T = calculate_transform(name)

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


def inverse_kinematics(self, xy, q_init=None):
    if q_init is None:
        q_init = self.q_sym

    def distance_to_target(q, xy):
        fxy = self.forward_kinematics(q)
        return np.sqrt((fxy[0] - xy[0]) ** 2 + (fxy[1] - xy[1]) ** 2)

    return scipy.optimize.minimize(fun=distance_to_target, x0=q_init, args=([xy[0], xy[1]]))['x']


def calculate_transform(name):
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
    dofs = 4
    q = [sp.Symbol('q1'), sp.Symbol('q2'), sp.Symbol('q3'), sp.Symbol('q4')]
    l = [sp.Symbol('l1'), sp.Symbol('l2'), sp.Symbol('l3'), sp.Symbol('l4')]

    # Homogeneous transformation matrices
    T01 = sp.Matrix([[-sp.cos(q[0]), sp.sin(q[0]), 0, -l[1] * sp.cos(q[0])],
                     [-sp.sin(q[0]), -sp.cos(q[0]), 0, -l[1] * sp.sin(q[0])],
                     [0, 0, 1, l[0]],
                     [0, 0, 0, 1]])

    T12 = sp.Matrix([[sp.sin(q[1]), sp.cos(q[1]), 0, l[2] * sp.sin(q[1])],
                     [-sp.cos(q[1]), sp.sin(q[1]), 0, -l[2] * sp.cos(q[1])],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])

    T23 = sp.Matrix([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, q[2] - l[3]],
                     [0, 0, 0, 1]])

    T34 = sp.Matrix([[-sp.sin(q[3]), sp.cos(q[3]), 0, 0],
                     [sp.cos(q[3]), sp.sin(q[3]), 0, 0],
                     [0, 0, -1, 0],
                     [0, 0, 0, 1]])

    # Transformation matrices of the COM in each link frame
    T0COM1 = sp.simplify(sp.Matrix([[-sp.cos(q[0]), sp.sin(q[0]), 0, -0.5 * l[1] * sp.cos(q[0])],
                                    [-sp.sin(q[0]), -sp.cos(q[0]), 0, -0.5 * l[1] * sp.sin(q[0])],
                                    [0, 0, 1, l[0]],
                                    [0, 0, 0, 1]]))

    T1COM2 = sp.simplify(sp.Matrix([[sp.sin(q[1]), sp.cos(q[1]), 0, 0.5 * l[2] * sp.sin(q[1])],
                                    [-sp.cos(q[1]), sp.sin(q[1]), 0, -0.5 * l[2] * sp.cos(q[1])],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]]))

    T2COM3 = sp.simplify(sp.Matrix([[1, 0, 0, 0],
                                    [0, 1, 0, 0],
                                    [0, 0, 1, q[2] - 0.5 * l[3]],
                                    [0, 0, 0, 1]]))

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
    Tx = calculate_transform('EE') * x
    print("\nForward kinematics positions:")
    print("X-coordinate: " + str(Tx[0]))
    print("Y-coordinate: " + str(Tx[1]))
    print("Z-coordinate: " + str(Tx[2]))

    # Compute transformation matrices from base to each joint
    print("\nTotal symbolic transformation matrix from base to joint1: \n" + str(T01))
    print("\nTotal symbolic transformation matrix from base to COM1: \n" + str(T0COM1))
    T02 = calculate_transform('joint2')
    print("\nTotal symbolic transformation matrix from base to joint2: \n" + str(T02))
    T0COM2 = calculate_transform('link2')
    print("\nTotal symbolic transformation matrix from base to COM2: \n" + str(T0COM2))
    T03 = calculate_transform('joint3')
    print("\nTotal symbolic transformation matrix from base to joint3: \n" + str(T03))
    T0COM3 = calculate_transform('link3')
    print("\nTotal symbolic transformation matrix from base to COM3: \n" + str(T0COM3))
    T04 = calculate_transform('joint4')
    print("\nTotal symbolic transformation matrix from base to joint4: \n" + str(T04))

    # Compute Jacobians from base to the COM of each link
    J0COM1 = calculate_jacobian('link1')
    print("\nTotal symbolic Jacobian matrix from base to COM1: \n" + str(J0COM1))
    J0COM2 = calculate_jacobian('link2')
    print("\nTotal symbolic Jacobian matrix from base to COM2: \n" + str(J0COM2))
    J0COM3 = calculate_jacobian('link3')
    print("\nTotal symbolic Jacobian matrix from base to COM3: \n" + str(J0COM3))
    J0COM4 = calculate_jacobian('link4')
    print("\nTotal symbolic Jacobian matrix from base to COM4: \n" + str(J0COM4))

    # Plot inverse kinematics iterative process
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = np.mat([-0.2, 0.25, 0.2])
    xd = np.mat([0.0, 0.4, 0.1])
    q_init = np.reshape([0, 0, 0, 0], (4, 1))
    q_ = q_init
    plt.ion()
    for i in range(5):
        delta_x = xd - x
        J_ = np.array(J0COM4.subs(
            [(q[0], q_[0][0]), (q[1], q_[1][0]), (q[2], q_[2][0]), (q[3], q_[3][0]), (l[0], 0.3), (l[1], 0.2),
             (l[2], 0.25), (l[3], 0.1)]), dtype='float')
        J_inv_ = np.linalg.pinv(J_)
        corr = 0.2 * J_inv_[:, 0:3] * np.reshape(delta_x, (3, 1))
        q_ = q_ + corr
        plot(q_, ax)
        plt.draw()
        plt.pause(1)
    plt.show()

    # Analytical inverse kinematics
    theta_2 = -acos((0.4 ** 2 - 0.2 ** 2 - 0.25 ** 2) / (2 * 0.2 * 0.25))
    print("Theta 2: " + str(theta_2))
    theta_1 = atan2(0.4, 0.0) - atan2(0.2 * sin(theta_2), (0.2 + 0.25 * cos(theta_2)))
    print("Theta 1: " + str(theta_1))
    theta_3 = 0.1 - 0.3 + 0.1
    print("Theta 3: " + str(theta_3))

    # Check correctness by computing forward kinematics
    print(Tx.subs(
        [(q[0], theta_1), (q[1], theta_2), (q[2], theta_3), (q[3], 0.0), (l[0], 0.3), (l[1], 0.2), (l[2], 0.25),
         (l[3], 0.1)]))
