from math import *

import numpy as np


def xyz_rpy_from_t(t):
    assert type(t) == np.ndarray and np.shape(t) == (4, 4)
    xyz = xyz_from_t(t)
    rpy = rpy_from_t(t)
    return [xyz[0], xyz[1], xyz[2], rpy[0], rpy[1], rpy[2]]


def xyz_from_t(t):
    assert type(t) == np.ndarray and np.shape(t) == (4, 4)
    x = t[0, 3]
    y = t[1, 3]
    z = t[2, 3]
    return x, y, z


def rpy_from_t(t):
    assert type(t) == np.ndarray and np.shape(t) == (4, 4)
    r = r_from_t(t)
    return rpy_from_r(r)


def r_from_t(t):
    assert type(t) == np.ndarray and np.shape(t) == (4, 4)
    r = t[0:3, 0:3]
    return np.array(r)


def r_from_rpy(_r, _p, _y):
    r_r = np.array([[1, 0, 0], [0, c(_r), -s(_r)], [0, s(_r), c(_r)]])
    r_p = np.array([[c(_p), 0, s(_p)], [0, 1, 0], [-s(_p), 0, c(_p)]])
    r_y = np.array([[c(_y), -s(_y), 0], [s(_y), c(_y), 0], [0, 0, 1]])
    return np.dot(np.dot(r_y, r_p), r_r)


def rpy_from_r2(r):
    assert type(r) == np.ndarray and np.shape(r) == (3, 3)
    roll = np.arctan2(r[2][1], r[2][2])
    yaw = np.arctan2(r[1][0], r[0][0])
    pitch = np.arctan2(-r[2][0], c(yaw) * r[0][0] + s(yaw) * r[1][0])
    return roll, pitch, yaw


def rpy_from_r(r):
    assert type(r) == np.ndarray and np.shape(r) == (3, 3)
    roll = np.arctan2(r[2][1], r[2][2])
    yaw = np.arctan2(r[1][0], r[0][0])
    pitch = np.arcsin(-r[2][0])
    return roll, pitch, yaw


def t_from_dh(a, alpha, d, theta):
    return np.array([[c(theta), -s(theta) * c(alpha), s(theta) * s(alpha), a * c(theta)],
                     [s(theta), c(theta) * c(alpha), -c(theta) * s(alpha), a * s(theta)],
                     [0., s(alpha), c(alpha), d],
                     [0., 0., 0., 1.]])


def t_from_xyz_r(x, y, z, r):
    assert type(r) == np.ndarray and np.shape(r) == (3, 3)
    return np.array(
        [[r[0][0], r[0][1], r[0][2], x], [r[1][0], r[1][1], r[1][2], y], [r[2][0], r[2][1], r[2][2], z], [0, 0, 0, 1]])


def c(inp):
    return np.cos(float(inp))


def s(inp):
    return np.sin(float(inp))


def normalize_angles(q):
    q_ = q
    for i in range(len(q)):

        # Remove multiple full turns
        while q_[i] > 2 * pi:
            q_[i] = q_[i] - 2 * pi
        while q_[i] < - 2 * pi:
            q_[i] = q_[i] + 2 * pi

        # Normalize to [-pi, pi]
        if q_[i] > pi:
            q_[i] = -(2 * pi - q_[i])
        if q_[i] < -pi:
            q_[i] = (2 * pi + q_[i])

    return q_
