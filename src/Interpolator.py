import numpy as np
from sympy import *


def interpolate_cubic(p1, p2, k_traj, t):
    '''
                Computes a smooth cubic polynomail between 2 N-dimensional points
                Input:
                    p1: Nx1 numpy array the first point
                    p2: Nx1 numpy array the second point
                    dp1: Nx1 numpy array of the required velocities at the first point
                    dp2: Nx1 numpy array of the required velocities at the second point
                    T: Scalar which denotes the time needed to traverse the polynomal from point 1 to point 2
                    f: Scalar which denotes the frequency of sampling
                Returns:
                    traj: (N+1) x (Txf) matrix with all interpolated position points for each axis + timesteps
                    dtraj: (N+1) x (Txf) matrix with all interpolated velocities for each axis + timesteps
                    ddtraj: (N+1) x (Txf) matrix with all interpolated accelerations for each axis + timesteps
            '''

    assert type(p1) == np.ndarray and type(p2) == np.ndarray
    assert type(k_traj) == int and (type(t) == float or type(t) == int)

    traj_list = []
    dtraj_list = []
    ddtraj_list = []
    dddtraj_list = []
    s, ds, dds, ddds = get_normalized_third_degree_polynomial(k_traj)
    for i in range(len(p1)):
        traj_ = [((p2[i] - p1[i]) * s[j] + p1[i]) for j in range(len(s))]
        dtraj_ = np.divide([((p2[i] - p1[i]) * ds[j]) for j in range(len(ds))], t)
        ddtraj_ = np.divide([((p2[i] - p1[i]) * dds[j]) for j in range(len(dds))], t ** 2)
        dddtraj_ = np.divide([((p2[i] - p1[i]) * ddds[j]) for j in range(len(ddds))], t ** 3)
        traj_list.append(traj_)
        dtraj_list.append(dtraj_)
        ddtraj_list.append(ddtraj_)
        dddtraj_list.append(dddtraj_)

    tv = np.linspace(0, t, k_traj)
    traj_list.append(tv)
    dtraj_list.append(tv)
    ddtraj_list.append(tv)
    dddtraj_list.append(tv)
    traj = np.asarray(traj_list)
    dtraj = np.asarray(dtraj_list)
    ddtraj = np.asarray(ddtraj_list)
    dddtraj = np.asarray(dddtraj_list)

    return traj, dtraj, ddtraj, dddtraj


def interpolate_quintic(p1, p2, k_traj, t):
    assert type(p1) == np.ndarray and type(p2) == np.ndarray
    assert type(k_traj) == int and (type(t) == float or type(t) == int)

    traj_list = []
    dtraj_list = []
    ddtraj_list = []
    dddtraj_list = []
    s, ds, dds, ddds = get_normalized_fifth_degree_polynomial(k_traj)
    for i in range(len(p1)):
        traj_ = [((p2[i] - p1[i]) * s[j] + p1[i]) for j in range(len(s))]
        dtraj_ = np.divide([((p2[i] - p1[i]) * ds[j]) for j in range(len(ds))], t)
        ddtraj_ = np.divide([((p2[i] - p1[i]) * dds[j]) for j in range(len(dds))], t ** 2)
        dddtraj_ = np.divide([((p2[i] - p1[i]) * ddds[j]) for j in range(len(ddds))], t ** 3)
        traj_list.append(traj_)
        dtraj_list.append(dtraj_)
        ddtraj_list.append(ddtraj_)
        dddtraj_list.append(dddtraj_)

    tv = np.linspace(0, t, k_traj)
    traj_list.append(tv)
    dtraj_list.append(tv)
    ddtraj_list.append(tv)
    dddtraj_list.append(tv)
    traj = np.asarray(traj_list)
    dtraj = np.asarray(dtraj_list)
    ddtraj = np.asarray(ddtraj_list)
    dddtraj = np.asarray(dddtraj_list)

    return traj, dtraj, ddtraj, dddtraj


def interpolate_septic(p1, p2, k_traj, t):
    assert type(p1) == np.ndarray and type(p2) == np.ndarray
    assert type(k_traj) == int and (type(t) == float or type(t) == int)

    traj_list = []
    dtraj_list = []
    ddtraj_list = []
    dddtraj_list = []
    s, ds, dds, ddds = get_normalized_seventh_degree_polynomial(k_traj)
    for i in range(len(p1)):
        traj_ = [((p2[i] - p1[i]) * s[j] + p1[i]) for j in range(len(s))]
        dtraj_ = np.divide([((p2[i] - p1[i]) * ds[j]) for j in range(len(ds))], t)
        ddtraj_ = np.divide([((p2[i] - p1[i]) * dds[j]) for j in range(len(dds))], t ** 2)
        dddtraj_ = np.divide([((p2[i] - p1[i]) * ddds[j]) for j in range(len(ddds))], t ** 3)
        traj_list.append(traj_)
        dtraj_list.append(dtraj_)
        ddtraj_list.append(ddtraj_)
        dddtraj_list.append(dddtraj_)

    tv = np.linspace(0, t, k_traj)
    traj_list.append(tv)
    dtraj_list.append(tv)
    ddtraj_list.append(tv)
    dddtraj_list.append(tv)
    traj = np.asarray(traj_list)
    dtraj = np.asarray(dtraj_list)
    ddtraj = np.asarray(ddtraj_list)
    dddtraj = np.asarray(dddtraj_list)

    return traj, dtraj, ddtraj, dddtraj


def interpolate_nonic(p1, p2, k_traj, t):
    assert type(p1) == np.ndarray and type(p2) == np.ndarray
    assert type(k_traj) == int and (type(t) == float or type(t) == int)

    traj_list = []
    dtraj_list = []
    ddtraj_list = []
    dddtraj_list = []
    s, ds, dds, ddds = get_normalized_ninth_degree_polynomial(k_traj)
    for i in range(len(p1)):
        traj_ = [((p2[i] - p1[i]) * s[j] + p1[i]) for j in range(len(s))]
        dtraj_ = np.divide([((p2[i] - p1[i]) * ds[j]) for j in range(len(ds))], t)
        ddtraj_ = np.divide([((p2[i] - p1[i]) * dds[j]) for j in range(len(dds))], t ** 2)
        dddtraj_ = np.divide([((p2[i] - p1[i]) * ddds[j]) for j in range(len(ddds))], t ** 3)
        traj_list.append(traj_)
        dtraj_list.append(dtraj_)
        ddtraj_list.append(ddtraj_)
        dddtraj_list.append(dddtraj_)

    tv = np.linspace(0, t, k_traj)
    traj_list.append(tv)
    dtraj_list.append(tv)
    ddtraj_list.append(tv)
    dddtraj_list.append(tv)
    traj = np.asarray(traj_list)
    dtraj = np.asarray(dtraj_list)
    ddtraj = np.asarray(ddtraj_list)
    dddtraj = np.asarray(dddtraj_list)

    return traj, dtraj, ddtraj, dddtraj


def interpolate_trapezoid(p1, p2, k_traj, t):
    assert type(p1) == np.ndarray and type(p2) == np.ndarray
    assert type(k_traj) == int and (type(t) == float or type(t) == int)

    traj_list = []
    dtraj_list = []
    ddtraj_list = []
    dddtraj_list = []
    s, ds, dds, ddds = get_normalized_trapezoid_polynomial(k_traj)
    for i in range(len(p1)):
        traj_ = [((p2[i] - p1[i]) * s[j] + p1[i]) for j in range(len(s))]
        dtraj_ = np.divide([((p2[i] - p1[i]) * ds[j]) for j in range(len(ds))], t)
        ddtraj_ = np.divide([((p2[i] - p1[i]) * dds[j]) for j in range(len(dds))], t ** 2)
        dddtraj_ = np.divide([((p2[i] - p1[i]) * ddds[j]) for j in range(len(ddds))], t ** 3)
        traj_list.append(traj_)
        dtraj_list.append(dtraj_)
        ddtraj_list.append(ddtraj_)
        dddtraj_list.append(dddtraj_)

    tv = np.linspace(0, t, k_traj)
    traj_list.append(tv)
    dtraj_list.append(tv)
    ddtraj_list.append(tv)
    dddtraj_list.append(tv)
    traj = np.asarray(traj_list)
    dtraj = np.asarray(dtraj_list)
    ddtraj = np.asarray(ddtraj_list)
    dddtraj = np.asarray(dddtraj_list)

    return traj, dtraj, ddtraj, dddtraj


def interpolate_minimum_jerk_derivative(p1, p2, k_traj, t):
    assert type(p1) == np.ndarray and type(p2) == np.ndarray
    assert type(k_traj) == int and (type(t) == float or type(t) == int)

    traj_list = []
    dtraj_list = []
    ddtraj_list = []
    dddtraj_list = []
    s, ds, dds, ddds = get_normalized_minimum_jerk_derivative_polynomial(k_traj)
    for i in range(len(p1)):
        traj_ = [((p2[i] - p1[i]) * s[j] + p1[i]) for j in range(len(s))]
        dtraj_ = np.divide([((p2[i] - p1[i]) * ds[j]) for j in range(len(ds))], t)
        ddtraj_ = np.divide([((p2[i] - p1[i]) * dds[j]) for j in range(len(dds))], t ** 2)
        dddtraj_ = np.divide([((p2[i] - p1[i]) * ddds[j]) for j in range(len(ddds))], t ** 3)
        traj_list.append(traj_)
        dtraj_list.append(dtraj_)
        ddtraj_list.append(ddtraj_)
        dddtraj_list.append(dddtraj_)

    tv = np.linspace(0, t, k_traj)
    traj_list.append(tv)
    dtraj_list.append(tv)
    ddtraj_list.append(tv)
    dddtraj_list.append(tv)
    traj = np.asarray(traj_list)
    dtraj = np.asarray(dtraj_list)
    ddtraj = np.asarray(ddtraj_list)
    dddtraj = np.asarray(dddtraj_list)

    return traj, dtraj, ddtraj, dddtraj


def get_normalized_first_degree_polynomial(k_traj):
    tau = np.linspace(0, 1, k_traj)
    stau = np.linspace(0, 1, k_traj)
    dstau_dtau = np.linspace(0, 0, k_traj)
    ddstau_ddtau = np.linspace(0, 0, k_traj)
    dddstau_dddtau = np.linspace(0, 0, k_traj)

    for i in range(k_traj):
        t = tau[i]
        stau[i] = t
        dstau_dtau[i] = 1
        ddstau_ddtau[i] = 0
        dddstau_dddtau[i] = 0

    return stau, dstau_dtau, ddstau_ddtau, dddstau_dddtau


def get_normalized_third_degree_polynomial(k_traj):
    tau = np.linspace(0, 1, k_traj)
    stau = np.linspace(0, 1, k_traj)
    dstau_dtau = np.linspace(0, 0, k_traj)
    ddstau_ddtau = np.linspace(0, 0, k_traj)
    dddstau_dddtau = np.linspace(0, 0, k_traj)

    for i in range(k_traj):
        t = tau[i]
        stau[i] = -2 * (t ** 3) + 3 * (t ** 2)
        dstau_dtau[i] = -6 * (t ** 2) + 6 * t
        ddstau_ddtau[i] = -12 * t + 6
        dddstau_dddtau[i] = -12

    return stau, dstau_dtau, ddstau_ddtau, dddstau_dddtau


def get_normalized_fifth_degree_polynomial(k_traj):
    tau = np.linspace(0, 1, k_traj)
    stau = np.linspace(0, 1, k_traj)
    dstau_dtau = np.linspace(0, 0, k_traj)
    ddstau_ddtau = np.linspace(0, 0, k_traj)
    dddstau_dddtau = np.linspace(0, 0, k_traj)

    for i in range(k_traj):
        t = tau[i]
        stau[i] = 6 * (t ** 5) - 15 * (t ** 4) + 10 * (t ** 3)
        dstau_dtau[i] = 30 * (t ** 4) - 60 * (t ** 3) + 30 * (t ** 2)
        ddstau_ddtau[i] = 120 * (t ** 3) - 180 * (t ** 2) + 60 * t
        dddstau_dddtau[i] = 360 * (t ** 2) - 360 * t + 60

    return stau, dstau_dtau, ddstau_ddtau, dddstau_dddtau


def get_normalized_seventh_degree_polynomial(k_traj):
    tau = np.linspace(0, 1, k_traj)
    stau = np.linspace(0, 1, k_traj)
    dstau_dtau = np.linspace(0, 0, k_traj)
    ddstau_ddtau = np.linspace(0, 0, k_traj)
    dddstau_dddtau = np.linspace(0, 0, k_traj)

    for i in range(k_traj):
        t = tau[i]
        stau[i] = -20 * (t ** 7) + 70 * (t ** 6) - 84 * (t ** 5) + 35 * (t ** 4)
        dstau_dtau[i] = -140 * (t ** 6) + 420 * (t ** 5) - 420 * (t ** 4) + 140 * (t ** 3)
        ddstau_ddtau[i] = -840 * (t ** 5) + 2100 * (t ** 4) - 1680 * (t ** 3) + 420 * (t ** 2)
        dddstau_dddtau[i] = -4200 * (t ** 4) + 8400 * (t ** 3) - 5040 * (t ** 2) + 840 * t

    return stau, dstau_dtau, ddstau_ddtau, dddstau_dddtau


def get_normalized_ninth_degree_polynomial(k_traj):
    tau = np.linspace(0, 1, k_traj)
    stau = np.linspace(0, 1, k_traj)
    dstau_dtau = np.linspace(0, 0, k_traj)
    ddstau_ddtau = np.linspace(0, 0, k_traj)
    dddstau_dddtau = np.linspace(0, 0, k_traj)

    for i in range(1, k_traj):
        t = tau[i]
        stau[i] = 70 * (t ** 9) - 315 * (t ** 8) + 540 * (t ** 7) - 420 * (t ** 6) + 126 * (t ** 5)
        dstau_dtau[i] = 630 * (t ** 8) - 2520 * (t ** 7) + 3780 * (t ** 6) - 2520 * (t ** 5) + 630 * (t ** 4)
        ddstau_ddtau[i] = 5040 * (t ** 7) - 17640 * (t ** 6) + 22680 * (t ** 5) - 12600 * (t ** 4) + 2520 * (t ** 3)
        dddstau_dddtau[i] = 35280 * (t ** 6) - 105840 * (t ** 5) + 113400 * (t ** 4) - 50400 * (t ** 3) + 7560 * (
                t ** 2)

    return stau, dstau_dtau, ddstau_ddtau, dddstau_dddtau


def get_normalized_trapezoid_polynomial(k_traj):
    t_acc = 1 / 10.
    t_ct = 1 - 2 * t_acc
    v_m = 1.0 / (t_acc + t_ct)
    x = t_acc

    tau = np.linspace(0, 1, k_traj)
    stau = np.linspace(0, 1, k_traj)
    dstau_dtau = np.linspace(0, 0, k_traj)
    ddstau_ddtau = np.linspace(0, 0, k_traj)
    dddstau_dddtau = np.linspace(0, 0, k_traj)

    for i in range(k_traj):
        t = tau[i]
        if 0 <= t <= x:
            res = 0.5 * v_m * (t ** 2) / t_acc
            vel = v_m * t / t_acc
        elif x < t <= 1 - x:
            res = 0.5 * v_m * (t_acc ** 2) / t_acc + v_m * (t - t_acc)
            vel = v_m
        elif t > 1 - x:
            res = 0.5 * v_m * (t_acc ** 2) / t_acc + v_m * t_ct + v_m * (t - t_acc - t_ct) - 0.5 * v_m / t_acc * (
                    t - t_acc - t_ct) ** 2
            vel = v_m - v_m / t_acc * (t - t_acc - t_ct)
        else:
            res = None
            vel = None
        stau[i] = res
        dstau_dtau[i] = vel

    for i in range(tau.size - 2):
        dstau_dtau[i] = (stau[i + 1] - stau[i]) / (tau[i + 1] - tau[i])

    for i in range(tau.size - 2):
        ddstau_ddtau[i] = (dstau_dtau[i + 1] - dstau_dtau[i]) / (tau[i + 1] - tau[i])

    for i in range(tau.size - 2):
        dddstau_dddtau[i] = (ddstau_ddtau[i + 1] - ddstau_ddtau[i]) / (tau[i + 1] - tau[i])

    return stau, dstau_dtau, ddstau_ddtau, dddstau_dddtau


def get_normalized_minimum_jerk_derivative_polynomial(k_traj):
    x = (1 - np.sqrt(0.5)) / 2

    tau = np.linspace(0, 1, k_traj)
    stau = np.linspace(0, 1, k_traj)
    dstau_dtau = np.linspace(0, 0, k_traj)
    ddstau_ddtau = np.linspace(0, 0, k_traj)
    dddstau_dddtau = np.linspace(0, 0, k_traj)

    res = None
    for i in range(k_traj - 1):
        t = tau[i]
        if 0 <= t <= x:
            res = 16 * (t ** 4)
        elif x < t <= 0.5:
            res = -16 * (t ** 4) + 128 * x * (t ** 3) - 192 * (x ** 2) * (t ** 2) + 128 * (x ** 3) * t - 32 * (x ** 4)
        elif 0.5 < t <= 1 - x:
            res = 1 + 16 * ((1 - t) ** 4) - 128 * x * ((1 - t) ** 3) + 192 * (x ** 2) * ((1 - t) ** 2) - 128 * (
                    x ** 3) * (1 - t) + 32 * (x ** 4)
        elif 1 - x < t <= 1:
            res = 1 - 16 * (1 - t) ** 4
        stau[i] = res

    for i in range(tau.size - 2):
        dstau_dtau[i] = (stau[i + 1] - stau[i]) / (tau[i + 1] - tau[i])

    for i in range(tau.size - 2):
        ddstau_ddtau[i] = (dstau_dtau[i + 1] - dstau_dtau[i]) / (tau[i + 1] - tau[i])

    for i in range(tau.size - 2):
        dddstau_dddtau[i] = (ddstau_ddtau[i + 1] - ddstau_ddtau[i]) / (tau[i + 1] - tau[i])

    return stau, dstau_dtau, ddstau_ddtau, dddstau_dddtau


def get_normalized_cubic_polynomial_coefficients():
    # Kinematic equations for a cubic polynomial
    x0 = [1, 0, 0, 0]
    xt = [1, 1, pow(1, 2), pow(1, 3)]
    v0 = [0, 1, 0, 0]
    vt = [0, 1, 2 * 1, 3 * pow(1, 2)]

    # Solve polynomial coefficients
    a = np.array([x0, xt, v0, vt], dtype='float')
    b = np.array([[0], [1], [0], [0]], dtype='float')
    polynomial = np.linalg.solve(a, b)
    return polynomial


def get_normalized_quintic_polynomial_coefficients():
    # Kinematic equations for a cubic polynomial
    x0 = [1, 0, 0, 0, 0, 0]
    xt = [1, 1, pow(1, 2), pow(1, 3), pow(1, 4), pow(1, 5)]
    v0 = [0, 1, 0, 0, 0, 0]
    vt = [0, 1, 2 * 1, 3 * pow(1, 2), 4 * pow(1, 3), 5 * pow(1, 4)]
    a0 = [0, 0, 2, 0, 0, 0]
    at = [0, 0, 2, 6 * 1, 12 * pow(1, 2), 20 * pow(1, 3)]

    # Solve polynomial coefficients
    a = np.array([x0, xt, v0, vt, a0, at], dtype='float')
    b = np.array([[0], [1], [0], [0], [0], [0]], dtype='float')
    polynomial = np.linalg.solve(a, b)
    return polynomial


def get_normalized_septic_polynomial_coefficients():
    # Kinematic equations for a cubic polynomial
    x0 = [1, 0, 0, 0, 0, 0, 0, 0]
    xt = [1, 1, pow(1, 2), pow(1, 3), pow(1, 4), pow(1, 5), pow(1, 6), pow(1, 7)]
    v0 = [0, 1, 0, 0, 0, 0, 0, 0]
    vt = [0, 1, 2 * 1, 3 * pow(1, 2), 4 * pow(1, 3), 5 * pow(1, 4), 6 * pow(1, 5), 7 * pow(1, 6)]
    a0 = [0, 0, 2, 0, 0, 0, 0, 0]
    at = [0, 0, 2, 6 * 1, 12 * pow(1, 2), 20 * pow(1, 3), 30 * pow(1, 4), 42 * pow(1, 5)]
    j0 = [0, 0, 0, 6, 0, 0, 0, 0]
    jt = [0, 0, 0, 6, 24 * 1, 60 * pow(1, 2), 120 * pow(1, 3), 210 * pow(1, 4)]

    # Solve polynomial coefficients
    a = np.array([x0, xt, v0, vt, a0, at, j0, jt], dtype='float')
    b = np.array([[0], [1], [0], [0], [0], [0], [0], [0]], dtype='float')
    polynomial = np.linalg.solve(a, b)
    return polynomial


def get_normalized_nonic_polynomial_coefficients():
    # Kinematic equations for a cubic polynomial
    x0 = [1, 0, 0, 0, 0, 0]
    xt = [1, 1, pow(1, 2), pow(1, 3), pow(1, 4), pow(1, 5)]
    v0 = [0, 1, 0, 0, 0, 0]
    vt = [0, 1, 2 * 1, 3 * pow(1, 2), 4 * pow(1, 3), 5 * pow(1, 4)]
    a0 = [0, 0, 2, 0, 0, 0]
    at = [0, 0, 2, 6 * 1, 12 * pow(1, 2), 20 * pow(1, 3)]
    j0 = [0, 0, 0, 6, 0, 0, 0, 0]
    jt = [0, 0, 0, 6, 24 * 1, 60 * pow(1, 2), 120 * pow(1, 3), 210 * pow(1, 4)]

    # Solve polynomial coefficients
    a = np.array([x0, xt, v0, vt, a0, at, j0, jt], dtype='float')
    b = np.array([[0], [1], [0], [0], [0], [0], [0], [0]], dtype='float')
    polynomial = np.linalg.solve(a, b)
    return polynomial


def interpolate_quint_2(p1, p2, dp1, dp2, ddp1, ddp2, k_traj, T):
    '''
            Computes a smooth quintic polynomial between 2 N-dimensional points
            Input:
              p1: Nx1 numpy array the first point
              p2: Nx1 numpy array the second point
              dp1: Nx1 numpy array of the required velocities at the first point
              dp2: Nx1 numpy array of the required velocities at the second point
              ddp1: Nx1 numpy array of the required accelerations the first point
              ddp2: Nx1 numpy array of the required accelerations the second point
              T: Scalar which denotes the time needed to traverse the polynomal from point 1 to point 2
              f: Scalar which denotes the frequency of sampling
            Returns:
              traj: (N+1) x (Txf) matrix with all interpolated position points for each axis + timesteps
              dtraj: (N+1) x (Txf) matrix with all interpolated velocities for each axis + timesteps
              ddtraj: (N+1) x (Txf) matrix with all interpolated accelerations for each axis + timesteps
    '''

    assert type(p1) == np.ndarray and type(p2) == np.ndarray
    assert type(dp1) == np.ndarray and type(dp2) == np.ndarray
    assert type(ddp1) == np.ndarray and type(ddp2) == np.ndarray
    assert type(k_traj) == int and (type(T) == float or type(T) == int)

    # Kinematic equations for a quintic polynomial
    x0 = [1, 0, 0, 0, 0, 0]
    xT = [1, T, pow(T, 2), pow(T, 3), pow(T, 4), pow(T, 5)]
    v0 = [0, 1, 0, 0, 0, 0]
    vT = [0, 1, 2 * T, 3 * pow(T, 2), 4 * pow(T, 3), 5 * pow(T, 4)]
    a0 = [0, 0, 2, 0, 0, 0]
    aT = [0, 0, 2, 6 * T, 12 * pow(T, 2), 20 * pow(T, 3)]

    # Kinematic matrix
    A = np.array([x0, xT, v0, vT, a0, aT], dtype='float')

    # Interpolate
    traj_list = []
    dtraj_list = []
    ddtraj_list = []
    t = Symbol('t')
    tv = np.linspace(0, T, k_traj)
    for i in range(len(p1)):
        B = np.array([[p1[i]], [p2[i]], [dp1[i]], [dp2[i]], [ddp1[i]], [ddp2[i]]], dtype='float')
        x = np.linalg.solve(A, B)
        traj = x[0, 0] + x[1, 0] * t + x[2, 0] * pow(t, 2) + x[3, 0] * pow(t, 3) + x[4, 0] * pow(t, 4) + x[
            5, 0] * pow(t, 5)
        dtraj = x[1, 0] + 2 * x[2, 0] * t + 3 * x[3, 0] * pow(t, 2) + 4 * x[4, 0] * pow(t, 3) + 5 * x[
            5, 0] * pow(t, 4)
        ddtraj = 2 * x[2, 0] + 6 * x[3, 0] * t + 12 * x[4, 0] * pow(t, 2) + 20 * x[5, 0] * pow(t, 3)
        traj_ = [traj.subs(t, tv_) for tv_ in tv]
        dtraj_ = [dtraj.subs(t, tv_) for tv_ in tv]
        ddtraj_ = [ddtraj.subs(t, tv_) for tv_ in tv]
        traj_list.append(traj_)
        dtraj_list.append(dtraj_)
        ddtraj_list.append(ddtraj_)

    traj_list.append(tv)
    dtraj_list.append(tv)
    ddtraj_list.append(tv)
    traj = np.asarray(traj_list)
    dtraj = np.asarray(dtraj_list)
    ddtraj = np.asarray(ddtraj_list)

    return traj, dtraj, ddtraj


def interpolate_cubic_2(p1, p2, k_traj, T, dp1=np.zeros((6, 1)), dp2=np.zeros((6, 1))):
    '''
            Computes a smooth cubic polynomal between 2 N-dimensional points
            Input:
                p1: Nx1 numpy array the first point
                p2: Nx1 numpy array the second point
                dp1: Nx1 numpy array of the required velocities at the first point
                dp2: Nx1 numpy array of the required velocities at the second point
                T: Scalar which denotes the time needed to traverse the polynomal from point 1 to point 2
                f: Scalar which denotes the frequency of sampling
            Returns:
                traj: (N+1) x (Txf) matrix with all interpolated position points for each axis + timesteps
                dtraj: (N+1) x (Txf) matrix with all interpolated velocities for each axis + timesteps
                ddtraj: (N+1) x (Txf) matrix with all interpolated accelerations for each axis + timesteps
        '''

    assert type(p1) == np.ndarray and type(p2) == np.ndarray
    assert type(dp1) == np.ndarray and type(dp2) == np.ndarray
    assert type(k_traj) == int and (type(T) == float or type(T) == int)

    # Kinematic equations for a cubic polynomial
    x0 = [1, 0, 0, 0]
    xT = [1, T, pow(T, 2), pow(T, 3)]
    v0 = [0, 1, 0, 0]
    vT = [0, 1, 2 * T, 3 * pow(T, 2)]

    # Kinematic matrix
    A = np.array([x0, xT, v0, vT], dtype='float')

    traj_list = []
    dtraj_list = []
    ddtraj_list = []
    t = Symbol('t')
    tv = np.linspace(0, T, k_traj)
    for i in range(len(p1)):
        B = np.array([[p1[i]], [p2[i]], [dp1[i]], [dp2[i]]], dtype='float')
        x = np.linalg.solve(A, B)
        traj = x[0, 0] + x[1, 0] * t + x[2, 0] * pow(t, 2) + x[3, 0] * pow(t, 3)
        dtraj = x[1, 0] + 2 * x[2, 0] * t + 3 * x[3, 0] * pow(t, 2)
        ddtraj = 2 * x[2, 0] + 6 * x[3, 0] * t
        traj_ = [traj.subs(t, tv_) for tv_ in tv]
        dtraj_ = [dtraj.subs(t, tv_) for tv_ in tv]
        ddtraj_ = [ddtraj.subs(t, tv_) for tv_ in tv]
        traj_list.append(traj_)
        dtraj_list.append(dtraj_)
        ddtraj_list.append(ddtraj_)
    traj_list.append(tv)
    dtraj_list.append(tv)
    ddtraj_list.append(tv)
    traj = np.array(traj_list)
    dtraj = np.array(dtraj_list)
    ddtraj = np.array(ddtraj_list)

    return traj, dtraj, ddtraj


def interpolate_viapoints(p, v1, vn, k_traj, t):
    '''
            Computes a smooth cubic polynomal between M N-dimensional points
            Input:
                p: MxN numpy array containing all points
                v1: Nx1 numpy array of the required velocities at the first point
                vn: Nx1 numpy array of the required velocities at the last point
                t: Mx1 numpy array of the timesteps at which the points should be reached
                f: Scalar which denotes the frequency of sampling
            Returns:
                traj: (N+1) x (Txf) matrix with all interpolated position points for each axis + timesteps
                dtraj: (N+1) x (Txf) matrix with all interpolated velocities for each axis + timesteps
                ddtraj: (N+1) x (Txf) matrix with all interpolated accelerations for each axis + timesteps
    '''

    assert type(p) == np.ndarray and type(k_traj) == int

    # Compute time interval matrix
    h = list(np.zeros((len(t) - 1, 1)))
    for i in range(len(t) - 1):
        h[i] = t[i + 1] - t[i]

    # Compute A(h) matrix
    A = np.zeros((len(h) - 1, len(h) - 1))
    for i in range(len(h) - 1):
        for j in range(len(h) - 1):
            if i == j:
                A[i][j] = 2 * (h[i] + h[i + 1])
            if i == j + 1:
                A[i][j] = h[i + 1]
            if j == i + 1:
                A[i][j] = h[i]

    # Compute known B(p0,p1,h,v1,vn) matrix
    B = np.zeros((len(h) - 1, len(p[0])))
    for i in range(len(h) - 1):
        B[i] = (3 / (h[i] * h[i + 1])) * (
                pow(h[i], 2) * (np.subtract(p[i + 2], p[i + 1])) + pow(h[i + 1], 2) * (np.subtract(p[i + 1], p[i])))
    B[0] = B[0] - np.dot(h[1], v1)
    B[-1] = B[-1] - np.dot(h[-2], vn)

    # Solve for all unknown velocities of intermediate knots
    x = np.linalg.solve(A, B)
    vel = [v1.copy()]
    [vel.append(x[i]) for i in range(len(x))]
    vel.append(vn.copy())

    # Compute N-1 polynomials using computed velocities
    traj = [[0], [0], [0], [0], [0], [0], [0]]
    dtraj = [[0], [0], [0], [0], [0], [0], [0]]
    ddtraj = [[0], [0], [0], [0], [0], [0], [0]]
    for i in range(len(p) - 1):
        traj_, dtraj_, ddtraj_ = interpolate_cubic_2(p[i], p[i + 1], k_traj, float(h[i]), vel[i], vel[i + 1])
        for j in range(len(traj) - 1):
            traj[j].extend(traj_[j])
            dtraj[j].extend(dtraj_[j])
            ddtraj[j].extend(ddtraj_[j])
        traj[-1].extend(traj_[-1] + traj[-1][-1])
        dtraj[-1].extend(dtraj_[-1] + dtraj[-1][-1])
        ddtraj[-1].extend(ddtraj_[-1] + ddtraj[-1][-1])
    traj = np.asarray(np.delete(traj, 0, 1))
    dtraj = np.asarray(np.delete(traj, 0, 1))
    ddtraj = np.asarray(np.delete(traj, 0, 1))

    return traj, dtraj, ddtraj
