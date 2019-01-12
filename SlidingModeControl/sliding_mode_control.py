"""
The MIT License (MIT)

Copyright (c) 2015 Issei Kuzumaki.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

"""スライディングモード制御器の設計と線形システムを用いたデモ."""


import scipy.linalg as sl
import numpy.linalg as nl
from scipy.signal import cont2discrete
from math import pi
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def initialize_system(A, B):
    """システムの状態方程式のA, B行列から状態量x, 入力uの初期化を行う."""
    nx = A.shape[1]
    nu = B.shape[1]

    x = np.zeros((nx,1), dtype=float)
    u = np.zeros((nu,1), dtype=float)

    return x, u


def update_dxdt(A, B, x, u):
    """連続時間の微分."""
    Ax = np.dot(A, x)
    Bu = np.dot(B, u)
    dxdt = Ax + Bu
    return dxdt


def update_discrete_x(A, B, x, u):
    """離散時間の状態量を求める."""
    Ax = np.dot(A, x)
    Bu = np.dot(B, u)
    x = Ax + Bu
    return x


def dimension_x_u(A, B):
    """x, uの次元を求める."""
    nx = A.shape[1]
    nu = B.shape[1]
    return nx, nu


def cannonical_transformation(A, B):
    """xを正準変換するための行列Tを求める."""
    nx, nu = dimension_x_u(A, B)

    B1 = B[0:nx-nu,:]
    B2 = B[nx-nu:nx,:]

    T = np.zeros((nx, nx), dtype=float)
    T[0:nx-nu,0:nx-nu] = np.eye(nx-nu, dtype=float)
    T[0:nx-nu, nx-nu:nx] = - np.dot(B1, nl.inv(B2))
    T[nx-nu:nx,0:nx-nu] = np.zeros((nu, nx-nu), dtype=float)
    T[nx-nu:nx, nx-nu:nx] = np.eye(nu, dtype=float)
    return T


def optimal_hyperplane_vector(A, B, W):
    """最適制御理論から，切替超平面を設計する"""
    nx, nu = dimension_x_u(A, B)

    x = np.zeros((nx,1), dtype=float)
    u = np.zeros((nx,1), dtype=float)

    B1 = B[0:nx-nu,:]
    B2 = B[nx-nu:nx,:]

    print(f'B1:\n{B1}\n')
    print(f'B2:\n{B2}\n')

    T = cannonical_transformation(A, B)

    print(f'W:\n{W}\n')
    print(f'T:\n{T}\n')

    A_ = np.dot(T, np.dot(A, nl.inv(T)))
    B_ = np.dot(T, B)

    A_11 = A_[0:nx-nu, 0:nx-nu]
    A_12 = A_[0:nx-nu, nx-nu:nx]
    A_21 = A_[nx-nu:nx, 0:nx-nu]
    A_22 = A_[nx-nu:nx, nx-nu:nx]

    B_1 = B_[0:nx-nu,:]
    B_2 = B_[nx-nu:nx,:]

    WT = np.dot(W, T)
    Q = np.dot(T.T, WT)

    Q11 = Q[0:nx-nu, 0:nx-nu]
    Q12 = Q[0:nx-nu, nx-nu:nx]
    Q21 = Q[nx-nu:nx, 0:nx-nu]
    Q22 = Q[nx-nu:nx, nx-nu:nx]

    QQ = np.dot(nl.inv(Q22), Q12.T)
    Q11_ = Q11 - np.dot(Q12, QQ)
    A_11_ = A_11 - np.dot(A_12, QQ)

    P = sl.solve_continuous_are(A_11_, A_12, Q11_, Q22)
    S1 = np.dot(A_12.T, P) + Q12.T
    S2 = Q22

    print(f'A_:\n{A_}\n')
    print(f'A_12:\n{A_12}\n')
    print(f'P:\n{P}\n')
    print(f'Q:\n{Q}\n')
    print(f'Q12:\n{Q12}\n')

    print(f'S1:\n{S1}\n')
    print(f'S2:\n{S2}\n')

    return S1, S2, A_, B_, T


def equivalent_feedback_gain(A, B, S):
    """切替超平面から線形フィードバック項のゲインを求める."""
    SB = np.dot(S, B)
    SA = np.dot(S, A)

    u_eq = np.dot(nl.inv(SB), SA)
    return u_eq


def linear_input(A, B, S, x):
    """線形フィードバック項をもとめる.
    スライディングモード制御は線形項と非線形項の合計で入力を決める．
    """
    gain = equivalent_feedback_gain(A, B, S)
    u = - np.dot(gain, x)
    return u


def switching_function(S, x):
    """切替関数の値を求める."""
    sigma = np.dot(S,x)
    return sigma


def nonlinear_input(B, S, k, x):
    """非線形項を求める.
    スライディングモード制御は線形項と非線形項の合計で入力を決める．
    """
    SB = np.dot(S, B)

    sigma = switching_function(S, x)
    n_sigma = sigma / nl.norm(sigma)

    u_ni = - np.sign(SB) * k * n_sigma

    return u_ni


def system_mass_spring_dumper():
    """マスバネダンパ系の設計例"""
    # define the system
    m = 1.0
    k = 1.0
    c = 1.0

    A = np.array([
        [0.0, 1.0],
        [-k/m, -c/m]
    ])

    B = np.array([
        [0],
        [1/m]
    ])
    C = np.eye(2)
    D = np.zeros((2,1),dtype=float)
    W = np.diag([1.0, 1.0])
    S1, S2, A_, B_, T = optimal_hyperplane_vector(A, B, W)
    S = np.hstack((S1, S2))
    x, u = initialize_system(A, B)
    x[0] = 0.0
    x[1] = 10.0
    # define the gain of
    k = 10
    return C, D, S, k, x, u, A_, B_, T


def system_program_7_4():
    """例題7.4"""
    A_ = np.array([
        [-2, 1, 0],
        [0, 0, 1],
        [0, 0, 1]
    ])
    B_ = np.array([
        [0.0],
        [0.0],
        [1.0]
    ])
    C = np.zeros((3,3),dtype=float)
    D = np.zeros((3,1),dtype=float)
    S = np.array([[1, 2, 1]])

    x, u = initialize_system(A_, B_)
    x[0] = 0.5
    x[1] = 1.0
    x[2] = 1.5

    k = 10

    return C, D, S, k, x, u, A_, B_


def system_program_7_2():
    """例題7.2の倒立振子モデル."""
    A = np.array([
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, -0.338, -16.261, 0.002],
        [0, 30.040, 71.852, -0.192]
    ],dtype=float)

    B = np.array([
        [0.0],
        [0.0],
        [28.727],
        [-129.937]
    ])
    C = np.zeros((4,4),dtype=float)
    D = np.zeros((4,1),dtype=float)
    #
    # define the weight of params
    W = np.diag([10.0, 10.0, 0.01, 0.01])

    S1, S2, A_, B_, T = optimal_hyperplane_vector(A, B, W)
    S = np.hstack((S1, S2))

    x, u = initialize_system(A, B)
    x[1] = 10 * pi/180

    # define the gain of
    k = 1

    return C, D, S, k, x, u, A_, B_, T


def system_program_7_1():
    """例題7.1のモデル."""
    A = np.array([
        [1, 1, 0],
        [0, 2, 1],
        [0, 0, 2]
    ],dtype=float)

    B = np.array([
        [0.0],
        [0.0],
        [1.0]
    ])
    C = np.zeros(A.shape,dtype=float)
    D = np.zeros(B.shape,dtype=float)
    #
    # define the weight of params
    W = np.diag([1.0, 1.0, 1.0])

    S1, S2, A_, B_, T = optimal_hyperplane_vector(A, B, W)
    S = np.hstack((S1, S2))

    x, u = initialize_system(A, B)
    x[0] = 1.0

    # define the gain of
    k = 1

    return C, D, S, k, x, u, A_, B_, T


def system_inversed_pendulum():
    """参考文献の倒立振子モデル.
    参考文献：
    『Model Predictive Controlによる倒立振子制御Pythonプログラム』
    https://myenigma.hatenablog.com/entry/2017/06/26/113719
    """
    l_bar = 2.0  # length of bar
    M = 1.0  # [kg]
    m = 0.3  # [kg]
    g = 9.8  # [m/s^2]

    nx = 4   # number of state
    nu = 1   # number of input
    delta_t = 0.1  # time tick

    # Model Parameter
    A = np.array([
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, m * g / M, 0.0],
        [0.0, 0.0, g * (M + m) / (l_bar * M), 0.0]
    ])
    A = np.eye(nx) + delta_t * A

    B = np.array([
        [0.0],
        [0.0],
        [1.0 / M],
        [1.0 / (l_bar * M)]
    ])
    B = delta_t * B

    C = np.zeros(A.shape,dtype=float)
    D = np.zeros(B.shape,dtype=float)
    #
    # define the weight of params
    W = np.diag([1.0, 1.0, 0.01, 0.01])

    S1, S2, A_, B_, T = optimal_hyperplane_vector(A, B, W)
    S = np.hstack((S1, S2))

    x, u = initialize_system(A, B)
    x[0] = 0.0
    x[1] = 1 * pi/180

    # define the gain of
    k = 10

    return C, D, S, k, x, u, A_, B_, T


if __name__ == '__main__':
    C, D, S, k, x, u, A_, B_, T = system_program_7_1()

    dt = 0.01
    step = 5000

    log_x = np.zeros((len(x), step), dtype=float)
    log_u = np.zeros((len(u), step), dtype=float)
    log_un = np.zeros((step), dtype=float)
    log_ul = np.zeros((step), dtype=float)
    log_s = np.zeros((step), dtype=float)

    Ad, Bd, Cd, Dd, dt = cont2discrete((A_, B_, C, D), dt)

    for n in range(step):
        x_ = np.dot(T, x)
        ul = linear_input(A_, B_, S, x_)
        un = nonlinear_input(B_, S, k, x_)
        u = ul + un
        x_ = update_discrete_x(Ad, Bd, x_, u)

        x = np.dot(nl.inv(T), x_)
        log_x[:,n] = x[:,0]
        log_u[:,n] = u[:,0]
        sigma = switching_function(S, x_)
        log_s[n] = sigma
        log_ul[n] = ul
        log_un[n] = un


    t = dt * np.array(range(step), dtype=float)
    plt.figure()
    plt.plot(t, log_x.T)
    plt.title('displacement')
    # plt.legend(['x1','x2'])
    # plt.ylim([-3,2])

    plt.figure()
    plt.plot(log_x[0,:],log_x[1,:])
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('phase plane trajectory')

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter3D(log_x[0,:],log_x[1,:],log_x[2,:])
    # ax.set_title('phase plane trajectory')

    plt.figure()
    plt.plot(t, log_s)
    plt.title('sigma')

    plt.figure()
    plt.plot(t, log_u.T)
    plt.title('total input')
    # plt.ylim([-15,20])

    plt.figure()
    plt.plot(t, log_un, t, log_ul)
    plt.legend(['nonlinear', 'linear'])
    # plt.ylim([-15,20])

    plt.show()
