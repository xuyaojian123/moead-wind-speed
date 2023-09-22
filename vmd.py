#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File  : vmd.py
@Author: XuYaoJian
@Date  : 2022/9/3 21:14
@Desc  : 
"""

# -*-coding:utf-8-*-
import numpy as np


def vmd(signal, alpha, tau, K, DC, init, tol):
    """
    :param signal: the time domain signal (1D) to be decomposed
    :param alpha: the balancing parameter of the data-fidelity constraint
    :param tau: time-step of the dual ascent ( pick 0 for noise-slack )
    :param K: the number of modes to be recovered
    :param DC: true if the first mode is put and kept at DC (0-freq) 是否有直流分量
    :param init: 0 = all omegas start at 0
                   1 = all omegas start uniformly distributed
                   2 = all omegas initialized randomly
    :param tol: tolerance of convergence criterion; typically around 1e-6
    :return: u       - the collection of decomposed modes
             u_hat   - spectra of the modes
             omega   - estimated mode center-frequencies
    """

    # 采样信号序列的长度和频率
    s_len = len(signal)
    s_f = 1 / s_len

    # 镜像扩展信号
    # matlab中索引(a:1:b)是从第a位到第b位，python中索引[a:b:1]是从a到b-1位。
    # matlab索引从1开始，python从0开始
    f_mirror = []
    half_len = int(np.round(s_len / 2))
    f_mirror[:half_len] = signal[half_len - 1::-1]
    f_mirror[half_len:3 * half_len] = signal
    f_mirror[3 * half_len:2 * s_len] = signal[s_len - 1:half_len - 1:-1]
    f = f_mirror

    # 求镜像信号的长度T和时间
    T = len(f_mirror)
    t = np.arange(1, T + 1) / T
    half_mir_len = int(np.round(T / 2))

    # 谱域离散化
    freqs = t - 0.5 - 1 / T

    # 最大迭代值
    N = 500

    # 归一化，为每个模态创建独立的alpha
    Alpha = alpha * np.ones((1, K))

    # 建造和中心化f_hat
    f_hat = np.fft.fftshift(np.fft.fft(f_mirror))
    f_hat_plus = f_hat.copy()  # Python中赋值是引用赋值！！！要复制变量使用a.copy()。用a[:]也是引用赋值
    f_hat_plus[:int(T / 2)] = 0

    # 保持迭代轨迹的矩阵
    u_hat_plus = np.zeros((N, len(freqs), K), np.complex128)

    # 初始化ωk，即中心频率
    omega_plus = np.zeros((N, K))
    if init == 1:
        for i in range(K):  # 0到K-1
            omega_plus[0, i] = (0.5 / K) * i
    elif init == 2:
        omega_plus[0, :] = np.sort(np.exp(np.log(s_f) + (np.log(0.5) - np.log(s_f)) * np.random.rand(1, K)))
    else:
        omega_plus[0, :] = 0

    if DC:
        omega_plus[0, 0] = 0

    # 空双重变量
    lambda_hat = np.zeros((N, len(freqs)), np.complex128)

    # 其他的初始化
    u_diff = tol + np.spacing(1)
    n = 0
    sum_uk = 0

    while u_diff > tol and n < N - 1:
        # 更新第一模态加速器
        k = 0
        sum_uk = u_hat_plus[n, :, K - 1] + sum_uk - u_hat_plus[n, :, 0]
        # 通过维纳滤波更新第一模态频谱
        b = (f_hat_plus - sum_uk - lambda_hat[n, :] / 2) / (
                1 + Alpha[0, k] * (freqs - omega_plus[n, k]) ** 2)

        u_hat_plus[n + 1, :, k] = b
        # 如果DC不为0，更新第一个ω
        if not DC:
            omega_plus[n + 1, k] = np.dot(freqs[half_mir_len:T],
                                          np.abs(u_hat_plus[n + 1, half_mir_len:T, k]) ** 2) / np.sum(
                np.abs(u_hat_plus[n + 1, half_mir_len:T, k]) ** 2)

        # 更新别的模态
        for k in range(1, K):
            # 加速器
            sum_uk = u_hat_plus[n + 1, :, k - 1] + sum_uk - u_hat_plus[n, :, k]

            # 模态谱
            c = (f_hat_plus - sum_uk - lambda_hat[n, :] / 2) / (
                    1 + Alpha[0, k] * (freqs - omega_plus[n, k]) ** 2)
            u_hat_plus[n + 1, :, k] = c

            # 中心频率
            omega_plus[n + 1, k] = np.dot(freqs[half_mir_len:T],
                                          np.abs(u_hat_plus[n + 1, half_mir_len:T, k]) ** 2) / np.sum(
                np.abs(u_hat_plus[n + 1, half_mir_len:T, k]) ** 2)

        # 双重上升
        d = lambda_hat[n, :] + tau * (np.sum(u_hat_plus[n + 1, :, :], 1) - f_hat_plus)
        lambda_hat[n + 1, :] = d
        # 计数加一
        n = n + 1

        # 合并
        u_diff = np.spacing(1)
        for i in range(K):  # np.conj为求复数共轭
            u_diff = u_diff + 1 / T * np.dot((u_hat_plus[n, :, i] - u_hat_plus[n - 1, :, i]),
                                             np.conj(u_hat_plus[n, :, i] - u_hat_plus[n - 1, :, i]))
        u_diff = np.abs(u_diff)

    # 如果提前合并则抛弃空闲空间
    N = np.min([N, n])
    omega = omega_plus[1:N - 1, :]

    # 信号重构
    u_hat = np.zeros((T, K), np.complex128)
    u_hat[half_mir_len:T, :] = np.squeeze(u_hat_plus[N, half_mir_len:T, :])
    u_hat[half_mir_len:0:-1, :] = np.squeeze(np.conj(u_hat_plus[N, half_mir_len:T, :]))
    u_hat[1, :] = np.conj(u_hat[-1, :])

    u = np.zeros((K, len(t)))

    for k in range(K):
        u[k, :] = np.real(np.fft.ifft(np.fft.ifftshift(u_hat[:, k])))

    # 移除镜像部分
    u = u[:, int(half_mir_len / 2):int(half_mir_len / 2 * 3)]

    # 重新计算频谱
    u_hat = []
    for k in range(K):
        u_hat.append(np.fft.fftshift(np.fft.fft(u[k, :])).T)
    u_hat = np.array(u_hat)
    # print("tau " + str(tau) + "  iteration times " + str(n))
    # print("converge accuracy" + str(u_diff))
    return [u, u_hat, omega]
