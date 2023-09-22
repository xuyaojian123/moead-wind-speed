#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File  : utils.py
@Author: XuYaoJian
@Date  : 2022/9/3 23:06
@Desc  :
"""

from vmd import vmd
import numpy as np
import matplotlib.pyplot as plt

# thansform excel and csv data to train and test data by vmd
def load_data_with_vmd(data, K, seq_len, path=""):
    # VMD分解
    alpha = 1000  # moderate bandwidth constraint
    DC = 0  # no DC part imposed
    init = 1  # initialize omegas uniformly
    tol = 0.5  # tolerance
    REI = np.inf
    tauo = 0
    for tau in [x / 10 for x in range(1, 11)]:
        [u, u_hat, omega] = vmd(data, alpha, tau, K, DC, init, tol)
        temp = np.sqrt(np.sum((np.sum(u, 0) - data) ** 2, 0) / len(data))  # 把分解的序列重构回去，和原序列作对比。保留误差最小的tau
        if temp < REI:
            REI = temp
            tauo = tau
    tau = tauo # noise-tolerance (no strict fidelity enforcement)
    print("tau:" + str(tau))
    [u, u_hat, omega] = vmd(data, alpha, tau, K, DC, init, tol)
    print('> vmd processed...')
    # 处理数据
    x_trains = []
    y_trains = []
    x_valids = []
    y_valids = []
    x_tests = []
    y_tests = []
    for i in range(K + 1):
        result = []
        if i < K:
            seq = u[i]
        else:
            seq = data
        for index in range(len(seq) - seq_len):
            result.append(seq[index:index + seq_len + 1])  # create train label
        result = np.array(result)  # 用numpy对其进行矩阵化
        # 划分训练集和验证集和测试集 (测试集为两天，风能间隔为10min,两天=6*24*2=288)
        len1 = int(result.shape[0] * 0.6)
        len2 = int(result.shape[0] * 0.8)
        # row = int(result.shape[0] * (1 - 288 / result.shape[0]))
        x_train = result[:len1, :-1]
        x_valid = result[len1:len2, :-1]
        x_test = result[len2:, :-1]
        y_train = result[:len1, -1]
        y_valid = result[len1:len2, -1]
        y_test = result[len2:, -1]

        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        x_valid = np.reshape(x_valid, (x_valid.shape[0], x_valid.shape[1], 1))
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        y_train = np.reshape(y_train, (y_train.shape[0], 1))
        y_valid = np.reshape(y_valid, (y_valid.shape[0], 1))
        y_test = np.reshape(y_test, (y_test.shape[0], 1))

        x_trains.append(x_train)
        y_trains.append(y_train)
        x_valids.append(x_valid)
        y_valids.append(y_valid)
        x_tests.append(x_test)
        y_tests.append(y_test)

    plot_vmd(u, data, path)
    return [x_trains, y_trains, x_valids, y_valids, x_tests, y_tests, u]


# 绘制分解后的序列图
def plot_vmd(u, data, path):
    recombine = np.sum(u, axis=0)
    # . Visualize decomposed modes
    plt.rc('font', family='Times New Roman')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), layout='constrained')
    ax1.plot(data, label='original')
    # ax1.plot(recombine, label='reconstituted')
    ax1.set_title('Original wind speed data',fontdict={'family' : 'Times New Roman', 'size'   : 12})
    ax1.set_xlabel('Time(10min/point)',fontdict={'family' : 'Times New Roman', 'size'   : 10})
    ax1.set_ylabel('Wind speed(m/s)',fontdict={'family' : 'Times New Roman', 'size'   : 11})
    ax1.legend()
    ax1.grid(True,alpha=0.4)

    ax2.plot(u.T)
    ax2.set_title('The subsequences decomposed by VMD',fontdict={'family' : 'Times New Roman', 'size'   : 12})
    ax2.set_xlabel('Time(10min/point)',fontdict={'family' : 'Times New Roman', 'size'   : 10})
    ax2.set_ylabel('Wind speed(m/s)',fontdict={'family' : 'Times New Roman', 'size'   : 11})
    ax2.legend(['Subsequence %d' % m_i for m_i in range(u.shape[0])])
    ax2.grid(True,alpha=0.4)
    plt.savefig(path, dpi=600, bbox_inches='tight')
    plt.show()


def maponezero(data, direction="normal", maxmin=None):
    if direction == "normal":
        if maxmin != None:
            maxval = maxmin[0]
            minval = maxmin[1]
            data = (data - minval) / (maxval - minval)
        else:
            maxval = np.max(data)
            minval = np.min(data)
            data = (data - minval) / (maxval - minval)
        return [data, [maxval, minval]]
    if direction == "apply":
        maxval = maxmin[0]
        minval = maxmin[1]
        data = (data - minval) / (maxval - minval)
        return data
    if direction == "reverse":
        maxval = maxmin[0]
        minval = maxmin[1]
        data = data * (maxval - minval) + minval
        return data


def PICP(bound_data, true_data):
    count = 0
    n = true_data.shape[0]
    for i in range(n):
        upper = np.maximum(bound_data[i, 0], bound_data[i, 1])
        lower = np.minimum(bound_data[i, 0], bound_data[i, 1])
        if true_data[i] >= lower and true_data[i] <= upper:
            count += 1
    return count / n


def PINRW(bound_data, true_data):
    R = np.max(true_data) - np.min(true_data)
    N = true_data.shape[0]
    return np.sqrt(np.sum((bound_data[:, 0] - bound_data[:, 1]) ** 2) / N) / R


def PINAW(bound_data, true_data):
    R = np.max(true_data) - np.min(true_data)
    N = true_data.shape[0]
    return np.sum(np.abs(bound_data[:, 0] - bound_data[:, 1])) / N / R

def FINAW(bound_data):
    N = len(bound_data)
    return np.sum(np.abs(bound_data[:, 0] - bound_data[:, 1])) / N


# 其中η1是PICP的惩罚因子，η2用于线性增加PINRW的影响，μ是用于评估的PINC，γ（PICP）是阶跃函数，当PICP低于μ时，将强制执行PICP的惩罚
# η1 = 6 ,η2 = 15, u = 0.9
def CWC(bound_data, true_data):
    pinaw = PINAW(bound_data, true_data)
    picp = PICP(bound_data, true_data)
    u = 0.90
    if picp >= u:
        cwc = 1 + 6 * pinaw
    else:
        cwc = (1 + 6 * pinaw) * (1 + np.exp(15 * (u - picp)))
    return cwc


# 其中η1是PICP的惩罚因子，η2用于线性增加PINRW的影响，μ是用于评估的PINC，γ（PICP）是阶跃函数，当PICP低于μ时，将强制执行PICP的惩罚
# η1 = 6 ,η2 = 15, u = 0.9
def CWC1(picp, pinrw):
    u = 0.90
    if picp >= u:
        cwc = 1 + 6 * pinrw
    else:
        cwc = (1 + 6 * pinrw) * (1 + np.exp(15 * (u - picp)))
    return cwc


def INAD(bound_data, true_data):
    count = 0
    N = true_data.shape[0]
    average = np.sum(np.abs(bound_data[:, 0] - bound_data[:, 1])) / N
    for i in range(N):
        upper = np.maximum(bound_data[i, 0], bound_data[i, 1])
        lower = np.minimum(bound_data[i, 0], bound_data[i, 1])
        if true_data[i] < lower:
            count += (lower - true_data[i]) / average
        elif true_data[i] > upper:
            count += (true_data[i] - upper) / average
    return count / N


def RMSE(predict_data, true_data):
    return np.sqrt(np.sum((true_data - predict_data) ** 2) / len(true_data))


def MAE(predict_data, true_data):
    return np.sum(np.abs(true_data - predict_data)) / len(true_data)


def MAPE(predict_data, true_data):
    return 100 * np.sum(np.abs((true_data - predict_data) / true_data)) / len(true_data)
