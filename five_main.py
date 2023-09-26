#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File  : main.py
@Author: XuYaoJian
@Date  : 2022/9/3 12:07
@Desc  : 
"""
from hurst import calcHurst2
from utils import load_data_with_vmd, CWC1, PICP, PINRW
from moead import initial, main_loop, MOEAD
from nsgaII import MyProblem

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import tensorflow as tf
import geatpy as ea

from gru import gru
from svr import svr
from arima import arima


def set_tf_device(device):
    if device == 'cpu':
        print("Training on CPU...")
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    elif device == 'gpu':
        gpus = tf.config.experimental.list_physical_devices("GPU")
        tf.config.set_visible_devices(gpus[2], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[2], True)
        print("Training on GPU...")
        # for gpu in gpus:
        #     tf.config.experimental.set_memory_growth(gpu, True)


# set_tf_device('gpu')
set_tf_device('cpu')


def main():
    filename = "washingtong_summer_20120701-20120707新"
    step = "/five"
    wind_speed = pd.read_csv("../data/week_data/data/" + filename + ".csv")[
        'wind speed at 100m (m/s)']

    fig1, ax1 = plt.subplots(figsize=(8, 4), layout='constrained')
    ax1.plot(wind_speed, label='Samples(10min/point)')
    ax1.legend()
    ax1.grid()
    plt.savefig("result/" + filename + step + "/" + filename + ".png")
    plt.show()

    wind_speed_cluster = np.array(wind_speed).reshape(-1, 1)
    n_clusters = 10
    clusters = KMeans(n_clusters=n_clusters, random_state=66).fit(wind_speed_cluster)
    width = cal_width(n_clusters, clusters.labels_, wind_speed_cluster)
    fig1, ax1 = plt.subplots(figsize=(8, 4), layout='constrained')
    ax1.scatter(clusters.labels_, wind_speed_cluster)
    ax1.legend()
    ax1.grid()
    plt.savefig("result/" + filename + step + "/cluster.png")
    plt.show()

    seq_len = 9
    vmd_k = 5
    path = "result/" + filename + step + "/VMD.png"
    x_trains, y_trains, x_valids, y_valids, x_tests, y_tests, u = load_data_with_vmd(wind_speed, vmd_k, seq_len, path)
    predict_valids = []
    models = []
    for i in range(0, vmd_k):
        print(f"《《---------------------------训练第{i+1}个模型-----------------------------》》")
        # hurst_value = calcHurst2(u[i])
        if i == 0:
            hurst_value = 0.8
        elif i % 2 == 0:
            hurst_value = 0.1
        else:
            hurst_value = 0.4

        if 0 <= hurst_value < 0.4:
            flag = 1
        elif 0.4 <= hurst_value < 0.6:
            flag = 2
        else:
            flag = 3

        if flag == 1:
            model = gru(seq_len, 0.01, 32)
            callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
            model.fit(x_trains[i], y_trains[i], validation_data=(x_valids[i], y_valids[i]), epochs=150, batch_size=64,
                      callbacks=[callback], verbose=1)
            predict_valid = model.predict(x_valids[i])
            predict_valid = tf.squeeze(predict_valid)
            models.append([model, 1])
        if flag == 2:
            model = svr()
            x = np.squeeze(x_trains[i])
            y = np.squeeze(y_trains[i])
            model.fit(x, y)
            x_valid = np.squeeze(x_valids[i])
            predict_valid = model.predict(x_valid)
            models.append([model, 2])
        if flag == 3:
            predict_valid = []
            sequence = []
            for idx in range(len(x_trains[i])):
                if idx == 0:
                    for s in range(seq_len):
                        sequence.append(x_trains[i][idx][s][0])
                sequence.append(y_trains[i][idx][0])
            model = arima(sequence)
            for idx in range(len(x_valids[i])):
                print(idx, len(x_valids[i]))
                model.fit(sequence)
                forecast = model.predict(n_periods=1)
                predict_valid.append(forecast)
                sequence.pop(0)
                sequence.append(y_valids[i][idx][0])
            models.append([model, 3])
        predict_valid = np.array(predict_valid)
        predict_valid = np.squeeze(predict_valid)
        predict_valids.append(predict_valid)

    for i in range(len(predict_valids)):
        val_mse = mean_squared_error(np.squeeze(y_valids[i]), predict_valids[i])
        val_mae = mean_absolute_error(np.squeeze(y_valids[i]), predict_valids[i])
        val_rmse = np.sqrt(val_mse)
        val_mape = mean_absolute_percentage_error(np.squeeze(y_valids[i]), predict_valids[i])
        print(f"mode{i}-------》val_mse:{val_mse},val_mae:{val_mae},val_rmse:{val_rmse},val_mape:{val_mape}")

        fig1, ax1 = plt.subplots(figsize=(6, 3), layout='constrained')
        ax1.plot(predict_valids[i], label='predict')
        ax1.plot(y_valids[i], label='true')
        ax1.set_title(f'model:{i}')
        ax1.set_xlabel('Time(10min/point)')
        ax1.legend()
        ax1.grid()
        plt.savefig("result/" + filename + step + f"/validation_model{i}.png")
        plt.show()
    predict_valids = np.array(predict_valids)
    predict_valid_reverse = np.sum(predict_valids, axis=0)
    val_mse = mean_squared_error(np.squeeze(y_valids[vmd_k]), predict_valid_reverse)
    val_mae = mean_absolute_error(np.squeeze(y_valids[vmd_k]), predict_valid_reverse)
    val_rmse = np.sqrt(val_mse)
    val_mape = mean_absolute_percentage_error(np.squeeze(y_valids[vmd_k]), predict_valid_reverse)
    print(f"reconstruct wind speed-------》val_mse:{val_mse},val_mae:{val_mae},val_rmse:{val_rmse},val_mape:{val_mape}")

    fig1, ax1 = plt.subplots(figsize=(6, 3), layout='constrained')
    ax1.plot(predict_valid_reverse, label='predict')
    ax1.plot(y_valids[vmd_k], label='true')
    ax1.set_title(f'validation reconstruct wind speed')
    ax1.set_xlabel('Time(10min/point)')
    ax1.legend()
    ax1.grid()
    plt.savefig("result/" + filename + step + "/validation_reconstruct_wind_speed.png")
    plt.show()

    MOEAD.variables = vmd_k
    MOEAD.save_filename = "result/" + filename + step + "/"
    MOEAD.kmeans = clusters
    MOEAD.predict = predict_valids
    MOEAD.true_data = np.squeeze(y_valids[vmd_k])
    MOEAD.width = width
    moead = MOEAD()
    # P, moead = initial()
    # main_loop(P, moead)

    # 实例化问题对象
    problem = MyProblem(moead)
    # 构建算法
    algorithm = ea.moea_NSGA2_templet(problem,
                                      ea.Population(Encoding='RI', NIND=40),
                                      MAXGEN=10,  # 最大进化代数
                                      logTras=0)  # 表示每隔多少代记录一次日志信息，0表示不记录。
    # 求解
    res = ea.optimize(algorithm, seed=1, verbose=False, drawing=1, outputMsg=True, drawLog=False, saveFlag=False,
                      dirName='result/' + filename)

    last_pop_obj = res['lastPop'].ObjV
    last_pop_var = res['lastPop'].Phen

    opt_pop_obj = res['optPop'].ObjV
    opt_pop_var = res['optPop'].Phen
    min_cwc = np.inf
    weight = []
    for i in range(0, len(opt_pop_obj)):
        picp = 1 - opt_pop_obj[i][0]
        pinrw = opt_pop_obj[i][1]
        cwc = CWC1(picp, pinrw)
        if cwc < min_cwc:
            min_cwc = cwc
            weight = opt_pop_var[i]

    save_opt_pop = np.column_stack((opt_pop_var, opt_pop_obj))
    save_opt_pop = pd.DataFrame(save_opt_pop)
    save_opt_pop.to_csv('result/' + filename + step + '/pareto.csv', header=False, index=False)

    ppp = len(x_tests[0]) % 5
    predict_tests = []
    for i in range(0, vmd_k):
        if models[i][1] == 1:
            predict_test = []
            for index in range(0, len(x_tests[i])-ppp, 5):
                x = x_tests[i][index].reshape(1, -1, 1)
                x = np.array(x)
                for num in range(5):
                    forecast = models[i][0].predict(x)
                    forecast = np.array(forecast).reshape(-1)
                    x = np.append(x, forecast)
                    x = x[1:]
                    x = x.reshape(1, -1, 1)
                    predict_test.append(forecast)
        elif models[i][1] == 2:
            predict_test = []
            for index in range(0, len(x_tests[i])-ppp, 5):
                x = x_tests[i][index].reshape(1,-1)
                for num in range(5):
                    forecast = models[i][0].predict(x)
                    x = np.append(x, forecast)
                    x = x[1:]
                    x = x.reshape(1, -1)
                    predict_test.append(forecast)
        elif models[i][1] == 3:
            predict_test = []
            for index in range(0, len(x_tests[i])-ppp, 5):
                print(index, len(x_tests[i]))
                for num in range(5):
                    models[i][0].fit(sequence)
                    forecast = models[i][0].predict(n_periods=1)
                    sequence.pop(0)
                    sequence.append(forecast)
                    predict_test.append(forecast)
                sequence = sequence[:-5]
                sequence.append(y_tests[i][index][0])
                sequence.append(y_tests[i][index+1][0])
                sequence.append(y_tests[i][index + 2][0])
                sequence.append(y_tests[i][index + 3][0])
                sequence.append(y_tests[i][index + 4][0])
        predict_test = np.array(predict_test)
        predict_test = np.squeeze(predict_test)
        predict_tests.append(predict_test)

    up = []
    down = []
    weight = np.array(weight)
    temp = weight.reshape(-1, 1)
    a = predict_tests * temp
    result = np.sum(a, axis=0)
    b = result.reshape(-1, 1)
    predict_class = clusters.predict(b)
    for idx, pre in enumerate(result):
        lambdas = width[predict_class[idx]]
        up.append(pre + lambdas)
        down.append(pre - lambdas)
    bound_data = np.column_stack((down, up))
    rest = len(bound_data)
    test_picp = PICP(bound_data, np.squeeze(y_tests[vmd_k][:rest]))
    test_pinrw = PINRW(bound_data, np.squeeze(y_tests[vmd_k][:rest]))
    test_cwc = CWC1(test_picp,test_pinrw)
    print("final result")
    print(test_picp,test_pinrw,test_cwc)

    fig1, ax1 = plt.subplots(figsize=(6, 3), layout='constrained')
    ax1.plot(down, label='down')
    ax1.plot(up, label='up')
    ax1.plot(y_tests[vmd_k], label='test true')
    ax1.set_title(f'test wind speed reconstruct')
    ax1.set_xlabel('Time(10min/point)')
    ax1.legend()
    ax1.grid()
    plt.savefig("result/" + filename + step + f"/test_reconstruct_wind_speed.png")
    plt.show()
    rest = len(bound_data)
    final_result = np.column_stack((bound_data, np.squeeze(y_tests[vmd_k][:rest])))
    final_result = pd.DataFrame(final_result)
    final_result.to_csv('result/' + filename + step + f'/picp:{test_picp:.5f}-pinrw:{test_pinrw:.5f}-cwc:{test_cwc:.5f}.csv', header=False, index=False)


def cal_width(n_clusters, labels, wind_speed_cluster):
    width = []
    print("聚类每个类的数目:")
    for x in range(n_clusters):
        count = []
        for idx, label in enumerate(labels):
            if label == x:
                count.append(wind_speed_cluster[idx])
        count = np.array(count)
        std = np.std(count)
        # %95的置信区间宽度
        alpha = (1.96 * std) / 2 + 0.45
        width.append(alpha)
        print(count.shape)
    return width


if __name__ == "__main__":
    main()
