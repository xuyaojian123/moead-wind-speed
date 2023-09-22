#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File  : fitness.py
@Author: XuYaoJian
@Date  : 2022/9/5 11:46
@Desc  : 
"""
import numpy as np
from utils import PICP, PINRW


def cal_fitness(true, predict, width, kmeans, individual):
    up = []
    down = []
    temp = individual[:-2].reshape(-1, 1)
    a = predict * temp
    result = np.sum(a, axis=0)
    b = result.reshape(-1, 1)
    predict_class = kmeans.predict(b)
    for idx, pre in enumerate(result):
        lambdas = width[predict_class[idx]]
        up.append(pre + lambdas)
        down.append(pre - lambdas)
    bound_data = np.column_stack((down, up))
    picp = PICP(bound_data, true)
    pinrw = PINRW(bound_data, true)
    return 1 - picp, pinrw


def cal_fitness1(true, predict, width, kmeans, individual):
    up = []
    down = []
    temp = individual.reshape(-1, 1)
    a = predict * temp
    result = np.sum(a, axis=0)
    b = result.reshape(-1, 1)
    predict_class = kmeans.predict(b)
    for idx, pre in enumerate(result):
        lambdas = width[predict_class[idx]]
        up.append(pre + lambdas)
        down.append(pre - lambdas)
    bound_data = np.column_stack((down, up))
    picp = PICP(bound_data, true)
    pinrw = PINRW(bound_data, true)
    return 1 - picp, pinrw
