#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File  : gru.py
@Author: XuYaoJian
@Date  : 2022/9/2 22:24
@Desc  : 
"""

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, GRU
from tensorflow.keras import Input
from tensorflow.keras.optimizers import Adam, SGD


def gru(seq_len, learning_rate, hidden_size):
    # 建立模型
    optimizer = Adam(learning_rate=learning_rate)
    model = Sequential()
    model.add(Input(shape=(seq_len, 1)))
    model.add(GRU(units=hidden_size, return_sequences=False, kernel_initializer='glorot_normal'))
    model.add(Dense(int(hidden_size/2), kernel_initializer='he_normal', activation='relu'))
    model.add(Dense(int(hidden_size/4), kernel_initializer='he_normal', activation='relu'))
    model.add(Dense(int(hidden_size/8), kernel_initializer='he_normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='he_normal'))
    model.compile(optimizer=optimizer, loss='mse')
    return model


# model = gru(9,0.001,32)
# total_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
# print("模型参数总量：", total_params)
# 
# # 输出每一层的参数数量
# print(model.summary())
