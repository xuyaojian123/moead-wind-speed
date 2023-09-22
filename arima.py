#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File  : arima.py
@Author: XuYaoJian
@Date  : 2022/9/2 22:24
@Desc  : 
"""

from pmdarima import auto_arima


def arima(train_data):
    # m:7 - daily
    arima_model = auto_arima(train_data, start_p=1, start_q=1,
                             max_p=3, max_q=3, m=7,
                             start_P=0, seasonal=True,
                             d=1, D=1, trace=True,
                             error_action='ignore',
                             suppress_warnings=True,
                             stepwise=True)
    return arima_model
