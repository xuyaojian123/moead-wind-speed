#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File  : svr.py
@Author: XuYaoJian
@Date  : 2022/9/2 22:24
@Desc  : 
"""

from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
import numpy as np


def svr():
    # 网格化搜索寻找最优参数
    model = GridSearchCV(SVR(kernel="rbf", gamma=0.1, verbose=True),
                         param_grid={"C": [1e0, 1e1, 1e2, 1e3], "gamma": np.logspace(-2, 2, 5)})
    return model
