#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File  : hurst.py
@Author: XuYaoJian
@Date  : 2022/9/3 21:17
@Desc  : 
"""
import numpy as np
from collections.abc import Iterable
from pandas import Series


def calcHurst2(ts):
    if not isinstance(ts, Iterable):
        print('error')
        return
    n_min, n_max = 2, len(ts) // 3
    RSlist = []
    for cut in range(n_min, n_max):
        children = len(ts) // cut
        children_list = [ts[i * children:(i + 1) * children] for i in range(cut)]
        L = []
        for a_children in children_list:
            Ma = np.mean(a_children)
            Xta = Series(map(lambda x: x - Ma, a_children)).cumsum()
            Ra = max(Xta) - min(Xta)
            Sa = np.std(a_children)
            rs = Ra / Sa
            L.append(rs)
        RS = np.mean(L)
        RSlist.append(RS)
    hurst_value = np.polyfit(np.log(range(2 + len(RSlist), 2, -1)), np.log(RSlist), 1)[0]
    return hurst_value
