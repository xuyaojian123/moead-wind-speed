#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File  : nsgaII.py
@Author: XuYaoJian
@Date  : 2022/9/5 20:20
@Desc  : 
"""
import geatpy as ea
import numpy as np
import pandas as pd
from fitness import cal_fitness1

class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self, moead):
        self.moead = moead
        name = 'MyProblem'  # 初始化name（函数名称，可以随意设置）
        M = 2  # 优化目标个数
        maxormins = [1] * M  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        Dim = moead.variables  # 初始化Dim（决策变量维数）
        varTypes = [0] * Dim  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        lb = [0] * Dim # 决策变量下界
        ub = [3] * Dim # 决策变量上界
        lbin = [1] * Dim # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1] * Dim # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        self.flag = 1
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def evalVars(self, Vars):  # 目标函数,Vars为二维矩阵，行表示种群大小，列表示每个个体变量取值
        f1 = []
        f2 = []
        for i in range(0, Vars.shape[0]):
            picp, pinrw = cal_fitness1(self.moead.true_data, self.moead.predict, self.moead.width, self.moead.kmeans, Vars[i])
            f1.append(picp)
            f2.append(pinrw)
        f1 = np.array(f1).reshape(-1, 1)
        f2 = np.array(f2).reshape(-1, 1)
        ObjV = np.hstack([f1, f2])  # 计算目标函数值矩阵
        if self.flag == 1:
            first_pop = np.column_stack((Vars, ObjV))
            first_pop = pd.DataFrame(first_pop)
            first_pop.to_csv(self.moead.save_filename + 'first_pop.csv', header=False, index=False)
        return ObjV





