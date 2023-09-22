import numpy as np


def Tchebycheff_dist(w, f, z):
    # 计算切比雪夫距离
    return w * abs(f - z)


def cpt_tchbycheff(moead, idx, X):
    # idx：X在种群中的位置
    # 计算X的切比雪夫距离（与理想点Z的）

    max = moead.Z[0]
    ri = moead.W[idx]
    # F_X = moead.Test_fun.Func(X)

    F_X = X[moead.variables:moead.variables + 2]
    # 返回远离理想点最远的
    for i in range(moead.object_num):
        fi = Tchebycheff_dist(ri[i], F_X[i], moead.Z[i])
        if fi > max:
            max = fi
    return max
