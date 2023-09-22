from MOEAD.MOEAD_Utils import *
import random
from MOEAD.normalization import normalize


def update_pop(P, moead, P_B, Y):
    # 根据Y更新P_B集内邻居
    for j in P_B:
        Xj = P[j]
        d_x = cpt_tchbycheff(moead, j, Xj)
        d_y = cpt_tchbycheff(moead, j, Y)
        if d_y <= d_x:
            # d_y 的切比雪夫距离更小
            if P[j][moead.variables] > Y[moead.variables]:
                P[j] = Y[:]
            elif random.random() > 0.7:
                P[j] = Y[:]

