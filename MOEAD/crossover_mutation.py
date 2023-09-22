import numpy as np
import random
from fitness import cal_fitness


def crossover_mutation(moead, p1, p2):
    y1 = np.copy(p1)
    y2 = np.copy(p2)

    # crossover
    if np.random.rand() < moead.c_rate:
        y1, y2 = crossover(moead, y1, y2)

    # # mutation
    # if np.random.rand()<moead.m_rate:
    #     y1=mutation(moead,y1)
    #     y2=mutation(moead,y2)
    y1 = mutation(moead, y1)
    y2 = mutation(moead, y2)

    # calculate the fitness of the new individual
    picp1, pinrw1 = cal_fitness(moead.true_data, moead.predict, moead.width, moead.kmeans, y1)
    picp2, pinrw2 = cal_fitness(moead.true_data, moead.predict, moead.width, moead.kmeans, y2)

    y1[moead.variables] = picp1
    y1[moead.variables + 1] = pinrw1

    y2[moead.variables] = picp2
    y2[moead.variables + 1] = pinrw2

    y1 = [y1]
    y2 = [y2]
    y1 = np.array(y1)
    y2 = np.array(y2)

    return y1[0], y2[0]


# two-point crossover
def crossover(moead, p1, p2):
    status = True
    # generate two crossover point
    while status:
        k1 = random.randint(0, moead.variables - 1)
        k2 = random.randint(0, moead.variables)
        if k1 < k2:
            status = False

    fragment1 = p1[k1: k2]
    fragment2 = p2[k1: k2]

    p1[k1: k2] = fragment2
    p2[k1: k2] = fragment1

    return p1, p2


# bitwise mutation
def mutation(moead, p):
    print('before mutation:', list(p))
    for j in range(moead.variables):
        if random.random() < moead.m_rate:
            new_gene = np.random.rand() * 3
            p[j] = new_gene
    print('after mutation', list(p))
    return p
