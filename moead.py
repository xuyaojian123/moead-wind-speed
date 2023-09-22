#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File  : moead.py
@Author: XuYaoJian
@Date  : 2022/9/2 22:24
@Desc  : 
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File  : cnn-lstm.py
@Author: XuYaoJian
@Date  : 2022/7/21 11:53
"""

import pandas as pd

from MOEAD.weight_initial import *
from initial_pop import initial_pop

from MOEAD.normalization import normalize2
from MOEAD.update_Z import update_Z
from MOEAD.generate_next import generate_next
from MOEAD.update_pop import update_pop

class MOEAD:
    individual_num = 30  # num of the population

    max_gen = 5  # num of the iteration

    gen = 0  # current generation
    object_num = 2  # num of the objective value
    epoch = 120  # num of the epoch

    c_rate = 1  # rate of crossover
    m_rate = 0.5  # rate of mutation
    Max, Min = 0, float('inf')  # max and min of the weight number

    W = []  # weight
    W_Bi_T = []  # T neighbour of the weight
    Z = []  # iedeal point
    T_size = 3  # num of the neighbour(crossover and update only in the neighbours)

    save_filename = None
    variables = None  # num of the parameters
    kmeans = None
    predict = None
    true_data = None
    width = None


############################## Initialization###############################################
def initial():
    # instantiate a MOEAD class
    moead = MOEAD()

    # load the reference vector
    load_weight(moead)

    # Calculate T neighbors for each weighted Wi
    cpt_W_Bi_T(moead)

    # generate initial population
    P = initial_pop(moead)

    update_Z(moead, P)
    return P, moead


def main_loop(P, moead):
    for gen in range(moead.max_gen):
        print('#######################################################################################')
        print("########################## Main loop: The %dth generation##############################" % (gen + 1))

        moead.gen = gen
        for pi, p in enumerate(P):
            print('#######################################################################################')
            print("###################### The %dth individual of the %dth generation ######################" % (
                (pi + 1), (gen + 1)))
            # neighbor sets of the pi th individual
            Bi = moead.W_Bi_T[pi]

            # randomly select two num as the neighbors of pi th
            k = np.random.randint(moead.T_size)
            l = np.random.randint(moead.T_size)

            ik = Bi[k]
            il = Bi[l]

            Xi = P[pi]
            Xk = P[ik]
            Xl = P[il]

            # generate new individual
            Y = generate_next(moead, gen, pi, Xi, Xk, Xl)

            # updata population
            update_pop(P, moead, Bi, Y)
            update_Z(moead, P)

        P_temp = pd.DataFrame(P)
        P_temp.to_csv(moead.save_filename + 'pop_iteration_' + str(gen + 1) + '.csv', header=False, index=False)


# if __name__ == "__main__":
#     P, moead = initial()
#     main_loop(P, moead)
