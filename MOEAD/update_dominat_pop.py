from MOEAD.fast_nondominated_sort import fast_non_dominated_sort as Sort
from MOEAD.fast_nondominated_sort import crowding_distance as Distance
# from MOEAD.double_sphere_crowding_distance import double_sphere_crowding_distance as DS_distance

import numpy as np


# def update_dominat_pop(pop, individual_num, gene_num, active_pop):
def update_dominat_pop(pop, moead):
    individual_num, gene_num, active_pop=moead.individual_num,moead.gene_num,moead.active_pop

    D = []

    Pop_distance = Distance(pop, len(pop), gene_num)
    # Pop_distance=DS_distance(pop,individual_num,gene_num,0.7)

    front,rank = Sort(Pop_distance, gene_num)

    for i in range(len(front)):
        if (len(D) < active_pop):
            if (active_pop - len(D)) >= len(front[i]):
                for j in range(len(front[i])):
                    D.append(Pop_distance[front[i][j], :])
            else:
                # pop_distance = Distance(pop, len(pop), gene_num)
                pop_temp = []
                for j in front[i]:
                    pop_temp.append(Pop_distance[j])
                pop_temp = sorted(pop_temp, key=lambda x: x[gene_num + 2],reverse=True)
                pop_temp = np.array(pop_temp)
                # pop_temp = np.delete(pop_temp, 10, axis=1)

                if (active_pop - len(D)) > (len(front[i])):
                    for index in range(len(front[i])):
                        D.append(pop_temp[index, :])
                else:
                    for index in range(active_pop - len(D)):
                        D.append(pop_temp[index, :])
        else:
            break



    # # choose the dominate population only by accuracy
    # pop = sorted(pop, key=lambda x: x[8])
    # D = pop[:active_pop]
    D=np.array(D)
    D = np.delete(D, 10, axis=1)
    # print(D)
    return D
