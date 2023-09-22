import numpy as np
from MOEAD.MOEAD_Utils import cpt_tchbycheff
from MOEAD.crossover_mutation import crossover_mutation


# generate new individual
# gen:  the 'gen' generation
# wi: the 'wi' th individual in the population   P0=P[wi]
# p1,p2: the neighbors of the Xi

def generate_next(moead, gen, wi, p0, p1, p2):
    # Before Mutation
    qbxf_p0 = cpt_tchbycheff(moead, wi, p0)
    qbxf_p1 = cpt_tchbycheff(moead, wi, p1)
    qbxf_p2 = cpt_tchbycheff(moead, wi, p2)

    qbxf = np.array([qbxf_p0, qbxf_p1, qbxf_p2])
    best = np.argmin(qbxf)

    # pick the smallest tchbycheff individual
    Y1 = [p0, p1, p2][best]

    # After mutation
    n_p0, n_p1, n_p2 = np.copy(p0), np.copy(p1), np.copy(p2)
    if np.random.rand() > 0.5:
        n_p1 = n_p2

    n_p0, n_p1 = crossover_mutation(moead, n_p0, n_p1)

    # calculate tchbycheff after crossover and mutation
    qbxf_np0 = cpt_tchbycheff(moead, wi, n_p0)
    qbxf_np1 = cpt_tchbycheff(moead, wi, n_p1)

    qbxf = np.array([qbxf_np0, qbxf_np1])
    best = np.argmin(qbxf)
    Y2 = [n_p0, n_p1][best]

    fm = 0
    # prevent local optimality
    if np.random.rand() < 0.25:
        FY1 = Y1[moead.variables + fm]
        FY2 = Y2[moead.variables + fm]

        return Y1 if FY1 < FY2 else Y2
    return Y2

