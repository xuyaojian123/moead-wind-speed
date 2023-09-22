from MOEAD.pop_init import pop_init
from fitness import cal_fitness
import pandas as pd


def initial_pop(moead):
    print(
        '***************************************************************************************************************************')
    print(
        '**************************************************** Initialization *******************************************************')

    # ############ 1.generate initial population #########################
    # randomly generate population
    pop = pop_init(moead.individual_num, moead.variables)
    # build model and calculate the fitness
    for i in range(moead.individual_num + 1):
        print("Initialization: The %dth individual" % (i + 1))
        picp, pinrw = cal_fitness(moead.true_data, moead.predict, moead.width, moead.kmeans, pop[i])
        pop[i][moead.variables] = picp
        pop[i][moead.variables + 1] = pinrw
    # save the initial population
    # print(pop)
    inv_y = pd.DataFrame(pop)
    inv_y.to_csv(moead.save_filename+'pop_init_1.csv', header=False, index=False)

    ################ 2.load existed initial population######################
    # pop = pd.read_csv('result/pop_init_1.csv', header=None)
    # pop = np.array(pop)
    return pop
