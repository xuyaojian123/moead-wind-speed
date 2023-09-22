import numpy as np
import random
import pandas as pd


# generate initial population
def pop_init(pop_num, variables):
    pop = np.zeros([pop_num + 1, variables + 2])
    for i in range(0, pop_num + 1):
        for j in range(0, variables):
            # [0, 3)
            pop[i][j] = np.random.rand()*3
        pop[i][variables] = 0.
        pop[i][variables+1] = 0.
    # print(pop)
    # inv_y = pd.DataFrame(pop)
    # inv_y.to_csv('result/washingtong_winter_20120101-20120107/pop_init_1.csv', header=False, index=False)
    return pop

# pop_init(20,5)
