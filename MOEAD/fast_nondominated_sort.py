# import random
# import winsound
# import math
# import operator
import numpy as np
import pandas as pd

# import matplotlib.pyplot as mpl
# from mpl_toolkits.mplot3d import Axes3D
#
# # General import structure
# # from CASE_STUDY import MOOP, nonlincnstr
#
#
# class Solution:
#     def __init__(self):
#         self.Position = []
#         self.Cost = []
#         self.Rank = 0
#         self.DominationSet = []
#         self.DominatedCount = 0
#         self.CrowdingInWaiting = []
#         self.CrowdingDistance = 0

# https://github.com/joejoseph007/2018_09_05/blob/master/Algorithms/2.NSGA/NSGA-II.py
# Function to carry out NSGA-II's fast non dominated sort
def fast_non_dominated_sort(pop, gene_num):
    fitness1, fitness2 = pop[:, gene_num], pop[:, gene_num + 1]

    S = [[] for i in range(0, len(fitness1))]
    front = [[]]
    n = [0 for i in range(0, len(fitness1))]
    rank = [0 for i in range(0, len(fitness1))]

    for p in range(0, len(fitness1)):
        S[p] = []
        n[p] = 0
        for q in range(0, len(fitness1)):
            if (fitness1[p] < fitness1[q] and fitness2[p] < fitness2[q]) or (
                    fitness1[p] <= fitness1[q] and fitness2[p] < fitness2[q]) or (
                    fitness1[p] < fitness1[q] and fitness2[p] <= fitness2[q]):
                if q not in S[p]:
                    S[p].append(q)
            elif (fitness1[q] < fitness1[p] and fitness2[q] < fitness2[p]) or (
                    fitness1[q] <= fitness1[p] and fitness2[q] < fitness2[p]) or (
                    fitness1[q] < fitness1[p] and fitness2[q] <= fitness2[p]):
                n[p] = n[p] + 1
        if n[p] == 0:
            rank[p] = 0
            if p not in front[0]:
                front[0].append(p)

    i = 0
    while (front[i] != []):
        Q = []
        for p in front[i]:
            for q in S[p]:
                n[q] = n[q] - 1
                if (n[q] == 0):
                    rank[q] = i + 1
                    if q not in Q:
                        Q.append(q)
        i = i + 1
        front.append(Q)

    # del front[len(front) - 1]
    # pop = np.insert(pop, 8, rank, axis=1)
    dominant=[]
    for i in front[0]:
        dominant.append(pop[i])
    Pop_temp = pd.DataFrame(pop)
    # sort(pop,gene_num)
    # save final population
    Pop_temp.to_csv( '../dominate.csv')



    # front: list each level 's individual
    # rank: illustrate the individual belong to some level
    # return front,rank



# # Function to calculate crowding distance
# def crowding_distance(values1, values2, front):
#     distance = [0 for i in range(0, len(front))]
#     sorted1 = sort_by_values(front, values1[:])
#     sorted2 = sort_by_values(front, values2[:])
#     distance[0] = 4444444444444444
#     distance[len(front) - 1] = 4444444444444444
#     for k in range(1, len(front) - 1):
#         distance[k] = distance[k] + (values1[sorted1[k + 1]] - values2[sorted1[k - 1]]) / (max(values1) - min(values1))
#     for k in range(1, len(front) - 1):
#         distance[k] = distance[k] + (values1[sorted2[k + 1]] - values2[sorted2[k - 1]]) / (max(values2) - min(values2))
#
#     return distance








# def CrowdDistance(B):
#     n=len(B)
#     for i in range(len(B[0].f)):
#         B.sort(key=lambda p:p.f[i])   #sorted by values of f1 and f2
#         B[0].crowd=float('inf')
#         B[n-1].crowd=float('inf')
#         h=B[n-1].f[i]-B[0].f[i]
#         for j in range(1,n-1):
#             B[j].crowd+=((B[j+1].f[i]-B[j-1].f[i])/h)

def crowding_distance(pop,individual_num,gene_num):
    crowded_distance=[0]*individual_num
    pop = np.insert(pop, 10, crowded_distance, axis=1)
    # 2: the number of the objective value
    for i in range(2):
        pop = sorted(pop, key=lambda x: x[gene_num+i])
        pop=np.array(pop)
        pop[0][10]=float('inf')
        pop[individual_num-1][10]=float('inf')
        h=(max(pop[:,8+i]) - min((pop[:,8+i])))
        for j in range(1,individual_num-1):
            # crowding_distance[j]+=((pop[][j+1].f[i]-B[j-1].f[i])/h)
            pop[j][10]+=((pop[:,i+gene_num][j+1]-pop[:,i+gene_num][j-1])/h)
    # print(pop)
    # inv_y = pd.DataFrame(pop)
    # inv_y.to_csv('result\\final\\b.csv')
    # print(crowding_distance)

    # crowded_distance=pop[:,10]
    return pop


# Function to sort by values

def sort_by_values(list1, values):
    sorted_list = []
    while(len(sorted_list)!=len(list1)):
        if index_of(min(values),values) in list1:
            sorted_list.append(index_of(min(values),values))
        values[index_of(min(values),values)] = float('inf')
    return sorted_list


# First function to optimize
def function1(x):
    value = -x ** 2
    return value


# Second function to optimize
def function2(x):
    value = -(x - 2) ** 2
    return value


# Function to find index of list
def index_of(a, list):
    for i in range(0, len(list)):
        if list[i] == a:
            # print(i)
            return i
    return -1
