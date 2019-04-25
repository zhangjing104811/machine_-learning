#encoding = utf-8
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from fylearn.ga import UnitIntervalGeneticAlgorithm,helper_fitness,helper_n_generations
from fylearn.jaya import JayaOptimizer
from sklearn.metrics import mean_squared_error
import math
import threading


def fitness(x,y):
    krr = KernelRidge(kernel='rbf',
                      alpha=x[0],
                      gamma=x[1])
    krr.fit(x,y)
    pred = krr.fit(x)
    loss = mean_squared_error(pred,y)
    return loss
low_bounds = np.array([1e-8,1e-8])
high_bounds = np.array([1e-8,1e-8])

jaya = JayaOptimizer(f=fitness,
                     lower_bound=low_bounds,
                     upper_bound=high_bounds,
                     n_population = 10)
krr = helper_n_generations(jaya,5)
best_solution,best_fitness = krr.best()
print("solution", best_solution, "fitness", best_fitness)