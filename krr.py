from __future__ import division
import numpy as np
from sklearn.kernel_ridge import KernelRidge
import matplotlib.pyplot as plt
# from fylearn.ga import UnitIntervalGeneticAlgorithm, helper_fitness, helper_n_generations
# from fylearn.jaya import JayaOptimizer
import math
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
# from newssa import SSA
# from ccc import SSA

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
rol=4
# load pima indians dataset
trainx = 'trainx.txt'
trainy = 'trainy.txt'
x_train = np.loadtxt(trainx)
y_train = np.loadtxt(trainy)
x = np.array(x_train)
y = np.array(y_train)
x_train = x.reshape(-1, rol)
y_train = y.reshape(-1, 1)
i,x0_train,x1_train,x2_train,y0_train,y1_train,y2_train=0,[],[],[],[],[],[]
while i<12000:
    x0_train.append(x_train[i])
    x1_train.append(x_train[i+1])
    x2_train.append(x_train[i+2])
    # ===============
    y0_train.append(y_train[i])
    y1_train.append(y_train[i+1])
    y2_train.append(y_train[i+2])
    i+=3

x_all= x_train[0:4000]
y_all= y_train[0:4000]
print(x_all)
print(y_all)

# =====================================================
# testx = 'testx.txt'
# testy = 'testy.txt'
# x_test = np.loadtxt(testx)
# y_test = np.loadtxt(testy)
# x = np.array(x_test)
# y = np.array(y_test)
# print(x.shape)
# x_test = x.reshape(-1, rol)
# y_test = y.reshape(-1, 1)
# i,x0_test,x1_test,x2_test,y0_test,y1_test,y2_test=0,[],[],[],[],[],[]
# while i<1200:
#     x0_test.append(x_test[i])
#     x1_test.append(x_test[i+1])
#     x2_test.append(x_test[i+2])
#     # ===============
#     y0_test.append(y_test[i])
#     y1_test.append(y_test[i+1])
#     y2_test.append(y_test[i+2])
#     i+=3
# print(np.logspace(-2, 2, 4))
# kr = GridSearchCV(KernelRidge(kernel='rbf', gamma=0.01), cv=5,
#                   param_grid={"alpha": [1e-0, 1e-1, 1e-2, 1e-3],
#                               "gamma": np.logspace(-2, 2, 10)})

# def fitness(x):
#     krr = KernelRidge(kernel='rbf',
#                       alpha=x[0],
#                       gamma=x[1])
#     krr.fit(x_all, y_all)
#     prediction = krr.predict(x2_test)
#     MSE = mean_squared_error(prediction, y2_test)
#     print("alpha,gamma,mse are:",x[0],x[1],MSE)
#     # print("mse is:",MSE)
#     return MSE
# # lower_bounds = np.array([1.0, 0.001, 100.0])
# # upper_bounds = np.array([100.0, 0.2, 2000.0])
#
# lower_bounds = np.array([1e-10, 1e-10])
# upper_bounds = np.array([1.0, 20.0])
# kr = JayaOptimizer(f=fitness,
#                   lower_bound=lower_bounds,
#                   upper_bound=upper_bounds,
#                    )
# kr = helper_n_generations(kr, 100)
# best_solution, best_fitness = kr.best()
# print("solution", best_solution, "fitness", best_fitness)
# k = KernelRidge(kernel='rbf',
#                   alpha=best_solution[0],
#                   gamma=best_solution[1])
#best sulotion=0.06393052,0.06393052
# k = KernelRidge(kernel='rbf',
#                   alpha=0.06393052,
#                   gamma=0.06393052)
# k.fit(x2_train, y2_train)
# p = k.predict(x2_test)
# i=0
# while i<400:
#     p[i]*=26.02
#     y2_test[i]*=26.02
#     i+=1
# np.savetxt("prediction_krr.txt", p)
# print("the mse is：",mean_squared_error(p, y2_test))
# print("the mse is：",mean_absolute_error(p, y2_test))
# x_ais=[i for i in range(0,400)]
# plt.plot(x_ais, y2_test, color='green',label='real_value',linewidth=1,markerfacecolor='black',markersize=2)
# plt.plot(x_ais, p, color='red',label='SVR',linewidth=1,markerfacecolor='blue',markersize=2)
# plt.ylabel("数据对比")
# plt.legend("128")  #show the figures
# #
# # # plt.savefig("361me.jpg")
# plt.show()



# Max_iteration,population,h=1000,5,2.5
#
# lower_bounds = np.array([1e-2, 1e-2, 1e-10, 1.0])
# # lower_bounds = np.array([-3, -1, -1, 1.0])
# upper_bounds = np.array([1, 1, 2, 500])
# def fit_ness(x):
#     # m = 1 - (1 / x[3])
#     # fitness =x[0] + (x[1] - x[0]) * math.pow(1 / (1 + x[2] * math.pow(abs(h), x[3])), m)
#     #
#     a = x[0]
#     b = x[1]
#     c = 1/(a*b)
#     fitness =math.pow(a,3)+math.pow(b,4)+math.pow(c,4)
#
#     return fitness
#
# Ssa = SSA(lower_bounds,upper_bounds,Max_iteration,population,h,fit_ness,2).loop_ineration()
# print(Ssa)





Max_iteration,population,h=100,50,2.5

lower_bounds = np.array([1e-10, 1e-10])
upper_bounds = np.array([1.0, 10])

def fit_ness(x):
    krr = KernelRidge(kernel='rbf',
                      alpha=x[0],
                      gamma=x[1])
    krr.fit(x2_train, y2_train)
    prediction = krr.predict(x2_test)
    MSE = mean_squared_error(prediction, y2_test)
    return MSE

Ssa = SSA(lower_bounds,upper_bounds,Max_iteration,population,h,fit_ness,2).loop_ineration()




plt.show()










