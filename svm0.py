from pandas import Series,DataFrame
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
# from fylearn.ga import UnitIntervalGeneticAlgorithm, helper_fitness, helper_n_generations
# from fylearn.jaya import JayaOptimizer

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
# fix random seed for reproducibility
# load pima indians dataset
trainx = 'trainx.txt'
trainy = 'trainy.txt'
x_train = np.loadtxt(trainx)
y_train = np.loadtxt(trainy)
i,x0_train,x1_train,x2_train,y0_train,y1_train,y2_train=0,[],[],[],[],[],[]
while i<len(x_train):
    x0_train.append(x_train[i])
    x1_train.append(x_train[i+1])
    x2_train.append(x_train[i+2])
    y0_train.append(y_train[i])
    y1_train.append(y_train[i+1])
    y2_train.append(y_train[i+2])
    i+=3
# print("x_train",x_train)
print(x_train)
print(y_train)
# =====================================================
# testx = 'testx.txt'
# testy = 'testy.txt'
# x_test = np.loadtxt(testx)
# y_test = np.loadtxt(testy)
# print("x_test",x_test)
# i,x0_test,x1_test,x2_test,y0_test,y1_test,y2_test=0,[],[],[],[],[],[]
# while i<len(x_test):
#     x0_test.append(x_test[i])
#     x1_test.append(x_test[i+1])
#     x2_test.append(x_test[i+2])
#     y0_test.append(y_test[i])
#     y1_test.append(y_test[i+1])
#     y2_test.append(y_test[i+2])
#     i+=3
# num,train_x,train_y,test_x,test_y=2,[],[],[],[]
# if num==0:
#     train_x,train_y = x0_train,y0_train
#     test_x,test_y = x0_test,y0_test
# if num==1:
#     train_x,train_y = x1_train,y1_train
#     test_x,test_y = x1_test,y1_test
# if num==2:
#     train_x,train_y = x2_train,y2_train
#     test_x,test_y = x2_test,y2_test

# clf = svm.SVR() #SVR方法
# def fitness(x):
#     svr = svm.SVR(kernel='rbf',
#                       C=x[0],
#                       gamma=x[1])
#     svr.fit(x_train[0:10000], y_train[0:10000])
#     prediction = svr.predict(test_x)
#     MSE = mean_squared_error(prediction, test_y)
#     print("C,gamma,MSE are:",x[0],x[1],MSE)
#     # print("mse is:",MSE)
#     return MSE
#
# lower_bounds = np.array([0.001, 1e-5])
# upper_bounds = np.array([500.0, 100.0])
# SVR_n = JayaOptimizer(f=fitness,
#                   lower_bound=lower_bounds,
#                   upper_bound=upper_bounds,
#                    )
# SVR_n = helper_n_generations(SVR_n, 10)
# best_solution, best_fitness = SVR_n.best()
# print("solution", best_solution, "fitness", best_fitness)
# k = svm.SVR(kernel='rbf',
#                   C=best_solution[0],
#                   gamma=best_solution[1])
#
# k.fit(x_train,y_train)
# x_t = k.predict(test_x)
# # print(x_t)
# i=0
# while i<400:
#     x_t[i]*=26.02
#     test_y[i]*=26.02
#     i+=1
# np.savetxt("prediction_svr.txt", x_t)
# print("the MSE is:",mean_squared_error(x_t,test_y))
# print("the MAE is:",mean_absolute_error(x_t,test_y))
# x_ais=[i for i in range(0,400)]
# plt.plot(x_ais, test_y, color='green',label='real_value',marker='o',linewidth=1,markerfacecolor='black',markersize=2)
# plt.plot(x_ais, x_t, color='red',label='prediction_value',linewidth=1,marker='o',markerfacecolor='blue',markersize=2)
# # plt.plot(x_ais, sub, color='black',label='sta',linewidth=1)
# # plt.plot(x_ais, sta, color='black',label='sub',linewidth=1)
# plt.ylabel("SVM数据对比")
# plt.legend()  # 让图例生效
#
# plt.show()
#
#
#
#
#
#
#











