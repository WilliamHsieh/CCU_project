import pickle
import math
import numpy as np
import matplotlib.pyplot as plt
from preprocess import getBase

## Variable
total_epochs = 500
input_dim = 4
window_size = 60
predict_days = 20
data_frequency = 5
path = f"./model/epoch_{total_epochs},dim_{input_dim},win_{window_size}/"

## Open file
file1 = "MSE"
# file2 = "RNN"
title = "LSTM vs. RNN"
with open("./model/draw/" + file1 , "rb") as fp:   
    MSE = pickle.load(fp)
#     tmp = []
#     for i in range(9, total_epochs, 10):
#         tmp += [math.sqrt(MSE[i])]
#     MSE1 = tmp

# with open("./model/draw/" + file2 , "rb") as fp:   
#     MSE = pickle.load(fp)
#     tmp = []
#     for i in range(9, total_epochs, 10):
#         tmp += [math.sqrt(MSE[i])]
#     MSE2 = tmp

# baseline
baseline = getBase(predict_days, data_frequency)
base = [baseline for i in range(total_epochs//10)]

## plot the MSE
# mpl.use('TkAgg')
# plt.style.use("ggplot")   # beautiful shit
plt.title(title)
plt.plot(MSE, 'r-', label = file1 , marker = "^")
# plt.plot(MSE2, 'g-', label = file2 , marker = "s")
# plt.plot(MSE3, 'b-', label = file3 , marker = "p")
plt.plot(base, 'c-', label = 'Baseline', marker = ".")
plt.xticks(range(total_epochs//10), [(i+1) for i in range(9, total_epochs, 10)])
plt.xlabel('epoch')
plt.ylabel('rmse')
plt.legend()
# plt.savefig(f'{path}mse.png')
plt.show()

