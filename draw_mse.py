import pickle
import math
import numpy as np
import matplotlib.pyplot as plt

## Variable
total_epochs = 300
input_dim = 4
window_size = 60
path = f"./model/epoch_{total_epochs},dim_{input_dim},win_{window_size}/"

## Open file
file1 = "LSTM"
file2 = "RNN"
title = "LSTM vs. RNN"
with open("./model/draw/" + file1 , "rb") as fp:   
    MSE = pickle.load(fp)
    tmp = []
    for i in range(9, total_epochs, 10):
        tmp += [math.sqrt(MSE[i])]
    MSE1 = tmp

with open("./model/draw/" + file2 , "rb") as fp:   
    MSE = pickle.load(fp)
    tmp = []
    for i in range(9, total_epochs, 10):
        tmp += [math.sqrt(MSE[i])]
    MSE2 = tmp

# baseline
MSE4 = [math.sqrt(17.125) for i in range(total_epochs//10)]

## plot the MSE
# plt.style.use("ggplot")   # beautiful shit
plt.title(title)
plt.plot(MSE1, 'r-', label = file1 , marker = "^")
plt.plot(MSE2, 'g-', label = file2 , marker = "s")
# plt.plot(MSE3, 'b-', label = file3 , marker = "p")
plt.plot(MSE4, 'c-', label = 'Baseline', marker = ".")
plt.xticks(range(30), [(i+1) for i in range(9, 300, 10)])
plt.xlabel('epoch')
plt.ylabel('rmse')
plt.legend()
# plt.savefig(f'{path}mse.png')
plt.show()

