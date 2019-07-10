import math
import pickle
import numpy as np
import matplotlib.pyplot as plt
from preprocess import getData

## Variable
total_epochs = 300
input_dim = 4
window_size = 60
path = f"./model/epoch_{total_epochs},dim_{input_dim},win_{window_size}/"

## Open file
file1 = "LOSS"
title = "rmse vs. epoch"
with open(path + "loss" , "rb") as fp:   
    LOSS = pickle.load(fp)
    tmp = []
    for i in range(9, total_epochs, 10):
        tmp += [math.sqrt(LOSS[i])]
    LOSS = tmp

    # rescale
#     [x_test, y_test], scaler_list = getData(input_dim, window_size, "train")
#     tmp = np.reshape(tmp, (1, -1))
#     tmp = scaler_list[0].inverse_transform(tmp)
#     LOSS = np.reshape(tmp, (-1))

## plot the LOSS
# plt.style.use("ggplot")   # beautiful shit
plt.title(title)
plt.plot(LOSS, 'r-', label = file1 , marker = "^")
# plt.plot(MSE2, 'g-', label = file2 , marker = "s")
# plt.plot(MSE3, 'b-', label = file3 , marker = "p")
# plt.plot(MSE4, 'c-', label = 'Baseline', marker = ".")
plt.xticks(range(30), [(i+1) for i in range(9, 300, 10)])
plt.xlabel('epoch')
plt.ylabel('rmse')
plt.legend()
# plt.savefig(f'{path}LOSS.png')
plt.show()

