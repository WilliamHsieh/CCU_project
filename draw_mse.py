import pickle
import numpy as np
import matplotlib.pyplot as plt

## Variable
total_epochs = 300
input_dim = 2
window_size = 60
path = f"./model/epoch_{total_epochs},dim_{input_dim},win_{window_size}/"

## Open file
with open("./model/draw/" + "60", "rb") as fp:   
    MSE1 = pickle.load(fp)
    tmp = []
    for i in range(9, total_epochs, 10):
        tmp += [MSE1[i]]
    MSE1 = tmp

with open("./model/draw/" + "120", "rb") as fp:   
    MSE2 = pickle.load(fp)
    tmp = []
    for i in range(9, total_epochs, 10):
        tmp += [MSE2[i]]
    MSE2 = tmp

# baseline
MSE3 = [17.125 for i in range(total_epochs//10)]

## plot the MSE
# plt.style.use("ggplot")   # beautiful shit
plt.title("window size 60 vs 120 days")
plt.plot(MSE1, 'r-', label = '60', marker = "s")
plt.plot(MSE2, 'g-', label = '120', marker = "^")
plt.plot(MSE3, 'b-', label = 'Baseline', marker = ".")
plt.xticks(range(30), [(i+1) for i in range(9, 300, 10)])
plt.xlabel('epoch')
plt.ylabel('error')
plt.legend()
# plt.savefig(f'{path}mse.png')
plt.show()

