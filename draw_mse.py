import pickle
import matplotlib.pyplot as plt

## Variable
total_epochs = 300
input_dim = 2
window_size = 60
path = f"./model/epoch_{total_epochs},dim_{input_dim},win_{window_size}/"

## Open file
with open("./model/draw/" + "2input", "rb") as fp:   
    MSE1 = pickle.load(fp)

with open("./model/draw/" + "4input", "rb") as fp:   
    MSE2 = pickle.load(fp)
    tmp = []
    for i in range(9, total_epochs, 10):
        tmp += [MSE2[i]]

# baseline
MSE3 = [17.125 for i in range(total_epochs//10)]

## plot the MSE
plt.title("2 input vs. 4 input")
plt.style.use("ggplot")   # beautiful shit
plt.plot(MSE1, 'r-', label = '2input', marker = "s")
plt.plot(tmp, 'g-', label = '4input', marker = "^")
plt.plot(MSE3, 'b-', label = 'Baseline', marker = ".")
plt.xticks([i for i in range(0, 30)])
plt.xlabel('epoch')
plt.ylabel('error')
plt.legend()
# plt.savefig(f'{path}mse.png')
plt.show()

