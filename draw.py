import pickle
import matplotlib.pyplot as plt

## Variable
input_dim = 4
total_epochs = 300
path = f"./model/epoch_{total_epochs},dim_{input_dim}/"

## Open file
with open("./model/MSE", "rb") as fp:   
    MSE1 = pickle.load(fp)

with open("./model/4.txt", "rb") as fp:   
    MSE2 = pickle.load(fp)

## plot the MSE
plt.plot(MSE1, 'r-', label = 'RNN')
plt.plot(MSE2, 'b-', label = 'LSTM')
plt.title('Mean Square Error')
plt.xticks([i for i in range(0, total_epochs + 1, 10)])
plt.xlabel('epoch')
plt.ylabel('error')
plt.legend()
# plt.savefig(f'{path}mse.png')
plt.show()

