import pickle
import matplotlib.pyplot as plt

## Variable
input_dim = 4
total_epochs = 300
path = f"./model/epoch_{total_epochs},dim_{input_dim}/"

## Open file
with open("./model/RNN60", "rb") as fp:   
    MSE1 = pickle.load(fp)

with open("./model/RNN120", "rb") as fp:   
    MSE2 = pickle.load(fp)

with open("./model/LSTM60", "rb") as fp:   
    MSE3 = pickle.load(fp)

with open("./model/LSTM120", "rb") as fp:   
    MSE4 = pickle.load(fp)

## plot the MSE
plt.plot(MSE1, 'r-', label = 'RNN_60days')
plt.plot(MSE2, 'g-', label = 'RNN_120days')
plt.plot(MSE3, 'b-', label = 'LSTM_60days')
plt.plot(MSE4, 'k-', label = 'LSTM_120days')
plt.title('Mean Square Error')
plt.xticks([i for i in range(0, total_epochs + 1, 10)])
plt.xlabel('epoch')
plt.ylabel('error')
plt.legend()
# plt.savefig(f'{path}mse.png')
plt.show()
# print(MSE3)

