import pickle
import matplotlib.pyplot as plt

# open file
with open("./model/2.txt", "rb") as fp:   
    MSE1 = pickle.load(fp)

with open("./model/4.txt", "rb") as fp:   
    MSE2 = pickle.load(fp)

# plot the MSE
plt.plot(MSE1, 'r-', label = '2 inputs')
plt.plot(MSE2, 'b-', label = '4 inputs')
plt.title('Mean Square Error')
plt.xticks([i for i in range(0, 301, 10)])
plt.xlabel('epoch')
plt.ylabel('error')
plt.legend()
plt.savefig('mse.png')
plt.show()

