import pickle
import matplotlib.pyplot as plt

# with open("./model/mse_100epochs.txt", "rb") as fp:   
#     MSE = pickle.load(fp)
# 
# filename = 'haha'
# 
# plot the MSE
# plt.plot(MSE, 'ro-')
# plt.title('Mean Square Error')
# plt.xlabel('epoch')
# plt.ylabel('error')
# plt.savefig(filename + '.png')
# plt.show()


plt.plot([1, 2, 3, 4], [1, 4, 9, 16], 'ro')
plt.xticks([i for i in range(10)])
plt.show()

