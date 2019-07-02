## Import
import pickle
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

## Variable
input_dim = 4
total_epochs = 300
window_size = 60
predict_days = 20

## Get dataset
dataset_test = pd.read_csv('./data/stock_test.csv')
close_price = dataset_test.iloc[:, 4:5].values   # close
real_stock_price = close_price[len(close_price) - predict_days:]
baseline_stock_price = close_price[len(close_price) - predict_days - 1 : len(close_price) - 1]

## Visualize the trend
def draw(real, pred, filename):
    mpl.use('TkAgg')
    plt.plot(real, 'ro-', label = 'Real Stock Price')  # red: real stock price
    plt.plot(pred, 'bo-', label = 'Predicted Stock Price') # blue: predicted stock price
    plt.title(f'Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.xticks([i for i in range(21)])
    plt.legend()
#     plt.savefig(filename + '.png')
    plt.show()
    plt.clf()

## Calculate && save MSE
path = f"./model/epoch_{total_epochs},dim_{input_dim}/"
MSE = []
tmp = 0
for i in range(len(real_stock_price)):
    tmp += (real_stock_price[i] - baseline_stock_price[i]) ** 2
print(tmp[0])
MSE += [tmp[0] / len(real_stock_price)]
print("MSE:", MSE)

# draw(real_stock_price, baseline_stock_price, path)
# with open(f"{path}MSE", "wb") as fp:   
#     pickle.dump(MSE, fp)

