## Import
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from preprocess import getData
from preprocess import getGT
from keras import backend as K
from keras.models import load_model

## Variable
MSE = []
total_epochs = 300
input_dim = 4
window_size = 60
predict_days = 20
data_frequency = 1

## Get data
real_stock_price = getGT(predict_days, data_frequency)
[x_test, y_test], scaler_list = getData(input_dim, window_size, predict_days, data_frequency, "test")

## Model predict
# load model
# path = f"./model/epoch_{total_epochs},dim_{input_dim},win_{window_size}/"
path = f"./model/draw/"
lstm = load_model(f'{path}000.h5')
label1 = 'test1'
# rnn = load_model(f'{path}50unit.h5')
# label2 = 'test2'
lstm_output = lstm.predict(x_test)
# rnn_output = rnn.predict(x_test)

# get all the close price
lstm_close_price = []
for j in range(len(lstm_output)):
    lstm_close_price.append(lstm_output[j][0])

# get all the close price
# rnn_close_price = []
# for j in range(len(rnn_output)):
#     rnn_close_price.append(rnn_output[j][0])

# re-scale back
lstm_close_price = np.reshape(lstm_close_price, (1, -1))
lstm_predicted_stock_price = scaler_list[0].inverse_transform(lstm_close_price)

# re-scale back
# rnn_close_price = np.reshape(rnn_close_price, (1, -1))
# rnn_predicted_stock_price = scaler_list[0].inverse_transform(rnn_close_price)
    
plt.clf()
# plt.style.use("ggplot")   # beautiful shit
plt.title('predicted stock price')
plt.plot(real_stock_price, 'ro-', label = 'Real Stock Price')
plt.plot(lstm_predicted_stock_price[0], 'bo-', label = label1, marker = "^")
# plt.plot(shit_predicted_stock_price[0], 'co-', label = label2, marker = "p")
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.xticks(range(20), [i+1 for i in range(20)])
plt.legend()
# plt.savefig(filename + '.png')
plt.show()

