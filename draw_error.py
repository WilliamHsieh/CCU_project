## Import
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from preprocess import getData
from keras import backend as K
from keras.models import load_model

## Variable
MSE = []
input_dim = 4
total_epochs = 300
window_size = 60
predict_days = 20

## Get data
csv_data = pd.read_csv('./data/stock_test.csv')
real_stock_price = csv_data.iloc[:, 4:5].values
real_stock_price = real_stock_price[len(real_stock_price) - predict_days:]
[x_test, y_test], scaler_list = getData(input_dim, window_size, "test", predict_days)

## Model predict
# load model
path = f"./model/draw/"
lstm = load_model(f'{path}LSTM.h5')
# rnn = load_model(f'{path}rnn.h5')
lstm_output = lstm.predict(x_test)
# rnn_output = rnn.predict(x_test)

# get all the close price
lstm_close_price = []
for j in range(len(lstm_output)):
    lstm_close_price.append(lstm_output[j][0])

# ge all the close price
# rnn_close_price = []
# for j in range(len(rnn_output)):
#     rnn_close_price.append(rnn_output[j][0])

# re-scale back
lstm_close_price = np.reshape(lstm_close_price, (1, -1))
lstm_predicted_stock_price = scaler_list[0].inverse_transform(lstm_close_price)

# calculate error
tmp = []
for i in range(len(real_stock_price)):
    tmp += [real_stock_price[i][0] - lstm_predicted_stock_price[0][i]]

plt.clf()
plt.title('error')
plt.bar(range(predict_days), tmp, align='center', alpha=0.5)

plt.xticks(range(predict_days), [i+1 for i in range(20)])
plt.xlabel('day')
plt.ylabel('Stock Price')
plt.legend()
# plt.savefig(filename + '.png')
plt.show()

