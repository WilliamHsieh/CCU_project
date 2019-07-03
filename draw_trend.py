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
total_epochs = 100
window_size = 60
predict_days = 20

## Get data
csv_data = pd.read_csv('./data/stock_test.csv')
real_stock_price = csv_data.iloc[:, 4:5].values
real_stock_price = real_stock_price[len(real_stock_price) - predict_days:]
[x_test, y_test], scaler_list = getData(input_dim, window_size, "test", predict_days)

## Model predict
# load model
# path = f"./model/epoch_{total_epochs},dim_{input_dim},win_{window_size}/"
path = f"./model/draw/"
lstm = load_model(f'{path}15unit.h5')
rnn = load_model(f'{path}50unit.h5')
shit = load_model(f'{path}100unit.h5')
lstm_output = lstm.predict(x_test)
rnn_output = rnn.predict(x_test)
shit_output = shit.predict(x_test)

# get all the close price
lstm_close_price = []
for j in range(len(lstm_output)):
    lstm_close_price.append(lstm_output[j][0])

# get all the close price
rnn_close_price = []
for j in range(len(rnn_output)):
    rnn_close_price.append(rnn_output[j][0])

# get all the close price
shit_close_price = []
for j in range(len(rnn_output)):
    shit_close_price.append(shit_output[j][0])

# re-scale back
lstm_close_price = np.reshape(lstm_close_price, (1, -1))
lstm_predicted_stock_price = scaler_list[0].inverse_transform(lstm_close_price)

# re-scale back
rnn_close_price = np.reshape(rnn_close_price, (1, -1))
rnn_predicted_stock_price = scaler_list[0].inverse_transform(rnn_close_price)

# re-scale back
shit_close_price = np.reshape(shit_close_price, (1, -1))
shit_predicted_stock_price = scaler_list[0].inverse_transform(shit_close_price)
    
plt.clf()
# plt.style.use("ggplot")   # beautiful shit
plt.title('predicted stock price')
plt.plot(real_stock_price, 'ro-', label = 'Real Stock Price')
plt.plot(lstm_predicted_stock_price[0], 'bo-', label = 'Predicted 15unit', marker = "^")
plt.plot(rnn_predicted_stock_price[0], 'go-', label = 'Predicted 50unit', marker = "s")
plt.plot(shit_predicted_stock_price[0], 'co-', label = 'Predicted 100unit', marker = "p")
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.xticks(range(20), [i+1 for i in range(20)])
plt.legend()
# plt.savefig(filename + '.png')
plt.show()

