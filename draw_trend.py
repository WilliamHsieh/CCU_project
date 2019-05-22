## Import
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras import backend as K
from keras.models import load_model

## Variable
MSE = []
input_dim = 4
total_epochs = 300
window_size = 60
predict_days = 20

## Parse data
### Feature Scaling
from sklearn.preprocessing import MinMaxScaler
scaler_list = []
for i in range(input_dim):
    scaler_list.append(MinMaxScaler(feature_range = (0, 1)))

### Import the testing set
dataset_test = pd.read_csv('./data/stock_data_test.csv')
testing_set = []
testing_set.append(dataset_test.iloc[:, 4:5].values)  # close
testing_set.append(dataset_test.iloc[:, 5:6].values)  # volumn
real_stock_price = dataset_test.iloc[:, 4:5].values
real_stock_price = real_stock_price[len(real_stock_price) - predict_days:]
dataset_testing = pd.read_csv('./data/nasdaq_test.csv')
testing_set.append(dataset_testing.iloc[:, 4:5].values)
dataset_testing = pd.read_csv('./data/dji_test.csv')    
testing_set.append(dataset_testing.iloc[:, 4:5].values)

### Scale testing set
testing_set_scaled = []
for x in range(input_dim):
    testing_set_scaled.append(scaler_list[x].fit_transform(testing_set[x]))

### Combine all dimension
testing_data = []
for data_length in range(len(testing_set_scaled[0])):
    tmp = []
    for data_count in range(input_dim):
        tmp.append(np.ndarray.tolist(testing_set_scaled[data_count][data_length])[0])
    testing_data.append(tmp)

### Create model input
inputs = testing_data[len(testing_data) - predict_days - window_size:]
x_test = []
y_test = []
for i in range(window_size, len(inputs)):
    x_test.append(inputs[i-window_size:i])
    y_test.append(inputs[i])
x_test = np.array(x_test)
y_test = np.array(y_test)

## Model predict
# load model
path = f"./model/draw/"
lstm = load_model(f'{path}lstm.h5')
rnn = load_model(f'{path}rnn.h5')
lstm_output = lstm.predict(x_test)
rnn_output = rnn.predict(x_test)

# get all the close price
lstm_close_price = []
for j in range(len(lstm_output)):
    lstm_close_price.append(lstm_output[j][0])

# get all the close price
rnn_close_price = []
for j in range(len(rnn_output)):
    rnn_close_price.append(rnn_output[j][0])

# re-scale back
lstm_close_price = np.reshape(lstm_close_price, (1, -1))
lstm_predicted_stock_price = scaler_list[0].inverse_transform(lstm_close_price)

# re-scale back
rnn_close_price = np.reshape(rnn_close_price, (1, -1))
rnn_predicted_stock_price = scaler_list[0].inverse_transform(rnn_close_price)
    
plt.clf()
plt.plot(real_stock_price, 'ro-', label = 'Real Stock Price')
plt.plot(lstm_predicted_stock_price[0], 'go-', label = 'Predicted lstm')
plt.plot(rnn_predicted_stock_price[0], 'bo-', label = 'Predicted rnn')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.xticks([i for i in range(21)])
plt.legend()
# plt.savefig(filename + '.png')
plt.show()

