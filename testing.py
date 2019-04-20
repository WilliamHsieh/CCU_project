# Import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## Variable
input_dim = 2
total_epoch = 100

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
scaler_list = []
for i in range(input_dim):
    scaler_list.append(MinMaxScaler(feature_range = (0, 1)))

# Import the training set
dataset_train = pd.read_csv('./data/stock_data_train.csv')
training_set = []
training_set.append(dataset_train.iloc[:, 4:5].values)  # close
training_set.append(dataset_train.iloc[:, 5:6].values)  # volumn

# Import the testing set
dataset_test = pd.read_csv('./data/stock_data_test.csv')
testing_set = []
testing_set.append(dataset_test.iloc[:, 4:5].values)  # close
testing_set.append(dataset_test.iloc[:, 5:6].values)  # volumn
real_stock_price = dataset_test.iloc[:, 4:5].values

# scale training set
training_set_scaled = []
for x in range(len(training_set)):
    training_set_scaled.append(scaler_list[x].fit_transform(training_set[x]))
    
# scale testing set
testing_set_scaled = []
for x in range(len(testing_set)):
    testing_set_scaled.append(scaler_list[x].fit_transform(testing_set[x]))

# combine all dim. input
training_data = []
for data_length in range(len(training_set_scaled[0])):
    tmp = []
    for data_count in range(len(training_set_scaled)):
        tmp.append(np.ndarray.tolist(training_set_scaled[data_count][data_length])[0])
    training_data.append(tmp)

# combine all dim. input
testing_data = []
for data_length in range(len(testing_set_scaled[0])):
    tmp = []
    for data_count in range(len(testing_set_scaled)):
        tmp.append(np.ndarray.tolist(testing_set_scaled[data_count][data_length])[0])
    testing_data.append(tmp)

dataset_total = training_data + testing_data
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:]
x_test = []
y_test = []
for i in range(60, 81):  # timesteps: 60, 80 = previous 61 + testing 21
    x_test.append(inputs[i-60:i])
    y_test.append(inputs[i])
x_test = np.array(x_test)
y_test = np.array(y_test)

# predict
# predicted_stock_price = []
# pre_60_days = x_test[0:1]

# for i in range(21):
#     tmp = model.predict(pre_60_days)
#     tmp = np.reshape(tmp, (1, tmp.shape[0], tmp.shape[1]))
#     pre_60_days = np.append(pre_60_days, tmp, axis=1)
#     predicted_stock_price.append(tmp.tolist()[0][0])
#     pre_60_days = [pre_60_days.tolist()[0][1:]]
#     pre_60_days = np.array(pre_60_days)

# Load model
from keras.models import load_model
# model = load_model("./data/epoch_40.h5")
# predicted_stock_price = model.predict(x_test)
# predicted_close = []
# for i in range(len(predicted_stock_price)):
#     predicted_close.append(predicted_stock_price[i][0])
# 
# np.array(predicted_close)
# predicted_close = np.reshape(predicted_close, (1, -1))
# predicted_close_price = scaler_list[0].inverse_transform(predicted_close)

# Visualising the prediction
def draw(real, pred):
    plt.plot(real, color = 'red', label = 'Real Stock Price')  # red: real stock price
    plt.plot(pred, color = 'blue', label = 'Predicted Stock Price') # blue: predicted stock price
#     plt.title(f'Stock Price Prediction, accuracy: {accuracy/21*100:.2f}%')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

MSE = []
for i in range(total_epoch):
    model = load_model(f'./data/epoch_{i}.h5')
    print(f'read model: ./data/epoch_{i}.h5')

    predicted_stock_price = model.predict(x_test)

    # re-scale and transfer back to original vector
    predicted_close = []
    for j in range(len(predicted_stock_price)):
        predicted_close.append(predicted_stock_price[j][0])

    np.array(predicted_close)
    predicted_close = np.reshape(predicted_close, (1, -1))
    predicted_close_price = scaler_list[0].inverse_transform(predicted_close)

    # calculate mean square error
    tmp = 0
    for j in range(len(real_stock_price)):
        tmp += (real_stock_price[j] - predicted_close_price[0][j]) ** 2
    MSE += [tmp / len(real_stock_price)]

    if (i + 1) % 10 == 0:
        draw(real_stock_price, predicted_close_price[0])

import pickle

#Pickling
with open("mse_100epochs.txt", "wb") as fp:   
    pickle.dump(MSE, fp)

# Unpickling
# with open("mse_100epochs.txt", "rb") as fp:   
#     b = pickle.load(fp)

# plot the MSE
plt.plot(MSE)
plt.title('Mean Square Error')
plt.xlabel('epoch')
plt.ylabel('error')
plt.show()

# accuracy
# accuracy = 0
# for i in range(0, 21):
#     if (real_stock_price[i] - predicted_close_price[0][i]) / real_stock_price[i] < 0.005 and ((real_stock_price[i]-real_stock_price[i-1]) * (predicted_close_price[0][i]-predicted_close_price[0][i-1]) > 0):
#         accuracy += 1

# Visualising the loss
# import json
# with open('./data/loss_and_acc.json') as infile:  
#     history = json.load(infile)
#     plt.plot(history['loss'])
#     plt.title('model loss')
#     plt.ylabel('loss')
#     plt.xlabel('epoch')
#     plt.show()

