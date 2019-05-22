## Import
import os
import csv
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import LSTM
from keras.layers import SimpleRNN
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

## Variable
scaler_list = []
training_set = []
training_set_scaled = []
x_train = []
y_train = []
input_dim = 4
total_epochs = 300
batchSize = 32
learning_rate = 0.001
loss_func = 'mean_squared_error'
window_size = 60

## Function
### Feature Scaling
def feature_scaling():
    # 0. close
    # 1. volumn
    # 2. Nasdaq
    # 3. dji
    global scaler_list
    for x in range(input_dim):
        scaler_list.append(MinMaxScaler(feature_range = (0, 1)))

### Import the training set
def get_training_data():
    global training_set
    # stock of choice
    dataset_train = pd.read_csv('./data/stock_data_train.csv')
    training_set.append(dataset_train.iloc[:, 4:5].values)  # close
    training_set.append(dataset_train.iloc[:, 5:6].values)  # volumn

    # Nasdaq
    dataset_train = pd.read_csv('./data/nasdaq_train.csv')
    training_set.append(dataset_train.iloc[:, 4:5].values)  # close

    # dji
    dataset_train = pd.read_csv('./data/dji_train.csv')
    training_set.append(dataset_train.iloc[:, 4:5].values)  # close

### Scale training set
def scale_data():
    global training_set_scaled
    for x in range(input_dim):
        training_set_scaled.append(scaler_list[x].fit_transform(training_set[x]))

### Orginize data (x_train, y_train)
def orginize_data():
    # combine all dimension data
    training_data = []
    for data_length in range(len(training_set_scaled[0])):
        tmp = []
        for data_count in range(input_dim):
            tmp.append(np.ndarray.tolist(training_set_scaled[data_count][data_length])[0])
        training_data.append(tmp)
    training_data = np.array(training_data)

    # get x_train, y_train
    global x_train, y_train
    for i in range(window_size, len(training_data)):
        x_train.append(training_data[i-window_size:i])
        y_train.append(training_data[i])

    x_train, y_train = np.array(x_train), np.array(y_train)

### Get model
def get_model():
    # Initialising the RNN
    model = Sequential()

    # Adding the first LSTM layer and some Dropout regularisation
    model.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1], x_train.shape[2])))
    model.add(Dropout(0.2))

    # Adding a second LSTM layer and some Dropout regularisation
    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(0.2))

    # Adding a third LSTM layer and some Dropout regularisation
    model.add(LSTM(units = 50))
    model.add(Dropout(0.2))

    # Adding the output layer
    model.add(Dense(units = input_dim))

    # Compiling
    opt = Adam(lr=learning_rate)
    model.compile(optimizer = opt, loss = loss_func)

    return model

### Model training
def training():

    model = get_model()
    loss = []

    # Fit && save model/history
    path = f"./model/epoch_{total_epochs},dim_{input_dim},win_{window_size}/"
    if not os.path.exists(path):
        os.mkdir(path, 755)

    # Visualize the model
#     print(model.summary())

    # train
    for i in range(total_epochs):
        print(f'epoch: {i + 1}/{total_epochs}')
        history = model.fit(x_train, y_train, epochs = 1, batch_size = batchSize)
        model.save(f'{path}epoch_{i}.h5')
        loss += history.history['loss']

    # save
    with open(f'{path}loss', 'wb') as fp:
        pickle.dump(loss, fp)

## Main function
if __name__ == "__main__":
    feature_scaling()
    get_training_data()
    scale_data()
    orginize_data()
    training()

