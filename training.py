## Import the libraries
import csv
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler

## Function
### Variable
scaler_list = []
training_set = []
training_set_scaled = []
x_train = []
y_train = []
total_epochs = 50
batchSize = 32

### Feature Scaling
def feature_scaling():
    # 0. close
    # 1. volumn
    global scaler_list
    scaler_list.append(MinMaxScaler(feature_range = (0, 1)))
    scaler_list.append(MinMaxScaler(feature_range = (0, 1)))

### Import the training set
def get_training_data():
    global training_set
    dataset_train = pd.read_csv('./data/stock_data_train.csv')
    training_set.append(dataset_train.iloc[:, 4:5].values)  # close
    training_set.append(dataset_train.iloc[:, 5:6].values)  # volumn

### Scale training set
def scale_data():
    global training_set_scaled
    for x in range(len(training_set)):
        training_set_scaled.append(scaler_list[x].fit_transform(training_set[x]))

### Orginize data (x_train, y_train)
def orginize_data():
    # combine all dimension data
    training_data = []
    for data_length in range(len(training_set_scaled[0])):
        tmp = []
        for data_count in range(len(training_set_scaled)):
            tmp.append(np.ndarray.tolist(training_set_scaled[data_count][data_length])[0])
        training_data.append(tmp)
    training_data = np.array(training_data)

    # get x_train, y_train
    global x_train, y_train
    for i in range(60, 1199):  # 1199 is total data amount
        x_train.append(training_data[i-60:i])
        y_train.append(training_data[i])

    x_train, y_train = np.array(x_train), np.array(y_train)

### model structure
def model_structure():
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
    model.add(Dense(units = 2))

    # Compiling
    model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=['accuracy'])

    # Fit
    history = model.fit(x_train, y_train, epochs = total_epochs, batch_size = batchSize, shuffle=True)

    # save model
    model.save('./data/my_model.h5')
    
    return history

    # Visualize the model
#     from keras.utils import plot_model
#     plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
#     print(model.summary())

### saving result
def result(history):
    # save the result to json file
    with open('./data/loss_and_acc.json', 'w') as outfile:
        json.dump(history.history, outfile)

## Main function
if __name__ == "__main__":
    feature_scaling()
    get_training_data()
    scale_data()
    orginize_data()
    history = model_structure()
    result(history)

