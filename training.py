## Import
import os
import pickle
import numpy as np
from preprocess import getData
from keras.layers import LSTM
from keras.layers import SimpleRNN
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Sequential
from keras.optimizers import Adam

## Variable
total_epochs = 50
input_dim = 4
window_size = 60
num_units = 100

batchSize = 32
learning_rate = 0.001
loss_func = 'mean_squared_error'

## Model definition
def get_model(x_train):
    # Initialising the RNN
    model = Sequential()

    # Adding the first LSTM layer and some Dropout regularisation
    model.add(LSTM(units = num_units, return_sequences = True, input_shape = (x_train.shape[1], x_train.shape[2])))
    model.add(Dropout(0.2))

    # Adding a second LSTM layer and some Dropout regularisation
    model.add(LSTM(units = num_units, return_sequences = True))
    model.add(Dropout(0.2))

    # Adding a third LSTM layer and some Dropout regularisation
    model.add(LSTM(units = num_units))
    model.add(Dropout(0.2))

    # Adding the output layer
    model.add(Dense(units = input_dim))
#     model.add(Dense(units = 1))

    # Compiling
    opt = Adam(lr=learning_rate)
    model.compile(optimizer = opt, metrics=['mean_squared_error'], loss = loss_func)

    return model

## Training
def training():

    # get data && model
    [x_train, y_train], s = getData(input_dim, window_size, "train")
    model = get_model(x_train)
    loss = []

    # Fit && save model/history
    path = f"./model/epoch_{total_epochs},dim_{input_dim},win_{window_size}/"
    if not os.path.exists(path):
        os.mkdir(path, 755)

    # Visualize the model
    print(model.summary())

    # train
    for i in range(total_epochs):
        print(f'epoch: {i + 1}/{total_epochs}')
        history = model.fit(x_train, y_train, epochs = 1, batch_size = batchSize)
        model.save(f'{path}epoch_{i}.h5')
        loss += history.history['mean_squared_error']

    # save
    with open(f'{path}loss', 'wb') as fp:
        pickle.dump(loss, fp)

## Main function
if __name__ == "__main__":
    training()

