# Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
scaler_list = []
scaler_list.append(MinMaxScaler(feature_range = (0, 1))) #close
scaler_list.append(MinMaxScaler(feature_range = (0, 1))) #volumn

# Import the training set
dataset_train = pd.read_csv('stock_data_train.csv')
training_set = []
training_set.append(dataset_train.iloc[:, 4:5].values)  # 'close'
training_set.append(dataset_train.iloc[:, 5:6].values)  # 'close'

# scale training set
training_set_scaled = []
for x in range(len(training_set)):
    training_set_scaled.append(scaler_list[x].fit_transform(training_set[x]))

# combine all dim. input
training_data = []
for data_length in range(len(training_set_scaled[0])):
    tmp = []
    for data_count in range(len(training_set_scaled)):
        tmp.append(np.ndarray.tolist(training_set_scaled[data_count][data_length])[0])
    training_data.append(tmp)
training_data = np.array(training_data)

x_train = []   # 60 days data
y_train = []   # predict 61th
for i in range(60, 1199):  # 1199 is total data amount
    x_train.append(training_data[i-60:i])
    y_train.append(training_data[i])

x_train, y_train = np.array(x_train), np.array(y_train)
# x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2]))
# y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1], 1))
# print(f'timesteps: {x_train.shape[1]}')

# Import the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
model = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1], x_train.shape[2])))
model.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50))
model.add(Dropout(0.2))

# Adding the output layer
model.add(Dense(units = 2))

# Visualize the model
from keras.utils import plot_model
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

# Compiling
model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=['accuracy'])
print(model.summary())

# Function for storing losses
from keras.callbacks import Callback
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

# Start training && save model/losses
# history = LossHistory()
# history = model.fit(x_train, y_train, epochs = 10, batch_size = 32, callbacks=[history])
history = model.fit(x_train, y_train, epochs = 100, batch_size = 32, validation_split=0.2, shuffle=True)
# list all data in history
# print(history.history.keys())

import json
with open('loss_acc.json', 'w') as outfile:  
    json.dump(history.history, outfile)
# plot history for loss
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()

