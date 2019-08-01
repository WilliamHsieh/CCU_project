## Import
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras import backend as K
from keras.models import load_model
from preprocess import getData
from preprocess import getGT

## Variable
MSE = []
total_epochs = 300
input_dim = 4
window_size = 49
predict_days = 20
data_frequency = 5

## Get data
real_stock_price = getGT(predict_days, data_frequency)
[x_test, y_test], scaler_list = getData(input_dim, window_size, predict_days, data_frequency, "test")

## Visualize the prediction
def draw(real, pred, filename, epoch):
    plt.plot(real, 'ro-', label = 'Real Stock Price', marker = "s")  # red: real stock price
    plt.plot(pred, 'bo-', label = 'Predicted Stock Price', marker = ".") # blue: predicted stock price
    plt.title(f'Stock Price Prediction, epoch: {epoch}')
    plt.xlabel('Days')
    plt.ylabel('Stock Price')
    plt.xticks(range(20), [i+1 for i in range(20)])
    plt.legend()
    plt.savefig(filename + '.png')
    plt.clf()

## Model predict
path = f"./model/epoch_{total_epochs},dim_{input_dim},win_{window_size},freq_{data_frequency}/"
for i in range(9, total_epochs, 10):
    start = time.time()
    K.clear_session()

    # load model
    model = load_model(f'{path}epoch_{i}.h5')
    print(f'read model: epoch_{i}.h5')
    output = model.predict(x_test)

    # get all the close price
    close_price = []
    for j in range(len(output)):
        close_price.append(output[j][0])

    # re-scale back
    close_price = np.reshape(close_price, (1, -1))
    predicted_stock_price = scaler_list[0].inverse_transform(close_price)[0]
	
    # calculate mean square error
    tmp = 0
    for j in range(predict_days):
        tmp += (real_stock_price[j] - predicted_stock_price[j]) ** 2
    MSE += [tmp / len(real_stock_price)]

    end = time.time()
    print(f'model complete: ./model/epoch_{i}.h5 ,  time: {end-start:.02f} secs')

    if (i + 1) % 10 == 0:
        draw(real_stock_price, predicted_stock_price, path + str(i+1), str(i+1))

print("done.")

## Save MSE
with open(f"{path}MSE", "wb") as fp:   
    pickle.dump(MSE, fp)

