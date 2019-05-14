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
input_dim = 2
total_epochs = 2

## Parse data
### Feature Scaling
from sklearn.preprocessing import MinMaxScaler
scaler_list = []
for i in range(input_dim):
    scaler_list.append(MinMaxScaler(feature_range = (0, 1)))

### Import
# Import the training set
dataset_train = pd.read_csv('./data/stock_data_train.csv')
training_set = []
training_set.append(dataset_train.iloc[:, 4:5].values)  # close
training_set.append(dataset_train.iloc[:, 5:6].values)  # volumn
dataset_train = pd.read_csv('./data/nasdaq_train.csv')
training_set.append(dataset_train.iloc[:, 4:5].values)
dataset_train = pd.read_csv('./data/dji_train.csv')    
training_set.append(dataset_train.iloc[:, 4:5].values)


# Import the testing set
dataset_test = pd.read_csv('./data/stock_data_test.csv')
testing_set = []
testing_set.append(dataset_test.iloc[:, 4:5].values)  # close
testing_set.append(dataset_test.iloc[:, 5:6].values)  # volumn
real_stock_price = dataset_test.iloc[:, 4:5].values
dataset_testing = pd.read_csv('./data/nasdaq_test.csv')
testing_set.append(dataset_train.iloc[:, 4:5].values)
dataset_testing = pd.read_csv('./data/dji_test.csv')    
testing_set.append(dataset_train.iloc[:, 4:5].values)

### Scale
# scale training set
training_set_scaled = []
for x in range(input_dim):
    training_set_scaled.append(scaler_list[x].fit_transform(training_set[x]))
    
# scale testing set
testing_set_scaled = []
for x in range(input_dim):
    testing_set_scaled.append(scaler_list[x].fit_transform(testing_set[x]))

### Combine all dimension
training_data = []
for data_length in range(len(training_set_scaled[0])):
    tmp = []
    for data_count in range(input_dim):
        tmp.append(np.ndarray.tolist(training_set_scaled[data_count][data_length])[0])
    training_data.append(tmp)

testing_data = []
for data_length in range(len(testing_set_scaled[0])):
    tmp = []
    for data_count in range(input_dim):
        tmp.append(np.ndarray.tolist(testing_set_scaled[data_count][data_length])[0])
    testing_data.append(tmp)

### Create model input
dataset_total = training_data + testing_data
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:]
x_test = []
y_test = []
for i in range(60, 81):  # timesteps: 60, 80 = previous 61 + testing 21
    x_test.append(inputs[i-60:i])
    y_test.append(inputs[i])
x_test = np.array(x_test)
y_test = np.array(y_test)

## Visualize the prediction
def draw(real, pred, filename):
    plt.plot(real, 'ro-', label = 'Real Stock Price')  # red: real stock price
    plt.plot(pred, 'bo-', label = 'Predicted Stock Price') # blue: predicted stock price
#     plt.title(f'Stock Price Prediction, accuracy: {accuracy/21*100:.2f}%')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.xticks([i for i in range(21)])
    plt.legend()
    plt.savefig(filename + '.png')
#     plt.show()
    plt.clf()

## Model predict
path = f"./model/epoch_{total_epochs},dim_{input_dim}/"
for i in range(total_epochs):
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
#     np.array(close_price)
    close_price = np.reshape(close_price, (1, -1))
    predicted_stock_price = scaler_list[0].inverse_transform(close_price)
	
    # calculate mean square error
    tmp = 0
    for j in range(len(real_stock_price)):
        tmp += (real_stock_price[j] - predicted_stock_price[0][j]) ** 2
    MSE += [tmp / len(real_stock_price)]

    end = time.time()
    print(f'model complete: ./model/epoch_{i}.h5 ,  time: {end-start:.02f} secs')

    if (i + 1) % 10 == 0:
        draw(real_stock_price, predicted_stock_price[0], path + str(i+1))

print("done.")

## Save (MSE) && plot
with open(f"{path}MSE", "wb") as fp:   
    pickle.dump(MSE, fp)

