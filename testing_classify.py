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
total_epochs = 10
input_dim = 4
window_size = 60

## Parse data
### Feature Scaling
from sklearn.preprocessing import MinMaxScaler
scaler_list = []
for i in range(input_dim):
    scaler_list.append(MinMaxScaler(feature_range = (0, 1)))

### Import
# Import the training set
dataset_train = pd.read_csv('./data/stock_train.csv')
training_set = []
training_set.append(dataset_train.iloc[:, 4:5].values)  # close
training_set.append(dataset_train.iloc[:, 5:6].values)  # volumn
dataset_train = pd.read_csv('./data/nasdaq_train.csv')
training_set.append(dataset_train.iloc[:, 4:5].values)
dataset_train = pd.read_csv('./data/dji_train.csv')    
training_set.append(dataset_train.iloc[:, 4:5].values)


# Import the testing set
dataset_test = pd.read_csv('./data/stock_test.csv')
testing_set = []
testing_set.append(dataset_test.iloc[:, 4:5].values)  # close
testing_set.append(dataset_test.iloc[:, 5:6].values)  # volumn
real_stock_price = dataset_test.iloc[:, 4:5].values
dataset_testing = pd.read_csv('./data/nasdaq_test.csv')
testing_set.append(dataset_test.iloc[:, 4:5].values)
dataset_testing = pd.read_csv('./data/dji_test.csv')    
testing_set.append(dataset_test.iloc[:, 4:5].values)

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
for i in range(window_size, len(inputs)):
    x_test.append(inputs[i-window_size:i])
    if (inputs[i][0] >= inputs[i-1][0]):
        y_test.append(0)
    elif (inputs[i][0] < inputs[i-1][0]):
        y_test.append(1)

x_test = np.array(x_test)

## Model predict
path = f"./model/class/epoch_{total_epochs},dim_{input_dim},win_{window_size}/"
for i in range(total_epochs):
    start = time.time()
    K.clear_session()

    # load model
    model = load_model(f'{path}epoch_{i}.h5')
    print(f'read model: epoch_{i}.h5')
    output = model.predict(x_test)

    acc = 0
    total_guess = len(inputs)
    for j in range(len(output)):
        x = output[j].tolist()
        if x.index(max(x)) == y_test[j]:
            acc += 1

    print(output)
    print(f"accuracy: {acc * 100 / total_guess:.2f}%")

    end = time.time()
    print(f'model complete: ./model/epoch_{i}.h5 ,  time: {end-start:.02f} secs')

print("done.")

## Save (MSE) && plot
with open(f"{path}MSE", "wb") as fp:   
    pickle.dump(MSE, fp)

