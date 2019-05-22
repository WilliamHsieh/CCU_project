## Import
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

## Feature Scaling
def feature_scaling(input_dim):
    # 0. close
    # 1. volumn
    # 2. Nasdaq
    # 3. dji
    scaler_list = []
    for x in range(input_dim):
        scaler_list.append(MinMaxScaler(feature_range = (0, 1)))
    return scaler_list

## Import the data set
def import_data(data_flag):
    raw_data = []
    # stock of choice
    csv_data = pd.read_csv('./data/stock_data_' + data_flag + '.csv')
    raw_data.append(csv_data.iloc[:, 4:5].values)  # close
    raw_data.append(csv_data.iloc[:, 5:6].values)  # volumn

    # Nasdaq
    csv_data = pd.read_csv('./data/nasdaq_' + data_flag + '.csv')
    raw_data.append(csv_data.iloc[:, 4:5].values)  # close

    # dji
    csv_data = pd.read_csv('./data/dji_' + data_flag + '.csv')
    raw_data.append(csv_data.iloc[:, 4:5].values)  # close

    return raw_data

## Scale training set
def scale_data(input_dim, scaler_list, raw_data):
    scaled_data = []
    for x in range(input_dim):
        scaled_data.append(scaler_list[x].fit_transform(raw_data[x]))

    return scaled_data

## Orginize data (x_train, y_train)
def orginize_data(input_dim, window_size, data_flag, predict_days, scaled_data):
    # combine all dimension data
    total_data = []
    for data_length in range(len(scaled_data[0])):
        tmp = []
        for data_count in range(input_dim):
            tmp.append(np.ndarray.tolist(scaled_data[data_count][data_length])[0])
        total_data.append(tmp)

    if data_flag == "test":
        total_data = total_data[len(total_data) - predict_days - window_size:]

    # get x_train, y_train
    x_train = []
    y_train = []
    for i in range(window_size, len(total_data)):
        x_train.append(total_data[i-window_size:i])
        y_train.append(total_data[i])

    return np.array(x_train), np.array(y_train)

## Return data
def getData(input_dim=4, window_size=60, data_flag="train", predict_days=0):

    # 1. feature scaling
    scaler_list = feature_scaling(input_dim)

    # 2. get dataset
    raw_data = import_data(data_flag)

    # 3. scale dataset
    scaled_data = scale_data(input_dim, scaler_list, raw_data)

    # 4. return data
    return orginize_data(input_dim, window_size, data_flag, predict_days, scaled_data), scaler_list

## Main function
if __name__ == "__main__":
    print("training data: ")
    [x, y], s = getData(4, 60, "train")
    print(x.shape)
    print(y.shape)
    print(s)

    print("\ntesting data: ")
    [x, y], s = getData(4, 60, "test", 20)
    print(x.shape)
    print(y.shape)
    print(s)

