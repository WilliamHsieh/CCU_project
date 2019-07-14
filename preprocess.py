## Import
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

## Feature Scaling
def featureScaling(input_dim):
    # 0. close
    # 1. volumn
    # 2. Nasdaq
    # 3. dji
    scaler_list = []
    for x in range(input_dim):
        scaler_list.append(MinMaxScaler(feature_range = (0, 1)))
    return scaler_list

## Import the data set
def importData(data_type):
    raw_data = []
    # stock of choice
    csv_data = pd.read_csv('./data/stock_' + data_type + '.csv')
    raw_data.append(csv_data.iloc[:, 4:5].values)  # close
    raw_data.append(csv_data.iloc[:, 5:6].values)  # volumn

    # Nasdaq
    csv_data = pd.read_csv('./data/nasdaq_' + data_type + '.csv')
    raw_data.append(csv_data.iloc[:, 4:5].values)  # close

    # dji
    csv_data = pd.read_csv('./data/dji_' + data_type + '.csv')
    raw_data.append(csv_data.iloc[:, 4:5].values)  # close

    return raw_data

## Scale training set
def scaleData(input_dim, scaler_list, raw_data):
    scaled_data = []
    for x in range(input_dim):
        scaled_data.append(scaler_list[x].fit_transform(raw_data[x]))

    return scaled_data

## Orginize data
def orginizeData(input_dim, window_size, data_type, predict_days, scaled_data, classify_flag):
    # combine all dimension data
    total_data = []
    for data_length in range(len(scaled_data[0])):
        tmp = []
        for data_count in range(input_dim):
            tmp.append(np.ndarray.tolist(scaled_data[data_count][data_length])[0])
        total_data.append(tmp)

    if data_type == "test":
        total_data = total_data[len(total_data) - predict_days - window_size:]

    # get x_data, y_data
    x_data = []
    y_data = []
    for i in range(window_size, len(total_data)):
        x_data.append(total_data[i-window_size:i])
        if classify_flag:
            y_data.append(1 if total_data[i][0] >= total_data[i-1][0] else 0)
        else:
            y_data.append(total_data[i])

    return np.array(x_data), np.array(y_data)

## Return data
def getData(input_dim=4, window_size=60, predict_days=20, data_type="train", classify_flag=False):

    # 1. feature scaling
    scaler_list = featureScaling(input_dim)

    # 2. get dataset
    raw_data = importData(data_type)

    # 3. scale dataset
    scaled_data = scaleData(input_dim, scaler_list, raw_data)

    # 4. return data
    return orginizeData(input_dim, window_size, data_type, predict_days, scaled_data, classify_flag), scaler_list

## Main function
if __name__ == "__main__":
    print("regression train: ")
    [x, y], s = getData(4, 60, 20, "train", False)  #default
    print(x.shape)
    print(y.shape)
    print(s)

    print("")
    print("regression test: ")
    [x, y], s = getData(4, 60, 20, "test", False)
    print(x.shape)
    print(y.shape)
    print(s)

    print("")
    print("classification train: ")
    [x, y], s = getData(4, 60, 20, "train", True)
    print(x.shape)
    print(y.shape)
    print(s)

    print("")
    print("classification test: ")
    [x, y], s = getData(4, 60, 20, "test", True)
    print(x.shape)
    print(y.shape)
    print(s)

