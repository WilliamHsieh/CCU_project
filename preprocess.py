## Import
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

## Data processing
### Feature Scaling
def featureScaling(input_dim):
    # 0. close
    # 1. volumn
    # 2. Nasdaq
    # 3. dji
    scaler_list = []
    for x in range(input_dim):
        scaler_list.append(MinMaxScaler(feature_range = (0, 1)))
    return scaler_list

### Import the data set
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

### Scale training set
def scaleData(input_dim, scaler_list, raw_data):
    scaled_data = []
    for x in range(input_dim):
        scaled_data.append(scaler_list[x].fit_transform(raw_data[x]))

    return scaled_data

### Orginize data
def orginizeData(input_dim, window_size, predict_days, data_frequency,
        data_type, scaled_data, classify_flag):

    # combine all dimension data
    total_data = []
    for data_length in range(0, len(scaled_data[0]), data_frequency):
        tmp = []
        for data_dim in range(input_dim):
            tmp.append(np.ndarray.tolist(scaled_data[data_dim][data_length])[0])
        total_data.append(tmp)

    # deal with test input
    if data_type == "test":
        total_data = total_data[len(total_data) - predict_days - window_size:]

    # get x_data, y_data
    x_data = []
    y_data = []
    #TODO: set the input dataset to fix size
    for i in range(window_size, len(total_data)):
        x_data.append(total_data[i-window_size:i])
        if classify_flag:
            y_data.append(1 if total_data[i][0] >= total_data[i-1][0] else 0)
        else:
            y_data.append(total_data[i])

    # format for classify training
    if data_type == "train" and classify_flag == True:
        y_data = to_categorical(y_data)

    return np.array(x_data), np.array(y_data)

## API
### Get data
def getData(input_dim=4, window_size=60, predict_days=20, data_frequency=1, 
        data_type="train", classify_flag=False):

    # 1. feature scaling
    scaler_list = featureScaling(input_dim)

    # 2. get dataset
    raw_data = importData(data_type)

    # 3. scale dataset
    scaled_data = scaleData(input_dim, scaler_list, raw_data)

    # 4. return data
    return orginizeData(input_dim, window_size, predict_days, data_frequency, 
            data_type, scaled_data, classify_flag), scaler_list

### Get ground truth
def getGT(predict_days=20, data_frequency=1):
    csv_data = pd.read_csv('./data/stock_test.csv')
    close_price = csv_data.iloc[:, 4:5].values    #close
    real_stock_price = []
    for i in range(0, len(close_price), data_frequency):
        real_stock_price += [close_price[i]]
    real_stock_price = real_stock_price[len(real_stock_price) - predict_days:]

    return np.reshape(real_stock_price, (1, -1))[0]

### Get baseline
def getBase(predict_days = 20, data_frequency = 1):
    # Get dataset
    gt = getGT(predict_days=predict_days, data_frequency=data_frequency)
    baseline = getGT(predict_days=predict_days+1, data_frequency=data_frequency)[:-1]

    # Calculate
    tmp = 0
    for i in range(predict_days):
        tmp += (gt[i] - baseline[i]) ** 2
    MSE = tmp / predict_days

    return MSE

## Main function
if __name__ == "__main__":
    print("# get ground truth")
    print(getGT())

    print("\n# get baseline value")
    print(getBase())

    print("\n# get data")
    # (input_dim, window_size, predict_days, data_frequency, data_type, classify_flag)
#     [x, y], s = getData(4, 60, 20, 1, "train", False) #default
    [x, y], s = getData(4, 49, 20, 5, "test", False)
    print(x.shape)
    print(y.shape)

