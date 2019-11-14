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
        data_type, scaled_data, classify_flag, data_size):

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
    elif data_size + window_size <= len(total_data):
        total_data = total_data[len(total_data) - data_size - window_size:]

    # get x_data, y_data
    x_data = []
    y_data = []
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
        data_type="train", classify_flag=False, data_size=1500):

    # 1. feature scaling
    scaler_list = featureScaling(input_dim)

    # 2. get dataset
    raw_data = importData(data_type)

    # 3. scale dataset
    scaled_data = scaleData(input_dim, scaler_list, raw_data)

    # 4. return data
    return orginizeData(input_dim, window_size, predict_days, data_frequency, 
            data_type, scaled_data, classify_flag, data_size), scaler_list

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

### Get predicted price
def getPrice(epoch=300, dim=4, win=60, pred=20, freq=1, fname=""):
    import os
    from keras import backend as K
    from keras.models import load_model
    
    # Get data
    [x_test, y_test], scaler_list = getData(dim, win, pred, freq, "test")

    # Model predict
    K.clear_session()
    model_name = f"./model/epoch_{epoch},dim_{dim},win_{win},freq_{freq}" + fname + f"/epoch_{epoch-1}.h5"
    if not os.path.isfile(model_name):
        return "no such model!"

    model = load_model(model_name)
    model_output = model.predict(x_test)

    # get all the close price
    close_price = []
    for j in range(len(model_output)):
        close_price.append(model_output[j][0])

    # re-scale back
    close_price = np.reshape(close_price, (1, -1))
    predicted_stock_price = scaler_list[0].inverse_transform(close_price)[0]
    return predicted_stock_price

### Get GT && Pred
def getGTandPred(epoch=300, dim=4, win=60, pred=20, freq=1, fname=""):
    real_price = getGT(pred, freq)
    pred_price = getPrice(epoch, dim, win, pred, freq, fname)
    return real_price, pred_price

## Draw trend
def drawTrend(real_price, pred_price, pred_days):
    import matplotlib.pyplot as plt
    label1 = "Real Stock Price"
    label2 = "Predicted Stock Price"
    #marker = "p"

    plt.clf()
    plt.style.use("ggplot")   # beautiful shit
    plt.title('predicted stock price')
    plt.plot(real_price, 'bo-', label = label1)
    plt.plot(pred_price, 'ro-', label = label2, marker = "^")
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.xticks(range(pred_days), [i+1 for i in range(pred_days)])
    plt.legend()
#     plt.savefig(filename + '.png')
    plt.show()

## Main function
if __name__ == "__main__":
    print("# get ground truth")
    real_price = getGT()
    print(real_price)

    print("\n# get baseline value")
    print(getBase())

    print("\n# get predicted price")
    pred_price = getPrice(epoch=300, dim=4, win=49, pred=20, freq=5)
    print(pred_price)

#     input_dim, window_size, predict_days, data_frequency, data_type, classify_flag, data_size
    print("\n# get data")
    [x, y], s = getData(4, 60, 20, 1, "train", False, 1500) #default
    print(x.shape)
    print(y.shape)

    print("\n# draw trend")
    print(drawTrend(real_price, pred_price, pred_days=20))

