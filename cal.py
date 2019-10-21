import pandas as pd
import numpy as np

## calculate correlation
def correlation():
    from scipy.stats import pearsonr
    data_type = "train"

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

# calculate Pearson's correlation
    raw_data = list(np.array(raw_data).reshape(4, -1))
    corr, _ = pearsonr(raw_data[0], raw_data[2])
    print('stock(close) / nasdaq : %.3f' % corr)
    corr, _ = pearsonr(raw_data[0], raw_data[3])
    print('stock(close) / dji : %.3f' % corr)

## calculate r2
def r2():
    import os
    from keras import backend as K
    from keras.models import load_model
    from sklearn.metrics import r2_score
    from preprocess import getGT, getData
    pass
    epoch = 300
    dim = 4
    win = 49
    pred = 20
    freq = 5
    # Get data
    real_stock_price = getGT(pred, freq)
    [x_test, y_test], scaler_list = getData(dim, win, pred, freq, "test")

    # Model predict
    K.clear_session()
    model_name = f"./model/draw/epoch_{epoch},dim_{dim},win_{win},freq_{freq}.h5"
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
    
    # r square
    print(r2_score(real_stock_price, predicted_stock_price))
    print(r2_score(real_stock_price, real_stock_price))

## main
if __name__ == "__main__":
    correlation()
    r2()

