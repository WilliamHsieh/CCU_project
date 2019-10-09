## Import
import numpy as np
from preprocess import getGT
from preprocess import getData
from keras.models import load_model
from keras import backend as K
from flask import Flask

app = Flask(__name__)

## "/"
@app.route("/")
def hello():
    tmp = predict()[0]
    return str(tmp)

## predict
def predict():
    # Variable
    MSE = []
    total_epochs = 300
    input_dim = 4
    window_size = 49
    predict_days = 20
    data_frequency = 5

    # Get data
    real_stock_price = getGT(predict_days, data_frequency)
    [x_test, y_test], scaler_list = getData(input_dim, window_size, predict_days, data_frequency, "test")

    # Model predict
    K.clear_session()
    path = f"./model/draw/"
    lstm = load_model(f'{path}test1.h5')
    lstm_output = lstm.predict(x_test)

    # get all the close price
    lstm_close_price = []
    for j in range(len(lstm_output)):
        lstm_close_price.append(lstm_output[j][0])

    # re-scale back
    lstm_close_price = np.reshape(lstm_close_price, (1, -1))
    lstm_predicted_stock_price = scaler_list[0].inverse_transform(lstm_close_price)
    return lstm_predicted_stock_price[0]

## Main function
if __name__ == "__main__":
    app.run()

