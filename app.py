## Import
import os
import numpy as np
from preprocess import getGT
from preprocess import getData
from keras import backend as K
from keras.models import load_model

from flask import Flask, render_template, request

app = Flask(__name__)

## "/"
@app.route("/", methods=['POST', 'GET'])
def index():
    if request.method == "POST":
        total_epochs = int(request.form['total_epochs'])
        input_dim = int(request.form['input_dim'])
        window_size = int(request.form['window_size'])
        predict_days = int(request.form['predict_days'])
        data_frequency = int(request.form['data_frequency'])

        result = predict(total_epochs, input_dim, window_size, predict_days, data_frequency)
        return str(result)
    else:
        return render_template("index.html")

## predict
def predict(epoch=300, dim=4, win=49, pred=20, freq=5):
#     epoch = 300
#     dim = 4
#     win = 60
#     pred = 20
#     freq = 5
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
    predicted_stock_price = scaler_list[0].inverse_transform(close_price)
    return predicted_stock_price[0]

## Main function
if __name__ == "__main__":
    app.run(debug=True)
#     print(predict(300, 4, 49, 20, 5))
#     print(predict(100, 2, 60, 20, 7))

