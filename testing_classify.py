## Import
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras import backend as K
from keras.models import load_model
from preprocess import getData

## Variable
MSE = []
total_epochs = 5
input_dim = 4
window_size = 60
predict_days = 20

# Get data
[x_test, y_test], scaler_list = getData(input_dim, window_size, predict_days,
        "test", True)

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
    total_guess = predict_days
    for j in range(len(output)):
        x = output[j].tolist()
        if x.index(max(x)) == y_test[j]:
            acc += 1

    print(output)
    print(y_test)
    print(f"accuracy: {acc * 100 / total_guess:.2f}%")

    end = time.time()
    print(f'model complete: ./model/epoch_{i}.h5 ,  time: {end-start:.02f} secs')

print("done.")

## Save MSE
with open(f"{path}MSE", "wb") as fp:   
    pickle.dump(MSE, fp)

