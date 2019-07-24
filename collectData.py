import os
import datetime
import pandas as pd
import pandas_datareader.data as web
from datafit import datafit

## Variable
# stock = "0050.TW"
stock = "2330.TW"
nasdaq = "^IXIC"
dji = "^DJI"

## Get training data
def get_training_data():

    # time
    start = datetime.datetime(2000, 1, 1)
    end = datetime.datetime(2016, 12, 31)

    # stock of choice
    data = web.DataReader(stock, 'yahoo', start, end)
    data.to_csv("./data/stock_train.csv")
    print("> train stock ...")
    print(data.head(3))

    # Nasdaq
    data = web.DataReader(nasdaq, 'yahoo', start, end)
    data.to_csv("./data/nasdaq_train.csv")
    print("> train nasdaq ...")
    print(data.head(3))

    # dji
    data = web.DataReader(dji, 'yahoo', start, end)
    data.to_csv("./data/dji_train.csv")
    print("> train dji ...")
    print(data.head(3))


## Get testing data
def get_testing_data():

    # time
    start = datetime.datetime(2017, 1, 1)
    end = datetime.datetime(2018, 12, 31)

    # stock of choice
    data = web.DataReader(stock, 'yahoo', start, end)
    data.to_csv("./data/stock_test.csv")
    print("> test stock ...")
    print(data.head(3))

    # Nasdaq
    data = web.DataReader(nasdaq, 'yahoo', start, end)
    data.to_csv("./data/nasdaq_test.csv")
    print("> test nasdaq ...")
    print(data.head(3))

    # dji
    data = web.DataReader(dji, 'yahoo', start, end)
    data.to_csv("./data/dji_test.csv")
    print("> test dji ...")
    print(data.head(3))


## Main function
if __name__ == "__main__":
    if not os.path.exists("./data"):
        os.mkdir("./data", 755)

    get_training_data()
    datafit('./data/nasdaq_train.csv', './data/stock_train.csv')
    datafit('./data/dji_train.csv', './data/stock_train.csv')

    get_testing_data()
    datafit('./data/nasdaq_test.csv', './data/stock_test.csv')
    datafit('./data/dji_test.csv', './data/stock_test.csv')

