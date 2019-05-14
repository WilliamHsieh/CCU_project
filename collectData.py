import os
import datetime
import pandas as pd
import pandas_datareader.data as web

## Variable
# stock = "0050.TW"
stock = "2330.TW"
nasdaq = "^IXIC"
dji = "^DJI"

## Get training data
def get_training_data():

    # time
    start = datetime.datetime(2014, 1, 1)
    end = datetime.datetime(2018, 12, 31)

    # stock of choice
    data = web.DataReader(stock, 'yahoo', start, end)
    data.to_csv("./data/stock_data_train.csv")
    print("> training data ...")
    print(data.head(5))

    # Nasdaq
    data = web.DataReader(nasdaq, 'yahoo', start, end)
    data.to_csv("./data/nasdaq_train.csv")
    print("> training data ...")
    print(data.head(5))

    # dji
    data = web.DataReader(dji, 'yahoo', start, end)
    data.to_csv("./data/dji_train.csv")
    print("> training data ...")
    print(data.head(5))


## Get testing data
def get_testing_data():

    # time
    start = datetime.datetime(2019, 4, 1)
    end = datetime.datetime(2019, 4, 30)

    # stock of choice
    data = web.DataReader(stock, 'yahoo', start, end)
    data.to_csv("./data/stock_data_test.csv")
    print("> training data ...")
    print(data.head(5))

    # Nasdaq
    data = web.DataReader(nasdaq, 'yahoo', start, end)
    data.to_csv("./data/nasdaq_test.csv")
    print("> training data ...")
    print(data.head(5))

    # dji
    data = web.DataReader(dji, 'yahoo', start, end)
    data.to_csv("./data/dji_test.csv")
    print("> training data ...")
    print(data.head(5))


## Main function
if __name__ == "__main__":
    if not os.path.exists("./data"):
        os.mkdir("./data", 755)
    get_training_data()
    get_testing_data()


