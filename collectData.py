import os
import datetime
import pandas as pd
import pandas_datareader.data as web

## variable
stock_of_choice = "2330.TW"
# stock_of_choice = "0050.TW"

## Get training data
def get_training_data():

    # get data from yahoo finance api
    start = datetime.datetime(2014, 1, 1)
    end = datetime.datetime(2018, 12, 31)
    data = web.DataReader(stock_of_choice, 'yahoo', start, end)

    # save data
    data.to_csv("./data/stock_data_train.csv")
    print("> training data ...")
    print(data.head(10))


## Get testing data
def get_testing_data():

    # get data from yahoo finance api
    start = datetime.datetime(2019, 1, 1)
    end = datetime.datetime(2019, 1, 31)
    data = web.DataReader(stock_of_choice, 'yahoo', start, end)

    # save data
    data.to_csv("./data/stock_data_test.csv")
    print("> testing data ...")
    print(data.head(10))


## Main function
if __name__ == "__main__":
    if not os.path.exists("./data"):
        os.mkdir("./data", 755)
    get_training_data()
    get_testing_data()

