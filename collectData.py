import pandas as pd
import pandas_datareader.data as web
import datetime

## train
# 台灣股市的話要用 股票代號 加上 .TW
start = datetime.datetime(1980, 1, 1)
end = datetime.datetime(2018, 12, 31)
data = web.DataReader('2330.TW', 'yahoo', start, end)
# data = web.DataReader('0050.TW', 'yahoo', start, end)
# data = web.DataReader('AAPL', 'iex', start, end)

# save data
# data.to_csv("stock_data_train.csv")
data.to_csv("tmp.csv")
print(data.head(10))
print(len(data))

## test data
# start = datetime.datetime(2019, 1, 1)
# end = datetime.datetime(2019, 1, 31)
# data = web.DataReader('0050.TW', 'yahoo', start, end)
# data.to_csv("stock_data_test.csv")

