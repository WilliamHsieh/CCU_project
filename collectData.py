import os
import datetime
import numpy as np
import pandas as pd
import pandas_datareader.data as web
from datafit import datafit

## Get stock data
def get_stock(start, end, datatype, s=0):
    stock = ["^DJI", "^IXIC", "2330.TW"]
    filename = ['./data/dji_' + datatype + '.csv', './data/nasdaq_' + datatype + '.csv', './data/stock_' + datatype + '.csv']
    for i in range(len(stock)):
        data = web.DataReader(stock[i], 'yahoo', start, end)[s:]
        data.to_csv(filename[i])
        print(">", datatype, stock[i])
        print(data.head(2))
        print(data.tail(2))
    print("")

## Cal all stock correlation
def get_corr(stock="2330.TW", sy=2017, sm=1, sd=1, ty=2018, tm=12, td=31):
    import sys
    from scipy.stats import pearsonr

    # time
    start = datetime.datetime(sy, sm, sd)
    end = datetime.datetime(ty, tm, td)
    gg = web.DataReader(stock, 'yahoo', start, end)
    gg = gg.iloc[:, 4:5].values
    gg = np.array(gg).reshape(-1)

    fp = open("stock.txt", "a")

    data = []
    for i in range(3020, 10000):
        x = str(str(i) + ".TW")
        try:
            # stock of choice
            tmp = web.DataReader(x, 'yahoo', start, end)
            tmp = tmp.iloc[:, 4:5].values
            tmp = np.array(tmp).reshape(-1)
            corr, _ = pearsonr(gg, tmp)
            if abs(corr) > 0.5:
                data += [[x, corr]]
                fp.write(str(data[-1]) + "\n")
                print(data[-1])
            else:
                print([x, corr], "shit")

        except Exception:
            print(x, "gg")
#         except KeyError:
#             pass
#             sys.stderr.write(x + " gg")
    fp.close()

## Main function
if __name__ == "__main__":
    if not os.path.exists("./data"):
        os.mkdir("./data", 755)

    # get stock data
#     datatype = "train"
    datatype = "test"
    cut = -500 if datatype == "test" else 0
#     start = datetime.datetime(2015, 10, 30)
#     end = datetime.datetime(2019, 10, 30)
#     start = datetime.datetime(2000, 5, 1)
#     end = datetime.datetime(2019, 5, 1)
    start = datetime.datetime(2015, 9, 11)
    end = datetime.datetime(2019, 9, 11)
    get_stock(start, end, datatype, cut)

    print("> fitting data ...")
    filename = ['./data/dji_' + datatype + '.csv', './data/nasdaq_' + datatype + '.csv', './data/stock_' + datatype + '.csv']
    for name in filename[:2]:
        datafit(name)
    print("done.\n")

    # append newest data
    print("> append newest data...")
    for name in filename:
        data = pd.read_csv(name)
        data = data.append(data.tail(1))
        data.to_csv(name, index=False)
        print(data.tail(3))
    print("done.")

