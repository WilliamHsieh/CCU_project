## import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import preprocess as P

from sklearn.metrics import r2_score
from scipy.stats import pearsonr

## calculate correlation
def correlation():
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
    raw_data = np.array(raw_data).reshape(4, -1)
    corr, _ = pearsonr(raw_data[0], raw_data[2])
    print('stock(close) / nasdaq : %.3f' % corr)
    corr, _ = pearsonr(raw_data[0], raw_data[3])
    print('stock(close) / dji : %.3f' % corr)
    corr, _ = pearsonr(raw_data[2], raw_data[3])
    print('nasdaq / dji : %.3f' % corr)

## calculate r2
def r2():
    epoch = 300
    dim = 4
    win = 60
    pred = 20
    freq = 5
    fname = ""
#     model_name = f"./model/draw/123/{epoch}_{dim}_{win}_{freq}.h5"

    # Get data
    real_price, pred_price = P.getGTandPred(epoch, dim, win, pred, freq, fname)
    print(real_price)
    print(pred_price)
    
    # r square
    print(r2_score(real_price, pred_price))
    P.drawTrend(real_price, pred_price, pred)

## draw boxplot
def drawBoxplot():
    draw = []

    real_price, pred_price = P.getGTandPred(300, 4, 60, 21, 1, ",default")
    draw += [[real_price[i] - real_price[i+1] for i in range(20)]]      #baseline
    draw += [[pred_price[i+1] - real_price[i+1] for i in range(20)]]    #predict

    real_price, pred_price = P.getGTandPred(300, 4, 60, 21, 1, ",3y")
    draw += [[pred_price[i+1] - real_price[i+1] for i in range(20)]]

    real_price, pred_price = P.getGTandPred(300, 4, 60, 21, 1, ",1y")
    draw += [[pred_price[i+1] - real_price[i+1] for i in range(20)]]

#     axes.set_ylabel('err')
    fig1, axes = plt.subplots()
    axes.set_title('Box Plot')
    axes.boxplot(draw)
    plt.xticks(range(1, len(draw)+1), ["baseline", "6y", "3y", "1y"])
    plt.show()
    
## main
if __name__ == "__main__":
#     correlation()
#     r2()
    drawBoxplot()

