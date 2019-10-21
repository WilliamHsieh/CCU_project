import pandas as pd
money = 10000000
bought = 0

# stock of choice
data_type = "test"
csv_data = pd.read_csv('./data/stock_' + data_type + '.csv')[:51]
PDC = csv_data.iloc[:, 4:5].values  # close
H = csv_data.iloc[:, 1:2].values  # high
L = csv_data.iloc[:, 2:3].values  # low
PDN = []

def TR(i):
    global H
    global L
    global PDC
    return max(H[i]-L[i], H[i]-PDC[i], PDC[i]-L[i])

def getN(i):
    global PDN
    return (19 * PDN[i-1] + TR(i)) / 20

def getN0(i):
    global H
    global L
    global PDC
    avg = 0
    for i in range(i-20, i):
        avg += TR(i)
    return avg / 20

def preLow(i):
    global H
    global L
    global PDC
    m = TR(i)
    for i in range(i-19, i):
        m = min(m, TR(i))
    return m

def preHigh(i):
    global H
    global L
    global PDC
    M = TR(i)
    for i in range(i-19, i):
        M = max(M, TR(i))
    return M

def unit(i):
    return money / getN(i) / 100

def trade(pred = 0):
    global PDC
    global bought
    global money
    pre_buy_price = 0
    for i in range(21, 50):
        if bought == 0:
            pre = preHigh(i)
            if PDC[i + pred] > pre:
                money -= PDC[i]
                bought += 1
                pre_buy_price = PDC[i]

        else:
            if PDC[i + pred] - pre_buy_price >= 0.5 * getN(i):
                money -= PDC[i]
                bought += 1
                pre_buy_price = PDC[i]

            # sell all
            if pre_buy_price - PDC[i + pred] >= 2 * getN(i) or PDC[i + pred] < preLow(i):
                money += PDC[i] * bought
                bought = 0

    if bought != 0:
        money += PDC[i] * bought
        bought = 0


def init():
    global PDN
    PDN += [getN0(21)]
    for i in range(1, 50):
        PDN += [getN(i)]


init()
trade(0)
print("balance:", money)
trade(1)
print("balance:", money)




