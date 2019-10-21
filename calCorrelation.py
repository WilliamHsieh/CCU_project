import pandas as pd
import numpy as np
from scipy.stats import pearsonr

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
raw_data = list(np.array(raw_data).reshape(4, -1))
corr, _ = pearsonr(raw_data[0], raw_data[2])
print('stock(close) / nasdaq : %.3f' % corr)
corr, _ = pearsonr(raw_data[0], raw_data[3])
print('stock(close) / dji : %.3f' % corr)

