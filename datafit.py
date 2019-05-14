import pandas as pd

# input_name = './data/nasdaq_train.csv'
input_name = './data/nasdaq_test.csv'
# input_name = './data/dji_train.csv'
# input_name = './data/dji_test.csv'
# std_name = './data/stock_data_train.csv'
std_name = './data/stock_data_test.csv'

data_ng = pd.read_csv(input_name)
data_sample = pd.read_csv(std_name)

for i in range(len(data_sample['Date'])):
#for i in range(6):
    if(data_sample['Date'][i] != data_ng['Date'][i]):
        flag_see=0
        for j in range(i,len(data_sample['Date'])):
            if(data_sample['Date'][j] == data_ng['Date'][i]):
                flag_see=1		

#         print(data_sample['Date'][i] , data_ng['Date'][i])

#data concate
        if(flag_see == 1):
            above = data_ng.loc[:i]
            below = data_ng.loc[i:]
            data_ng = above.append(below,ignore_index=True)		
            i=i+1
        if(flag_see  == 0):
            above = data_ng.loc[:i]
            below = data_ng.loc[i+2:]
            data_ng = above.append(below,ignore_index=True)
            i=i-1
        
#         print(data_ng.loc[i-3:i+3])
			
data_ng.to_csv(input_name, index=False)
