import pandas as pd

def datafit(input_name):

    # train or test
    if (input_name.find("train") != -1):
        sample_name = './data/stock_train.csv'
    else:
        sample_name = './data/stock_test.csv'

    # input
    data_input = pd.read_csv(input_name)
    data_sample = pd.read_csv(sample_name)

    input_ptr = 0
    sample_ptr = 0
    data_output = pd.DataFrame(columns=data_sample.columns)

    for i in range(len(data_sample['Date'])):

        # loop through input data to find the match
        index = -1
        for j in range(input_ptr, len(data_input['Date'])):
            if (data_sample['Date'][i] == data_input['Date'][j]):
                index = j
                break

        # same record not found -> skip it
        if (index == -1):
            continue

        # length between previous record and current one
        sample_tmp = i - sample_ptr
        input_tmp = index - input_ptr

        # append the previous and current record
        if (input_tmp >= sample_tmp):
            data_output = data_output.append(data_input.loc[index-sample_tmp:index], ignore_index=True)
        else:
            data_output = data_output.append(data_input.loc[index-input_tmp:index], ignore_index=True)
            for k in range(sample_tmp - input_tmp):
                data_output = data_output.append(data_input.loc[index], ignore_index=True)

#         print(data_sample['Date'][i], data_input['Date'][index])
        sample_ptr = i + 1
        input_ptr = index + 1

    # fill in missing record in the end
    while (len(data_output) > len(data_sample)):
        data_output = data_output.drop(data_output.index[-1])
    while (len(data_output) < len(data_sample)):
        data_output = data_output.append(data_output.loc[data_output.index[-1]], ignore_index=True)

    # output for debug
#     for i in range(5):
#         print(data_sample['Date'][i], data_output['Date'][i])
#     for i in range(len(data_output)-5,len(data_output)):
#         print(data_sample['Date'][i], data_output['Date'][i])
#     print("data input:", len(data_input))
#     print("data sample:", len(data_sample))
#     print("data output:", len(data_output))

    # write to file
    data_output.to_csv(input_name, index=False)
                    
if __name__ == "__main__":
    input_name = './data/nasdaq_test.csv'
    datafit(input_name)

