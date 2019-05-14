from torch import nn

class LSTM_signal(nn.Module):
    #建立LSTM class
    def __init__(self,input_feature_dim,hidden_feature_dim,hidden_layer_num,batch_size,classes_num):
        super(LSTM_signal,self).__init__()
        self.input_feature_dim=input_feature_dim
        self.hidden_feature_dim=hidden_feature_dim
        self.hidden_layer_num=hidden_layer_num
        self.batch_size=batch_size 

        #初始化LSTM       
        self.lstm=nn.LSTM(input_feature_dim,hidden_feature_dim,hidden_layer_num)

        #LSTM的輸出藉由單層的線性神經網路層分類~
        self.linear1=nn.Linear(hidden_feature_dim,classes_num)

    def init_hidden(self):
        h0=t.randn(self.hidden_layer_num,self.batch_size,self.hidden_feature_dim)
        c0=t.randn(self.hidden_layer_num,self.batch_size,self.hidden_feature_dim)

    def forward(self,input):
        output,(hn,cn)=self.lstm(input,(h0,c0))  
        output=self.linear1(output[-1])         
        return output,(hn,cn)

lstm=LSTM_signal(3, 4, 5, 6, 7)
print(lstm)


tmp = True
if tmp is not True:
    print("haha")
else:
    print("lala")

