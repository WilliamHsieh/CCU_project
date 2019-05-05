import pickle

with open(f"./model/epoch_2,dim_2/loss", "rb") as fp:   
    loss = pickle.load(fp)
    print(loss)

