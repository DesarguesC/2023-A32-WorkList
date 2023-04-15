import numpy as np
import torch
import pandas as pd
from torch import nn
from torch.utils import data
from sklearn.preprocessing import MinMaxScaler
from scipy import stats

Epoch_num = 5000
Learning_Rate = 0.001

class mlp(nn.Module):
    
    def __init__(self, d_in, d_hidden, d_out, dropout= 0.05):
        
        super(mlp, self).__init__()

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(d_in, d_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(d_hidden, d_out),
            torch.nn.ReLU()
        )
        self.dropout = dropout
        
        #self.mlp.weight.data = get_weight_initial(d_out, d_in)
        
    def forward(self, x):
        H_out = self.mlp(x)
        return H_out
    
def normalization1(x):
    x = x.detach().numpy()
    scalar = MinMaxScaler(feature_range=(0, 1))
    x_nor = scalar.fit_transform(x)
    return x_nor

def df2arr(x) -> np.ndarray:
    return np.array(x, dtype=np.float32)

def GetDataset(input_arr: list, output_arr: list, seq: int):
    assert(len(input_arr)==len(output_arr)), "Different size of input and output!"
    Input = []
    Output = []
    for i in range(input_arr.shape[0]-seq):
        Input.append(input_arr[i:i+seq][:])
        Output.append(output_arr[i:i+seq][:])
    return torch.tensor(Input, dtype=torch.float32), torch.tensor(Output, dtype=torch.float32)

def load_array(data_arrays, batch_size, is_train=True):
    # data-iter
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

def rsquared(x, y): 
    _, _, r_value, _, _ = stats.linregress(x.detach().numpy(), y.detach().numpy()) 
    return r_value**2

def get_torch_col(x,i):
    arr = x[:,i]
    return arr

excel = pd.read_excel('./Data/A32.xlsx', header=None)
sp = [1486, 2972, 4458]
station_1 = excel.iloc[1:sp[0]+1,1:6]
station_2 = excel.iloc[sp[0]+1:sp[1]+1,1:6]
standard = excel.iloc[sp[1]+1:sp[2]+1,1:6]
station_1 = df2arr(station_1)
station_2 = df2arr(station_2)
standard = df2arr(standard)

station_1 = torch.from_numpy(station_1)
station_2 = torch.from_numpy(station_2)
standard = torch.from_numpy(standard)

def Loss_func(pred1, labels):
    mmse = torch.nn.MSELoss()
    Loss= mmse(pred1, labels)
    return Loss

st=station_1

model = mlp(d_in = station_2.shape[1], d_hidden = 32, d_out = standard.shape[1])
optimzer = torch.optim.Adam(model.parameters(), lr=Learning_Rate)

r=0
i=1

#for epoch in range(Epoch_num):
while(r<=1):
    pred = model(station_2)
    loss = Loss_func(pred, standard)
    optimzer.zero_grad()
    loss.backward()
    optimzer.step()
    #print("Epoch: [{}]/[{}]".format(epoch + 1, Epoch_num))
    k=1
    r = rsquared(get_torch_col(pred,k), get_torch_col(standard,k))
    print("Epoch {}, R² = {}".format(i, r))
    i+=1

# station_1  0:0.70  1:0.649  2:0.61  3:0.79  4:0.55
# station_2  0:0.72  1:0.67   2:0.52  3:0.73  4:0.58