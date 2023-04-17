import numpy as np
import pandas as pd
from utils.util import df2arr, ArrNorm
import torch

SEQUENCE = 5

def load_array(data_arrays, batch_size):
    dataset = torch.utils.data.TensorDataset(*data_arrays)
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle=False)

def create(data_arr, length=SEQUENCE, TYPE=torch.float32) -> torch.Tensor:
    # length: sequence length
    # TODO: create sequence data for convolution

    if isinstance(data_arr, list):
        data_arr = np.array(data_arr)
    assert data_arr.shape[0] >= length, 'data length is too small to create sequence data'
    RE = []
    for i in range(data_arr.shape[0] - length):
        RE.append(data_arr[i:i+length])
    RE = torch.tensor(RE, dtype=TYPE)
    # print(RE.shape)
    return RE




def GetDataset(opt, few_shot=False):
    print('Data loading status: Start.')
    path = opt.data_path if not few_shot else opt.few_shot_path
    if path.endswith('xlsx'):
        try:
            read_data = pd.read_excel(path, header=None)
        except:
            raise NameError('excel file does not exist.')
    elif path.endswith('csv'):
        try:
            read_data = pd.read_csv(path, header=None)
        except:
            raise NameError('csv file does not exist.')
    else:
        raise NameError('invalid file format.')
    
    NAME = ['微站'+str(i+1) for i in range(opt.mini_station_num)]
    NAME.append('标准站')
    rows, cols = read_data.shape
    # print(read_data)
    assert cols>=7, 'invalid feature number'
    name = list(read_data.iloc[:, 0])
    sp = [0]
    for row in range(rows-1):
        for i in range(opt.mini_station_num):
            if name[row]==NAME[i] and name[row+1]==NAME[i+1]:
                sp.append(row)

    sp.append(rows)
    station_minus_list = []
    station_list = []
    
    standard_data = df2arr(read_data.iloc[sp[-2]+1:sp[-1]+1,1:6])
    # standard station data -> numpy.ndarray

    for i in range(len(sp)-2):
        station_data = df2arr(read_data.iloc[sp[i]+1:sp[i+1]+1, 1:6])
        station_minus_list.append(station_data - standard_data)     # output
        station_list.append(station_data)                           # input
    
    assert len(station_minus_list)==opt.mini_station_num, 'invalid: len(station_list) = ' + str(len(station_list))
    assert len(station_minus_list) == len(station_list)
    # one element in 'station_minus_list' contains the whole data in tiny staion
    # station_data[i] -> the (i+1)-th tiny station data

    Max_list = []
    Min_list = []
    Norm_Station_List = []

    for i in range(len(station_minus_list)):
        assert isinstance(station_minus_list, list)
        M, m, s = ArrNorm(station_minus_list[i], opt.norm)
        Max_list.append(M)
        Min_list.append(m)
        Norm_Station_List.append(s)
    # print('type(s) = ', type(s))
    
    print('Data loading status: Finished.')

    assert isinstance(Max_list, list)
    assert isinstance(Min_list, list)
    assert isinstance(station_minus_list, list)
    assert isinstance(Norm_Station_List, list)

    assert len(station_minus_list) == len(Norm_Station_List), 'Invalid Normalization or Illegal Input Data'
    
    # I_O = [(in,out), (in,out), ..., (in,out)]
    # for each tuple (in,out) corresponds to *data in load_array function
    I_O = [(create(station_minus_list[i]).unsqueeze(1), create(np.array(Norm_Station_List[i])).unsqueeze(1)) for i in range(len(station_minus_list))]
    # print(I_O[0][0].shape)
    return Max_list, Min_list, I_O, \
        torch.tensor(standard_data, dtype=torch.float32)
