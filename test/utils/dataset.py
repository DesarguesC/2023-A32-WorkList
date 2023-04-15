import numpy as np
import pandas as pd
from utils.util import df2arr, ArrNorm
import torch

def load_array(data_arrays, batch_size, is_train=True):
    dataset = torch.utils.data.TensorDataset(*data_arrays)
    return torch.utils.ata.DataLoader(dataset, batch_size, shuffle=is_train)



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
    assert cols==6, 'invalid feature number'
    name = list(read_data.iloc[:, 0])
    sp = [0]
    for row in range(rows-1):
        for i in range(opt.mini_station_num):
            if name[row]==NAME[i] and name[row+1]==NAME[i+1]:
                sp.append(row)

    sp.append(rows)
    # print(read_data.shape)
    # print(name[sp[0]-1], name[sp[0]])
    # print('debug: sp = ', sp)

    station_minus_list = []
    station_list = []
    
    standard_data = df2arr(read_data.iloc[sp[-2]+1:sp[-1]+1,1:6])

    for i in range(len(sp)-2):
        station_data = df2arr(read_data.iloc[sp[i]+1:sp[i+1]+1, 1:6])
        station_minus_list.append(station_data - standard_data)
        station_list.append(station_data)

    # station_minus_list.append(standard_data)

    assert len(station_minus_list)==opt.mini_station_num, 'invalid: len(station_list) = ' + str(len(station_list))

    # print(station_list)

    Max_list = []
    Min_list = []
    Norm_Station_List = []

    for i in range(len(station_minus_list)-1):
        temp_name = 'station' + str(i+1)
        M, m, s = ArrNorm(station_minus_list[i][temp_name], opt.norm)
        Max_list.append(M)
        Min_list.append(m)
        Norm_Station_List.append(s)
    
    print('Data loading status: Finished.')

    assert isinstance(Max_list, list)
    assert isinstance(Min_list, list)
    assert isinstance(station_minus_list, list)
    assert isinstance(Norm_Station_List, list)

    return Max_list, Min_list, \
        torch.tensor(station_minus_list, dtype=torch.float32), torch.tensor(Norm_Station_List, dtype=torch.float32)
