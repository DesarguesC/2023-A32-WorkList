import numpy as np
import pandas as pd
from utils.util import df2arr, ArrNorm



def GetDataset(opt):
    print('Data loading status: Start.')
    path = opt.data_path.lower()
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

    station_list = []
    standard_data = df2arr(read_data.iloc[sp[-2]+1:sp[-1]+1,1:6])

    for i in range(len(sp)-2):
        station_list.append({
            'station'+str(i+1): \
                df2arr(read_data.iloc[sp[i]+1:sp[i+1]+1, 1:6]) - standard_data
        })

    station_list.append({
        'standard': standard_data
    })

    assert len(station_list)-1==opt.mini_station_num, 'invalid: len(station_list) = ' + str(len(station_list))

    # print(station_list)

    Max_list = []
    Min_list = []
    Norm_Station_List = []

    for i in range(len(station_list)-1):
        temp_name = 'station' + str(i+1)
        M, m, s = ArrNorm(station_list[i][temp_name], opt.norm)
        Max_list.append({
            temp_name: M
        })
        Min_list.append({
            temp_name: m
        })
        Norm_Station_List.append({
            temp_name: s
        })
    
    print('Data loading status: Finished.')
    return Max_list, Min_list, Norm_Station_List