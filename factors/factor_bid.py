# -*- coding: utf-8 -*-
'''
calculate each stock's closing bid price of each tick
'''

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, time

param = {'path_data': 'F:\\Class - Statistical Machine Learning II\\project\\'
                      + 'HFT\\1_data_cleaning\\L1.cleaned.20190404.csv',
         'path_project': 'F:\\Class - Statistical Machine Learning II\\'
                      + 'project\\HFT\\Statistical_Machine_Learning_Project'}

dataset = pd.read_csv(param['path_data'], header=0, index_col=0)
dataset['time'] = dataset['time'].apply(lambda t: datetime.strptime(t,
                                                   '%Y-%m-%d %H:%M:%S.%f'))
dataset['h_m'] = [time(t.hour, t.minute) for t in dataset['time']]
stocks = np.array(dataset['symbol'].unique())
ticks = datetime(2019,4,4,9,30,0,0) + np.arange(23400)*timedelta(0,1,0)
delta_t = timedelta(0,1,0)

for stock in stocks:
    subdata = dataset.loc[dataset['symbol']==stock]
    subdata.index = range(len(subdata))
    test.groupby(by='a').last()['b']
    tick_end_ind = [list(subdata['time']>=t+delta_t).index(True)-1 for t in ticks]



test = pd.DataFrame([[1,2,3],[1,5,6],[1,5,2],[2,3,6],[2,3,5],[2,6,2]], columns=['a','b','c'])
test.groupby(by=['a','b']).last()
