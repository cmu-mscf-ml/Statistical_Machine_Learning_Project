# -*- coding: utf-8 -*-
'''
calculate each stock's closing executed price of each tick
'''

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, time

param = {'path_data': 'F:\\Class - Statistical Machine Learning II\\project\\'
                      + 'HFT\\1_data_cleaning\\L1.cleaned.20190404.csv',
         'path_project': 'F:\\Class - Statistical Machine Learning II\\'
                      + 'project\\HFT\\Statistical_Machine_Learning_Project'}

dataset = pd.read_csv(param['path_data'], header=0, index_col=0)
dataset = dataset.loc[dataset['type']=='trade']
dataset.index = range(len(dataset))

dataset['time'] = dataset['time'].apply(lambda t: datetime.strptime(t,
                                                   '%Y-%m-%d %H:%M:%S.%f'))
dataset['h_m_s'] = [datetime(t.year, t.month, t.day, t.hour, t.minute, t.second) for t in dataset['time']]
stocks = np.array(dataset['symbol'].unique())
ticks = datetime(2019,4,4,9,30,0,0) + np.arange(int(6.5*3600))*timedelta(0,1,0)
delta_t = timedelta(0,1,0)


exeprice = dataset.groupby(by=['symbol','h_m_s']).last()['tradePrice']
exeprice = exeprice.reset_index()
exeprice = exeprice.pivot(index='h_m_s',columns='symbol',values='tradePrice')
exeprice = exeprice.reindex(ticks)
# exeprice = exeprice.fillna(method='ffill')
exeprice.to_csv(param['path_project']+'\\factors\\exeprice.csv')