# -*- coding: utf-8 -*-
'''
calculate each stock's closing smart price of each tick
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
dataset['h_m_s'] = [datetime(t.year, t.month, t.day, t.hour, t.minute, t.second) for t in dataset['time']]
stocks = np.array(dataset['symbol'].unique())
ticks = datetime(2019,4,4,9,30,0,0) + np.arange(int(6.5*3600))*timedelta(0,1,0)
delta_t = timedelta(0,1,0)


smart = dataset.groupby(by=['symbol','h_m_s']).last()[['bidSize','bid','ask','askSize']]
smart = smart['bid']/smart['bidSize']+smart['ask']/smart['askSize']
smart = smart.reset_index()
smart = smart.rename(columns={0:'smart'})
smart = smart.pivot(index='h_m_s',columns='symbol',values='smart')
smart = smart.reindex(ticks)
smart = smart.fillna(method='ffill')
smart.to_csv(param['path_project']+'\\factors\\smart_price.csv')