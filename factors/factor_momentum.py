# -*- coding: utf-8 -*-
'''
calculate each stock's closing bid ask spread of each tick
'''

import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta, time



dataset = pd.read_csv('L1.cleaned.20190404.csv', header=0, index_col=0)
dataset['time'] = dataset['time'].apply(lambda t: datetime.strptime(t,
                                                   '%Y-%m-%d %H:%M:%S.%f'))
dataset['h_m_s'] = [datetime(t.year, t.month, t.day, t.hour, t.minute, t.second) for t in dataset['time']]
stocks = np.array(dataset['symbol'].unique())
ticks = datetime(2019,4,4,9,30,0,0) + np.arange(int(6.5*3600))*timedelta(0,1,0)
delta_t = timedelta(0,1,0)

dataset['mid'] = (dataset['ask']+dataset['bid'])/2
dataset['smart_price'] = (dataset['ask']*dataset['bidSize']+dataset['bid']*dataset['askSize'])/(dataset['bidSize']+dataset['askSize'])


s_list = [1,2,5,10,30]
order_list = [5,10,20,50]
# smartPrice
factor = 'smartPrice_insensitive'
os.mkdir(factor)
smartPrice = dataset.groupby(by=['symbol','h_m_s']).last()['smart_price']
smartPrice = smartPrice.reset_index()
smartPrice = smartPrice.pivot(index='h_m_s',columns='symbol',values='smart_price')
smartPrice = smartPrice.reindex(ticks)
smartPrice = smartPrice.fillna(method='ffill')
for stock in smartPrice.columns:
    stock_factor = smartPrice[[stock]]
    stock_factor.to_csv(os.path.join(factor, stock+'.csv'))



s_list = [1,2,5,10,30]
order_list = [5,10,20,50] 
order_list = [5]
# momentum, order
for ptype in ['mid','smart_price']:
    for s in order_list:
        dir_name = ptype + '_momentum_' + str(s) + 'ord'
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
     
        def f(x):
            y = pd.DataFrame({'symbol': x['symbol'], 'h_m_s': x['h_m_s']})
            y['momentum'] = (x[ptype]-x[ptype].shift(s))/x[ptype].shift(s)
            return y

        momentum = dataset.groupby('symbol').apply(f)
        momentum = momentum.groupby(['h_m_s','symbol']).last()
        momentum = momentum.reset_index()
        momentum = momentum.pivot(index='h_m_s',columns='symbol',values='momentum')
        for stock in momentum.columns:
            stock_factor = momentum[[stock]]
            stock_factor.to_csv(os.path.join(dir_name, stock+'.csv'))

