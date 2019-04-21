# -*- coding: utf-8 -*-
'''
calculate each stock's closing bid ask spread of each tick
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

dataset['spread'] = dataset['ask']-dataset['bid']

# snapshot
factor_name = 'spread_snapshot'
spread_snapshot = dataset.groupby(by=['symbol','h_m_s']).last()['spread']
spread_snapshot = spread_snapshot.reset_index()
spread_snapshot = spread_snapshot.pivot(index='h_m_s',columns='symbol',values='spread')
spread_snapshot = spread_snapshot.reindex(ticks)
spread_snapshot = spread_snapshot.fillna(method='ffill')
for stock in spread_snapshot.columns:
    stock_factor = spread_snapshot[[stock]]
    stock_factor.to_csv(param['path_project']+'\\factors\\'+factor_name+'\\'+stock+'.csv')

# time sensitive
# 1s
factor_name = 'spread_1s'
factor = dataset.groupby(by=['symbol','h_m_s']).mean()['spread']
factor = factor.reset_index()
factor = factor.pivot(index='h_m_s',columns='symbol',values='spread')
factor = factor.reindex(ticks)
factor = factor.fillna(method='ffill')
for stock in factor.columns:
    stock_factor = factor[[stock]]
    stock_factor.to_csv(param['path_project']+'\\factors\\'+factor_name+'\\'+stock+'.csv')

# 2s
factor_name = 'spread_2s'
factor = dataset.groupby(by=['symbol','h_m_s']).mean()['spread']
factor = factor.reset_index()
factor = factor.pivot(index='h_m_s',columns='symbol',values='spread')
factor = factor.reindex(ticks)
factor = factor.fillna(method='ffill')
factor = factor.rolling(2).mean()
for stock in factor.columns:
    stock_factor = factor[[stock]]
    stock_factor.to_csv(param['path_project']+'\\factors\\'+factor_name+'\\'+stock+'.csv')

# 5s
factor_name = 'spread_5s'
factor = dataset.groupby(by=['symbol','h_m_s']).mean()['spread']
factor = factor.reset_index()
factor = factor.pivot(index='h_m_s',columns='symbol',values='spread')
factor = factor.reindex(ticks)
factor = factor.fillna(method='ffill')
factor = factor.rolling(5).mean()
for stock in factor.columns:
    stock_factor = factor[[stock]]
    stock_factor.to_csv(param['path_project']+'\\factors\\'+factor_name+'\\'+stock+'.csv')

# 10s
factor_name = 'spread_10s'
factor = dataset.groupby(by=['symbol','h_m_s']).mean()['spread']
factor = factor.reset_index()
factor = factor.pivot(index='h_m_s',columns='symbol',values='spread')
factor = factor.reindex(ticks)
factor = factor.fillna(method='ffill')
factor = factor.rolling(10).mean()
for stock in factor.columns:
    stock_factor = factor[[stock]]
    stock_factor.to_csv(param['path_project']+'\\factors\\'+factor_name+'\\'+stock+'.csv')

# 30s
factor_name = 'spread_30s'
factor = dataset.groupby(by=['symbol','h_m_s']).mean()['spread']
factor = factor.reset_index()
factor = factor.pivot(index='h_m_s',columns='symbol',values='spread')
factor = factor.reindex(ticks)
factor = factor.fillna(method='ffill')
factor = factor.rolling(30).mean()
for stock in factor.columns:
    stock_factor = factor[[stock]]
    stock_factor.to_csv(param['path_project']+'\\factors\\'+factor_name+'\\'+stock+'.csv')

# order sensitive
# 5 orders
# 10 orders
# 20 orders
# 50 orders





spread_diff = spread-spread.shift(1)
spread_diff.to_csv(param['path_project']+'\\factors\\spread_diff.csv')