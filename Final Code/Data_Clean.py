# -*- coding: utf-8 -*-
'''
Perform data cleaning on original data.
'''

import numpy as np
import pandas as pd
from datetime import datetime, date, time

param = {'path_init_data': 'F:\\Class - Statistical Machine Learning II\\'
             + 'project\\HFT\\initial_data\\',
         'path_output': 'F:\\Class - Statistical Machine Learning II\\'
             + 'project\\HFT\\1_data_cleaning\\',
         'date': '20190404'}

# load L1 trade data
trade = pd.read_csv(param['path_init_data']+'L1Trade.'+param['date']+'.csv',
                    header=0, index_col=None)
columns = [c.strip() for c in trade.columns]
columns[0] = 'time'
trade.columns = columns
fmt = '%Y%m%d-%H:%M:%S.%f'
trade['time'] = trade['time'].apply(lambda x: datetime.strptime(x[:-15], fmt))
trade.insert(1, 'type', 'trade')
trade['symbol'] = trade['symbol'].apply(lambda x: x.strip())

# load L1 book change data
bk_chg = pd.read_csv(param['path_init_data']+'L1BookChange.'
                     +param['date']+'.csv', 
                     header=0, index_col=None)
columns = [c.strip() for c in bk_chg.columns]
columns[0] = 'time'
bk_chg.columns = columns
bk_chg['time'] = bk_chg['time'].apply(lambda x: datetime.strptime(x[:-20], fmt))
bk_chg.insert(1, 'type', 'bookChange')
bk_chg['symbol'] = bk_chg['symbol'].apply(lambda x: x.strip())

# synchronize them by time stamp, if they have same time stamp, 
# then sort by seqnum 
data = pd.concat([trade,bk_chg], axis=0, join='outer', sort=False)
data = data.sort_values(by=['time', 'seqnum'], ascending=True)
data.index = range(len(data))

# save
data.to_csv(param['path_output']+'L1.'+param['date']+'.csv')

# extract records within continuous trading period
date_ = param['date'].strftime('%Y%m%d')
cts_trade = [datetime.combine(date_, time(9,30,0)), 
             datetime.combine(date_, time(16,0,0))]
n_seconds = int(6.5*3600)
data = data.loc[(data['time']>=cts_trade[0]) & (data['time']<=cts_trade[1])]
data.index = range(len(data))

# extract stocks with enough records
cnt_threshold = n_seconds*3
records_cnt = data.groupby('symbol').count()['time']
stocks = np.array(records_cnt.index[records_cnt > cnt_threshold])
data_filtered = None
for s in stocks:
    subdata = data.loc[data['symbol']==s]
    subdata.indx = range(len(subdata))
    data_filtered = pd.concat([data_filtered,subdata], axis=0)
data_filtered.index = range(len(data_filtered))

# save
data_filtered.to_csv(param['path_output']+'L1.cleaned.'+param['date']+'.csv')


'''
# save for future usage
# load data
data = pd.read_csv(param['path_output']+'L1.cleaned.'+param['date']+'.csv',
                   header=0, index_col=0)
fmt = '%Y-%m-%d %H:%M:%S.%f'
data['time'] = data['time'].apply(lambda t: datetime.strptime(t, fmt)
'''