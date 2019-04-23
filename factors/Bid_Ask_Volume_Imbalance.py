import numpy as np
import pandas as pd
from datetime import datetime, timedelta, time

'''
dataset = pd.read_csv('L1.cleaned.20190404.csv', header=0, index_col=0)

dataset['time'] = dataset['time'].apply(lambda t: datetime.strptime(t,
                                                   '%Y-%m-%d %H:%M:%S.%f'))
dataset['h_m_s'] = [time(t.hour, t.minute, t.second) for t in dataset['time']]
stocks = np.array(dataset['symbol'].unique())
ticks = datetime(2019,4,4,9,30,0,0) + np.arange(int(6.5*3600))*timedelta(0,1,0)
delta_t = timedelta(0,1,0)
'''

# Volume_Imbalance

dataset['size_spread'] = (dataset[dataset['type'] == 'bookChange']['bidSize'] - dataset[dataset['type'] == 'bookChange']['askSize'])/(dataset[dataset['type'] == 'bookChange']['bidSize'] + dataset[dataset['type'] == 'bookChange']['askSize'])
spread = dataset.groupby(by=['symbol','h_m_s'])['size_spread'].last().reset_index()
spread = spread.pivot(index='h_m_s',columns='symbol',
                      values='size_spread').reindex(ticks)
spread.fillna(method = 'ffill', inplace = True)
spread.to_csv('./factors/Volume_Imbalance.csv')




# TRADE SIGN - by time

dataset['trade_sign'] = dataset[dataset['type'] == 'trade']['tradeSide'].map(lambda x: -1 if ' SELL' in x else 1 if ' BUY' in x else 0)
spread = dataset.groupby(by=['symbol','h_m_s']).sum()['trade_sign'].reset_index()
spread = spread.pivot(index='h_m_s',columns='symbol',
                      values='trade_sign').reindex(ticks)

spread.fillna(0, inplace = True)

for column in spread.columns.tolist():
    #print('./factors/trade_sign/'+column+'.csv')
    spread[[column]].to_csv('./factors/trade_sign/'+column+'.csv')

spread_2 = spread.ewm(span=2).mean()
for column in spread_2.columns.tolist():
    spread_2[[column]].to_csv('./factors/trade_sign_2/'+column+'.csv')

spread_3 = spread.ewm(span=3).mean()
for column in spread_3.columns.tolist():
    spread_3[[column]].to_csv('./factors/trade_sign_3/'+column+'.csv')

spread_10 = spread.ewm(span=10).mean()
for column in spread.columns.tolist():
    spread_10[[column]].to_csv('./factors/trade_sign_10/'+column+'.csv')

spread_30 = spread.ewm(span=30).mean()
for column in spread.columns.tolist():
    spread_30[[column]].to_csv('./factors/trade_sign_30/'+column+'.csv')


#TRADE SIGN - by order

factornames = ['trade_sign_snapshot', 'trade_sign_2ord', 'trade_sign_5ord', 'trade_sign_10ord']
num = [1, 2, 5, 10]
for i, factor_name in enumerate(factornames):
    for stock, group in dataset[dataset['type'] == 'trade'].groupby('symbol'):
        stock_factor = group[['trade_sign']].rolling(num[i]).sum()
        stock_factor['h_m_s'] = group['h_m_s']
        stock_factor = stock_factor.groupby('h_m_s').last()[['trade_sign']]
        stock_factor = stock_factor.reindex(ticks)
        stock_factor = stock_factor.fillna(method='ffill')
        stock_factor.to_csv('./factors/'+factor_name+'/'+stock+'.csv')
# TtE SIGN - by time

# transaction - by order
factornames = ['transaction_spread_snapshot', 'transaction_spread_2ord', 'transaction_spread_5ord', 'transaction_spread_10ord']
num = [1, 2, 5, 10]
for i, factor_name in enumerate(factornames):
    for stock, group in dataset[dataset['type'] == 'trade'].groupby('symbol'):
        stock_factor = group[['transaction_spread']].rolling(num[i]).sum()
        stock_factor['h_m_s'] = group['h_m_s']
        stock_factor = stock_factor.groupby('h_m_s').last()[['transaction_spread']]
        stock_factor = stock_factor.reindex(ticks)
        stock_factor = stock_factor.fillna(method='ffill')
        stock_factor.to_csv('./factors/'+factor_name+'/'+stock+'.csv')
