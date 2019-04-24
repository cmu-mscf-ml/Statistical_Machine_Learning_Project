# -*- coding: utf-8 -*-
'''
Summaries all factors and calculates returns of each stock
'''

import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta, time

param = {'path_data': 'F:\\Class - Statistical Machine Learning II\\project\\'
                      + 'HFT\\1_data_cleaning\\L1.cleaned.20190404.csv',
         'path_factors': 'F:\\Class - Statistical Machine Learning II\\'
                      + 'project\\HFT\\Statistical_Machine_Learning_Project\\'
                      + 'factors',
         'path_output': 'F:\\Class - Statistical Machine Learning II\\'
                      + 'project\\HFT\\Statistical_Machine_Learning_Project\\'
                      + 'factors\\all_factors'}

def summarize_factors(param):
    ''' summarize all factors in the directory and save '''
    factors = [f for f in os.listdir(param['path_factors']) if '.' not in f]
    factors.remove('all_factors')
    stocks = [s[:-4] for s in os.listdir(param['path_factors']+'\\'+factors[0])]
    for s in stocks:
        print(s)
        stock_factors = None
        for f in factors:
            path = '\\'.join([param['path_factors'],f,s]) + '.csv'
            f_data = pd.read_csv(path, header=0, index_col=0)
            f_data.columns = [f]
            if (len(f_data)!=23400):
                print('error: '+f)
            stock_factors = pd.concat([stock_factors,f_data], axis=1, 
                                      join='outer', sort=True)
        stock_factors.to_csv(param['path_output']+'\\'+s+'.csv')

def calc_return(param):
    ''' calculate all stocks' future return '''
    # load data
    dataset = pd.read_csv(param['path_data'], header=0, index_col=0)
    fmt = '%Y-%m-%d %H:%M:%S.%f'
    dataset['time'] = dataset['time'].apply(lambda t: datetime.strptime(t, fmt))
    dataset['h_m_s'] = [datetime(t.year, t.month, t.day, t.hour, 
                                 t.minute, t.second) for t in dataset['time']]
    dataset['mid_price'] = (dataset['ask']+dataset['bid'])/2
    ticks = datetime(2019,4,4,9,30,0,0) + np.arange(int(6.5*3600))*timedelta(0,1,0)
    stocks = np.array(dataset['symbol'].unique())
    
    # caclulate price of each stock at the end of each second 
    # (equivalent to the price at the beginning of next second)
    end_price = dataset.groupby(by=['symbol','h_m_s']).last()['mid_price']
    end_price = end_price.reset_index()
    end_price = end_price.pivot(index='h_m_s',columns='symbol',values='mid_price')
    end_price = end_price.reindex(ticks)
    end_price = end_price.fillna(method='ffill')
    
    # calculate future 1s, 2s, 5s, 10s log return
    futs = [1,2,5,10]
    fut_ret = pd.concat([np.log(end_price.shift(-fut)/end_price) for fut in futs],
                         axis=1, join='outer')
    columns1 = []
    for fut in futs:
        columns1 = columns1+['_'.join(['fut',str(fut),'ret'])]*len(stocks)
    columns2 = list(fut_ret.columns)
    columns = list(zip(*[columns1,columns2]))
    fut_ret.columns = pd.MultiIndex.from_tuples(columns, names=['fut_ret','symbol'])
    return fut_ret, stocks

    
def generate_model_data(param):
    ''' generate full dataset for modeling, including factors and y value '''
    summarize_factors(param)
    fut_ret, stocks = calc_return(param)
    # combine stock return with each stock's factors dataframe and save
    for s in stocks:
        stock_factors = pd.read_csv(param['path_output']+'\\'+s+'.csv')
        ret = fut_ret.loc[:, (slice(None), s)]
        ret.columns = ret.columns.get_level_values('fut_ret')
        ret.index = range(len(ret))
        stock_factors = pd.concat([stock_factors,ret], axis=1, join='outer')
        stock_factors.to_csv(param['path_output']+'\\'+s+'.csv')
        
if __name__ == '__main__':
    generate_model_data(param)