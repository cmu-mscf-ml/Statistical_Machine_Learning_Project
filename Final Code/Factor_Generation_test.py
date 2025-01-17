# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 20:37:20 2019

This file accesses the clean data (combined infomation, selected time and 
stocks), uses it to calculate factors and stores them into local filefolder.

Each result file is a csv file with values of a stock's factor value. 

@author: THINKPAD
"""

# import packages
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, time, date
import os

# relevant params
param = {# speicify path, should modify to the local path
         'read_path': '/Users/xiyuzhao/Documents/GitHub/Statistical_Machine_Learning_Project',
         #'read_path': 'D:\\Project\\data\\',
         'write_path': './factors',
         #'write_path': 'D:\\Project\\data\\'+'factors\\',
         # specify file names and current time, should modify based on the data's date
         'filename': 'L1.cleaned.20190404.csv',
         'date': date(2019,4,4),
         # continuous trading time
         'cts_trade': [time(9,30,0), time(16,0,0)],
         # time horizons to construct factors
         's_list': [1,2,5,10,30],
         'order_list': [5,10,20,50], 
         'trade_list': [1,2,5,10]
         }
         

### Data preprocess ###
def preprocess(param, write = True):
    '''
    Get, preprocess, and store preprocessed data to the read path (same as the
    data read in)
    '''
    # construct continuous trading time at this date
    param['cts_trade_time'] = datetime.combine(param['date'], param['cts_trade'][0])
    
    # read and preprocess data
    dataset = pd.read_csv(param['read_path']+param['filename'], header=0, index_col=0)
    
    # change the form of time
    dataset['time'] = dataset['time'].apply(lambda t: datetime.strptime(t,
                                                       '%Y-%m-%d %H:%M:%S.%f'))
    
    # discretize based on seconds
    dataset['h_m_s'] = [datetime(t.year, t.month, t.day, t.hour, 
           t.minute, t.second) for t in dataset['time']]
    
    # get symbol list
    param['stocks'] = np.array(dataset['symbol'].unique())
    
    # constrcut seconds index of a trading day
    delta_t = timedelta(0,1,0) # one second
    param['ticks'] = param['cts_trade_time'] + np.arange(int(6.5*3600))*delta_t
    
    ### Construct basic features
    # construct 'spread' information
    dataset['spread'] = dataset['ask']-dataset['bid']
    # construct 'mid'
    dataset['mid'] = (dataset['ask']+dataset['bid'])/2
    # construct 'smart_price'
    dataset['smart_price'] = (dataset['ask']*dataset['bidSize']+
           dataset['bid']*dataset['askSize'])/(dataset['bidSize']+dataset['askSize'])

    dataset['size_spread'] = (dataset[dataset['type'] == 'bookChange']['bidSize'] -
                              dataset[dataset['type'] == 'bookChange']['askSize']) / (
                             dataset[dataset['type'] == 'bookChange']['bidSize'] +
                             dataset[dataset['type'] == 'bookChange']['askSize'])

    dataset['trade_sign'] = dataset[dataset['type'] == 'trade']['tradeSide'].map(
        lambda x: -1 if ' SELL' in x else 1 if ' BUY' in x else 0)

    dataset['transaction_spread'] = (dataset[dataset['type'] == 'trade']['tradeSize']) * \
                                    dataset[dataset['type'] == 'trade']['tradeSide'].map(
                        lambda x: -1 if ' SELL' in x else 1 if ' BUY' in x else 0)


    if write:
        dataset.to_csv(param['read_path']+'Processed.'+param['filename'])
    
    return dataset


###############################################################################
### Functions to generate factors based on the preprocessed data and store them
### into the target path
    

# factor：smart price and it's momentum 
def fac_smartPrice(dataset, param):
    '''
    Calculate and store smart price related factor;
    (1) Snapshot smart price
    (2) Momentum of smart price (order based)
    '''
    factor_name1 = 'smart_price_snapshot'
    factor_name2 = 'smart_price_momentum_'

    ### (1) snaptshot data
    # file
    target_path1 = param['write_path']+factor_name1
    if not os.path.exists(target_path1): 
        os.mkdir(target_path1)
    # calculate
    smartPrice = dataset.groupby(by=['symbol','h_m_s']).last()['smart_price']
    smartPrice = smartPrice.reset_index()
    smartPrice = smartPrice.pivot(index='h_m_s',columns='symbol',values='smart_price')
    smartPrice = smartPrice.reindex(param['ticks'])
    smartPrice = smartPrice.fillna(method='ffill')
    # generate and store
    for stock in smartPrice.columns:
        stock_factor = smartPrice[[stock]]
        stock_factor.to_csv(target_path1+'\\'+stock+'.csv')
    
    
    ### (2) momentum data
    ## (2.1) time-based    
    for s in param['s_list']:
        # file
        target_path21 = param['write_path']+factor_name2+str(s)+'s'
        if not os.path.exists(target_path21):
            os.mkdir(target_path21)
        # calculate
        momentum = (smartPrice-smartPrice.shift(s))/smartPrice.shift(s)
        for stock in smartPrice.columns:
            stock_factor = momentum[[stock]]
            stock_factor.to_csv(target_path21+'\\'+stock+'.csv')

    ## (2.2) order-based
    for s in param['order_list']:
        # file
        target_path22 = param['write_path']+factor_name2+str(s)+'ord'
        if not os.path.exists(target_path22):
            os.mkdir(target_path22)
        # caculate 
        for stock, group in dataset.groupby('symbol'):
            # first get pre-value
            pre_value  = group[['smart_price']].shift(s)
            # get h_m_s information
            pre_value ['h_m_s'] = group['h_m_s']
            # group by data in the same second, get values
            pre_value  = pre_value.groupby('h_m_s').last()[['smart_price']]
            # use full ticks to reindex so that we have all the records
            pre_value  = pre_value .reindex(param['ticks'])
            # fill missing values
            pre_value = pre_value .fillna(method='ffill')
            
            # use this prevalue and current value to create momentum factor 
            stock_factor = (smartPrice[stock]-
                            pre_value['smart_price'])/pre_value['smart_price'] 
            # store
            stock_factor.to_csv(target_path1+'\\'+stock+'.csv')
    
    return

# factor：mid-price and it's momentum 
def fac_midPrice(dataset, param):
    '''
    Calculate and store mid price related factor;
    (1) Snapshot mid price
    (2) Momentum of mid price (order based)
    '''
    factor_name1 = 'mid_snapshot'
    factor_name2 = 'mid_momentum_'

    ### (1) snaptshot data
    # file
    target_path1 = param['write_path']+factor_name1
    if not os.path.exists(target_path1): 
        os.mkdir(target_path1)
    # calculate
    smartPrice = dataset.groupby(by=['symbol','h_m_s']).last()['mid']
    smartPrice = smartPrice.reset_index()
    smartPrice = smartPrice.pivot(index='h_m_s',columns='symbol',values='mid')
    smartPrice = smartPrice.reindex(param['ticks'])
    smartPrice = smartPrice.fillna(method='ffill')
    # generate and store
    for stock in smartPrice.columns:
        stock_factor = smartPrice[[stock]]
        stock_factor.to_csv(target_path1+'\\'+stock+'.csv')
    
    
    ### (2) momentum data
    ## (2.1) time-based    
    for s in param['s_list']:
        # file
        target_path21 = param['write_path']+factor_name2+str(s)+'s'
        if not os.path.exists(target_path21):
            os.mkdir(target_path21)
        # calculate
        momentum = (smartPrice-smartPrice.shift(s))/smartPrice.shift(s)
        for stock in smartPrice.columns:
            stock_factor = momentum[[stock]]
            stock_factor.to_csv(target_path21+'\\'+stock+'.csv')

    ## (2.2) order-based
    for s in param['order_list']:
        # file
        target_path22 = param['write_path']+factor_name2+str(s)+'ord'
        if not os.path.exists(target_path22):
            os.mkdir(target_path22)
        # caculate 
        for stock, group in dataset.groupby('symbol'):
            # first get pre-value
            pre_value  = group[['mid']].shift(s)
            # get h_m_s information
            pre_value ['h_m_s'] = group['h_m_s']
            # group by data in the same second, get values
            pre_value  = pre_value.groupby('h_m_s').last()[['mid']]
            # use full ticks to reindex so that we have all the records
            pre_value  = pre_value .reindex(param['ticks'])
            # fill missing values
            pre_value = pre_value .fillna(method='ffill')
            
            # use this prevalue and current value to create momentum factor 
            stock_factor = (smartPrice[stock]-
                            pre_value['mid'])/pre_value['mid'] 
            # store
            stock_factor.to_csv(target_path1+'\\'+stock+'.csv')
    
    return


# spread factor
def fac_spread(dataset, param, write = True):
    '''
    Calculate each stock's closing bid ask spread of each tick;
    This is a snapshot data
    '''
    factor_name = 'spread_snapshot'
    
    # create filefolder
    target_path = param['write_path']+factor_name
    if not os.path.exists(target_path): 
        os.mkdir(target_path)
    
    # calculate
    # construct a "group by" object based on stock symbol and the second it is in
    data = dataset.groupby(by=['symbol','h_m_s'])
    spread_snapshot = data.last()['spread']
    spread_snapshot = spread_snapshot.reset_index()
    spread_snapshot = spread_snapshot.pivot(index='h_m_s',
                                            columns='symbol',values='spread')
    # the index generated above may have missing records, so reindex it
    spread_snapshot = spread_snapshot.reindex(param['ticks'])
    spread_snapshot = spread_snapshot.fillna(method='ffill')
    
    if write:
        for stock in spread_snapshot.columns:
            stock_factor = spread_snapshot[[stock]]
            stock_factor.to_csv(target_path+'\\'+stock+'.csv')

    return spread_snapshot


# spred difference and average spread factor
def fac_spread_diff(dataset, param):
    '''
    Calculate spread difference factor;
    Current value - recent mean value;
    This is order-based data;
    Meanwhile, store recent spread mean values as another factor
    '''
    factor_name1 = 'spread_'
    factor_name2 = 'spread_diff_'
    spread_snapshot = fac_spread(dataset, param, False)
    
    ## (1) time-based
    # construct a "group by" object based on stock symbol and the second it is in
    data = dataset.groupby(by=['symbol','h_m_s'])
    
    # pre-calculate
    factor = data.mean()['spread']
    factor = factor.reset_index()
    factor = factor.pivot(index='h_m_s',columns='symbol',values='spread')
    factor = factor.reindex(param['ticks'])
    factor = factor.fillna(method='ffill')
    
    for s in param['s_list']:
        # create filefolder
        target_path1 = param['write_path']+factor_name1+str(s)+'s'
        target_path2 = param['write_path']+factor_name2+str(s)+'s'
        if not os.path.exists(target_path1): 
            os.mkdir(target_path1)
        if not os.path.exists(target_path2): 
            os.mkdir(target_path2)
            
        # calculate
        factor_s = factor.rolling(s).mean()
        factor_diff = spread_snapshot-factor_s
        for stock in factor_s.columns:
            stock_factor = factor_s[[stock]]
            stock_factor.to_csv(target_path1+'\\'+stock+'.csv')
            stock_factor_diff = factor_diff[[stock]]
            stock_factor_diff.to_csv(target_path2+'\\'+stock+'.csv')

    ## (2) order_based
    for s in param['order_list']:
        # create filefolder
        target_path1 = param['write_path']+factor_name1+str(s)+'ord'
        target_path2 = param['write_path']+factor_name2+str(s)+'ord'
        if not os.path.exists(target_path1): 
            os.mkdir(target_path1)
        if not os.path.exists(target_path2): 
            os.mkdir(target_path2)
            
        # calculate
        for stock, group in dataset.groupby('symbol'):
            stock_factor = group[['spread']].rolling(s).mean()
            stock_factor['h_m_s'] = group['h_m_s']
            stock_factor = stock_factor.groupby('h_m_s').last()[['spread']]
            stock_factor = stock_factor.reindex(param['ticks'])
            stock_factor = stock_factor.fillna(method='ffill')
            stock_factor_diff = spread_snapshot[stock]-stock_factor['spread']
            stock_factor_diff = pd.DataFrame(stock_factor_diff, columns=[stock])
            # save
            stock_factor.to_csv(target_path1+'\\'+stock+'.csv')
            stock_factor_diff.to_csv(target_path2+'\\'+stock+'.csv')
    return

def fac_volimbalance(dataset, param):

    factor_name1 = 'volum_imbalance'

    ### (1) snaptshot data
    # file
    target_path1 = param['write_path']+factor_name1
    if not os.path.exists(target_path1):
        os.mkdir(target_path1)

    spread = dataset[dataset['type'] == 'bookChange'].groupby(by=['symbol', 'h_m_s'])[
        'size_spread'].last().reset_index()
    spread = spread.pivot(index='h_m_s', columns='symbol',
                          values='size_spread').reindex(ticks)
    spread.fillna(method='ffill', inplace=True)

    for stock in spread.columns.tolist():
        spread[[column]].to_csv(target_path1+'\\'+stock+'.csv')

    return


def fac_trade_sign(dataset, param):
    '''
    Calculate spread difference factor;
    Current value - recent mean value;
    This is order-based data;
    Meanwhile, store recent spread mean values as another factor
    '''
    factor_name1 = 'trade_sign_'

    ## (1) time-based
    # construct a "group by" object based on stock symbol and the second it is in
    data = trade.groupby(by=['symbol','h_m_s'])

    # pre-calculate
    factor = data.sum()['trade_sign']
    factor = factor.reset_index()
    factor = factor.pivot(index='h_m_s',columns='symbol',values='trade_sign')
    factor = factor.reindex(param['ticks'])
    factor = factor.fillna(0, inplace = True)

    for s in param['trade_list']:
        # create filefolder
        target_path1 = param['write_path'] + factor_name1 + str(s) + 's'
        if not os.path.exists(target_path1):
            os.mkdir(target_path1)


        # calculate
        factor_s = factor.rolling(s).mean()
        for stock in factor_s.columns:
            stock_factor = factor_s[[stock]]
            stock_factor.to_csv(target_path1 + '\\' + stock + '.csv')

    ## (2) order_based
    for s in param['trade_list']:
        # create filefolder
        target_path1 = param['write_path'] + factor_name1 + str(s) + 'ord'

        if not os.path.exists(target_path1):
            os.mkdir(target_path1)


        # calculate
        for stock, group in dataset.groupby('symbol'):
            stock_factor = group[['trade_sign']].rolling(s).mean()
            stock_factor['h_m_s'] = group['h_m_s']
            stock_factor = stock_factor.groupby('h_m_s').last()[['spread']]
            stock_factor = stock_factor.reindex(param['ticks'])
            stock_factor = stock_factor.fillna(0, inplace = True)
            # save
            stock_factor.to_csv(target_path1 + '\\' + stock + '.csv')
    return

def fac_trans_spread(dataset, param):
    '''
    Calculate spread difference factor;
    Current value - recent mean value;
    This is order-based data;
    Meanwhile, store recent spread mean values as another factor
    '''
    factor_name1 = 'ttransaction_spread_'

    ## (1) time-based
    # construct a "group by" object based on stock symbol and the second it is in
    data = trade.groupby(by=['symbol','h_m_s'])

    # pre-calculate
    factor = data.sum()['transaction_spread']
    factor = factor.reset_index()
    factor = factor.pivot(index='h_m_s',columns='symbol',values='transaction_spread')
    factor = factor.reindex(param['ticks'])
    factor = factor.fillna(0, inplace = True)

    for s in param['trade_list']:
        # create filefolder
        target_path1 = param['write_path'] + factor_name1 + str(s) + 's'
        if not os.path.exists(target_path1):
            os.mkdir(target_path1)


        # calculate
        factor_s = factor.rolling(s).mean()
        for stock in factor_s.columns:
            stock_factor = factor_s[[stock]]
            stock_factor.to_csv(target_path1 + '\\' + stock + '.csv')

    ## (2) order_based
    for s in param['trade_list']:
        # create filefolder
        target_path1 = param['write_path'] + factor_name1 + str(s) + 'ord'

        if not os.path.exists(target_path1):
            os.mkdir(target_path1)


        # calculate
        for stock, group in dataset.groupby('symbol'):
            stock_factor = group[['transaction_spread']].rolling(s).mean()
            stock_factor['h_m_s'] = group['h_m_s']
            stock_factor = stock_factor.groupby('h_m_s').last()[['spread']]
            stock_factor = stock_factor.reindex(param['ticks'])
            stock_factor = stock_factor.fillna(0, inplace = True)
            # save
            stock_factor.to_csv(target_path1 + '\\' + stock + '.csv')
    return



################### main ##########################
dataset = preprocess(param, write = False)

# smart price
#fac_smartPrice(dataset, param)
# mid price
#fac_midPrice(dataset, param)

fac_volimbalance(dataset, param)

fac_trade_sign(dataset, param)

ac_trans_spread(dataset, param)