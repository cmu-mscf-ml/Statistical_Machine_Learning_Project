# -*- coding: utf-8 -*-
'''
modeling
'''

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, time
import re
import matplotlib.pyplot as plt 
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet

param = {'path_model_data': ####### fill your path here ######
                 'D:\\'
                 + 'Statistical_Machine_Learning_Project\\'
                 + 'factors\\all_factors\\',
         'train_ratio': 0.6}

    
def evaluate_model(reg, X_train, y_train, X_test, y_test): 
    reg.fit(X_train, y_train)
    
    y_train_fit = reg.predict(X_train)
    y_test_fit = reg.predict(X_test)
    
    insample_result = {}
    insample_result['score'] = reg.score(X_train,y_train)
    insample_result['pnl'] = y_train_fit*y_train
    insample_result['revenue'] = np.sum(insample_result['pnl'])
    insample_result['sharpe'] = np.mean(insample_result['pnl'])/np.std(insample_result['pnl'])
        
    outofsample_result = {}
    outofsample_result['score'] = reg.score(X_test,y_test)
    outofsample_result['pnl'] = y_test_fit*y_test
    outofsample_result['revenue'] = np.sum(outofsample_result['pnl'])
    outofsample_result['sharpe'] = np.mean(outofsample_result['pnl'])/np.std(outofsample_result['pnl'])
    # outofsample_result['bench_pnl'] = y_test
    # outofsample_result['bench_sharpe'] = np.mean(y_test)/np.std(y_test)
    
    result = {'insample': insample_result, 'outofsample': outofsample_result}
    return result
    
    
def run_model(stocks, factors, y_horizon, model, 
              param, rolling=False, **model_param):
    model_dict = {'random_forest': RandomForestRegressor,
                  'elastic_net': ElasticNet}
    all_result = []
    for stock in stocks:
        # fetch stock's modeling data
        data = pd.read_csv(param['path_model_data']+stock+'.csv', index_col=0)
        fmt = '%Y-%m-%d %H:%M:%S'
        data['h_m_s'] = data['h_m_s'].apply(lambda x: datetime.strptime(x,fmt))
        data.index = data['h_m_s']
        del data['h_m_s']
        data = data.dropna()
        X = data.loc[:,factors]
        # X = data.loc[:,[not c.endswith('_ret') for c in data.columns]]
        ret_name = '_'.join(['fut',str(int(y_horizon)),'ret'])
        y = data[ret_name]
        
        # run model

        n_obs = len(X)
        n_train = int(n_obs*param['train_ratio'])
        if rolling:
            train_index = [range(i,i+n_train) for i in range(n_obs-n_train)]
            test_index = range(n_train, n_obs)
            for train_ind, test_ind in zip(*[train_index, test_index]):
                X_train, y_train = X.iloc[train_ind], y[train_ind]
                X_test, y_test = X.iloc[test_ind], y[test_ind]
                
                func = model_dict.get(model, lambda: 'nothing')
                reg = func(**model_param)
                result = evaluate_model(reg, X_train, y_train, X_test, y_test)
                result['stock'] = stock
           
        else:
            n_obs = len(X)
            X_train, y_train = X.iloc[:n_train], y[:n_train]
            X_test, y_test = X.iloc[n_train:], y[n_train:]
            
            func = model_dict.get(model, lambda: 'nothing')
            reg = func(**model_param)
            result = evaluate_model(reg, X_train, y_train, X_test, y_test)
            result['stock'] = stock
            
        all_result.append(result)
    return all_result

stocks = ['ABX', 'ACB', 'AEM', 'BAM.A', 'BNS', 'CNQ',
          'CNR', 'CRON', 'CVE', 'ECA', 'ENB', 'FOOD']

all_factors = [
 'mid_momentum_10ord',
 'mid_momentum_10s',
 'mid_momentum_1s',
 'mid_momentum_20ord',
 'mid_momentum_2s',
 'mid_momentum_30s',
 'mid_momentum_50ord',
 'mid_momentum_5ord',
 'mid_momentum_5s',
 'smartPrice_insensitive',
 'smartPrice_snapshot',
 'smart_price_momentum_10ord',
 'smart_price_momentum_10s',
 'smart_price_momentum_1s',
 'smart_price_momentum_20ord',
 'smart_price_momentum_2s',
 'smart_price_momentum_30s',
 'smart_price_momentum_50ord',
 'smart_price_momentum_5ord',
 'smart_price_momentum_5s',
 'spread_10ord',
 'spread_10s',
 'spread_1s',
 'spread_20ord',
 'spread_2s',
 'spread_30s',
 'spread_50ord',
 'spread_5ord',
 'spread_5s',
 'spread_diff_10ord',
 'spread_diff_10s',
 'spread_diff_1s',
 'spread_diff_20ord',
 'spread_diff_2s',
 'spread_diff_30s',
 'spread_diff_50ord',
 'spread_diff_5ord',
 'spread_diff_5s',
 'spread_snapshot',
 'trade_sign_10ord',
 'trade_sign_10s',
 'trade_sign_1ord',
 'trade_sign_1s',
 'trade_sign_2ord',
 'trade_sign_2s',
 'trade_sign_30s',
 'trade_sign_5ord',
 'trade_sign_5s',
 'transaction_spread_10ord',
 'transaction_spread_10s',
 'transaction_spread_1ord',
 'transaction_spread_1s',
 'transaction_spread_2ord',
 'transaction_spread_2s',
 'transaction_spread_30s',
 'transaction_spread_5ord',
 'transaction_spread_5s',
 'volume_imbalance']

y_horizon = 5


model = 'random_forest'
model_param = {'max_depth': 2, 'random_state': 0, 'n_estimators': 100}
results = run_model(stocks, all_factors, y_horizon, model, param, **model_param) 
                
'''

model = 'elastic_net'
model_param = {'random_state':0,'alpha':1e-2}
results = run_model(stocks, all_factors, y_horizon, model, param, **model_param)

'''
key_results = [[res['stock'],res['outofsample']['score'],res['outofsample']['sharpe']] for 
              res in results]

key_results = pd.DataFrame(columns=['Stock','Score','Sharpe'],data=key_results)
## give the result a name
key_results.name = model+'; '+str(model_param)+'; '+str(y_horizon)+" days"
