# -*- coding: utf-8 -*-
'''
modeling
'''

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, time
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn import ensemble
#from sklearn.linear_model import ElasticNet

param = {'path_model_data': ####### fill your path here ######
                 '/Users/xiyuzhao/Documents/GitHub/Statistical_Machine_Learning_Project/factors/all_factors/',
         'train_ratio': 0.6}

def evaluate_model(y_test_fit, y_test, y_train_fit=None, y_train=None):
    result = {}
    if (y_train_fit is not None) and (y_train is not None):
        train_result = {}
        u = np.sum((y_train-y_train_fit)**2)
        v = np.sum((y_train-np.mean(y_train))**2)
        train_result['score'] = 1-u/v
        train_result['pnl'] = y_train_fit*y_train
        train_result['revenue'] = np.sum(train_result['pnl'])
        train_result['sharpe'] = np.mean(train_result['pnl'])/np.std(train_result['pnl'])
        result['train'] = train_result
    
    test_result = {}
    u = np.sum((y_test-y_test_fit)**2)
    v = np.sum((y_test-np.mean(y_test))**2)
    test_result['score'] = 1-u/v
    test_result['pnl'] = y_test_fit*y_test
    test_result['revenue'] = np.sum(test_result['pnl'])
    test_result['sharpe'] = np.mean(test_result['pnl'])/np.std(test_result['pnl'])
    # outofsample_result['bench_pnl'] = y_test
    # outofsample_result['bench_sharpe'] = np.mean(y_test)/np.std(y_test)
    result['test'] = test_result
    return result
    

def run_model(stocks, factors, y_horizon, model, 
              param, rolling=False, **model_param):
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
        ret_name = '_'.join(['fut',str(int(y_horizon)),'ret'])
        y = data[ret_name]
        
        # run model

        n_obs = len(X)
        n_train = int(n_obs*param['train_ratio'])
        if rolling:
            train_index = [range(i,i+n_train) for i in range(n_obs-n_train)]
            test_index = range(n_train, n_obs)
            y_test_fit = np.zeros(len(test_index))
            for i in range(len(train_index)):
                print(i)
                train_ind, test_ind = train_index[i], test_index[i]
                X_train, y_train = X.iloc[train_ind].copy(), y.iloc[train_ind].copy()
                X_test, y_test = X.iloc[[test_ind]].copy(), y.iloc[test_ind].copy()
                
                reg = model(**model_param)
                reg.fit(X_train, y_train)
                
                y_train_fit = reg.predict(X_train)
                y_test_fit[i] = reg.predict(X_test)[0]
                
            y_test = np.array(y.iloc[test_index])
            result = evaluate_model(y_test_fit, y_test)
            result['stock'] = stock
           
        else:
            n_obs = len(X)
            X_train, y_train = X.iloc[:n_train].copy(), y[:n_train].copy()
            X_test, y_test = X.iloc[n_train:].copy(), y[n_train:].copy()
            
            reg = model(**model_param)
            reg.fit(X_train, y_train)
            y_train_fit = reg.predict(X_train)
            y_test_fit = reg.predict(X_test)
            result = evaluate_model(y_test_fit, y_test, y_train_fit, y_train)
            result['stock'] = stock
            
        all_result.append(result)
    return all_result

stocks = ['ABX', 'ACB', 'AEM', 'BAM.A', 'BNS', 'CNQ',
          'CNR', 'CRON', 'CVE', 'ECA', 'ENB', 'FOOD']
stocks = ['ABX', 'ACB', 'AEM']

all_factors = [
       'mid_momentum_10ord', 'mid_momentum_10s',
       'mid_momentum_1s', 'mid_momentum_20ord', 'mid_momentum_2s',
       'mid_momentum_30s', 'mid_momentum_50ord', 'mid_momentum_5ord',
       'mid_momentum_5s', 'mid_snapshot', 'smart_price_momentum_10ord',
       'smart_price_momentum_10s', 'smart_price_momentum_1s',
       'smart_price_momentum_20ord', 'smart_price_momentum_2s',
       'smart_price_momentum_30s', 'smart_price_momentum_50ord',
       'smart_price_momentum_5ord', 'smart_price_momentum_5s',
       'smart_price_snapshot', 'spread_10ord', 'spread_10s', 'spread_1s',
       'spread_20ord', 'spread_2s', 'spread_30s', 'spread_50ord',
       'spread_5ord', 'spread_5s', 'spread_diff_10ord', 'spread_diff_10s',
       'spread_diff_1s', 'spread_diff_20ord', 'spread_diff_2s',
       'spread_diff_30s', 'spread_diff_50ord', 'spread_diff_5ord',
       'spread_diff_5s', 'spread_snapshot', 'trade_sign_10ord',
       'trade_sign_10s', 'trade_sign_1ord', 'trade_sign_1s', 'trade_sign_2ord',
       'trade_sign_2s', 'trade_sign_5ord', 'trade_sign_5s',
       'transaction_spread_10ord', 'transaction_spread_10s',
       'transaction_spread_1ord', 'transaction_spread_1s',
       'transaction_spread_2ord', 'transaction_spread_2s',
       'transaction_spread_5ord', 'transaction_spread_5s', 'volum_imbalance']

y_horizon = 10

'''
model = RandomForestRegressor
model_name = 'random_forest'
model_param = {'max_depth': 2, 'random_state': 0, 'n_estimators': 100}
results = run_model(stocks, all_factors, y_horizon, model, param, **model_param) 
'''
model = ensemble.GradientBoostingRegressor
model_name = 'elastic_net'
model_param = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 3,
          'learning_rate': 0.01, 'loss': 'huber'}
results = run_model(stocks, all_factors, y_horizon, model, param, False, **model_param)

key_results = [[res['stock'],res['test']['score'],res['test']['sharpe']] for res in results]
key_results = pd.DataFrame(columns=['Stock','Score','Sharpe'],data=key_results)
## give the result a name
key_results.name = model_name+'; '+str(model_param)+'; '+str(y_horizon)+" days"
print(key_results)

for res in results:
    res['test']['pnl'].plot()

model_param = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}
results = run_model(stocks, all_factors, y_horizon, model, param, False, **model_param)

key_results = [[res['stock'],res['test']['score'],res['test']['sharpe']] for res in results]
key_results = pd.DataFrame(columns=['Stock','Score','Sharpe'],data=key_results)
## give the result a name
key_results.name = model_name+'; '+str(model_param)+'; '+str(y_horizon)+" days"
print(key_results)

for res in results:
    res['test']['pnl'].plot()
