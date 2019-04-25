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
from sklearn.model_selection import TimeSeriesSplit

param = {'path_model_data': ####### fill your path here ######
                 'F:\\Class - Statistical Machine Learning II\\project\\HFT\\'
                 + 'Statistical_Machine_Learning_Project\\'
                 + 'factors\\all_factors\\',
         'train_ratio': 0.6}


for train_index, test_index in tscv.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

def random_forest(X_train, y_train, X_test, y_test):
    reg = RandomForestRegressor(max_depth=2, random_state=0, n_estimators=100)
    reg.fit(X_train, y_train)
    y_train_fit = reg.predict(X_train)
    y_test_fit = reg.predict(X_test)
    
    insample_result = {}
    insample_result['accuracy_score'] = reg.score(X_train,y_train)
    insample_result['pnl'] = 1000*y_train_fit*y_train
    insample_result['benchmark'] = y_train
    insample_result['actual_revenue'] = np.sum(pnl)
    
    outofsample_result = {}
    outofsample_result['accuracy_score'] = reg.score(X_test,y_test)
    outofsample_result['pnl'] = 1000*y_test_fit*y_test
    outofsample_result['cum_pnl'] = np.cumsum(outofsample_result['pnl'])
    outofsample_result['revenue'] = np.sum(outofsample_result['pnl'])
    outofsample_result['sharpe'] = np.mean(outofsample_result['pnl'])/np.std(outofsample_result['pnl'])
    outofsample_result['bench_pnl'] = y_test
    outofsample_result['bench_cum_pnl'] = np.cumsum(outofsample_result['benchmark'])
    outofsample_result['bench_revenue'] = np.sum(outofsample_result['bench_pnl'])
    outofsample_result['bench_sharpe'] = np.mean(y_test)/np.std(y_test)
    
    result = {'insample': insample_result, 'outofsample': outofsample_result}
    return result
    
    
def run_model(stocks, factors, y_horizon, model, rolling=True, evaluator, param, **kwargs):
    model_dict = {'random_forest': random_forest}
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
        stock_results = []
        n_obs = len(X)
        n_train = int(n_obs*param['train_ratio'])
        if rolling:
            train_index = [range(i,i+n_train) for i in range(n-n_train)]
            test_index = range(n_train, n)
            for train_ind, test_ind in zip(*[train_index, test_index]):
                X_train = X.iloc[train_ind]
                y_train = y[train_ind]
                X_test = X.iloc[test_ind]
                y_test = y[test_ind]
                func = model_dict.get(model, lambda: 'nothing')
                result = func(X_train, y_train, X_test, y_test)
                stock_results.append(result)            
        else:
            n_obs = len(X)
            X_train = X.iloc[:n_train]
            y_train = y[:n_train]
            X_test = X.iloc[n_train:]
            y_test = y[n_train:]
            func = model_dict.get(model, lambda: 'nothing')
            result = func(X_train, y_train, X_test, y_test)
            stock_results.append(result)
        all_result.append(stock_results)
    return all_results
    
    
    
    
    
    
    