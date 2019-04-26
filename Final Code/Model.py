# -*- coding: utf-8 -*-
'''
modeling
'''

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, time
import matplotlib.pyplot as plt 
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet

param = {'path_model_data': ####### fill your path here ######
                 'D:\\'
                 + 'Statistical_Machine_Learning_Project\\'
                 + 'factors\\all_factors\\',
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
                # print(i)
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

stocks = ['ABX', 'ACB', 'AEM', 'BAM.A', 'BNS', 'CNQ', 'CNR', 'CRON', 'CVE',
       'ECA', 'ENB', 'FOOD', 'G', 'HQU', 'HSD', 'LUG', 'MFC', 'PTG',
       'PVG', 'RY', 'SHOP', 'SMU.UN', 'SU', 'TD', 'TECK.B', 'WPM', 'XIU',
       'XSP']


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

y_horizon = 5

#############  Research ####################
'''
The code below is process-oriented and can be a litte messy
There are some code re-using (copy and paste)


First try random forest.
It's complex and have selection power.
'''
model = RandomForestRegressor
model_name = 'random_forest'
model_param = {'max_depth': 2, 'random_state': 0, 'n_estimators': 100}
results = run_model(stocks, all_factors, y_horizon, model, param, **model_param) 

key_results = [[res['stock'],res['train']['score'],res['train']['sharpe'],
                res['test']['score'],res['test']['sharpe']] for res in results]
key_results = pd.DataFrame(columns=['Stock','Score_train','Sharpe_train',
                                    'Score_test','Sharpe_test'],data=key_results)
## give the result a name
key_results.name = model_name+'_'+str(model_param)+'_'+str(y_horizon)+" days"

# Since the model is complex, we first eliminate stocks whose train set score 
# is even lower than 0.01
stock_subset = key_results.loc[key_results['Score_train']>0.01]['Stock']

print(stock_subset)
# ACB, CRON, ECA, FOOD, PVG, SHOP, TECK.B are good

# Look at each’s feature importance
factors = all_factors
subset_importance = pd.DataFrame(columns=factors,
                                 data=np.zeros((len(stock_subset),len(factors))))
subset_importance.index = stock_subset.values
for stock in stock_subset.values:
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

    X_train, y_train = X.iloc[:n_train].copy(), y[:n_train].copy()
    X_test, y_test = X.iloc[n_train:].copy(), y[n_train:].copy()
            
    reg = model(**model_param)
    reg.fit(X_train, y_train)
    subset_importance.loc[stock] = reg.feature_importances_

# select the ones with feature importances higher than 0.01
# for the chosen model, we use random forest to refit it
sub_stocks = stock_subset.values
sub_results = []

for stock in sub_stocks:
    # print(stock)
    subset_features = subset_importance.columns[subset_importance.loc[stock]>0.01].values
    sub_results.append(run_model([stock], subset_features, y_horizon, model, param, **model_param)[0]) 

sub_key_results = [[res['stock'],res['train']['score'],res['train']['sharpe'],
                res['test']['score'],res['test']['sharpe']] for res in sub_results]

sub_key_results = pd.DataFrame(columns=['Stock','Score_train','Sharpe_train',
                                    'Score_test','Sharpe_test'],data=sub_key_results)

################## single research ##########
######### ACB
stock = 'ACB'
factors = ['volum_imbalance']
model = RandomForestRegressor
model_name = 'random_forest'
model_param = {'max_depth': 2, 'random_state': 0, 'n_estimators': 100}
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

X_train, y_train = X.iloc[:n_train].copy(), y[:n_train].copy()
X_test, y_test = X.iloc[n_train:].copy(), y[n_train:].copy()
            
reg = model(**model_param)
reg.fit(X_train, y_train)
y_train_fit = reg.predict(X_train)
y_test_fit = reg.predict(X_test)
result = evaluate_model(y_test_fit, y_test, y_train_fit, y_train)


## result demo ##

plt.plot(np.cumsum(result['train']['pnl']),label='train')
train_revenue = np.cumsum(result['train']['pnl'])[-1]

plt.plot(train_revenue+np.cumsum(result['test']['pnl']),label='test')
plt.title("ACB, PnL")
plt.legend()
plt.show()

print("ACB:")
print("Train Score: ", round(result['train']['score'],5))
print("Train Sharpe: ", round(result['train']['sharpe'],5))
print("Test Score: ", round(result['test']['score'],5))
print("Test Sharpe: ", round(result['test']['sharpe'],5))

########## CRON
stock = 'CRON'
factors = ['mid_momentum_50ord', 'spread_30s', 'volum_imbalance']
model = RandomForestRegressor
model_name = 'random_forest'
model_param = {'max_depth': 2, 'random_state': 0, 'n_estimators': 100}
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

X_train, y_train = X.iloc[:n_train].copy(), y[:n_train].copy()
X_test, y_test = X.iloc[n_train:].copy(), y[n_train:].copy()
            
reg = model(**model_param)
reg.fit(X_train, y_train)
y_train_fit = reg.predict(X_train)
y_test_fit = reg.predict(X_test)
result = evaluate_model(y_test_fit, y_test, y_train_fit, y_train)


## result demo ##

plt.plot(np.cumsum(result['train']['pnl']),label='train')
train_revenue = np.cumsum(result['train']['pnl'])[-1]

plt.plot(train_revenue+np.cumsum(result['test']['pnl']),label='test')
plt.title("CRON, PnL")
plt.legend()
plt.show()

print("Cron:")
print("Train Score: ", round(result['train']['score'],5))
print("Train Sharpe: ", round(result['train']['sharpe'],5))
print("Test Score: ", round(result['test']['score'],5))
print("Test Sharpe: ", round(result['test']['sharpe'],5))

########## PVG
stock = 'PVG'
factors = ['mid_momentum_50ord', 'mid_snapshot', 'spread_10s']
model = RandomForestRegressor
model_name = 'random_forest'
model_param = {'max_depth': 2, 'random_state': 0, 'n_estimators': 100}
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

X_train, y_train = X.iloc[:n_train].copy(), y[:n_train].copy()
X_test, y_test = X.iloc[n_train:].copy(), y[n_train:].copy()
            
reg = model(**model_param)
reg.fit(X_train, y_train)
y_train_fit = reg.predict(X_train)
y_test_fit = reg.predict(X_test)
result = evaluate_model(y_test_fit, y_test, y_train_fit, y_train)


## result demo ##

plt.plot(np.cumsum(result['train']['pnl']),label='train')
train_revenue = np.cumsum(result['train']['pnl'])[-1]

plt.plot(train_revenue+np.cumsum(result['test']['pnl']),label='test')
plt.title("PVG, PnL")
plt.legend()
plt.show()

print("PVG:")
print("Train Score: ", round(result['train']['score'],5))
print("Train Sharpe: ", round(result['train']['sharpe'],5))
print("Test Score: ", round(result['test']['score'],5))
print("Test Sharpe: ", round(result['test']['sharpe'],5))

##### Test on Apr. 5