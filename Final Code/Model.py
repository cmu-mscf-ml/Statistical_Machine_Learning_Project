# -*- coding: utf-8 -*-
'''
modeling
'''

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, time
import re
import matplotlib.pyplot as plt 

param = {'path_model_data': ####### fill your path here ######
                 + 'project\\Statistical_Machine_Learning_Project\\'
                 + 'factors\\all_factors\\',
         'train_ratio': 0.6}

stock = 'ACB'
data = pd.read_csv(param['path_model_data']+stock+'.csv', index_col=0)
fmt = '%Y-%m-%d %H:%M:%S'
data['h_m_s'] = data['h_m_s'].apply(lambda x: datetime.strptime(x, fmt))
data.index = data['h_m_s']
del data['h_m_s']
data = data.dropna()


X = data.loc[:,[not c.endswith('_ret') for c in data.columns]]
y = data['fut_5_ret']



n_obs = len(X)

print(float(np.sum(y==0))/len(y))


X_train = X.iloc[:int(n_obs*param['train_ratio'])]
y_train = y[:int(n_obs*param['train_ratio'])]
X_test = X.iloc[int(n_obs*param['train_ratio']):]
y_test = y[int(n_obs*param['train_ratio']):]

model = 'random_forest'
from sklearn.ensemble import RandomForestRegressor

reg = RandomForestRegressor(max_depth=2, random_state=0, n_estimators=100)
reg.fit(X_train, y_train)

print("In sample:")
print("Accuracy score: ", reg.score(X_train,y_train))
y_train_fit = reg.predict(X_train)
pnl = 1000*y_train_fit*y_train
bench = y_train
print("Actual revenue: ", np.sum(pnl))
print("Actual sharpe: ", np.mean(pnl)/np.std(pnl))
plt.plot(np.cumsum(pnl), label='Train PnL')
plt.plot(np.cumsum(bench), label='Train benchmark')
plt.legend()
plt.show()



print("Out sample:")
print("Accuracy score: ", reg.score(X_test,y_test))
y_test_fit = reg.predict(X_test)
pnl = 1000*y_test_fit*y_test
bench = y_test
print("Actual revenue: ", np.sum(pnl))
print("Actual sharpe: ", np.mean(pnl)/np.std(pnl))
plt.plot(np.cumsum(pnl), label='Test PnL')
plt.plot(np.cumsum(bench), label='Test benchmark')
plt.legend()
plt.show()