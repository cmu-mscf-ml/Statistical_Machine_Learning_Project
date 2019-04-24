# -*- coding: utf-8 -*-
'''
modeling
'''

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, time
import re

param = {'path_model_data': 'F:\\Class - Statistical Machine Learning II\\'
                 + 'project\\HFT\\Statistical_Machine_Learning_Project\\'
                 + 'factors\\all_factors\\'}

stock = 'ABX'
data = pd.read_csv(param['path_model_data']+stock+'.csv', index_col=0)
fmt = '%Y-%m-%d %H:%M:%S'
data['h_m_s'] = data['h_m_s'].apply(lambda x: datetime.strptime(x, fmt))
data.index = data['h_m_s']
del data['h_m_s']

X = data.loc[:,[not c.endswith('_ret') for c in data.columns]]
y = data['fut_2_ret']
n_obs = len(X)
train_ratio = 0.6

X_train = X.iloc[:int(n_obs*train_ratio)]
y_train = y[:int(n_obs*train_ratio)]
X_test = X.iloc[int(n_obs*train_ratio):]
y_test = y[int(n_obs*train_ratio):]


