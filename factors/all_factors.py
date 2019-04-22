# -*- coding: utf-8 -*-
'''
Summaries all factors of each stock
'''

import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta, time

param = {'path_factors': 'F:\\Class - Statistical Machine Learning II\\'
                      + 'project\\HFT\\Statistical_Machine_Learning_Project\\'
                      + 'factors',
         'path_output': 'F:\\Class - Statistical Machine Learning II\\'
                      + 'project\\HFT\\Statistical_Machine_Learning_Project\\'
                      + 'factors\\all_factors'}


factors = [f for f in os.listdir(param['path_factors']) if '.' not in f]
factors.remove('all_factors')
stocks = [s[:-4] for s in os.listdir(param['path_factors']+'\\'+factors[0])]

for s in stocks:
    stock_factors = None
    for f in factors:
        path = '\\'.join([param['path_factors'],f,s]) + '.csv'
        f_data = pd.read_csv(path, header=0, index_col=0)
        f_data.columns = [f]
        if (len(f_data)!=23400):
            print('error: '+f)
        stock_factors = pd.concat([stock_factors,f_data], axis=1, join='outer', sort=True)
    stock_factors.to_csv(param['path_output']+'\\'+s+'.csv')
