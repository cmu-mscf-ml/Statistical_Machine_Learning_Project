# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

param = {'path_data': 'F:\\Class - Statistical Machine Learning II\\project\\'
                      + 'HFT\\1_data_cleaning\\L1.cleaned.20190404.csv',
         'path_project': 'F:\\Class - Statistical Machine Learning II\\'
                      + 'project\\HFT\\Statistical_Machine_Learning_Project'}

dataset = pd.read_csv(param['path_data'], header=0, index_col=0)
dataset['time'] = dataset['time'].apply(lambda t: datetime.strptime(t,
                                                   '%Y-%m-%d %H:%M:%S.%f'))
stocks = np.array(dataset['symbol'].unique())
delta_t = 

datetime(2019,4,4,9,30,0,0) + np.arange(23400)*delta_t
ticks = np.arange(datetime(2019,4,4,9,30,0,0), 
                  datetime(2019,4,4,16,0,0,0), 
                  timedelta(0,1,0))