# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 14:30:57 2019

@author: renze
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# Backtesting strategy: long top 20%, short bottom 20%, equal weights
def backtest(data, price, holding_time=10):
    UpDownTable = ((price - price.shift(ahead))>0).astype(int).shift(-ahead_num)
    
    