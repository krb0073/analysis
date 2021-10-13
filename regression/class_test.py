# @Author: Kyle Billings <kbillings>
# @Date:   2020-12-29T21:47:20-05:00
# @Email:  krb0073@mix.wvu.edu
# @Filename: class_test.py
# @Last modified by:   kbillings
# @Last modified time: 2021-01-13T14:42:14-05:00
import sys
sys.path.append('/media/bak12/Analysis/regression/')
from V2_regression import *
import numpy as np
import pandas as pd
import os
import glob
import random
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error
import shutil
import statsmodels.api as sm
import loos
import loos.pyloos
import scipy
#OLS = OLS_regression(st,end,Lmax,DATA_ALL ,seed,3)
Lmax = [570,590,550,412,560,640]
DATA_ALL =os.path.abspath('../')
path = os.path.abspath('../goodRangeDCD/')
seed = 53
st = 0
end = 10000
#end =  558940
num = 10
loc = os.getcwd()
OLS = OLS_regression(st,end,Lmax,DATA_ALL ,seed,,keep_num=5)
OLS.RandomSeed()
states = ['bR_all','K_all','L_all','M_all','N_all','O_all']
data = []
for s in states:
    data.append(OLS.Make_data_frames(s,'dat'))
OLS.Create_final_dataset_splits(data,num)
systems = ['bR','K','L','M','N','O']
OLS.get_AUC('./53_RMSF/',systems)
OLS.pandafy(systems)
OLS.pick_modeling_index()
#OLS.untouchedData()
#OLS.run_leave_one_out()
OLS.kfold_runner(flipp=True)
