# @Author: Kyle Billings <kbillings>
# @Date:   2021-01-12T21:29:01-05:00
# @Email:  krb0073@mix.wvu.edu
# @Filename: Main_regression_code.py
# @Last modified by:   kbillings
# @Last modified time: 2021-03-04T23:47:12-05:00
from V2_regression import *
import os
import sys
import glob
import random
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import shutil
import statsmodels.api as sm
import loos
import loos.pyloos
import pandas as pd
import numpy as np
import scipy
from math import log, ceil
sys.path.append('/media/bak12/Analysis/regression/')
config_file = sys.argv[1]


def main(config_file):
    config = readConfig(config_file)
    # define the common stuff for all the metods
    reg = OLS_regression(
        config.data_range[0],
        config.data_range[1],
        config.Lmax,
        config.dirloc,
        config.seedNum,
        config.num_var)
    if reg.keep != config.keep_num:
        reg.keep = config.keep_num
    # gen the random seed
    reg.RandomSeed()
    # check if skip is not 0
    if len(config.data_range) == 3:
        if config.data_range[2] != 0:
            reg.skip = config.data_range[2]
    # loop into the file locations to get the data
    data = []
    if config.suffix is None:
        config.suffix = 'dat'
    for prefix in config.sim_list:
        data.append(reg.Make_data_frames(prefix, suffix=config.suffix))
    # split data into groups
    reg.Create_final_dataset_splits(data, config.n_sets, mix=config.mix)
    reg.plot_distrabution(
        data,
        config.n_sets,
        config.systems_list,
        mix=config.mix)
    # add in the AUC if needed
    if config.path_auc is not None:
        reg.get_AUC(config.path_auc, config.systems_list)
    reg.pandafy(config.systems_list)
    if config.method == "LOOCV":
        reg.run_leave_one_out()
    elif config.method == "KFOLD":
        # $untouched only matters in k fold
        if config.untouched_num is not None:
            reg.untouchedData(nset=config.untouched_num)
        reg.kfold_runner(
            tolerance=config.data_tol,
            purge_cutoff=config.purge_cutoff,
            cv_metric=config.metric,
            flipp=config.flipp)


main(config_file)
