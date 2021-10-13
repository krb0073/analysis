# @Author: Kyle Billings <kbillings>
# @Date:   2020-06-24T16:40:11-04:00
# @Email:  krb0073@mix.wvu.edu
# @Filename: LR_randomvalues.py
# @Last modified by:   kbillings
# @Last modified time: 2020-08-07T14:10:31-04:00
import sys
import numpy as np
import os
sys.path.append('/media/bak12/Analysis/regression/')
from LRfunctions import *
# using bR_all_C10C11C12C13.dat to test
seed = sys.argv[1]
st = 0
end = 600000
r = 10
X = randArray(seed,st,end,10)
Fname = 'randvals_seed{}.csv'.format(seed)
np.savetxt(Fname, X, delimiter=',', fmt='%d') #save to a CSV file to
## be used in the RMSD collection stuff
