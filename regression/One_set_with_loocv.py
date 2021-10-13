# @Author: Kyle Billings <kbillings>
# @Date:   2020-09-01T15:35:34-04:00
# @Email:  krb0073@mix.wvu.edu
# @Filename: One_set_with_loocv.py
# @Last modified by:   kbillings
# @Last modified time: 2020-09-15T08:38:02-04:00
import sys
import numpy as np
import os
sys.path.append('/media/bak12/Analysis/regression/')
from LRfunctions import *
import seaborn as sns
import matplotlib.pyplot as plt
import time
import scipy
run_name = sys.argv[1]
path = str(sys.argv[2])
st = 0
end = 558940
#X = randArray(seed,st,end,10) # will have ten rows in it for us
state = [ 'bR' , 'K' , 'L' , 'M' , 'N' , 'O'] # all intermedaties
# all file must be in the same place rn
Lmax = [570,590,550,412,560,640]
L = np.array((Lmax))
ALL = []
ATOMS  =[['C13','C14','C15','NZ'],['C20','C13','C14','C15'],['C19','C9','C10','C11'],['C12','C13','C14','C15'],['C11','C12','C13','C14'],['C11','C12','C13','C20'],['C9','C10','C11','C12'],['C10','C11','C12','C13'],['C8','C9','C10','C11'],['C7','C8','C9','C10'],['C7','C8','C9','C19'],['C5','C6','C7','C8'],['C6','C7','C8','C9'],['C16','C1','C6','C5'],['C17','C1','C6','C5'],['C17','C1','C2','C3'],['C4','C5','C6','C7'],['C6','C1','C2','C3'],['C16','C1','C2','C3'],['C2','C1','C6','C7'],['C1','C6','C7','C8'],['C14','C15','NZ','CE'],['CG','CD','CE','NZ'],['N','CA','CB','CG'],['N','CA','C','O'], ['C8','C7','C6'] , ['CA','NZ','C3'] , ['C3','C7','C8'] , ['CA','C11','C3'] ,['C9'] , ['C13']]
# aray for the names of the atoms as seen in the code that runs the dihe and angs
# make storage matrix
mm = len(ATOMS) # the lenght of the array to pass to the function that makes
## the state by the totl number of obsevartions
Q1 = blankArray(mm) # rand values set 1 -> 9
for i in ATOMS:
    nameDihe = ''.join(i)
    ALL.append(nameDihe)

spos = 0 # state counter 0=bR , 1=k ..
for s in state: # for intermedatie
    print(s)
    pos = 0 # postion of the vaule in the matrix
    for l in ATOMS:
        nameDihe = ''.join(l) #get the name of the file by joing the dihe/ang
        # file name to grab
        F = "{0}_all_{1}.dat".format(s,nameDihe) # file name
        data = grabCol(F,1) # reaad the column name with the data
        set1 = med(data)
        Q1[spos][pos] = set1
        pos += 1
    spos += 1
# addaing the RMSF data The file has to be the dir
# array list
dQ1 = buildingDicts(L,ALL,Q1)
# lets use pandas to make a data frame
import pandas as pd
colls = getList(dQ1) # from dict get the coulmns name
def np_to_dataFrame(mydict , colls=colls , index=state):
    return pd.DataFrame(mydict , columns=colls ,index=state)
# making the data frames for all the stuff df3 is the training set
df1 = np_to_dataFrame(dQ1)
df1.to_csv("all_data_{}.csv".format(run_name))
# making the data set
df_cor = df1.corr() # this is to check that the values are not correlated
S = "{}_corr_maxtrix.csv".format(run_name)
df_cor.to_csv(S) # makes it a csv file for later use
X = df1.drop("Lmax",1) # All data but Lmax
Y = df1["Lmax"] # getting the target data

mods = []
## returns the model with the best RSS values sum  = (perdicted - actual)**2
for k in range(1,5):
    mods.append(LeaveOneOutCV_OLS(X,Y,k))
mods = pd.DataFrame(mods)
for combo in list(mods['combo']):
    title = str(path)+ "/" + 'LOOCV_OLS_' +"".join(combo) + "Xtal.png"
    act ,pred = loocv_collect_guesss("OLS",X,Y,combo)
    A = processSubset(Y,X,combo)
    real_model = A["model"]
    print(real_model.summary())
    line = EQFunction("OLS",real_model,combo)
    print_eq("OLS",line)
    data = X[combo]
    data = sm.add_constant(data)
    pred_real = list(real_model.predict(data))
    act = np.array((act))
    pred = np.array((pred))
    pred_real = np.array((pred_real))
    plt.figure(figsize=(10, 10))
    ax = plt.axes()
    ax.scatter(act,pred_real,c="black",label="model")
    ax.scatter(act,pred,c="r", label="leave one out")
    ax.set_xlabel("actual Lmax(nm)")
    leg1 = ax.legend(loc = 'lower right')
    ax.add_artist(leg1)
    m, b , r_value, p_value, std_err = scipy.stats.linregress(act, pred_real)
    m2, b2 , r_value2, p_value2, std_err2 = scipy.stats.linregress(act, pred)
    plt.plot(act,m*act + b,'black')
    plt.plot(act,m2*act + b2,'r-')
    plt.legend(('model | R squared ' + str(r_value) , 'leave one out cv| R sqaured ' + str(r_value2)) )
    plt.savefig(title)
