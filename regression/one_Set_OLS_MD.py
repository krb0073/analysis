# @Author: Kyle Billings <kbillings>
# @Date:   2020-08-31T19:58:21-04:00
# @Email:  krb0073@mix.wvu.edu
# @Filename: one_Set_OLS_MD.py
# @Last modified by:   kbillings
# @Last modified time: 2020-09-09T11:10:14-04:00
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
print("made new all data with title AAAAA_MD_only")
# making the data set
df_cor = df1.corr() # this is to check that the values are not correlated
S = "{}_corr_maxtrix.csv".format(run_name)
df_cor.to_csv(S) # makes it a csv file for later use
X = df1.drop("Lmax",1) # All data but Lmax
Y = df1["Lmax"] # getting the target data
models_best = pd.DataFrame(columns=["model","adj_R","BIC", "AIC","RSS"])
for i in range(1,5): # this is for the number of variables at one time
    models_best.loc[i] = bestModel(Y,X,i) # add top the model_best data frame
    print("done with {}".format(i))
ranSetFinal = pd.DataFrame(columns=["parmaLabel" ,"pramaVals" , "set", "personCorr"])
TestingData = []
#setList = [df1,df2,df4, df5,df6 ,df7,df8 ,df9 ,df10]
act = list(Y)
for mod in models_best.itertuples():
    print("\n")
    print(personCorr(mod[1],X))
    print(mod[1].summary())
    p = list(mod[1].params.index)[1:]
    title = str(path) + "All_data_OLS_MD"+"".join(p) + ".png"
    eq = EQFunction("OLS",mod[1],p)
    print_eq("OLS" , eq)
    ans = eq['const']
    for key in eq.keys():
        if key != "const":
            ans += X[key] * eq.get(key)
    pred = (list(ans))
    print(mod[1].bic)
    plt.figure(figsize=(10, 10))
    ax = plt.axes()
    ax.scatter(act,pred)
    ax.set_xlabel("actual Lmax(nm)")
    ax.set_ylabel("predcited Lmax(nm)")
    m,b = np.polyfit(act,pred,1)
    m, b , r_value, p_value, std_err = scipy.stats.linregress(act, pred)
    print(r_value)
    act = np.array((act))
    plt.plot(act,m*act + b,'r')
    plt.savefig(title)
