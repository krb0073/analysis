# @Author: Kyle Billings <kbillings>
# @Date:   2020-09-09T11:10:49-04:00
# @Email:  krb0073@mix.wvu.edu
# @Filename: oneset_ridge.py
# @Last modified by:   kbillings
# @Last modified time: 2020-09-09T14:10:31-04:00
import sys
import numpy as np
import os
sys.path.append('/media/bak12/Analysis/regression/')
from LRfunctions import *
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
# using bR_all_C10C11C12C13.dat to test
run_name = sys.argv[1]
path = str(sys.argv[2])
print('start')
#seed = sys.argv[1]
#st = 0
#end = 558940
#X = randArray(seed,st,end,10) # will have ten rows in it for us
state = [ 'bR' , 'K' , 'L' , 'M' , 'N' , 'O'] # all intermedaties
# all file must be in the same place rn
Lmax = [570,590,550,412,560,640]
L = np.array((Lmax))
ALL = []
ATOMS  =[['C13','C14','C15','NZ'],['C20','C13','C14','C15'],['C19','C9','C10','C11'],['C12','C13','C14','C15'],['C11','C12','C13','C14'],['C11','C12','C13','C20'],['C9','C10','C11','C12'],['C10','C11','C12','C13'],['C8','C9','C10','C11'],['C7','C8','C9','C10'],['C7','C8','C9','C19'],['C5','C6','C7','C8'],['C6','C7','C8','C9'],['C16','C1','C6','C5'],['C17','C1','C6','C5'],['C17','C1','C2','C3'],['C4','C5','C6','C7'],['C6','C1','C2','C3'],['C16','C1','C2','C3'],['C2','C1','C6','C7'],['C1','C6','C7','C8'],['C14','C15','NZ','CE'],['CG','CD','CE','NZ'],['N','CA','CB','CG'],['N','CA','C','O'], ['C8','C7','C6'] , ['CA','NZ','C3'] , ['C3','C7','C8'] , ['CA','C11','C3'] ,['C9'] , ['C13']]
# aray for the names of the atoms as seen in the code that runs the dihe and angs # aray for the names of the atoms as seen in the code that runs the dihe and angs
# make storage matrix
mm = len(ATOMS) # the lenght of the array to pass to the function that makes
## the state by the totl number of obsevartions
Q1 = blankArray(mm) # rand values set 1 -> 9

for i in ATOMS:
    nameDihe = ''.join(i)
    ALL.append(nameDihe)

spos = 0 # state counter 0=bR , 1=k ..
print('Filling Random arrays')
for s in state: # for intermedatie
    #print(s)
    pos = 0 # postion of the vaule in the matrix
    for l in ATOMS:
        nameDihe = ''.join(l) #get the name of the file by joing the dihe/ang
        # file name to grab

        F = "{0}_all_{1}.dat".format(s,nameDihe) # file name
        data = med(grabCol(F,1)) # reaad the column name with the data
        #print(len(data))
        Q1[spos][pos] = data
        set1 = data # randvals gets an array then loops

        pos += 1
    spos += 1
# addaing the RMSF data The file has to be the dir
# array list
dQ1 = buildingDicts(L,ALL,Q1)
import pandas as pd
colls = getList(dQ1) # from dict get the coulmns name
#df = pd.DataFrame(dQ3 , columns=colls ,index=state) # makes data frame
def np_to_dataFrame(mydict , colls=colls , index=state):
    return pd.DataFrame(mydict , columns=colls ,index=state)
df1 = np_to_dataFrame(dQ1)
Y = df1["Lmax"] # getting the target data
from sklearn.linear_model import Ridge , RidgeCV
from sklearn.metrics import mean_squared_error
alphas = 10**np.linspace(10,-2,100)*0.5
#ridge.set_params(alpha = A)
Rn = Ridge(normalize = True)
X = df1.drop("Lmax",axis=1)
y = Y
print(y)
res = []
for k in range(1,5):
    res.append(ridgeNoKfold(df1,y,k))
final = pd.DataFrame(res)
final_later = final[["model","parm"]]
for i , j in  final_later.iterrows():
    title = str(path)  + 'Ridge_' +"".join(combo) + ".png"
    combo = j[1]
    mod = j[0]
    fxn = EQFunction("R",mod,PAR)
    print_eq("R",fxn)
    ans = fxn['const'] + fxn['Error']
    for key in fxn.keys():
        if key != 'const':
            if key != 'Error':
                ans += X[key] * fxn.get(key)
    pred = list(ans)
    pred = np.array((ans))
    act = np.array(list(Y))
    plt.figure(figsize=(10, 10))
    ax = plt.axes()
    ax.set_xlabel("actual Lmax(nm)")
    ax.set_ylabel("predicted Lmax(nm)")
    ax.scatter(act,pred,c="r", label="leave one out")
    m, b , r_value, p_value, std_err = scipy.stats.linregress(act, pred)
    plt.plot(act,m*act + b,'black')
    plt.legend("model | R squared" + str(r_value))
    plt.savefig(title)
