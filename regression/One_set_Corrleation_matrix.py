# @Author: Kyle Billings <kbillings>
# @Date:   2020-09-23T14:08:08-04:00
# @Email:  krb0073@mix.wvu.edu
# @Filename: One_set_Corrleation_matrix.py
# @Last modified by:   kbillings
# @Last modified time: 2020-09-24T20:45:15-04:00

import sys
import numpy as np
import os
sys.path.append('/media/bak12/Analysis/regression/')
from LRfunctions import *
import seaborn as sns
import matplotlib.pyplot as plt
import time
import scipy
import seaborn as sns
import pandas as pd
st = 0
end = 558940
#X = randArray(seed,st,end,10) # will have ten rows in it for us
state = [ 'bR' , 'K' , 'L' , 'M' , 'N' , 'O'] # all intermedaties
# all file must be in the same place rn
Lmax = [570,590,550,412,560,640]
L = np.array((Lmax))
ALL = []
ATOMS  =[['C13','C14','C15','NZ'],['C20','C13','C14','C15'],['C19','C9','C10','C11'],['C12','C13','C14','C15'],['C11','C12','C13','C14'],['C11','C12','C13','C20'],['C9','C10','C11','C12'],['C10','C11','C12','C13'],['C8','C9','C10','C11'],['C7','C8','C9','C10'],['C7','C8','C9','C19'],['C5','C6','C7','C8'],['C6','C7','C8','C9'],['C16','C1','C6','C5'],['C17','C1','C6','C5'],['C17','C1','C2','C3'],['C4','C5','C6','C7'],['C6','C1','C2','C3'],['C16','C1','C2','C3'],['C2','C1','C6','C7'],['C1','C6','C7','C8'],['C14','C15','NZ','CE'],['CG','CD','CE','NZ'],['N','CA','CB','CG'],['N','CA','C','O'], ['C8','C7','C6'] , ['CA','NZ','C3'] , ['C3','C7','C8'] , ['CA','C11','C3'] ,['C9'] , ['C13']]

# make storage matrix
mm = len(ATOMS) # the lenght of the array to pass to the function that makes
## the state by the totl number of obsevartions
Q1 = blankArray(mm) # rand values set 1 -> 9

for i in ATOMS:
    nameDihe = ''.join(i)
    ALL.append(nameDihe)
spos = 0
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
colls = list(dQ1.keys())  # from dict get the coulmns name
#df = pd.DataFrame(dQ3 , columns=colls ,index=state) # makes data frame
def np_to_dataFrame(mydict , colls=colls , index=state):
    return pd.DataFrame(mydict , columns=colls ,index=state)
df1 = np_to_dataFrame(dQ1)
plt.subplots(figsize=(20,15))
X = df1.drop("Lmax",axis=1)
matrix = np.triu(X.corr())
#print(matrix)
sns.heatmap(X.corr(), annot=True, cmap= 'coolwarm')
plt.title("Correlation between data points for all data")
#sns.heatmap(df1.corr(), annot=True, mask=matrix)
#plt.show()
plt.savefig("All_coor.png")
