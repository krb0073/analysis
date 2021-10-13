# @Author: Kyle Billings <kbillings>
# @Date:   2020-09-09T02:06:43-04:00
# @Email:  krb0073@mix.wvu.edu
# @Filename: LR_LOOCV_ridge.py
# @Last modified by:   kbillings
# @Last modified time: 2020-10-12T18:17:25-04:00
# @Last modified time: 2020-10-12T18:17:25-04:00
import sys
import numpy as np
import os
sys.path.append('/media/bak12/Analysis/regression/')
from LRfunctions import *
import seaborn as sns
import matplotlib.pyplot as plt
import time
import scipy
import random
# using bR_all_C10C11C12C13.dat to test
#F = sys.argv[1]
print('start')
seed = sys.argv[1]
path = str(sys.argv[2])
st = 0
end = 558940
X = randArray(seed,st,end,10) # will have ten rows in it for us
state = [ 'bR' , 'K' , 'L' , 'M' , 'N' , 'O'] # all intermedaties
# all file must be in the same place rn
Lmax = [568,590,550,412,560,640]
L = np.array((Lmax))
ALL = []
ATOMS  =[['C13','C14','C15','NZ'],['C20','C13','C14','C15'],['C19','C9','C10','C11'],['C12','C13','C14','C15'],['C11','C12','C13','C14'],['C11','C12','C13','C20'],['C9','C10','C11','C12'],['C10','C11','C12','C13'],['C8','C9','C10','C11'],['C7','C8','C9','C10'],['C7','C8','C9','C19'],['C5','C6','C7','C8'],['C6','C7','C8','C9'],['C16','C1','C6','C5'],['C17','C1','C6','C5'],['C17','C1','C2','C3'],['C4','C5','C6','C7'],['C6','C1','C2','C3'],['C16','C1','C2','C3'],['C2','C1','C6','C7'],['C1','C6','C7','C8'],['C14','C15','NZ','CE'],['CG','CD','CE','NZ'],['N','CA','CB','CG'],['N','CA','C','O'], ['C8','C7','C6'] , ['CA','NZ','C3'] , ['C3','C7','C8'] , ['CA','C11','C3'] ,['C9'] , ['C13']]
# aray for the names of the atoms as seen in the code that runs the dihe and angs# aray for the names of the atoms as seen in the code that runs the dihe and angs
# make storage matrix
mm = len(ATOMS) # the lenght of the array to pass to the function that makes
## the state by the totl number of obsevartions
Q1 = blankArray(mm) # rand values set 1 -> 9
Q2 = blankArray(mm)
Q3 = blankArray(mm)
Q4 = blankArray(mm)
Q5 = blankArray(mm)
Q6 = blankArray(mm)
Q7 = blankArray(mm)
Q8 = blankArray(mm)
Q9 = blankArray(mm)
Q10= blankArray(mm)
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

        F = "cleaned_data/clean_{0}_all_{1}.dat".format(s,nameDihe) # file name
        data = grabCol(F,1) # reaad the column name with the data
        #print(len(data))
        set1 = med(randomVals(data,X[0])) # randvals gets an array then loops
        ## into each to get the value that the index then from that aray of the
        ### the med function takes an n array then returns the medina values for
        #### for the randmized index which realte to the index of the frame
        set2 = med(randomVals(data,X[1]))
        set3 = med(randomVals(data,X[2]))
        set4 = med(randomVals(data,X[3]))
        set5 = med(randomVals(data,X[4]))
        set6 = med(randomVals(data,X[5]))
        set7 = med(randomVals(data,X[6]))
        set8 = med(randomVals(data,X[7]))
        set9 = med(randomVals(data,X[8]))
        set10 = med(randomVals(data,X[9]))
        Q1[spos][pos] = set1
        Q2[spos][pos] = set2
        Q3[spos][pos] = set3
        Q4[spos][pos] = set4
        Q5[spos][pos] = set5
        Q6[spos][pos] = set6
        Q7[spos][pos] = set7
        Q8[spos][pos] = set8
        Q9[spos][pos] = set9
        Q10[spos][pos] = set10
        pos += 1
    spos += 1
# addaing the RMSF data The file has to be the dir
# array list
print("done filling rand arays")
#Qlist = [Q1,Q2,Q3,Q4,Q5,Q6,Q7,Q8,Q9,Q10] # list of possable arrays
AUC = ['Auc' , 'ringAuc']
ALL.append(AUC[0])
ALL.append(AUC[1])
print("grabing a randomize vaules from RMSF")
A , B = grabAUC(seed,1) # grad the AUC froom the file i made
Q1 = np.column_stack((Q1,A,B)) #adding columns to the things
A , B = grabAUC(seed,2)
Q2 = np.column_stack((Q2,A,B))
A , B = grabAUC(seed,3)
Q3 = np.column_stack((Q3,A,B))
A , B = grabAUC(seed,4)
Q4 = np.column_stack((Q4,A,B))
A , B = grabAUC(seed,5)
Q5 = np.column_stack((Q5,A,B))
A , B = grabAUC(seed,6)
Q6 = np.column_stack((Q6,A,B))
A , B = grabAUC(seed,7)
Q7 = np.column_stack((Q7,A,B))
A , B = grabAUC(seed,8)
Q8 = np.column_stack((Q8,A,B))
A , B = grabAUC(seed,9)
Q9 = np.column_stack((Q9,A,B))
A , B = grabAUC(seed,10)
Q10 = np.column_stack((Q10,A,B))
dQ1 = buildingDicts(L,ALL,Q1)
dQ2 = buildingDicts(L,ALL,Q2)
dQ3 = buildingDicts(L,ALL,Q3)
dQ4 = buildingDicts(L,ALL,Q4)
dQ5 = buildingDicts(L,ALL,Q5)
dQ6 = buildingDicts(L,ALL,Q6)
dQ7 = buildingDicts(L,ALL,Q7)
dQ8 = buildingDicts(L,ALL,Q8)
dQ9 = buildingDicts(L,ALL,Q9)
dQ10 = buildingDicts(L,ALL,Q10)
# lets use pandas to make a data frame
import pandas as pd
colls = getList(dQ1) # from dict get the coulmns name
#df = pd.DataFrame(dQ3 , columns=colls ,index=state) # makes data frame
def np_to_dataFrame(mydict , colls=colls , index=state):
    return pd.DataFrame(mydict , columns=colls ,index=state)
print('Training set made ')
# making the data frames for all the stuff df3 is the training set
df3 = np_to_dataFrame(dQ3)
df1 = np_to_dataFrame(dQ1)
df2 = np_to_dataFrame(dQ2)
df4 = np_to_dataFrame(dQ4)
df5 = np_to_dataFrame(dQ5)
df6 = np_to_dataFrame(dQ6)
df7 = np_to_dataFrame(dQ7)
df8 = np_to_dataFrame(dQ8)
df9 = np_to_dataFrame(dQ9)
df10 = np_to_dataFrame(dQ10)
# making the data set
df_cor = df3.corr() # this is to check that the values are not correlated
#df_cor.to_csv('test_coor.csv') # makes it a csv file for later use
Dfs = [df1,df2,df3,df4,df5,df6,df7,df8,df9,df10]
# making one big ol' array from it
allDF = pd.concat(Dfs,ignore_index=True)
X = df3.drop("Lmax",axis = 1)
Y = df3["Lmax"] # getting the target data
Xs = allDF.drop("Lmax",axis=1)
Ys = allDF["Lmax"]
mods = []
DF = [df1,df2,df4, df5,df6 ,df7,df8 ,df9 ,df10]
testing_data =  pd.concat(DF,ignore_index=True)
for k in range(1,5):
    mods.append(LeaveOneOutCV_Ridge(X,Y,k))
mods = pd.DataFrame(mods)
All_A = list(mods['Alpha'])
cnt = 0
for combo in list(mods['combo']):
    title = str(path) + 'Ridge_LOOCV' +"".join(combo) + ".png"
    A = All_A[cnt]
    cnt += 1
    pred_model , mets = process_subset_ridge(X,Y,A,combo)
    print("BIC" , mets['BIC'] )
    print("AIC" , mets["AIC"])
    print("AdjR" , mets["adj_R"])
    print("RSS" , mets["RSS"])
    act_model = list(Y)
    pred_kfold ,act_kfold = precent_10_LOOCV_ridge_grab_all(X,Y,A,combo)
    # grab the equation
    m = mets["model"]
    line = EQFunction("R",m,combo)
    print_eq("R" ,line)
    #ans = line["const"] + line["Error"]
    ans = 0
    for key in line.keys():
        if key != "const":
            if key != "Error":
                ans += testing_data[key] * line.get(key)
    ans = np.array((ans))
    ans = ans + float(line["const"]) + float(line["Error"])
    pred_test = ans
    act_test = testing_data["Lmax"]
    plt.figure(figsize=(10, 10))
    ax = plt.axes()
    ax.scatter(act_model,pred_model,c="black",label="model")
    ax.scatter(act_kfold,pred_kfold,c='red',label="LOOCV")
    ax.scatter(act_test,pred_test,c="blue",label="test_data")
    leg1 = ax.legend(loc = 'lower right')
    act_model = np.array((act_model))
    act_kfold = np.array((act_kfold))
    m, b , r_value, p_value, std_err = scipy.stats.linregress(act_model, pred_model)
    m2, b2 , r_value2, p_value2, std_err2 = scipy.stats.linregress(act_kfold,pred_kfold)
    plt.plot(act_model,m*act_model + b , 'black')
    plt.plot(act_kfold,m*act_kfold+b, 'red')
    plt.legend(('model | R squared ' + str(r_value) , 'leave one out cv| R sqaured ' + str(r_value2)) )
    plt.savefig(title)
