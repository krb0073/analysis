# @Author: Kyle Billings <kbillings>
# @Date:   2020-09-13T21:47:52-04:00
# @Email:  krb0073@mix.wvu.edu
# @Filename: one_set_test.py
# @Last modified by:   kbillings
# @Last modified time: 2020-09-24T11:27:39-04:00
# @Author: Kyle Billings <kbillings>
# @Date:   2020-09-01T15:35:34-04:00
# @Email:  krb0073@mix.wvu.edu
# @Filename: One_set_with_loocv.py
# @Last modified by:   kbillings
# @Last modified time: 2020-09-24T11:27:39-04:00
import sys
import numpy as np
import os
sys.path.append('/media/bak12/Analysis/regression/')
from LRfunctions import *
import seaborn as sns
import matplotlib.pyplot as plt
import time
import scipy
#run_name = sys.argv[1]
#path = str(sys.argv[2])
st = 0
end = 558940
#X = randArray(seed,st,end,10) # will have ten rows in it for us
state = [ 'bR' , 'K' , 'L' , 'M' , 'N' , 'O'] # all intermedaties
# all file must be in the same place rn
Lmax = [570,590,550,412,560,640]
L = np.array((Lmax))
ALL = []
ATOMS  =[['C13','C14','C15','NZ'],['C20','C13','C14','C15'],['C19','C9','C10','C11'],['C12','C13','C14','C15'],['C11','C12','C13','C14'],['C11','C12','C13','C20'],['C9','C10','C11','C12'],['C10','C11','C12','C13'],['C8','C9','C10','C11'],['C7','C8','C9','C10'],['C7','C8','C9','C19'],['C5','C6','C7','C8'],['C6','C7','C8','C9'],['C16','C1','C6','C5'],['C17','C1','C6','C5'],['C17','C1','C2','C3'],['C4','C5','C6','C7'],['C6','C1','C2','C3'],['C16','C1','C2','C3'],['C2','C1','C6','C7'],['C1','C6','C7','C8'],['C14','C15','NZ','CE'],['CG','CD','CE','NZ'],['N','CA','CB','CG'],['N','CA','C','O'], ['C8','C7','C6'] , ['CA','NZ','C3'] , ['C3','C7','C8'] , ['CA','C11','C3'] ,['C9'] , ['C13']]
fig1 , axs1 = plt.subplots(4,4,figsize = (10,10))
fig2 , axs2 = plt.subplots(4,4, figsize = (10,10))
left  = 0.125  # the left side of the subplots of the figure
right = 0.9    # the right side of the subplots of the figure
bottom = 0.1   # the bottom of the subplots of the figure
top = 0.9      # the top of the subplots of the figure
wspace = 0.2   # the amount of width reserved for blank space between subplots
hspace = 0.2   # the amount of height reserved for white space between subplots
# aray for the names of the atoms as seen in the code that runs the dihe and angs

# make storage matrix
mm = len(ATOMS) # the lenght of the array to pass to the function that makes
## the state by the totl number of obsevartions
Q1 = blankArray(mm) # rand values set 1 -> 9
for i in ATOMS:
    nameDihe = ''.join(i)
    ALL.append(nameDihe)
spos = 0 # state counter 0=bR , 1=k ..
A = 0
B = 0
cnt = 0
fig_num = 0
for l in ATOMS: # for intermedatie
    C = 0
    for s in state:
        colors = ['black' , 'red' , 'green', "purple" ,"darkgoldenrod" , "slategrey"]
        nameDihe = ''.join(l) #get the name of the file by joing the dihe/ang
        # file name to grab
        subtitle = "-".join(l)

        F = "clean_{0}_all_{1}.dat".format(s,nameDihe) # file name
        data = grabCol(F,1) # reaad the column name with the data
        # getting the histgram
        hist , bin = np.histogram(data, bins=(400*2),  density=True)
        if fig_num == 0:

            axs1[B,A].set_title(subtitle)
            axs1[B,A].plot(bin[:-1] ,hist , alpha=0.75 , c=colors[C])
        else:
            axs2[B,A].set_title(subtitle)
            axs2[B,A].plot(bin[:-1] ,hist , alpha=0.75 , c=colors[C])
        C += 1
    if B == 3 and A != 3:
        A += 1
        B = 0
    elif B == 3 and A == 3:
        fig_num += 1
        A = 0
        B =0

    else:
        B += 1
fig1.tight_layout()
fig2.tight_layout()
fig1.savefig("Dihe_1.png")
fig2.savefig("Dihe_2.png")

#        set1 = med(data)
#        Q1[spos][pos] = set1
#        pos += 1
#    spos += 1
# addaing the RMSF data The file has to be the dir
# array list
#dQ1 = buildingDicts(L,ALL,Q1)
# lets use pandas to make a data frame
