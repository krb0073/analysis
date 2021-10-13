# @Author: Kyle Billings <kbillings>
# @Date:   2020-11-05T16:29:20-05:00
# @Email:  krb0073@mix.wvu.edu
# @Filename: 2nd_clean.py
# @Last modified by:   kbillings
# @Last modified time: 2020-11-05T16:32:28-05:00
import sys
import numpy as np
import os
sys.path.append('/media/bak12/Analysis/regression/')
from LRfunctions import *
state = [ 'bR' , 'K' , 'L' , 'M' , 'N' , 'O'] # all intermedaties
# all file must be in the same place rn
Lmax = [570,590,550,412,560,640]
ATOMS  =[['C13','C14','C15','NZ'],['C20','C13','C14','C15'],['C19','C9','C10','C11'],['C12','C13','C14','C15'],['C11','C12','C13','C14'],['C11','C12','C13','C20'],['C9','C10','C11','C12'],['C10','C11','C12','C13'],['C8','C9','C10','C11'],['C7','C8','C9','C10'],['C7','C8','C9','C19'],['C5','C6','C7','C8'],['C6','C7','C8','C9'],['C16','C1','C6','C5'],['C17','C1','C6','C5'],['C17','C1','C2','C3'],['C4','C5','C6','C7'],['C6','C1','C2','C3'],['C16','C1','C2','C3'],['C2','C1','C6','C7'],['C1','C6','C7','C8'],['C14','C15','NZ','CE'],['CG','CD','CE','NZ'],['N','CA','CB','CG'],['N','CA','C','O'], ['C8','C7','C6'] , ['CA','NZ','C3'] , ['C3','C7','C8'] , ['CA','C11','C3'] ,['C9'] , ['C13']]
# aray for the names of the atoms as seen in the code that runs the dihe and angs# aray for the names of the atoms as seen in the code that runs the dihe and angs
# make storage matrix
for s in state:
    for A in ATOMS:
        nameDihe = ''.join(A)
        F_old = "../{0}_all_{1}.dat".format(s,nameDihe)
        F_new = "clean_{0}_all_{1}.dat".format(s,nameDihe)
        # grab the data in the file that is ther
        frames = grabCol(F_old,0)
        data = grabCol(F_old,1)
        # grab min and max
        MIN  = min(data)
        MAX  = max(data)
        if  MAX > 180:
            #cleaned_data = []
            A="Y" #if ture we need to process that data
            for i in range(len(data)):
                if data[i] > 180:
                    data[i] = data[i] - 360
        else:
            A = "N"
        F = open(F_new , 'w')
        print(len(data), len(frames),A)
        for i in range(len(data)):
            W = "{0} {1}\n".format(frames[i],data[i])
            F.write(W)
        F.close()
