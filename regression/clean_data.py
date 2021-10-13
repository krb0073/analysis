# @Author: Kyle Billings <kbillings>
# @Date:   2020-09-24T10:29:26-04:00
# @Email:  krb0073@mix.wvu.edu
# @Filename: clean_data.py
# @Last modified by:   kbillings
# @Last modified time: 2020-11-12T14:12:55-05:00
""" data starts out between 0 360 when we start"""
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
        clean = []
        MIN  = min(data)
        MAX  = max(data)
        if MIN < 1 and MAX > 350:
             for i in range(len(data)):
                 if data[i] < 100:
                     data[i] = data[i] + 360
                     if data[i] > 410:
                         data[i] = data[i] - 360
        F = open(F_new , 'w')
        print(len(data), len(frames),A)
        for i in range(len(data)):
            W = "{0} {1}\n".format(frames[i],data[i])
            F.write(W)
        F.close()
