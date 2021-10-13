# @Author: Kyle Billings <kbillings>
# @Date:   2021-09-09T15:49:00-04:00
# @Email:  krb0073@mix.wvu.edu
# @Filename: genrated_seq_file.py
# @Last modified by:   kbillings
# @Last modified time: 2021-09-21T14:14:48-04:00

import sys
import os
from modeller import *
from modeller.automodel import *

log.verbose()
# load a given pdb file
# will only work for a monomer pdb
pdb = sys.argv[1]
CHAIN = sys.argv[2] # this is the chain we want to work on
outfile = sys.argv[3]
AA = {"ALA": "A", "ARG":"R","ASN":"N","ASP": "D","CYS":"C",'GLU':'E','GLN':'Q','GLY':'G','HIS':'H',"ILE":"I" , 'LEU' : 'L' , 'LYS':'K' , 'MET':'M','PHE':'F' , 'PRO':'P' , 'SER':'S', 'THR':'T', 'TRP':'W','TYR':'Y','VAL':'V'}
# get the pdb code from the pdb
code = code = pdb.split('/')[-1].split('.')[0]
def find_missing_range(l):
    RAN = []
    base = [l[0]]
    for i in range(1,len(l)): # each item on the list
        # check if RAN[i] -1 == RAN[i-1]
        if l[i] -1 == l[i -1]: # if this is ture then we can add it to
            # add the number to base
            base.append(l[i])
        else:
            RAN.append(base)
            base = [l[i]]
    return RAN

def find_parts(pdb,CHAIN):
    # empty ditcs for later
    good = {} # AA in the pdb
    missing = {} # ditc for the dash line used latter
    remark = {} # the AA for the missing
    # with opens the file and closes once the loop is done
    with open(pdb,'r') as f :
        # for each line in the pdb
        for line in f.readlines(): #for each line
            # split the line
            # find the good atoms
            if line.split()[0] == 'ATOM':
                if line.split()[2] == "CA": # do not want repates using CA to get one for each
                    if line.split()[4] == CHAIN:
                        good[int(line.split()[5])] = AA[line.split()[3]] # give {resid: single letter AA}
            elif line.split()[0] == "REMARK":  # checking for line 465
                if line.split()[1] == '465':
                    if len(line.split()) == 5:
                        if line.split()[3] == CHAIN:
                            remark[int(line.split()[4])] = AA[line.split()[2]]
                            missing[int(line.split()[4])] = '-'

    return good,missing,remark

def seq_builder(A,B):
    ans = '' # empty string to store stuff
    a_list = list(A.keys()) # lsit of keys in the list
    cnt = 0
    for i in sorted(A|B): # this will give a list of sorted resid for the mol
        if i in a_list:
            ans += A[i]
        else:
            ans += B[i]
        cnt += 1
        if cnt == 75:
            ans += '\n'
            cnt = 0
    ans += "*"
    return ans
def get_seq(pdb):
    e = Environ()
    m = Model(e, file=code,model_segment=("FIRST:"+CHAIN,'LAST:'+CHAIN))
    aln = Alignment(e)
    aln.append_model(m, align_codes=code)
    aln.write(file=code+'.seq')

# use the seq info to build the ali file
def make_align(pdb,CHAIN):
    get_seq(pdb)
    # this makes the thing we need for the header
    S = str(code) + '.seq'
    F = open(S,'r')
    F.readline()
    LINE1 = F.readline()
    LINE2 = F.readline()
    F.close()
    # need a blank line
    line = ''
    line += LINE1
    line += LINE2
    # need to maked modfied
    G,M,R = find_parts(pdb,CHAIN)
    data = [a for a in R.keys()]
    cont_res = find_missing_range(data)
    missing = seq_builder(G,M)
    filled = seq_builder(G,R)
    line += missing + '\n'
    line += f'>P1;{code}_fill\n'
    line += 'sequence:::::::::\n'
    line += filled
    with open(outfile+'.ali','w') as Q:
        Q.write(line)
    return cont_res
cont_res_ranges = make_align(pdb,CHAIN)

class MyModel(AutoModel):
    def select_atoms(self):
        s = Selection()
        global cont_res_ranges
        global CHAIN
        for r in cont_res_ranges:
            if len(r) == 1:
                s.add(self.residue(f"{r}:{CHAIN}"))
            elif len(r) > 1:
                s.add(self.residue_range(f"{r[0]}:{CHAIN}" , f"{r[-1]}:{CHAIN}"))
        return s
# putting the stuff together
def make_modell(code,n=20):
    env = Environ()
    print(code)
    env.io.atom_files_directory = ['.']
    a = MyModel(env, alnfile = outfile + '.ali' , knowns= code ,sequence= code + '_fill')
    a.starting_model = 1
    a.ending_model  = n
    a.make()
make_modell(code)
