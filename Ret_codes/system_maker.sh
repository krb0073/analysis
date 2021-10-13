# @Author: Kyle Billings <kbillings>
# @Date:   2021-09-16T14:50:52-04:00
# @Email:  krb0073@mix.wvu.edu
# @Filename: system_maker.sh
# @Last modified by:   kbillings
# @Last modified time: 2021-09-16T19:44:57-04:00



#!/bin/bash


# command line args for the pdb
A='1IW6'
B='1M0K'
mkdir $A $B
# run them into the modeller program
loc=$(pwd)
for i in $A $B ; do
	cd $i
	wget https://files.rcsb.org/download/${i}.pdb
	run_conda_loos
	python3.9 /media/bak12/Analysis/Missing_loop_tool/genrated_seq_file.py ${i}.pdb A new_test
	vmd `ls -1v *_fill*.pdb`
	cd $loc
done
