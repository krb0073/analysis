# @Author: Kyle Billings <kbillings>
# @Date:   2021-09-09T20:45:46-04:00
# @Email:  krb0073@mix.wvu.edu
# @Filename: do_all.sh
# @Last modified by:   kbillings
# @Last modified time: 2021-09-09T20:51:46-04:00
#!/bin/#!/usr/bin/env bash

for i in 5ZIH 5AHZ 3DDL 3T45 1UAZ 4QID 6GUY 4jR8 7BMH 6NWD 4HYJ 5AZD 3X3C 5AWZ 7B03 6EDQ 1H2S 4jQ6 6CSM 6EID; do
  mkdir ${i}
  cd $i
  wget https://files.rcsb.org/download/${i}.pdb
  python3.9 /media/bak12/Analysis/Missing_loop_tool/genrated_seq_file.py ${i}.pdb A new_test
  cd ../
done   
