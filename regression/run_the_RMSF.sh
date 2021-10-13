# @Author: Kyle Billings <kbillings>
# @Date:   2020-06-29T16:39:18-04:00
# @Email:  krb0073@mix.wvu.edu
# @Filename: run_the_RMSF.sh
# @Last modified by:   kbillings
# @Last modified time: 2020-08-07T14:26:05-04:00
#!/usr/bin/env bash
#  this works in the good DCD file
state=$1
seed=$2
for i in $(seq 1 10); do
python3.8 /media/bak12/Analysis/regression/kyleRMSF.py ../${state}_goodAll.pdb '(resname =~ "^(RET|RTNH)") || (resname == "LYS" && resid == 216) && !hydrogen' ${i}.dat  ../${state}_goodAll.dcd > ${state}_set_${i}.dat
done
