# @Author: Kyle Billings <kbillings>
# @Date:   2020-06-25T12:14:37-04:00
# @Email:  krb0073@mix.wvu.edu
# @Filename: runRandRMSF.sh
# @Last modified by:   kbillings
# @Last modified time: 2020-06-29T16:34:30-04:00
#will not work

#!/usr/bin/env bash
#PBS -N randRMSF
#PBS -m bae
#PBS -M krb0073@mix.wvu.edu
#PBS -q jbmertz
#PBS -l nodes=40:ppn=1
module purge
module load loos/latest
module load mertz_conda/latest
conda activate loos
cd $PBS_O_WORKDIR
cnt=1 # counter
while read line ; do # this loop into every line
frames=$line # this helps my head
for i in bR K L M N O; do # for intermedatie
  for j in $(seq 1 6); do # for respilacate
    cd ${i}/SYS_${j}/ # open that dir
    # run the rmsf given the k(skip the eq) -s === sel and -r frames we want
    rmsf -k 50000 -s '(resname =~ "^(RET|RTNH)") || (resname == "LYS" && resid == 216) && !hydrogen' -r $frames final.psf merged.dcd > rmsf_${i}-${j}-${cnt}.dat
    cd ../../ # exit that dir
  done
done
cnt=$((cnt + 1)) # count to the next line
done  <  randvals.csv
