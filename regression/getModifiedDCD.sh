# @Author: Kyle Billings <kbillings>
# @Date:   2020-06-29T14:04:47-04:00
# @Email:  krb0073@mix.wvu.edu
# @Filename: getModifiedDCD.sh
# @Last modified by:   kbillings
# @Last modified time: 2020-06-29T16:38:12-04:00


#!/usr/bin/env bash
#PBS -N randRMSF
#PBS -m bae
#PBS -M krb0073@mix.wvu.edu
#PBS -q jbmertz
#PBS -l nodes=10:ppn=1
module purge
module load loos/latest
module load mertz_conda/latest
conda activate loos
cd $PBS_O_WORKDIR
for i in bR K L M N O ; do
subsetter -C 'name == "CA"' -s '(resname =~ "^(RET|RTNH)") || (resname == "LYS" && resid == 216) && !hydrogen' -r 50000:150000,50000:150000,50000:150000,50000:150000,50000:150000,50000:150000 goodRRangeDCD/${i}_goodAll ${i}/SYS_1/final.psf ${i}/SYS_1/merged.dcd ${i}/SYS_2/merged.dcd ${i}/SYS_3/merged.dcd ${i}/SYS_4/merged.dcd ${i}/SYS_5/merged.dcd ${i}/SYS_6/merged.dcd
done
