# @Author: Kyle Billings <kbillings>
# @Date:   2020-12-02T11:45:40-05:00
# @Email:  krb0073@mix.wvu.edu
# @Filename: Gen_time.py
# @Last modified by:   kbillings
# @Last modified time: 2020-12-02T12:09:03-05:00
import sys
import loos
sys.path.append('/media/bak12/Analysis/loos_timeseries/')
# @Last modified by:   kbillings
from time_series_loos import *
# given a list of files get the averages
Col = int(sys.argv[1])
skip = int(sys.argv[2])
F = sys.argv[3:]
FILES =  [f for f in F ] # all file strinfgs into one list
#print(FILES)
final = []
print("#FileName AVG STDEV STDERR VARINCE")
for file in FILES:
    AVG , STDEV , STDERR , VARINCE = Basic_time_fxn(file,col=Col,skip=skip)
    final.append(f"{file} {AVG} {STDEV} {STDERR} {VARINCE}")
for i in final:
    print(i)
