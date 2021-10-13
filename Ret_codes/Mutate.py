# @Author: Kyle Billings <kbillings>
# @Date:   2021-08-14T15:43:39-04:00
# @Email:  krb0073@mix.wvu.edu
# @Filename: Mutate.py
# @Last modified by:   kbillings
# @Last modified time: 2021-09-17T08:33:33-04:00
# this coed makes the BR mustions that i want

# 1 fetch the pdb
from sys import argv

#cmd.fetch(str(sys.argv[1])) #  K code
cmd.load('/tmp/step1_pdbreader.pdb')
#2 clean the pdb
#cmd.load(sys.argv[1])
cmd.select('Bad','not (resname RET or resname HOH or polymer)')
cmd.remove('Bad')
cmd.delete('Bad')
# 3 split to a RET , and other files
cmd.select('RET','resname RET') # slect atoms
cmd.save('ret_only.pdb','RET') #make a pdb for alter
cmd.remove('RET') # water and protein only left
cmd.delete('RET')

# now for HOH
cmd.select("WAT",'resname HOH')
cmd.save('water_only.pdb','WAT')
cmd.remove('WAT')
cmd.delete('WAT')
# svave the top level as 1IW6_cleaned.pdb
cmd.select("PRO",'polymer')
cmd.save('1M0K_cleaned.pdb','PRO')
cmd.remove('PRO')
cmd.delete('PRO')
#4 make the muatations
resid_list = ['182','185','189']
res="PHE"
cmd.wizard("mutagenesis")
wiz = cmd.get_wizard()
wiz.set_mode(res) # all muataions are PHE
# every mutation
for R in resid_list:
    # load the clean boy
    cmd.load('1M0K_cleaned.pdb',"1M0K")
    cmd.do("refresh_wizard")
    wiz.do_select(f"{R}/")
    wiz.apply()
    cmd.save(f'{R}.pdb','1M0K')
    cmd.delete('all')
