# @Author: Kyle Billings <kbillings>
# @Date:   2021-08-14T20:13:31-04:00
# @Email:  krb0073@mix.wvu.edu
# @Filename: algn_ret_to_xtal.py
# @Last modified by:   kbillings
# @Last modified time: 2021-09-17T09:27:07-04:00

#will need to do this for each system
pdbs=['ret_182.pdb','ret_185.pdb','ret_189.pdb' ]

# load into the file  make the c_wat.pdb
cmd.load('cnt_xtal.pdb','Base')
cmd.select("WAT",'resname HOH')
cmd.save("c_wat.pdb",'WAT')
cmd.remove("resname HOH")
cmd.delete("WAT")

# now we can use base to algin the stucutres up and right new pdb files
for p in pdbs:
    cmd.load(p,'pro')
    cmd.align('pro','Base')
    cmd.save('cnt'+p , 'pro')
    cmd.remove('pro')
    cmd.delete('pro')
