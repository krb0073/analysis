"""
this code will take the systems  with the regenatred system and place
the new retinal
"""
systes = "/media/bak11/KylesStuff/newer_mutants/cis/"
mut = ['182' , '185' , '189']
for m in mut: # loop into all mutants
	# there are three replacates of each
	for st in range(1,4):
		cmd.load(f'ret_1IW6_{m}.pdb','good')
		# load the  full system and the new boys 
		cmd.load(f"{systes}/{m}/{str(st)}/pro_only.pdb",f"{m}-{str(st)}")
		# select the protein atoms
		cmd.align("good",f"{m}-{str(st)}")
		cmd.save(f"{systes}/{m}/{str(st)}/regen.pdb" , "good")
		cmd.remove('all')
		cmd.delete('all')
