# @Author: Kyle Billings <kbillings>
# @Date:   2020-12-06T18:41:43-05:00
# @Email:  krb0073@mix.wvu.edu
# @Filename: Config_get.py
# @Last modified by:   kbillings
# @Last modified time: 2020-12-14T11:47:57-05:00
import sys
import loos
import subprocess
import os
import sys
class Read_config:
    """
    This code will read the confie and check for vaildity
    """
    def __init__(self,FILE):
        # all for these are the vaiables that need to be in there
        ## all of which need to be put into loos selections later
        self.water = None
        self.salt =[]
        self.protein = []
        self.psf = None
        self.pdb = None
        self.namd_binary = []
        self.parameters = []
        self.lipid = []
        file = open(FILE) # open file
        # looping into the file
        for line in file.readlines():
            # skip blanks and commonets
            if line.startswith("#") or line.isspace() or len(line) == 0:
                continue
            elif line.upper().startswith("PARAMETERS"):
                (par,filename) = line.split()
                print(filename)
                self.parameters.append(os.path.abspath(filename))

            elif line.upper().startswith("WATER"):
                (a,waters ) = line.split()
                self.water = waters
            elif line.upper().startswith("SALT"):
                (a,salts) = line.split()
                self.salt.append(salts)
            elif line.upper().startswith("PROTEIN"):
                (a, pro) = line.split()
                self.protein.append(pro)
            elif line.upper().startswith("PSF"):
                (a,psf) = line.split()
                self.psf = psf
            elif line.upper().startswith("PDB"):
                (a,pdb) = line.split()
                self.pdb = pdb
            elif line.upper().startswith("LIPID"):
                (a,lipid) = line.split()
                self.lipid.append(lipid)
            elif line.upper().startswith("NAMD"):
                namd = " ".join(line.split()[1:])
                self.namd_binary = namd
            else:
                sys.stderr.write("Unrecognized line type: %s" % line)



        # now we check the the nessary parts are in the config
        if self.psf is None:
            sys.stderr.write("No psf was provided , exiting")
            sys.exit(1)
        if self.pdb is None:
            sys.stderr.write("No psf was provided , exiting")
            sys.exit(1)
        if self.namd_binary is None:
            sys.stderr.write("No namd binary was provided , exiting")
            sys.exit(1)
        # we are going to check for the resname for the loos stuff
        if len(self.parameters) == 0: #pars
            sys.stderr.write("No paramter to run MD exting")
            sys.exit(1)
        if self.water is None: # water
            sys.stderr.write("No solvent resname idetifed ....exiting")
            sys.exit(1)
        else:
            self.water_sel = 'resname == ' + f'\"{self.water}\"'
        if len(self.lipid) == 0: # lipid
            sys.stderr.write("No Lipid resname idetifed ....exiting")
            sys.exit(1)
        else:
            s = 'resname == ' + f'\"{self.lipid[0]}\"'
            for sel in self.lipid[1:]:
                s += f'|| resname == \"{sel}\"'
            self.lipid_sel = s
        if len(self.salt) == 0:# salt
            sys.stderr.write("No Salt resname idetifed" )
        else:
            s = 'resname == ' + f'\"{self.salt[0]}\"'
            for sel in self.salt[1:]:
                s += f'|| resname == \"{sel}\"'
            self.salt_sel = s
        # the protein sels are different b/c we need backbone ,CA and segname for the proein
        if self.protein is None:
            sys.stderr.write("No protein resname idetifed" )
        else:
            self.protein_sel = f'segname == \"{self.protein[0]}\"'
            for pro in self.protein[1:]:
                self.protein_sel += f'|| resname == \"{pro}\"'
            self.backbone = 'backbone'
            self.ca = 'name == \"CA\"'
# testing read Config
