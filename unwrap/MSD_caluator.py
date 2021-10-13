# @Author: Kyle Billings <kbillings>
# @Date:   2021-09-25T17:46:36-04:00
# @Email:  krb0073@mix.wvu.edu
# @Filename: MSD_caluator.py
# @Last modified by:   kbillings
# @Last modified time: 2021-10-12T15:49:02-04:00
#!/usr/bin/env python3

"""Calculate the Mean Sqaured Displacmnet of a the lipid_unwrap text file output
"""
import numpy as np
import sys
import argparse
def fullhelp():
    print("""MSD_caluator.py takes the output text file given by lipid_unwrap.py
    and calcuates the Mean Sqaured Displacmnet for each frame of a trejectory.""")


class FullHelp(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        kwargs['nargs'] = 0
        super(FullHelp, self).__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        fullhelp()
        parser.print_help()
        setattr(namespace, self.dest, True)
        parser.exit()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Lipid Mean Squared Displacmnet")
    parser.add_argument("text_file",
                        help='File containg the centers of each lipid')
    #
    args = parser.parse_args()
    # use numpy to read the txt file we made
    centers = np.loadtxt(args.text_file)
    time_zero = centers.T[0] # will be all the lipids at the at time 0 postion
    displacmnet = (centers.T - time_zero.T).T
    displacmnet_sq = displacmnet**2
    MSD_timebased = displacmnet_sq.mean(axis=0)
    header = ""
    header += "#"  +  " ".join([f"{i}" for i in sys.argv]) + '\n'
    header += "#frame MSD"
    print(header)
    string = ""
    for M in range(len(MSD_timebased)):
        string += f"{M} {MSD_timebased[M]}\n"
    print(string)
