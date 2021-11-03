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

def obtain_info(file,axis_val):
    """ given the file outputted from the  unwrapp code we can obtain the number
    of frames, the number of lipids, and the data which we have to play with to
    get X,Y and Z stuff to use for the given mode that the user gives
    """
    data_file = np.loadtxt(file).astype('float64')
    X_data = data_file[:,::3]
    Y_data = data_file[:,1::3]
    Z_data = data_file[:,2::3]
    n_frames , n_lipids = X_data.shape
    dim = 3
    if axis_val == 0:
        # need all three to do the job
        data = np.array((X_data,Y_data,Z_data))
    elif axis_val == 1:
        # x y martix
        dim = 2
        data = np.array((X_data,Y_data))
    elif axis_val == 2:
        # x z
        dim = 2
        data = np.array((X_data,Z_data))
    elif axis_val == 3:
        # y z
        dim = 2
        data = np.array((Y_data,Z_data))
    elif axis_val == 4:
        # x
        dim = 1
        data = X_data
    elif axis_val == 5:
        # y
        dim = 1
        data = Y_data
    elif axis_val == 6:
        # z
        dim = 1
        data = Z_data
    return n_frames, n_lipids, dim , data

def simple_MSD(data,n_frames,dim):
    """Take the data which is the in format index1=x,y,z index2=the frame we are on
    index3=the atom that we have"""
    lagtimes = np.arange(1,n_frames)
    msd_out = np.zeros((len(lagtimes)+1))
    for lag in lagtimes:
        # find the differences
        diff = data[:,:-lag,:] - data[:,lag:,:]
        mag_sq = 0
        for i in range(dim):
            mag_sq += np.square(diff[i])
        # need to find the mean of of each of the lipids
        msd_out[lag] = mag_sq.mean(axis=0).mean()
    return msd_out

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
    parser.add_argument("--axes_to_use",
                        type=int,default=0,help="This flagg is used to tell the code which axes to do the analyis on x,y,z = 0 x,y = 1 x,z = 2 y ,z= 3 x only =4 y only =5 z only = 6")
    parser.add_argument("--dim_type",type=int,default=0,help="either the simple(0) version of the calacuation of or the fft using tidynamics which needs to be imported(1)")
    args = parser.parse_args()
    n_frames , n_lipids , dim , data = obtain_info(args.text_file,args.axes_to_use)
    if args.dim_type == 0:
        # use the simple version of the code
        msd = simple_MSD(data,n_frames,dim)
    else:
        pass


    for i in range(len(msd)):
        print(i,msd[i])
