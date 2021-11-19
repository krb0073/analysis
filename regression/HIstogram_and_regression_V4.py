import os
import sys
import glob
import random
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import shutil
import statsmodels.api as sm
import loos
import loos.pyloos
import pandas as pd
import numpy as np
import scipy
import scipy.stats
from scipy.sparse import csr_matrix
from math import log ,ceil


def scramble(array,seedNum,axis=-1):
    """ Function to use a random seed to mix up a numpy array"""
    # allows you to mix up a array and stay the same shape
    np.random.seed() # set the random seed
    swapped = array.swapaxes(axis,-1) #call a command to swap axes in np
    n = array.shape[axis] # get the hsape of the array to refreance
    idx = np.random.choice(n, n, replace=False) # choseing valus no doubles
    swapped = swapped[..., idx] #making array
    return swapped.swapaxes(axis, -1)



# class to read a config
class readConfig:
    """ Class will alwo you to read a config for your anaylsis"""
    # reading a config file to run the analysis
    def __init__(self,File):
        self.seedNum = None
        self.dir_path = None
        self.start_index = 0
        self.end_index = -1
        self.stride = 1
        self.data_range = None
        self.num_devs = 2
        self.n_sets = 1
        self.mix = False
        self.n_folds = None
        self.dataloc = None
        self.suffix = 'dat'
        self.systems_list = []
        self.LMAXS = []
        self.LMAX_dev = []
        self.hist = False
        self.clac_type = 'mean'
        self.skew_tol = None
        self.skew_calc = False
        self.nbins = 0
        self.norm_test = None
        self.p_val = 0.05
        self.per_check = False
        self.per_check_val = 0.5
        self.max_features = 2
        self.diff_check = False
        self.diff_tol = 0.8
        # open the file read only
        file = open(File,'r')
        for line in file.readlines():
            if line.startswith("#") or line.isspace() or len(line) == 0 or "@" in line:
                continue
            elif line.upper().startswith("SEED"):
                # find the seed number
                self.seedNum = int(line.split()[1])
            elif line.upper().startswith("MIX"):
                self.mix = True
            elif line.upper().startswith('OUTPUTDIR'):
                self.dir_path = line.split()[1]
            elif line.upper().startswith("DATA_RANGE"):
                self.data_range = line.split()[1:]
            elif line.upper().startswith("SKEW_CALC_VALS"):
                # the user wants to use the skew to find the method of clac_type
                self.skew_calc = True
            elif line.upper().startswith("NUM_DEVATIONS"):
                self.num_devs = int(line.split()[1])
            elif line.upper().startswith("USE_HIST"):
                self.hist = True
            elif line.upper().startswith("N_SETS"):
                self.n_sets = int(line.split()[1])
            elif line.upper().startswith("N_FOLDS"):
                self.n_folds = int(line.split()[1])
            elif line.upper().startswith("DATA_LOCATION"):
                self.dataloc = line.split()[1]
            elif line.upper().startswith("SUFFIX"):
                self.suffix = line.split()[1]
            elif line.upper().startswith("SYSTEM_PREFIXES"):
                self.systems_list = line.split()[1:]
            elif line.upper().startswith("LMAXS"):
                lvals = line.split()[1:]
                for l in lvals:
                    self.LMAXS.append(float(l))
            elif line.upper().startswith("DEVS"):
                ldevs = line.split()[1:]
                for l in ldevs:
                    self.LMAX_dev.append(float(l))
            elif line.upper().startswith("CALC_TYPE"):
                self.clac_type = line.split()[1]
            elif line.upper().startswith("SKEW_TOLERANCE"):
                self.skew_tol = float(line.split()[1])
            elif line.upper().startswith("NORM_TEST"):
                self.norm_test = int(line.split()[1])
            elif line.upper().startswith("P_VALUE"):
                self.p_val = float(line.split()[1])
            elif line.upper().startswith("FILTER_CORR"):
                self.per_check = True
                if len(line.split()) == 2:
                    self.per_check_val = line.split()[1]
            elif line.upper().startswith("NUM_FEATURES"):
                self.max_features = int(line.split()[1])
            elif line.upper().startswith("DIFF_TOL"):
                self.diff_tol = float(line.split()[1])

        # now do check to make sure that what is needeed is there
        if self.dir_path == None:
            # need to know where all the files for the regression are
            print("Missing the path for the regression data ... will use currnet working dir")
        if self.mix: # if this is ture but we do not have a seed exit
            if self.seedNum == None:
                print("Data randomization was requested but no seed number was given ... exititng")
                sys.exit(1)
        if self.data_range != None:
            d = len(self.data_range)
            if d == 2:
                self.start_index,self.end_index = [int(x) for x in self.data_range]
            elif d == 3:
                self.start_index,self.end_index,self.stride = [int(x) for x in self.data_range]
            else:
                print("data range is given but nubmer of items in not lenght of 2 or 3...exiting")
                sys.exit(1)
        if len(self.LMAXS) == 0:
            print("The vaues for Lmax are not given...exiting")
            sys.exit(1)
        # check for if we are doing the histogram method
        if self.hist == True:
            # check if they are the same legnht of the lmax values and num_devs
            if len(self.LMAXS) != len(self.LMAX_dev):
                print("the vaues for the LMAX and the devations given are not the same size ..exiting")
                sys.exit(1)
            # we can now find the number of bins using the number of the dev to
            # calcute the the number of bins 2* n_devs +1
            if self.num_devs != 0:
                self.nbins = 2 * self.num_devs + 1
            else:
                print("the number of standard devations wanted can not be 0...exit")
                sys.exit(1)

# class to setup the data for the regression
class Regression_setup(readConfig):
    "intinal setup for the regression"

    def __init__(self,*args, **kwargs):
        self.measures = []
        self.final_sets = None
        self.nbins = 0
        super().__init__(*args,**kwargs)
    def RandomSeed(self):
        """ make the dir for the seed we are running  and randomize the state"""
        random.seed(self.seedNum) # set the random seed
        # now we build a DIR for that seed in the location ran in
        if self.dir_path == None:
            self.dir_path = os.getcwd()
        # checking if the the given path will work
        try:
            os.path.isdir(self.dir_path)
        except:
            print("the output dir location is not vaild ...exiting")
            sys.exit()
        self.dir_path = self.dir_path + "/" + f"Regression_results_for_seed_{self.seedNum}"
        # checking for the direcotry
        try:
            os.mkdir(self.dir_path)
        except:
            print(f"Dir {self.dir_path} exsist ..removing and makeing a new one")
            shutil.rmtree(self.dir_path)
            os.mkdir(self.dir_path)

    def obtainData(self,file_to_use):
        """ for a singel file we will get out the the data for that files"""
        data = np.loadtxt(file_to_use)
        return data[self.start_index:self.end_index:self.stride]

    def checkSkew(self,data):
        """ check if the distrabution given is systmerics and passes a given"""
        skew = scipy.stats.skew(data,bias=False)
        # if the user has a skew tol we will see if the swek given is with in a
        # range then output fail
        # data here will jsut be the values for all of the stuff
        #https://pyshark.com/skewness-in-python/
        if self.skew_calc: # check if we want to use the skew to figure out the
        # metod to give you the higest values
            if skew >= 0:
                self.clac_type = 'median'
            else:
                self.clac_type = 'mean'
        if self.skew_tol != None:
            # we will then check if it too large either way
            if abs(skew) > self.skew_tol:
                return skew ,1
            else:
                return skew, 0
        else:
            return skew , 0

    def histogramData(self,data):
        """ Given a read txt file use np histogram to collect the and the frame
        indexes that belong that bin """
        # find the histogram of ranges to use in the data
        _, bin_edges = np.histogram(data[:,1],bins=self.nbins,density=True)
        # use digirize to find the index that each bin belongs to
        index_of_bins = np.digitize(data[:,1],bin_edges)
        # sort the data into binns filled with indexes
        empty = []
        # making a empty list to sort frames into
        index_frames = []
        for edge in range(len(bin_edges) - 1):
            frame_numbers = []
            for frame in data:
                if frame[1] > bin_edges[edge] and frame[1] <= bin_edges[edge + 1]:
                    # reutrn a list of the indexes of the thing are
                    # this means in the all file if you have one we can not have frame numbers
                    # repeat
                    frame_numbers.append(frame[1])
            index_frames.append(frame_numbers)
        return index_frames

    def splitLmaxByStandrdDev(self,lmax,devs):
        """ this fuctinon will take a given Lmax values and make a list of Lmax values"""
        absorbances = []
        for val in range(-self.num_devs,self.num_devs+1):
            absorbances.append(lmax + devs * val)
        return absorbances # this is all of the values of absorbace for the
          # given lmax values

    def subDivideData(self,list_of_frames):
        """ Take a given data frame and split into a number of different groups"""
        np.random.seed(self.seedNum) # sets the seed
        # if requested mix up the frames in the list
        if self.mix != False:
            # mix up the indexes that we want
            list_of_frames = scramble(np.array(list_of_frames),self.seedNum)
        if self.n_sets != 1:
            list_of_frames = np.array_split(list_of_frames,self.n_sets)
        return list_of_frames

    def checkNormality(self,data):
        """
        test the dsn to see if the dsn is noraml of not
        """
        if self.norm_test == 0:
            return scipy.stats.jarque_bera(data[:,1])
        elif self.norm_test == 1:
            return scipy.stats.kstest(data[:,1],cdf='norm')
        elif self.norm_test == 2:
            return scipy.stats.shapiro(data[:,1])
        elif self.norm_test > 2:
            print("Normtest is deisred but values in larger than 2 ... exiting")
            sys.exit(1)

    def passFailNorm(self,data):
        """ run the test for the give norm if the p value is too big output a one"""
        _, p = self.checkNormality(data)
        if self.norm_test == 0 or self.norm_test == 2:
            # a value bigger than a p vaule is then a nomral disn
            if p > self.p_val:
                test = 0
            else:
                test = 1
        elif self.norm_test == 1:
            # if the value is bigger we have a normal for this boy
            if p < self.p_val:
                test = 0
            else:
                test = 1
        return test

    def makeAllDataFrames(self):
        """
        using each of the given prefxies and the split stff comple all datasets
        """
        self.RandomSeed()
        # step 1: find all the files that will be used to do the run
        # use the path to the output

        file_loc = os.path.abspath(self.dataloc)

        # we have a list of prefixes to use in the system
        # top level find the parmaeters that will be used to run

        # check to see if the dev list in empty
        if len(self.LMAX_dev) != 0:
            # this tell us that we will do the user method of histgoram binning
            Hist_method = True
            tot = len(self.splitLmaxByStandrdDev(self.LMAXS[0],self.LMAX_dev[0]))
            self.nbins = len(self.splitLmaxByStandrdDev(self.LMAXS[0],self.LMAX_dev[0]))
            lmax_list = []
            for lmax,my_devs in zip(self.LMAXS,self.LMAX_dev):
                all_vales = self.splitLmaxByStandrdDev(lmax,my_devs)
                tot += len(all_vales)
                for v in all_vales:
                    lmax_list.append(v)
            # this new list is all the LMAX values that we have
            self.LMAXS = lmax_list
            print(lmax_list)

        else:
            Hist_method = False
            # only need to worry about the lmax values
        tot = len(self.LMAXS)
        # find the measures that they will all use
        #check if we are looking for skew
        filenames = glob.glob(file_loc + f"/{self.systems_list[0]}*{self.suffix}")
        for fline in filenames:
            self.measures.append(fline.split("/")[-1].split("_")[-1].split(".")[0])
        # use the features and the total number of items to bluild a blank np
        # array then given the number of sets we want add that to a master list
        blank_np_array = np.zeros((tot,len(self.measures)+1))
        # check if we are doing more than one set make a list of list
        all_data = [blank_np_array for i in range(self.n_sets)]
        for a in all_data: # each sets
            cnt = 0
            for l in self.LMAXS:
                a[cnt][0] = l
                cnt += 1
        if self.n_sets == 1:
            all_data = all_data[0]
        skew_array = np.zeros((len(self.LMAXS),len(self.measures)))
        skew_tol_array = np.zeros((len(self.LMAXS),len(self.measures)))
        norm_test_array = np.zeros((len(self.LMAXS),len(self.measures)))
        row = 0
        sk_cnt = 0
        for i in self.systems_list:
            filenames = glob.glob(file_loc + f"/{i}*{self.suffix}")
            # print to the user all the flies that we are loading in incase we
            # we need them
            print(f"for Prefix {i} the follwing file are being used:")
            print("\n".join(filenames))
            if not Hist_method: # no new data neing added
                if self.n_sets == 1: # if we only want one set of all the data
                    col = 0
                    for file in filenames:
                    # read in the data as a varaibel that we will kill later
                        data = self.obtainData(file)
                    # find the skew of the data
                        skew_val , test = self.checkSkew(data[:,1])
                        # fill the arrays
                        skew_array[sk_cnt][col] = skew_val
                        skew_tol_array[sk_cnt][col] = test
                        if self.norm_test != None:
                            norm_test_array[sk_cnt][col] = self.passFailNorm(data)
                    # we don not need to hsitograme the data
                    # we do not need to split the set up more
                        if self.clac_type == 'mean':
                            all_data[row][col+1] = np.mean(data[:,1])
                        elif self.clac_type == 'median':
                            all_data[row][col+1] = np.median(data[:,1])
                        col += 1
                else:
                    # we need to run the subset function but not histogram
                    col = 0
                    for file in filenames:
                        # load the data
                        data = self.obtainData(file)
                        # now split the data into the sets that we want to do
                        skew_val , test = self.checkSkew(data[:,1])
                        # fill the arrays
                        skew_array[sk_cnt][col] = skew_val
                        skew_tol_array[sk_cnt][col] = test
                        if self.norm_test != None:
                            norm_test_array[sk_cnt][col] = self.passFailNorm(data)
                        my_sets = self.subDivideData(data[:,1]) # give it the frames
                        # we parsred the data now fill the boys
                        for s in range(len(my_sets)):
                            if self.clac_type == 'mean':
                                all_data[s][row][col+1] = np.mean(my_sets[s])
                            elif self.clac_type == 'median':
                                all_data[s][row][col+1] = np.median(my_sets[s])
                        col += 1
                row += 1
                sk_cnt += 1
            elif Hist_method:
                if self.n_sets == 1:
                    col = 0
                    # doing the histogram method but not makeing more than one set of the stuff
                    for file in filenames:
                        # read the data
                        data = self.obtainData(file)
                        # use the histogram vaules of the code
                        hist = self.histogramData(data)
                        # we retrun bins filled with each histograms values
                        skew_val , test = self.checkSkew(data[:,1])
                        if self.norm_test != None:
                            norm_test_array[sk_cnt][col] = self.passFailNorm(data)
                        # fill the arrays
                        skew_array[sk_cnt][col] = skew_val
                        skew_tol_array[sk_cnt][col] = test
                        for h in range(len(hist)):
                            temp = np.array(hist[h])
                            if len(hist[h]) != 0:
                                if self.clac_type == 'mean':
                                    all_data[row+h][col+1] = np.mean(temp)
                                elif self.clac_type == 'median':
                                    all_data[row+h][col+1] = np.median(temp)
                            else:
                                all_data[row+h][col+1] = np.nan # avoid the run time error
                        col += 1
                else:
                    col = 0
                    # this one we ill do splits if the users want
                    for file in filenames:
                        # read the data
                        data = self.obtainData(file)
                        # use the histogram vaules of the code
                        hist = self.histogramData(data)
                        # now we will split the values given from hist
                        skew_val , test = self.checkSkew(data[:,1])
                        if self.norm_test != None:
                            norm_test_array[sk_cnt][col] = self.passFailNorm(data)
                        # fill the arrays
                        skew_array[sk_cnt][col] = skew_val
                        skew_tol_array[sk_cnt][col] = test
                        for h in range(len(hist)):
                            # do the sub sets
                            my_sets = self.subDivideData(hist[h])
                            for s in range(len(my_sets)):
                                if self.clac_type == 'mean':
                                    all_data[s][row+h][col+1] = np.mean(my_sets[s])
                                elif self.clac_type == 'median':
                                    all_data[s][row+h][col+1] = np.median(my_sets[s])
                row += len(hist) # add in the extra to the things
                sk_cnt += 1
        # save the item into class to be inheratned later
        full_set = all_data
        #print(full_set)
        if self.hist:
            print(np.isnan(full_set[0]).any(axis=0))
            full_set = [x[:, ~np.isnan(x).any(axis=0)] for x in full_set]
        if self.n_sets == 1:
            full_set = []
            full_set.append(all_data)
        # add values in the columns to the set
        tot_tol_skew = np.sum(skew_tol_array,axis=0)[np.newaxis]
        # take the skew set an make it into a pandas data frame
        tot_tol_skew = pd.DataFrame(tot_tol_skew,columns=self.measures)
        # find the column of the data frmae that are greater than z
        failed_skew = tot_tol_skew[tot_tol_skew > 0].columns
        # add l max to the list of measure at index 0
        m = ["LMAXS"]
        for i in self.measures:
            m.append(i)
        # make all the data frame
        #self.measures = np.array(m)[np.newaxis]
        full_set = [pd.DataFrame(x,columns=m) for x in full_set]
        if self.skew_tol != None:
            # add values in the columns to the set
            tot_tol_skew = np.sum(skew_tol_array,axis=0)[np.newaxis]
            # take the skew set an make it into a pandas data frame
            tot_tol_skew = pd.DataFrame(tot_tol_skew,columns=m[1:])
            # find the column of the data frmae that are greater than z
            failed_skew = tot_tol_skew[tot_tol_skew > 0].columns
            temp_set = []
            for df in full_set:
                for index in failed_skew:
                    df.drop(index,axis=1)
        if self.norm_test != None:
            # add values in the columns to the set
            tot_tol_norm = np.sum(skew_tol_array,axis=0)[np.newaxis]
            # take the skew set an make it into a pandas data frame
            tot_tol_norm = pd.DataFrame(norm_test_array,columns=m[1:])
            # find the column of the data frmae that are greater than z
            failed_norm = tot_tol_skew[tot_tol_skew > 0].columns
            # need to to trim the set if we do the thing
            if self.skew_tol != None:
                temp = []
                for i in failed_norm:
                    if i not in failed_skew:
                        temp.append(i)
                failed_norm = temp
            temp_set = []
            for df in full_set:
                for index in failed_norm:
                    df.drop(index,axis=1)
        self.final_sets = full_set

# class for doing regression
class lamxRegression(Regression_setup):
    def __init__(self,*args, **kwargs):
        self.model_index = 0
        super().__init__(*args,**kwargs)

    def randomSet(self):
        """ picks a random model to use as the base """
        if self.n_sets != 1:
            self.model_index = random.randint(0,self.n_folds -1)

    def collectNSubset(self,var_num):
        """ make all the combos of subsets that are possable """
        return [x for x in itertools.combinations(self.measures,var_num) ]

    def pearsonTolCheck(self,var_num):
        """ check if the pearson correation between the values are too high"""
        print(f'The following combonation are purged acoording to correlations:')
        cols = self.final_sets[0].columns
        # find the correlations
        corrs = self.final_sets[self.model_index].drop("LMAXS",1).corr()
        combos = self.collectNSubset(var_num)
        # we have a given number of combos that we ahve to look at for the correaltions
        bad_string = ""
        for i in self.collectNSubset(var_num): # for subset of combos
            column_names = list(i) # make a list
            # get all the values of the correation from the matrix
            subset_corr = corrs.loc[column_names,column_names].values.flatten()
            check = 1
            for corr in subset_corr:
                if corr != 1:
                    if corr > self.per_check_val:
                        check = 0
                if check == 0: # at some point the item is bad
                # add the bad combos to the string we are keeping
                 print(f"combos :{' '.join(i)} removed")
                 combos.remove(i)
        return combos

    def diffBetweenSets(self):
        """ simple check given a tolerance to check if the data in each of the split
         frames is less than a tolerance"""
        if self.diff_check:
            # check if the set is more than one item
            if self.n_sets > 1:
                model_sets = self.final_sets[self.model_set]
                all = np.zeros((st.shape))
                for sets in self.final_sets[1:]:
                    ans =  model_sets - sets
                    print(ans[ans<self.diff_tol])

# making a main fuctuion
def main(config):
    reg = lamxRegression(config)
    reg.makeAllDataFrames()
    reg.diffBetweenSets()
    combos = []
    if reg.per_check:
        for i in range(1,reg.max_features):
            combos.exstend(reg.pearsonTolCheck(i))
    print(combos)

if __name__ == '__main__':
    config = sys.argv[1]
    main(config)
