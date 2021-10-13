# @Author: Kyle Billings <kbillings>
# @Date:   2020-12-16T12:57:30-05:00
# @Email:  krb0073@mix.wvu.edu
# @Filename: V2_regression.py
# @Last modified by:   kbillings
# @Last modified time: 2021-01-21T17:54:58-05:00
# found this scarmbel code at https://stackoverflow.com/questions/36272992/numpy-random-shuffle-by-row-independently
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
from math import log ,ceil
def scramble(array,seedNum,axis=-1):
    import numpy as np
    np.random.seed()
    swapped = array.swapaxes(axis,-1)
    n = array.shape[axis]
    idx = np.random.choice(n, n, replace=False)
    swapped = swapped[..., idx]
    return swapped.swapaxes(axis, -1)
# making OLS sub class of Regression setup
class readConfig:
    # reading a config file to run the analysis
    def __init__(self,File):
        self.data_range = []
        self.Lmax = []
        self.num_var = 1
        self.dataloc = None
        self.seedNum = None
        self.sim_list = []
        self.suffix = None
        self.n_sets = 1
        self.path_auc = None
        self.keep_num = None
        self.systems_list = []
        self.data_tol = 0
        self.purge_cutoff = None
        self.metric = "MSE"
        self.untouched_num = None
        self.method = None
        self.flipp = False
        self.log = 'output.log'
        # open the config
        file = open(File)
        # loop into every line
        for line in file.readlines():
            # skip blanks and commonets
            if line.startswith("#") or line.isspace() or len(line) == 0 or "@" in line:
                continue
            elif line.upper().startswith("DATA_RANGE"):
                # check line of the line without the name of the the line
                checker = len(line.split()[1:]) # all but the name
                self.data_range.append(float(line.split()[1]))
                self.data_range.append(float(line.split()[2]))
                if checker > 2:
                    # we have a number to skip by so we will append that in as well
                    self.data_range.append(float(line.split()[3]))
            elif line.upper().startswith("LMAX"):
                vals = line.split()[1:] # this will get one to the end w/o lmax
                for v in vals:
                    self.Lmax.append(int(v))
            elif line.upper().startswith("DIR_NAME"):
                (delim,path) = line.split()
                self.dirloc = path
            elif line.upper().startswith("MAX_VAR"):
                (delim,var) = line.split()
                self.num_var = int(var)
            elif line.upper().startswith("SEED"):
                (delim,seed) = line.split()
                self.seedNum = int(seed)
            elif line.upper().startswith("FILE_PREFIXES"):
                 for pre in line.split()[1:]: #loop from the frist prefix to the end
                    self.sim_list.append(pre)
            elif line.upper().startswith("SUFFIX"):
                (delim,ender) = line.split()
                self.suffix = ender
            elif line.upper().startswith("NFOLDS"):
                (delim,folds) = line.split()
                self.n_sets = int(folds) # to do intail spilt
            elif line.upper().startswith("AUC_PATH"):
                (delim,auc_path) = line.split()
                self.path_auc = auc_path
            elif line.upper().startswith("KEEP_NUM"):
                (delim,keep) = line.split()
                self.keep_num = int(keep)
            elif line.upper().startswith("SYSTEMS"):
                for system in line.split()[1:]:
                    self.systems_list.append(system)
            elif line.upper().startswith("TOLERANCE"):
                (delim,tol) = line.split()
                self.data_tol = float(tol)
            elif line.upper().startswith("PURGE_CUTOFF"):
                (delim,purge) = line.split()
                self.purge_cutoff = float(purge)
            elif line.upper().startswith("SCORING_METRIC"):
                (delim,score) = line.split()
                self.metric = score
            elif line.upper().startswith("NUMBER_LEFT_OUT"):
                (delim,out) = line.split()
                self.untouched_num = int(out)
            elif line.upper().startswith("METHOD"):
                delim,met = line.split()
                self.method = met
            elif line.upper().startswith("FLIPP"):
                (delim,fp) = line.split()
                if fp == "True":
                    self.flipp = True
            elif line.upper().startswith("LOG"):
                (delim,log) = line.split()
                self.log = log
        # now we can check for the nessarcy conditons
        if len(self.data_range) <= 1: # if the data range is no at least 2
            print("data range is not good ...exiting\n")
            self.config_error_message()
        if self.num_var == None:
            print("missing the total of number of varables to try (e.i. 1 ,2 ) ...exiting")
        if len(self.Lmax) == 0:
            print("No Lmax list given ... exiting \n")
            self.config_error_message()
        if len(self.sim_list) == 0:
            print("there are no file prefixes given ... exiting\n")
            self.config_error_message()
        if len(self.systems_list) == 0:
            print("the systems list is not filled ... exiting \n")
            self.config_error_message()
        if self.dirloc == None:
            print("need path to where data is ... exiting \n")
            self.config_error_message()
        # these three must be the same legnth
        if len(self.Lmax) == len(self.sim_list):
            if len(self.sim_list) == len(self.systems_list):
                pass
            else:
                print("Lenght of systems list is not the same as lmax, or prefix list... exiting \n")
                self.config_error_message()
        else:
            print("Lenght of Lmax and prefix list not equal... exiting \n")

        if self.seedNum == None:
            print("No random seed given ... exiting\n")
            self.config_error_message()
        if self.method != "LOOCV" and self.method != "KFOLD":
            print("Mehod Type is not correct/not given ... exting\n")
            self.config_error_message()
        # working with the optinal stuff will not cause (in most cases the failing)
        if self.data_tol == None:
            print("No tolerance is given all data being used")
            self.tolerance = 0
        if self.purge_cutoff == None:
            print("correaltions are not being trimmed setting pruge to defule ")
            self.purge_cutoff = 0
        if self.n_sets == None:
            print("the data is being treated as one set")
            # now if they have set the parameter for the
            if self.keep_num != None:
                print("the untouched data set is not for uaing one big set ... esiting\n")
                sys.exit(1)
            else:
                self.keep_num = 5
        if self.metric == "MSE":
            print("using defult socring system MSE")
        if self.method == "LOOCV" or self.method == "KFOLD": # if we good here
            if self.keep_num != None or self.untouched_num != None:
                if self.n_sets == None: # this is if the user wanted to keep with spliting up
                    print("cant split a set that is one set ...exiting")
                    sys.exit(1)
        else:
            print(f"the Line: {line} is not reconized ...exiting\n")
    def config_error_message(self):
        """ this will proctor every time a error in the config is there"""
        print("One or more items was inncrorrect in the confing ...printing config example")
        config = """
# Kyle Billings 1/12/21
# linear regression confing example
####### required paramters ###############
data_range 0 55894

seed 10000000000

# max number of observations to use
max_var 3
# for the data range if you want to only count ever 10 frames you can put tem after for skip
#data_range 0 55894 10

Lmax 570 590 550 412 560 640

#this will list the file prefixes to look for
FILE_PREFIXES bR_all K_all L_all M_all N_all O_all

# provide the list of what to call the indexes
systems bR K L M N O

#NOTE: inorder for the program to run Lamx, systems and FILE_PREFIXES must tbe the same lenght
#Note 2: the program WILL NOT check to see if the order for each is correct. thats your job.

# there are two methods that are supported at this time

# this will only use one of the sets and do leave one out crossvaialdtion for modeling
#method LOOCV

method KFOLD

####### paramters that can be added ########################

# setting a tolerance in the file will ensire that the data point of an obseravtion will be
## spread out at leat that much e.i if TOLERANCE is set to 0.5 all values must be greater than 0.5 from each other

#TOLERANCE 0.5

# if the purge_cutoff is on then the values with preason coerrleations greater than it will be removed
## the min values is 0.01

#puge_cutoff 0.5

# if Nfolds is on the data will be split into that many sets before the medain values are taken

Nfolds 10

# change the output dir loaction using

#DIR_NAME /path/to/dir

# if the suffix of your data files is not dat use this to cahnge it

#suffix txt

# to include the area under the curve  use this setting

#AUC /path/to/auc/dir

# to cahnge the number of regression graphs for each number of varaibles use

KEEP_NUM 5

# the defult scoring method to campre between line from the same number of var is MSE
# but BIC AIC and RSS are all supproted

#SCORING_METRIC BIC

# when doing the 10% validation method we set the flipp to ture by

Flipp True

# if you want to do a tran test split and not use a set of data for modeling then

#NUMBER_LEFT_OUT 5
"""

        print(config)
        sys.exit(1)
class Regression_setup:

    def __init__(self,start_index,end_index,LMAXS,Data_location,seedNum,skip=0,LOG='output.log'):
        self.start_index = start_index
        self.end_index = end_index -1
        self.LMAXS = LMAXS
        self.Data_location = Data_location
        self.seedNum = seedNum
        self.skip = skip
        self.log = LOG
    def logger(self,words):
        file = open(self.log,'a') # will append to the file
        line = words + '\n'
        file.write(line)
        file.close()
    def RandomSeed(self,path=None):
        """ make the dir for the seed we are running  and randomize the state"""
        random.seed(self.seedNum) # set the random seed
        # now we build a DIR for that seed in the location ran in
        if path == None:
            path = os.getcwd()
        # checking if the the given path will work
        try:
            os.path.isdir(path)
        except:
            print("the output dir location is not vaild ...exiting")
            sys.exit()
        self.dir_path = path + "/" + f"Regression_results_for_seed_{self.seedNum}"
        # update the log with the path
        self.log = self.dir_path + "/" + self.log
        # checking for the direcotry
        try:
            os.mkdir(self.dir_path)
        except:
            print(f"Dir {self.dir_path} exsist ..removing and makeing a new one")
            shutil.rmtree(self.dir_path)
            os.mkdir(self.dir_path)
    def Make_data_frames(self,prefix,suffix='dat',skip=0):
        """ for a given state make arrays all that data assume the files wiil be large """
        # TODO: move the RMSF under the curve with the rest of the data and fix format
        file_loc = os.path.abspath(self.Data_location) # gets full path of data
        print(file_loc +f"/{prefix}*.{suffix}")
        filenames = glob.glob(file_loc +f"/{prefix}*.{suffix}") # list all the prefix*.suffix files to use
        n_cols = len(filenames)
        n_rows = self.start_index
        self.measures = []
        for f in filenames:
            ans = f.split("/")[-1].split("_")[-1].split(".")[0]
            self.measures.append(ans)
        ## i.e. if i want bR*.dat I could get all of those files in a list
        print(f"The follewing file will be used to run linear regression of {prefix}:")
        print("\n".join(filenames)) # lists all file names being used
        # to help with memory I am going to use np.memmap https://numpy.org/doc/stable/reference/generated/numpy.memmap.html
        loc = 0
        for file in filenames: # loop into files
            with open(file) as txt_file:
                temp_data = [] # store the data of the file in question
                line_num = 0 # counter of the nuber or line
                for line in txt_file: # loop into each line
                    if line_num >= self.start_index and line_num <= self.end_index: # checkin if we are at the right index
                        if line.startswith("#") or line.isspace() or len(line) == 0: # no spcaes or #
                            continue # jsut ingoring thos lines
                        else:
                            frame , value = line.split() # split into two variables
                            temp_data.append(float(value)) # append measurment as a float
                    if skip != 0:
                        line_num += skip
                    else:
                        line_num += 1
                if file == filenames[0]:
                    data = np.array(temp_data)
                else:
                    data = np.row_stack((data,temp_data))
                del temp_data # free this from the memory
        print(f"done reading {prefix} for regression")
        return data #, all_data_medains , all_data_mean , all_data_std , all_data_stderr
    def Create_final_dataset_splits(self,list_of_np_arrays,number_of_datasets):
        """splits thae data into n groups after minding the meadian values """
        np.random.seed(self.seedNum)
        self.n_folds = number_of_datasets
        all_data = list_of_np_arrays
        set_1 = all_data[0]
        set_1 = scramble(set_1,self.seedNum)
        try:
            n_temp_array = np.hsplit(set_1,self.n_folds)
        except:
            print(f"The number of sets wanted is not disiavble by {self.n_folds}")
            sys.exit(1)
        # using the first to build the array
        n_array = []
        # moving the pick array to here
        self.model_index = 0
        if self.n_folds != 1:
            self.model_index = random.randint(0,self.n_folds -1)
            print(f"n_fold is lager than 1 using set  index {self.model_index}")
            self.logger(f"n_fold is lager than 1 using set  index {self.model_index}")
        for sets in range(len(n_temp_array)):
            n_array.append(np.median(n_temp_array[sets],axis=1))
        # now  for the rest of the sets we put it in the n_array
        for d2 in all_data[1:]:
            temp_array = scramble(d2,self.seedNum)
            d2_array = np.hsplit(temp_array,self.n_folds) # spilt into the sets we want
            for df in range(len(n_array)):
                meds = np.median(d2_array[df],axis=1)
                n_array[df] = np.row_stack((n_array[df],meds))
        self.med_array = n_array # set a varaibles for the medains of each set
    def plot_distrabution(self,list_of_np_arrays,number_of_datasets,systems_list):
        np.random.seed(self.seedNum)
        self.n_folds = number_of_datasets
        all_data = list_of_np_arrays
        set_1 = all_data[0] # grabbing the things
        set_1 = scramble(set_1,self.seedNum)
        color_list = ['black','blue','red','darkgreen','goldenrod','indigo','deeppink','darkoramge','grey','tan','aqua','plum','magenta','crimson','lawngreen','saddelbrown','lavender','gainsbro','teal','deepskyblue','pink','palegreen','oldlace']
        try:
            n_temp_array = np.hsplit(set_1,self.n_folds)[self.model_index]
        except:
            print(f"The number of sets wanted is not disiavble by {self.n_folds}")
            sys.exit(1)
        h =np.apply_along_axis(lambda a: np.histogram(a,bins=10,density=True)[0],1,n_temp_array)
        b =np.apply_along_axis(lambda a: np.histogram(a,bins=10,density=True)[1],1,n_temp_array)
        hists = [x for x in h]
        bins = [x for x in b]
        for data in all_data[1:]:
            data = scramble(data,self.seedNum)
            data = np.hsplit(data,self.n_folds)[self.model_index]
            h =np.apply_along_axis(lambda a: np.histogram(a,bins=10,density=True)[0],1,data)
            b =np.apply_along_axis(lambda a: np.histogram(a,bins=10,density=True)[1],1,data)
            for item in range(len(h)):
                hists[item] = np.row_stack((hists[item],h[item]))
                bins[item] = np.row_stack((bins[item],b[item]))
        # now we can loop into the each array
        plt.rc('xtick',labelsize=6)
        plt.rc('ytick',labelsize=6)
        fig = plt.figure(figsize=(20,20))
        cnt = 0
        cols = 5
        rows = len(bins)/5
        if rows != len(bins) // 5:
            rows = len(bins) // 5 + 1 # adds an extra row when needed
        for H,B in zip(hists,bins): # all varaibles
            name = self.measures[cnt]
            cnt += 1
            alpha = 1
            ax = fig.add_subplot(rows,cols,cnt) # setup the plotter for the things
            ax.set_xlabel('angel',fontsize=6)
            ax.set_ylabel('probablity',fontsize=6)
            for item in range(len(systems_list)): # all systems
                system = systems_list[item]
                h_data = H[item]
                b_data = B[item]
                centers = []
                for i in range(len(b_data)-1):
                    cent = (b_data[i] + b_data[i+1])/2
                    centers.append(cent)
                ax.plot(centers,h_data,c=color_list[item],label=systems_list[item])
            ax.legend(loc='best',prop={"size":6}).set_title(name,prop={'size':6})
        fig.tight_layout()
        fig.savefig(self.dir_path+ '/'+'metrics_histograms.png',dpi=400)
    def get_AUC(self,AUC_path,order_of_systems):
        """Add is the median vaules for the system """
        # the path with the RMSF data will only have the data we want
        # the dir must be run with my pacakge to work
        full_path = os.path.abspath(AUC_path)
        for run in range(1,self.n_folds +1):
            ring = []
            auc =  []
            for system in order_of_systems:
                filename = full_path + f"/{system}_{self.seedNum}_set_{run}.dat"
                #print(filename)
                for line in open(filename,'r').readlines():
                    if line.upper().startswith("AUC"):
                        auc.append(line.split()[1])
                    elif line.upper().startswith("RING_AUC"):
                        ring.append(line.split()[1])
            ring = np.array(ring)
            auc = np.array(auc)
            AUC = np.column_stack((auc,ring))
            self.med_array[run-1] = np.column_stack((self.med_array[run-1],AUC))
        self.measures.append('AUC')
        self.measures.append('ring')
# making OLS sub class of Regression setup
class OLS_regression(Regression_setup):
    """This class is for doing ordinay least squares on data. it is part of set up """
    def __init__(self,start_index,end_index,LMAXS,Data_location,seedNum,max_variables_num,keep_num=5):
        self.max_var = max_variables_num # number of pars to use
        self.keep = keep_num
        super().__init__(start_index,end_index,LMAXS,Data_location,seedNum) # allows the regression to get all things of Regression_setup
    def pandafy(self,systems): # this works
        """turn data that we found in setup into an n lenght array of data frames """
        self.measures.insert(0,'Lmax') # put lmax infront of the array
        self.data_frames = []
        for array in self.med_array:
            self.LMAXS = np.array(self.LMAXS) # lmax into np array
            array = np.column_stack((self.LMAXS,array))
            array = pd.DataFrame(data=array , index=systems,columns=self.measures,dtype='float64')
            self.data_frames.append(array)
        self.logger("priting values from the final model ")
        self.logger(self.data_frames[self.model_index])
        del self.measures # do not need these no more
        del self.LMAXS
        del self.med_array
    def best_subset_runer(self,var_num):
        """this function will genrate a list of varaibles"""
        cols = self.data_frames[self.model_index].drop("Lmax",axis=1).columns # list all open columns to do regression
        return [x for x in itertools.combinations(cols,var_num)]
    def peasron_correaltion_heatmap(self):
        non_lmax_data = self.data_frames[self.model_index].drop('Lmax',1)
        filename = f"heat_Map_{self.seedNum}_for_final_set.png"
        fig = plt.figure(figsize=(12,8))
        heatmap = sns.heatmap(non_lmax_data.corr().abs(),cmap='coolwarm' ,vmin=0, vmax=1, annot=False)
        fig.savefig(self.dir_path+ '/'+filename,dpi=400)
        print("heat map image made")
    def tolerance_cheack(self,tolerance):
        """If woried about how close that vlues are togther given a tolerance we trim the data set. looks for vaules
        too close only cause the other way is better for what we are doing
        """
        Bad_cols = []
        for col in self.data_frames[self.model_index].columns:
            tracker = 'good'
            temp = (self.data_frames[self.model_index][col].values) # get the values of the columns
            # substract each values from the rest of the arrays. if the difference of the values are not in
            ## tolerance range cahnge traker to bad if one vaule is too close then the whole thing is left out
            for val in temp:
                test = abs(temp-val)
                rest = []
                for num in test:
                    if num != 0:
                        rest.append(num)
                rest = np.array(rest)
                clean = np.delete(rest,np.argwhere((rest < tolerance)))
                if len(rest) != len(clean):
                    tracker = 'bad'
            if tracker == 'bad':
                Bad_cols.append(col)
        for frame in range(0,len(self.data_frames)):
            #for bad in Bad_cols:
            self.data_frames[frame] = self.data_frames[frame].drop(columns=Bad_cols,axis=1)
        print(f"found {len(Bad_cols)} parameters that where the points were < {tolerance}")
        print(f"removed:")
        print("\n".join(Bad_cols))
    def purge_corr_too_high(self,var_num,cut_off):
        """given a comboanation of things the fuction will find a remove items from the best sub set that are too
        corealted
        """
        cols = self.data_frames[self.model_index].columns
        all_corrs = self.data_frames[self.model_index].drop('Lmax',1).corr()
        bad_combos = []
        good_combs = []
        combos = self.best_subset_runer(var_num)
        for val in combos:
            checker = 'good'
            temp_cols = list(val)
            used = all_corrs.loc[temp_cols,temp_cols].values.flatten()
            for new_val in used:
                if new_val != 1:
                    if new_val > cut_off:
                        checker = 'bad'
            if checker == 'bad':
                bad_combos.append(val)
        temp_all = []
        for t in combos:
            string_combo = " ".join(list(t))
            temp_all.append(string_combo)
        temp_bad = []
        for t in bad_combos:
            string_bad_combo = " ".join(list(t))
            temp_bad.append(string_bad_combo)
        for com in temp_all:
            if com not in temp_bad:
                good_combs.append(com.split())
        if len(bad_combos) != 0:
            print(f"found {len(bad_combos)} combanation of data that are above the correaltion threshold {cut_off}")
            print("bad combos are:")
            print("\n".join(temp_bad))
        else:
            print("No bad combos found")
        return good_combs
    def cal_RSS(self,actual,predicted):
        """ function to calc the RSS for a given set """
        return ((actual - predicted)**2).sum() # this works for arrays
    def SST(self):
        """ calc the bottom portion of the r sqaured forumla"""
        self.SST_val =((self.data_frames[self.model_index]['Lmax'] - self.data_frames[self.model_index]['Lmax'].mean())**2).sum()
    def process_subset_statsmodels(self,data_frame,cols):
        """ Given a data frame prefrom the linear regression via stats models
        This will reutrn the model odject from statsmodels , the combo used ,adjusted R squared ,
        , the R squared BIC,AIC RSS"""
        if len(cols) != 1:
            x = data_frame.drop('Lmax',1)[list(cols)] # define x for the model suing the cols that we want
        else:
            x = data_frame.drop('Lmax',1)[cols[0]]
        Y = data_frame['Lmax']
        x = sm.add_constant(x) # for this versioon to hvae a intercept we need to add a constant
        model = sm.OLS(Y,x)
        fit_data_model = model.fit()
        return {'model': fit_data_model,'combo': list(cols),'adj_R': fit_data_model.rsquared_adj ,"R": fit_data_model.rsquared ,'BIC' : fit_data_model.bic,'AIC' : fit_data_model.aic , 'RSS': self.cal_RSS(fit_data_model.predict(x),Y)}
    def gen_linear_equation(self,result):
        """ given the result from the process subset function write a str for the equation
        of the line"""
        # result is the result of the sm linear regression
        model = result['model']
        variables = list(model.params.index)
        values = list(model.params.values)
        val =[]
        var = []
        eq_string = "LMAX = "
        string_list = []
        for val,var in zip(values,variables):
            s = f"{var}*{val}"
            string_list.append(s)
        s = " + ".join(string_list)
        return eq_string + s
    def predict_vaules(self,result,data):
        """given the result from the process subset function and the data frame we are looking
        at return the predicted values"""
        variables = list(result['model'].params.index)
        values = list(result['model'].params.values)
        ans = 0
        for val , var in zip(values,variables):
            if var == 'const':
                ans += val
            else:
                ans += data[var] * val

        return ans.values
    def leave_one_out(self,data_frame,combo,cv_metric='MSE'):
        """ Prefromes leave one out crosws validation on a given set of data"""
        LOOCV = LeaveOneOut()
        cv_array = []
        PRESS = 0
        c = [x for x in combo]
        c.append('Lmax')
        data_used = data_frame[c]
        LOOCV.get_n_splits(data_used)
        predictions =[]
        cnt_np = 0
        for train_index , test_index in LOOCV.split(data_used):
            X_train , X_test = data_used.iloc[train_index] , data_used.iloc[test_index]
            result = self.process_subset_statsmodels(X_train,combo)
            pre = self.predict_vaules(result,X_test)
            predictions.append(pre)
            cnt_np += 1
            result['MSE'] = mean_squared_error(X_test['Lmax'],pre)
            cv_array.append(result[cv_metric])
            PRESS += (pre - X_test['Lmax'].values)**2
        p =[p[0] for p in predictions]
        return np.mean(cv_array),np.std(cv_array)/(len(cv_array)**0.5), (1 -(PRESS/self.SST_val))[0] , p
    def loocv_for_one_variable(self,data_frame,var,purge_cutoff=None,cv_metric='MSE'):
        """ for a given number of data points return the top self.keep models varaified by
        LOOCV"""
        if purge_cutoff != None and var != 1:
            combos = self.purge_corr_too_high(var,purge_cutoff)
        else:
            combos = self.best_subset_runer(var)
        cv = []
        pred = []
        for combo in combos:
            val , _ ,pred_R, _= self.leave_one_out(data_frame,combo) # run leave one out
            cv.append(val)
            pred.append(pred_R)
        end_data = {"combo":combos,"cv_scorce" : cv,"pred_rsquared":pred}
        return pd.DataFrame(data=end_data).nsmallest(self.keep,'cv_scorce')
    def run_leave_one_out(self,tolerance=0,purge_cutoff=None,cv_metric='MSE'):
        """a wrapper that prforms the LOOCV and graphs the wanted results"""
        # if there is a tolerance given trim set=
        if tolerance != 0:
            self.tolerance_cheack(tolerance)
        # plot heat map
        self.peasron_correaltion_heatmap()
        # after trim we need to loop into the number of vaiables that we want
        good_par = []
        pred_R = []
        self.SST()
        for num_par in range(1,self.max_var+ 1):
            top_n_cv = self.loocv_for_one_variable(self.data_frames[self.model_index],num_par,purge_cutoff=purge_cutoff,cv_metric=cv_metric)
            for x in top_n_cv['combo']:
                if len(x) != 1:
                    x_list = list(x)
                else:
                    x_list = []
                    x_list.append(x[0])
                good_par.append(x_list)
            for p in top_n_cv['pred_rsquared']:
                pred_R.append(p)
        # we have to rerun the top models with the full set to get the final models
        for par,pred in zip(good_par,pred_R):
            result = self.process_subset_statsmodels(self.data_frames[self.model_index],par)
            model = result['model']
            self.logger(model.summary())
            eq = self.gen_linear_equation(result)
            self.logger(f"the eqeuation for {par} is:")
            self.logger(eq)
            self.logger(f"The redicted rsqaured vaules is {pred}")
        # plot the whole thing
        nrows = self.keep
        ncols = self.max_var
        plt.rc('xtick',labelsize=8)
        plt.rc('ytick',labelsize=8)
        fig = plt.figure(figsize=(20,20))
        cnt = 0
        for c in good_par:
            cnt += 1
            result = self.process_subset_statsmodels(self.data_frames[self.model_index],c)
            data_not_used = []
            expermental_lmax = list(self.data_frames[self.model_index]['Lmax']) * (len(self.data_frames) -1)
            for mod in range(len(self.data_frames)):
                 if mod != self.model_index:
                        test_data = list(self.predict_vaules(result,self.data_frames[mod]))
                        for point in test_data:
                            data_not_used.append(point)
            mse_not_used =  mean_squared_error(expermental_lmax,data_not_used)
            rmse_test = (mse_not_used)**0.5
            self.logger(f'rmse for combo {c}: {rmse_test}')
            pred = self.predict_vaules(result,self.data_frames[self.model_index])
            eq = self.gen_linear_equation(result)
            ax = fig.add_subplot(nrows,ncols,cnt)
            _,_,_,loocv_data = self.leave_one_out(self.data_frames[self.model_index],c)
            ax.scatter(self.data_frames[self.model_index]['Lmax'],pred,c="black", label="model")
            ax.scatter(expermental_lmax,data_not_used,c='red',marker="P",label=f'test-data|RSME {rmse_test}')
            ax.scatter(self.data_frames[self.model_index]['Lmax'],loocv_data,c='blue',marker="*",label='loocv')
            m, b , r_value, p_value, std_err = scipy.stats.linregress(self.data_frames[self.model_index]['Lmax'], pred)
            m2, b2 , r_value2, p_value2, std_err2 = scipy.stats.linregress(self.data_frames[self.model_index]['Lmax'],loocv_data)
            ax.plot(self.data_frames[self.model_index]['Lmax'],m*self.data_frames[self.model_index]['Lmax']+ b , 'black',label='model | R squared ' + str(r_value**2))
            ax.plot(self.data_frames[self.model_index]['Lmax'],m2*self.data_frames[self.model_index]['Lmax']+ b2 , 'blue' , linestyle=":",linewidth=1, label='loocv | R squared ' + str(r_value2**2))
            ax.set_title(c, fontsize=10)
            ax.set_ylabel("predicted Lmax")
            ax.set_xlabel("expermental lmax")
            ax.legend( prop={"size":8},loc='best')
            fig.tight_layout()
        fig.savefig(self.dir_path+ '/'+'loocv.png',dpi=400)
    def untouchedData(self,nset=5):
        """ If the user wants to randomdly index numbers to not use we can split
        The data to be in to two groups MODS: used to vaialdate/make the line
        ,or untouched """
        # check if possable to grab the item
        remain = len(self.data_frames) - nset
        if remain < 0:
            print("the number untouched data is bigger than the total number of sets")
            sys.exit(1)
        remain -= 1 # since we are keeping the item of model index
        keeping = []
        keeping.append(self.data_frames[self.model_index]) # making sure that index is 0
        # changing the index of the model index
        index_list = [x for x in range(len(self.data_frames) -1)] # list all possable indexes
        # lets remove a number of that index
        keep_index = []
        for index in index_list:
            if index != self.model_index:
                keep_index.append(index)
        # using this lest we choose the ones to use
        keep_index = np.array((keep_index))
        rans = np.random.choice(keep_index,remain,replace=False) # get the rest of the using set
        testing_index = []
        # find the index that we are going to use
        for test in keep_index:
            if test not in rans:
                testing_index.append(test)
        for r in rans:
            keeping.append(self.data_frames[r])
        test = []
        for t in testing_index:
            test.append(self.data_frames[t])
        self.data_frames = keeping
        self.untouched = test
        self.n_folds = len(self.data_frames) # change the number of fold to equal the numder
        ## of things we have in the new dataframes
        self.model_index = 0
    def calc_aic(self,n,mse,num_par):
        return n * log(mse) + 2 *num_par
    def calc_bic(self,n, mse, num_params):
        return n * log(mse) + num_params * log(n)
    def KfoldCV(self,combo,flipp=False,metric="MSE"):
        """ this will run the the Kfold cross validation for us if FLip
        we flipp the test and train variables for the ten precent"""
        cv = KFold(n_splits=self.n_folds,shuffle=False)
        dataFrame = pd.concat(self.data_frames,ignore_index=True)
        cv_score = []
        temp_p = []
        if len(combo) != 1:
            cols = list(combo)
        else:
            cols = []
            cols.append(list(combo)[0])
        cols.append('Lmax')
        data = dataFrame[cols]
        for train, test in cv.split(data):
            if flipp == False:
                tr , te  = data.iloc[train] , data.iloc[test]
            elif flipp == True:
                te , tr  = data.iloc[train] , data.iloc[test]
            else:
                print("exting due to incorrect flipp invoations")
                sys.exit(1)
            result = self.process_subset_statsmodels(tr,combo)
            pred = self.predict_vaules(result,te) # predict values of the test set
            temp_p.append(pred)
            ## this is just a list of values
            # have to do these by hand to find the effect of the test set(s)
            MSE = mean_squared_error(te['Lmax'],pred)
            BIC = self.calc_bic(len(te["Lmax"]),MSE,len(result.get('model').params))
            AIC = self.calc_aic(len(te["Lmax"]),MSE,len(result.get('model').params))
            RSS = self.cal_RSS(te['Lmax'],pred)
        modChoice = {"RSS": RSS , "BIC": BIC , "AIC":AIC , "MSE": MSE}
        cv_score.append(modChoice.get(metric)) # adding the metric to use
        p = []
        for j in temp_p:
            for k in j:
                p.append(k)
        return np.mean(np.array((cv_score))) , np.std(np.array((cv_score))), p
    def kfold_for_one_variable(self,var,flipp=False,purge_cutoff=None,metric="MSE"):
        if purge_cutoff != None and var != 1:
            combos = self.purge_corr_too_high(var,purge_cutoff)
        else:
            combos = self.best_subset_runer(var)
        cv = []
        for combo in combos:
            cv_val ,_,_ = self.KfoldCV(combo,flipp=flipp,metric=metric)
            cv.append(cv_val)
        end_data = {"combo":combos,"cv_scorce" : cv }
        return pd.DataFrame(data=end_data).nsmallest(self.keep,'cv_scorce')
    def kfold_runner(self,tolerance=0,flipp=False,purge_cutoff=None,cv_metric='MSE'):
        #if there is a tolerance given trim set=
        if tolerance != 0:
            self.tolerance_cheack(tolerance)
        self.peasron_correaltion_heatmap()
        good_par = []
        pred_R = []
        self.SST()
        for num_par in range(1,self.max_var+ 1): # run all the number of parmaters
            top_n_cv = self.kfold_for_one_variable(num_par,flipp=flipp,purge_cutoff=purge_cutoff)
            for x in top_n_cv['combo']:
                if len(x) != 1:
                    x_list = list(x)
                else:
                    x_list = []
                    x_list.append(x[0])
                good_par.append(x_list) # collecting the good partmaters
        nrows = self.keep
        ncols = self.max_var
        plt.rc('xtick',labelsize=8)
        plt.rc('ytick',labelsize=8)
        fig = plt.figure(figsize=(20,20))
        cnt = 0
        for par in good_par: # now will build the model with the one we want (Meesy)
            if flipp ==True: # this is the case where we are using only ten perent of the model
                modset = self.data_frames[self.model_index]
                try:
                    len(self.untouched)
                except:
                    print("print you are using a 10% model without leaving data out")
                    print("using the rest of the data to test the final model")
                    self.untouched = []
                    for frame in range(len(self.data_frames)):
                        if frame != self.model_index:
                            self.untouched.append(self.data_frames[frame])
                _,_,kfoldData = self.KfoldCV(par,flipp=flipp,metric=cv_metric)
                kfold_expre = list(self.data_frames[self.model_index]['Lmax']) * (len(self.data_frames) * (len(self.data_frames) - 1))
                kfold_expre = np.array((kfold_expre),dtype='float64')

            else:
                modset = pd.concat(self.data_frames) # will mush the used data into one set for later
                # this case would require that we have a set held back before we finilze the model
                try:
                    len(self.untouched)
                except:
                    print("you are using the normal fold method and not doing a test train split ")
                    print("need vaules for testing the model. pelase run untouchedData and try again")
                    sys.exit(1)
                _,_,kfoldData = self.KfoldCV(par,flipp=flipp,metric=cv_metric)
                kfold_expre = modset['Lmax']
            # with the data now frimily de fined we can build the model next time
            cnt += 1
            result = self.process_subset_statsmodels(modset,par)
            model = result['model']
            self.logger(model.summary())
            eq = self.gen_linear_equation(result)
            self.logger(f"the eqeuation for {par} is: {eq}")
            # now we run the leave one out for our stuff
            _,_,pred_R,loocv_data = self.leave_one_out(modset,par)
            self.logger(f"The predicted R squared value for {par}: {pred_R}")
            # now we loop into the untested data sets
            test_data = []
            expermental_lmax = []
            for dataset in self.untouched:
                lmax_values = dataset['Lmax']
                for lmax , val in  zip(lmax_values,list(self.predict_vaules(result,dataset))):
                    expermental_lmax.append(lmax)
                    test_data.append(val)
            mse_not_used =  mean_squared_error(expermental_lmax,test_data)
            rmse_test = (mse_not_used)**0.5
            self.logger(f'rmse for combo {par}: {rmse_test}')
            pred = self.predict_vaules(result,modset)
            ax = fig.add_subplot(nrows,ncols,cnt)
            ax.scatter(modset['Lmax'],pred,c="black", label="model")
            ax.scatter(expermental_lmax,test_data,c='blue',marker="P",label=f'test-data|RSME {rmse_test}' ,alpha=0.5)
            #ax.scatter(modset['Lmax'],loocv_data,c='blue',marker="*",label='loocv')
            ax.scatter(kfold_expre,kfoldData,c="red",marker="D",label="Kfold_resutls", s=10, alpha=0.25)
            # line of fits between points
            m, b , r_value, p_value, std_err = scipy.stats.linregress(modset['Lmax'], pred)
            m2, b2 , r_value2, p_value2, std_err2 = scipy.stats.linregress(expermental_lmax,test_data)
            m3, b3 , r_value3, p_value3, std_err3 = scipy.stats.linregress(kfold_expre,kfoldData)
            ax.plot(modset['Lmax'], m * modset['Lmax'] + b , c ='black',label='model | R squared ' + str(r_value**2))
            #ax.plot(modset['Lmax'], m2 * modset['Lmax'] + b2 , c ='red',linestyle="--",label='loocv | R squared ' + str(r_value2**2))
            ax.plot(kfold_expre,kfold_expre*m3 +b3,c="red",label="Kfold_cv |R_squared" + str(r_value3**2),linestyle="-.",alpha=0.25)
            ax.set_title(list(par), fontsize=8)
            #if cnt == ceil(nrows/2):
            ax.set_ylabel("predicted Lmax",fontsize=8)
            #if cnt == ceil(ncols/2):
            ax.set_xlabel("expermental lmax",fontsize=8)
            ax.legend(prop={"size":8},loc='best',framealpha=0.25)
        fig.tight_layout()
        plt.savefig(self.dir_path+ '/'+'kfold.png',dpi=400)
