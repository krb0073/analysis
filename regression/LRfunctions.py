# @Author: Kyle Billings <kbillings>
# @Date:   2020-06-16T12:37:44-04:00
# @Email:  krb0073@mix.wvu.edu
# @Filename: LRfunctions.py
# @Last modified by:   kbillings
# @Last modified time: 2020-12-29T23:04:39-05:00
####### first make arrays of random values that will be filled with indexes
######## of postions in the file
import numpy as np
import statsmodels.api as sm
import itertools
import pandas as pd
from sklearn.linear_model import Ridge , RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from math import log
import matplotlib.pyplot as plt
# function to load in the data from the file
def grabCol(FILE,col1,delim=' '): # this will be all the data in the file
    import csv
    allVals=[]
    with open( FILE , 'r' ) as csvfile:
        plots = csv.reader(csvfile, delimiter=delim) # hard code the delimaiter
        ## so we can change it
        for row in plots:
            if "#"  not in row: # will pass the "#" sybmol if it is therw
                allVals.append(float(row[col1]))
    return allVals
# fxn to grab the given indexs
def randomVals(allVals,B): # given a list of values a
    V = [] # storage
    for i in B: # for each index in the random indexs
        V.append(allVals[i]) # append the vaules
    return V # return vals
# this find the median of the values of the random stuff
def med(randVals):
    from  statistics import median
    return median(randVals) # find and return the random median from the frames
# building the nump array for storage
def blankArray(numVals):
    return np.zeros((6,numVals)) # reruns a blank array
# numpy random stuff to make the values for each test
def randArray(seedNum,startFrame, lastFrame,numRow):
    # making the numpy seed so we can get the same number from the random gen
    ## each time
    np.random.seed(int(seedNum))
    # build a np arrange with all the possable indexs
    All_nums = np.arange(startFrame,lastFrame)
    # number of frames per line need to check that the number of frames is evenly
    ## divisable by the num of trails we want to run
    tot = lastFrame - startFrame
    if tot <= 0: # chec if the tot number of frmes we want is not neg or 0
        return "neagtive or no frames given from the tot = lastFrame -startFrame"
    elif tot % numRow != 0: # make sure teh number of rows we want works so all
    ## have the same number of vaules
        return "the total number of frames is not divisable by the num of rows"
    else: ## if both are meeet
        np.random.shuffle(All_nums) # mixes all the vlaues in the arrays
        return All_nums.reshape((numRow,-1)) # create a 2d aray only know the n
        ## number of rows we want
def buildingDicts(Lmax,ALL,QQ): # makes a dict from the Lmax , all data and then
# set
    d = {} # empty dict
    L = {'Lmax' : Lmax} # lmax
    d.update(L) # add L to dict
    K = QQ.T # transpose the array for work for is
    for i in range(len(ALL)):
        tempD = {ALL[i] : K[i]} # making a part of the dict via indexa
        d.update(tempD)
    return d
def grabAUC(seed,qCnt):
    Fname = "goodRangeDCD/RMSF_datasets_{0}/my_set_{1}.dat".format(seed,qCnt) # getting the file name
    # grab AUC coulmn
    AUC = grabCol(Fname,0) # grabs the AUC all
    aucRing = grabCol(Fname,1) # just ring AUC
    return AUC , aucRing

def getList(dict):
    return list(dict.keys()) # get the keys of a dict as a lsit
def processSubset(Y,X,subSet): # Subset is the columns used
    x = X[list(subSet)] # grab desired set
    x = sm.add_constant(x) # add a const for the intercept
    model = sm.OLS(Y,x) # making the model
    regr = model.fit() # do a linear fit
    RSS = ((regr.predict(x) - Y)**2).sum()
    return {'model': regr ,'adj_R': regr.rsquared_adj ,'BIC' : regr.bic,'AIC' : regr.aic , 'RSS': RSS  }
def nBestModels(Y,X,k): # find the best model using the best subsets
    res = []
    for combo in itertools.combinations(X.columns, k): # each possable combo
        res.append(processSubset(Y,X,combo))
    models = pd.DataFrame(res).sort_values(by=['RSS'],ascending=True) # dit to data frames
    return models[:5] # returns the top 5 models
def bestModel(Y,X,k):
    res = []
    for combo in itertools.combinations(X.columns, k):
        res.append(processSubset(Y,X,combo))
    models = pd.DataFrame(res)
    best_model = models.loc[models['RSS'].argmin()]

    return best_model # same as above to only the top model is given back
def findPredictedvlaues(data,M,names):
    tX = data.drop("Lmax",1)
    r=M.params
    usedCols = list(r.index)[1:]
    tX = tX[usedCols]
    tX = sm.add_constant(tX)
    pre = M.predict(tX)
    #print(raw[usedCols].corr())
    return {"parmaLabel" : list(r), "pramaVals" : list(r) , "set" :names , "PRE" : pre  }
def personCorr(M,raw): # prefroms the person corraction from the raw data
# M is a OLS model
    r=M.params
    usedCols = list(r.index)[1:]
    return raw[usedCols].corr()
def BICbestModel(Y,X,k): # best model detrined by BIC
    res = []
    for combo in itertools.combinations(X.columns, k):
        res.append(processSubset(Y,X,combo))
    models = pd.DataFrame(res)
    best_model = models.loc[models['BIC'].argmin()]
    return best_model
def AICbestModel(Y,X,k): # bset model found by AIC
    res = []
    for combo in itertools.combinations(X.columns, k):
        res.append(processSubset(Y,X,combo))
    models = pd.DataFrame(res)
    best_model = models.loc[models['AIC'].argmin()]
    return best_model
def ridgeNoKfold(data,y,k): #uses only one data set
    alphas = 10**np.linspace(10,-2,100)*0.5 # list of alphas possable
    X = data.drop("Lmax",axis=1)
    res = []
    for combo in itertools.combinations(X.columns,k): # for combo
        vars = list(combo)
        X = data[vars] # data via column
        if k == 1: # if only one column is being used we have to change the Data
        ## shape
            X = np.array((X)).reshape(-1,1) # to comply with skearlns fromat
        RnCv = RidgeCV(alphas = alphas, scoring = 'neg_mean_squared_error', normalize = True)
        # uses Crossvaidation to find alpha
        RnCv.fit(X,y) # fit the RnCV to a linear model
        A = RnCv.alpha_ # retrun alpha params
        Rn = Ridge(normalize = True) # set a variable to run regression in sklearn ridge
        Rn.set_params(alpha = A) # now we found A we can now use it to do the real analysis
        Rn.fit(X,y) # fit ridge model
        pedict = Rn.predict(X) # predict the values from that
        MSE = mean_squared_error(y,pedict)
        RSS = ((Rn.predict(X) - y)**2).sum()
        coef = Rn.coef_ # grabing in the coef from tghe thing
        AIC = calc_aic(len(y),MSE,(len(Rn.coef_) +1 ))
        BIC = calc_bic(len(y),MSE,(len(Rn.coef_) +1 ))
        myDict = {"model" : Rn.fit(X,y) ,"parm" : list(combo) , "MSE" : MSE , "alpha" : A , "R" : Rn.score(X,y), "RSS": RSS ,"BIC" : BIC , "AIC" :AIC}
        res.append(myDict)
    all_mods = pd.DataFrame(res)
    best = all_mods.loc[all_mods["MSE"].argmin()] # find the best model by mse
    #Rn = best.model
    #Rn.fit(data[best.parm],y)
    #print(Rn.predict(data[list(best.parm)]))
    return best
def calc_aic(n,mse,num_par):
    return n * log(mse) + 2 *num_par
def calc_bic(n, mse, num_params):
    return n * log(mse) + num_params * log(n)
def adj_R_square(n,num_par,R_val):
    return 1 -(((1- R_val) * (n-1)) /(n- num_par - 1))
    # 1 -( ((1-R_val)(n - 1))/(n -num_par - 1) )
def ridgeNoKfold_2(data,y,k): #uses only one data set will not make a large frame
    alphas = 10**np.linspace(10,-2,100)*0.5 # see above for how it works
    X = data.drop("Lmax",axis=1)
    mseLow = 10000000000
    for combo in itertools.combinations(X.columns,k):
        vars = list(combo)
        X = data[vars]
        if k == 1:
            X = np.array((X)).reshape(-1,1) # to comply with skearlns fromat
        RnCv = RidgeCV(alphas = alphas, scoring = 'neg_mean_squared_error', normalize = True)
        RnCv.fit(X,y)
        A = RnCv.alpha_
        Rn = Ridge(normalize = True)
        Rn.set_params(alpha = A)
        Rn.fit(X,y)
        pedict = Rn.predict(X)
        MSE = mean_squared_error(y,pedict)
        RSS = ((Rn.predict(X) - y)**2).sum()
        coef = Rn.coef_
        AIC = calc_aic(len(y),MSE,(len(Rn.coef_) +1 ))
        BIC = calc_bic(len(y),MSE,(len(Rn.coef_) +1 ))
        myDict = {"model" : Rn.fit(X,y) ,"parm" : combo , "MSE" : MSE , "alpha" : A , "R" : Rn.score(X,y), "RSS": RSS ,"BIC" : BIC , "AIC" :AIC}
        if MSE < mseLow:
            mseLow = MSE
            best = myDict
    best = pd.DataFrame(best)
    #all_mods = pd.DataFrame(res)
    #best = all_mods.loc[all_mods["MSE"].argmin()]
    #Rn = best.model
    #Rn.fit(data[best.parm],y)
    #print(Rn.predict(data[list(best.parm)]))
    return best
def Kfold_smOLS(data ,Y,k,nfolds=10,metric="RSS"): # use kfold corss and OLS
    print(f"start Kfolding now {nfolds}fold K is {k}")
    cv = KFold(n_splits=nfolds,shuffle=False) #set the cross vaildation model
    X = data.drop("Lmax", axis = 1)
    cols  = list(X.columns) # get feldis involed
    all_RSS =[]
    for combo in itertools.combinations(cols,k): # for column and K parma
        v = list(combo) # turn to list
        sele = data[v] # tgrab needed data
        Rss_sum = []
        #start_Rss = 1000000000000000
        for train_index , test_index in cv.split(data): # test train split over all
        ## ten
            X_train , X_test , Y_train , Y_test = sele.iloc[train_index] , sele.iloc[test_index] , Y.iloc[train_index] ,Y.iloc[test_index]
            # split the data in the data frame by given index in order
            trained = processSubset(Y_train,X_train,combo) # run regression
            mod = trained.get("model") # get model
            X_test = sm.add_constant(X_test) # add const
            RSS = ((mod.predict(X_test) - Y_test)**2).sum() # find perctors
            MSE = RSS / 6
            BIC =  calc_bic(len(Y_test),MSE,len(trained.get("model").params))
            AIC = calc_aic(len(Y_test),MSE,len(trained.get("model").params))
            modChoice = {"RSS": RSS , "BIC": BIC , "AIC":AIC , "MSE": MSE}
            #print(BIC, RSS)
            Rss_sum.append(modChoice.get(metric)) # add the prdictor to a list
        RSSnp = np.array((Rss_sum)) # to np from array
        finalList ={"combo" : list(combo), "avg" : np.average(RSSnp)}  # put all in a data frame
        all_RSS.append(finalList)
    allMods = pd.DataFrame(all_RSS)
    #print(allMods)
    bestMod = allMods.loc[allMods['avg'].argmin()] # retrun lowest avarge pridictor
    return bestMod
def Kfold_skRidge(data,Y,k,nfolds=10,metric="RSS"): # see above for the this w/o
## comments
    alphas = 10**np.linspace(10,-2,100)*0.5 # list alpha
    print(f"starting ridge version of K-fold with {nfolds}folds with k of {k}")
    cv = KFold(n_splits=nfolds,shuffle=False) # Crossvaidation method 10fold
    X = data.drop("Lmax", axis = 1)
    cols = list(X.columns)
    all_RSS =[]
    for combo in itertools.combinations(cols,k):
        v = list(combo)
        sele = data[v]
        Rss_sum = []
        all_alphas = []
        # find the best alpha
        for train_index ,test_index in cv.split(data):
            X_train , X_test , Y_train , Y_test = sele.iloc[train_index] , sele.iloc[test_index] , Y.iloc[train_index] ,Y.iloc[test_index]
            if k == 1:
                X_train = np.array((X_train)).reshape(-1,1)
                X_test = np.array((X_test)).reshape(-1,1)
            RnCv = RidgeCV(alphas = alphas, scoring = 'neg_mean_squared_error', normalize = True)
            RnCv.fit(X_train,Y_train)
            all_alphas.append(RnCv.alpha_)
        all_alphas = np.array((all_alphas))
        A = np.average(all_alphas) # best A parameter from all
        for train_index ,test_index in cv.split(data):
            X_train , X_test , Y_train , Y_test = sele.iloc[train_index] , sele.iloc[test_index] , Y.iloc[train_index] ,Y.iloc[test_index]
            if k == 1:
                X_train = np.array((X_train)).reshape(-1,1)
                X_test = np.array((X_test)).reshape(-1,1)
            Rn = Ridge(normalize = True)
            Rn.set_params(alpha = A)
            Rn.fit(X_train,Y_train)
            pedict = Rn.predict(X_test)
            RSS = ((pedict - Y_test)**2).sum()
            MSE = mean_squared_error(Y_test,pedict)
            AIC = calc_aic(len(Y_test),MSE,(len(Rn.coef_) +1 ))
            BIC = calc_bic(len(Y_test),MSE,(len(Rn.coef_) +1 ))
            modChoice = {"RSS": RSS , "BIC": BIC , "AIC":AIC , "MSE": MSE}
            Rss_sum.append(modChoice.get(metric))
        RSSnp = np.array((Rss_sum))
        finalList ={"combo" : combo, "avg" : np.average(RSSnp), "Alpha" : A}
        all_RSS.append(finalList)
    allMods = pd.DataFrame(all_RSS)
    bestMod = allMods.loc[allMods['avg'].argmin()]
    return bestMod
def LeaveOneOutCV_OLS(X,Y,k):
    print("start loo")
    cols  = list(X.columns)
    loo = LeaveOneOut()
    all_mods = []
    for combo in itertools.combinations(cols,k):
        v = list(combo)
        sele = X[v]
        loo.get_n_splits(sele)
        Rss_sum = 0
        for tr , te in  loo.split(X):
            Xtr , Xte = sele.iloc[tr] , sele.iloc[te]
            Ytr , Yte = Y.iloc[tr] , Y.iloc[te]
            trained = processSubset(Ytr,Xtr,combo)
            mod = trained.get("model") # get model
            variables = (list(mod.params.index))
            values = list(mod.params)
            formula = {}
            for i in range(0,len(values) ): # do it by index
                formula[str(variables[i])] = values[i]
            result = formula['const']
            for key in formula.keys():
                if key != 'const':
                    result += Xte[key] * formula.get(key)
            guess = list(result)[0]
            ans = list(Yte)[0]
            Rss_sum += (guess -ans)**2
        CV = Rss_sum / loo.get_n_splits(sele)
        modDict = {"combo" : list(combo) , "CV_score" : CV}
        all_mods.append(modDict)
    ALLmods = pd.DataFrame(all_mods)
    bestMod = ALLmods.loc[ALLmods['CV_score'].argmin()]
    return bestMod
def loocv_collect_guesss(type,X,Y,combo):
    loo = LeaveOneOut()
    all_act = []
    all_pred = []
    sele = X[list(combo)]
    if type == "OLS":
        for tr ,te in  loo.split(X):
            Xtr , Xte = sele.iloc[tr] , sele.iloc[te]
            Ytr , Yte = Y.iloc[tr] , Y.iloc[te]
            trained = processSubset(Ytr,Xtr,combo)
            mod = trained.get("model")
            lineEQ = EQFunction(type,mod,combo)
            predVal = lineEQ['const']
            for key in lineEQ.keys():
                if key != 'const':
                    predVal +=  Xte[key] * lineEQ.get(key)
            all_pred.append(list(predVal)[0])
            all_act.append(list(Yte)[0])
    if type == "R":
        v = list(combo)
        sele = X[v]
        loo.get_n_splits(sele)
        Rss_sum = 0
        all_alphas = []
        alphas = 10**np.linspace(10,-2,100)*0.5 # list alpha
        for tr , te in  loo.split(X):
            Xtr , Xte = sele.iloc[tr] , sele.iloc[te]
            if len(list(combo)) == 1:
                Xtr = np.array((Xtr)).reshape(-1,1)
                Xte = np.array((Xte)).reshape(-1,1)
            Ytr , Yte = Y.iloc[tr] , Y.iloc[te]
            RnCv = RidgeCV(alphas = alphas, scoring = 'neg_mean_squared_error', normalize = True)
            RnCv.fit(Xtr,Ytr)
            all_alphas.append(RnCv.alpha_)
        all_alphas = np.array((all_alphas))
        A = np.average(all_alphas) # best A parameter from all
        for tr , te in  loo.split(X):
            Xtr, Xte = sele.iloc[tr] , sele.iloc[te]
            print (len(list(combo)) )
            if len(list(combo)) == 1:
                Xtr = np.array((Xtr)).reshape(-1,1)
                Xte = np.array((Xte)).reshape(-1,1)
            Ytr , Yte = Y.iloc[tr] , Y.iloc[te]
            Rn = Ridge(normalize = True)
            Rn.set_params(alpha = A)
            Rn.fit(Xtr,Ytr)
    return all_act , all_pred
def LeaveOneOutCV_Ridge(X,Y,k):
    print(f"ridge SKlearn leave one out CV w/ k {k}")
    alphas = 10**np.linspace(10,-2,100)*0.5 # list alpha
    cols = list(X.columns)
    loo = LeaveOneOut()
    all_mod = []
    for combo in itertools.combinations(cols,k):

        v = list(combo)
        sele = X[v]
        loo.get_n_splits(sele)
        Rss_sum = 0
        all_alphas = []
        # find the best alpha for each combo
        for tr , te in  loo.split(X):
            Xtr , Xte = sele.iloc[tr] , sele.iloc[te]
            if k == 1:
                Xtr = np.array((Xtr)).reshape(-1,1)
                Xte = np.array((Xte)).reshape(-1,1)
            Ytr , Yte = Y.iloc[tr] , Y.iloc[te]
            RnCv = RidgeCV(alphas = alphas, scoring = 'neg_mean_squared_error', normalize = True)
            RnCv.fit(Xtr,Ytr)
            all_alphas.append(RnCv.alpha_)
        all_alphas = np.array((all_alphas))
        A = np.average(all_alphas)
        for tr , te in  loo.split(X):
            Xtr, Xte = sele.iloc[tr] , sele.iloc[te]
            if k == 1:
                Xtr = np.array((Xtr)).reshape(-1,1)
                Xte = np.array((Xte)).reshape(-1,1)
            Ytr , Yte = Y.iloc[tr] , Y.iloc[te]
            Rn = Ridge(normalize = True)
            Rn.set_params(alpha = A)
            Rn.fit(Xtr,Ytr)
            #print(Rn.coef_ ,Rn.get_params()['alpha'])
            test = EQFunction("R",Rn,combo)
            #print(test)
            guess = Rn.predict(Xte)[0]
            ans = list(Yte)[0]
            Rss_sum += (guess -ans)**2
        CV = Rss_sum / loo.get_n_splits(sele)
        modDict = {"combo" : combo , "CV_score" : CV , "Alpha" : A}
        all_mod.append(modDict)
    ALLmods = pd.DataFrame(all_mod)
    bestMod = ALLmods.loc[ALLmods['CV_score'].argmin()]
    print(f"done with loocv with k of {k}")
    return bestMod
# inoder to veiw and graph to result better lets make a thing to sort the
## eqiation of the line
def EQFunction(types,mod,combo):
    lineEQ = {}
    if types == "OLS":
        variables = (list(mod.params.index))
        values = list(mod.params)
        for i in range(0,len(values)):
            lineEQ[str(variables[i])] = values[i]
        return lineEQ
    if types == "R":
        const = mod.intercept_
        Alpha = mod.get_params()['alpha']
        values = list(mod.coef_)
        variables = list(combo)
        for i in range(0,len(values)):
            lineEQ[str(variables[i])] = values[i]
        B_squared = [x **2 for x in values]
        lineEQ['const'] = const
        lineEQ['Error'] = Alpha * (sum(B_squared) - const)
        #print(lineEQ["Error"])
        return lineEQ

def Kfold_smOLS_10precent(data ,Y,k,nfolds=10,metric="RSS"): # use kfold corss and OLS
    print(f"start Kfolding now {nfolds}fold K is {k}")
    cv = KFold(n_splits=nfolds,shuffle=False) #set the cross vaildation model
    X = data.drop("Lmax", axis = 1)
    cols  = list(X.columns) # get feldis involed
    all_RSS =[]
    for combo in itertools.combinations(cols,k): # for column and K parma
        v = list(combo) # turn to list
        sele = data[v] # tgrab needed data
        Rss_sum = []
        #start_Rss = 1000000000000000
        for  test_index , train_index  in cv.split(data): # test train split over all
        ## ten
            X_train , X_test , Y_train , Y_test = sele.iloc[train_index] , sele.iloc[test_index] , Y.iloc[train_index] ,Y.iloc[test_index]
            # split the data in the data frame by given index in order
            trained = processSubset(Y_train,X_train,combo) # run regression
            mod = trained.get("model") # get model
            X_test = sm.add_constant(X_test) # add const
            RSS = ((mod.predict(X_test) - Y_test)**2).sum() # find perctors
            MSE = RSS / 6
            BIC =  calc_bic(len(Y_test),MSE,len(trained.get("model").params))
            AIC = calc_aic(len(Y_test),MSE,len(trained.get("model").params))
            modChoice = {"RSS": RSS , "BIC": BIC , "AIC":AIC , "MSE": MSE}
            #print(BIC, RSS)
            Rss_sum.append(modChoice.get(metric)) # add the prdictor to a list
        RSSnp = np.array((Rss_sum)) # to np from array
        finalList ={"combo" : list(combo), "avg" : np.average(RSSnp)}  # put all in a data frame
        all_RSS.append(finalList)
    allMods = pd.DataFrame(all_RSS)
    #print(allMods)
    bestMod = allMods.loc[allMods['avg'].argmin()] # retrun lowest avarge pridictor
    return bestMod

def Ols_kfold_grab_pred_10precent(type,data,Y, combo):
    all_pred = []
    all_act = []
    if type == "OLS":
        cv = KFold(n_splits=10,shuffle=False) #set the cross vaildation model
        X = data.drop("Lmax", axis = 1)
        v = list(combo) # turn to list
        sele = data[v] # tgrab needed data
        Rss_sum = []
        for  test_index , train_index  in cv.split(data):
            X_train , X_test , Y_train , Y_test = sele.iloc[train_index] , sele.iloc[test_index] , Y.iloc[train_index] ,Y.iloc[test_index]

            trained = processSubset(Y_train,X_train,combo) # run regression
            mod = trained.get("model") # get model
            X_test = sm.add_constant(X_test) # add const
            temp = mod.predict(X_test)
        all_pred += list(temp)
        all_act += list(Y_test)
    return all_act , all_pred
def Kfold_skRidge_10precent(data,Y,k,nfolds=10,metric="RSS"): # see above for the this w/o
## comments
    alphas = 10**np.linspace(10,-2,100)*0.5 # list alpha
    print(f"starting ridge version of K-fold with {nfolds}folds with k of {k}")
    cv = KFold(n_splits=nfolds,shuffle=False) # Crossvaidation method 10fold
    X = data.drop("Lmax", axis = 1)
    cols = list(X.columns)
    all_RSS =[]
    for combo in itertools.combinations(cols,k):
        v = list(combo)
        sele = data[v]
        Rss_sum = []
        all_alphas = []
        # find the best alpha
        for test_index ,train_index in cv.split(data):
            X_train , X_test , Y_train , Y_test = sele.iloc[train_index] , sele.iloc[test_index] , Y.iloc[train_index] ,Y.iloc[test_index]
            if k == 1:
                X_train = np.array((X_train)).reshape(-1,1)
                X_test = np.array((X_test)).reshape(-1,1)
            RnCv = RidgeCV(alphas = alphas, scoring = 'neg_mean_squared_error', normalize = True)
            RnCv.fit(X_train,Y_train)
            all_alphas.append(RnCv.alpha_)
        all_alphas = np.array((all_alphas))
        A = np.average(all_alphas) # best A parameter from all
        for train_index ,test_index in cv.split(data):
            X_test , X_train , Y_test , Y_train = sele.iloc[train_index] , sele.iloc[test_index] , Y.iloc[train_index] ,Y.iloc[test_index]
            if k == 1:
                X_train = np.array((X_train)).reshape(-1,1)
                X_test = np.array((X_test)).reshape(-1,1)
            Rn = Ridge(normalize = True)
            Rn.set_params(alpha = A)
            Rn.fit(X_train,Y_train)
            pedict = Rn.predict(X_test)
            RSS = ((pedict - Y_test)**2).sum()
            MSE = mean_squared_error(Y_test,pedict)
            AIC = calc_aic(len(Y_test),MSE,(len(Rn.coef_) +1 ))
            BIC = calc_bic(len(Y_test),MSE,(len(Rn.coef_) +1 ))
            modChoice = {"RSS": RSS , "BIC": BIC , "AIC":AIC , "MSE": MSE}
            Rss_sum.append(modChoice.get(metric))
        RSSnp = np.array((Rss_sum))
        finalList ={"combo" : combo, "avg" : np.average(RSSnp), "Alpha" : A}
        all_RSS.append(finalList)
    allMods = pd.DataFrame(all_RSS)
    bestMod = allMods.loc[allMods['avg'].argmin()]
    return bestMod
def process_subset_ridge(X,Y,A,combo):
    # grad the X data from the frames
    c = list(combo)
    data = X[c]
    alphas = 10**np.linspace(10,-2,100)*0.5 # list alpha
    if  len(c) == 1:
        data = np.array((data)).reshape(-1,1)
    Rn = Ridge(normalize = True)
    Rn.set_params(alpha = A)
    Rn.fit(data,Y)
    R = Rn.score(data,Y)
    adjR = adj_R_square(len(list(Y)),len(c),R)
    MSE = mean_squared_error(Rn.predict(data),Y)
    RSS = ((Rn.predict(data) - Y)**2).sum()
    BIC = calc_bic(len(list(Y)),MSE,(len(Rn.coef_) +1 ))
    AIC = calc_aic(len(list(Y)),MSE,(len(Rn.coef_) +1 ))
    mets = {'model': Rn ,'adj_R': adjR ,'BIC' : BIC,'AIC' : AIC , 'RSS': RSS  }
    pred = Rn.predict(data)
    return pred , mets
def precent_10_kfold_ridge_grab_all(X,Y,A,combo):
    c = list(combo)
    sele = X[c]
    pred_all = []
    act_all =  []
    cv = KFold(n_splits=10,shuffle=False) #set the cross vaildation model
    for test_index ,train_index in cv.split(sele):
        X_test , X_train , Y_test , Y_train = sele.iloc[train_index] , sele.iloc[test_index] , Y.iloc[train_index] ,Y.iloc[test_index]
        if  len(c) == 1:
            X_test = np.array((X_test)).reshape(-1,1)
            X_train = np.array((X_train)).reshape(-1,1)
        Rn = Ridge(normalize = True)
        Rn.set_params(alpha = A)
        Rn.fit(X_train,Y_train)
        pred_all += list(Rn.predict(X_test))
        act_all += list(Y_test)
    return pred_all ,act_all
def precent_10_LOOCV_ridge_grab_all(X,Y,A,combo):
    c = list(combo)
    sele = X[c]
    pred_all = []
    act_all =  []
    loo = LeaveOneOut()
    loo.get_n_splits(sele)
    #cv = KFold(n_splits=10,shuffle=False) #set the cross vaildation model
    for train_index ,test_index in loo.split(sele):
        X_test , X_train , Y_test , Y_train = sele.iloc[train_index] , sele.iloc[test_index] , Y.iloc[train_index] ,Y.iloc[test_index]
        if  len(c) == 1:
            X_test = np.array((X_test)).reshape(-1,1)
            X_train = np.array((X_train)).reshape(-1,1)
        Rn = Ridge(normalize = True)
        Rn.set_params(alpha = A)
        Rn.fit(X_train,Y_train)
        pred_all += list(Rn.predict(X_test))
        act_all += list(Y_test)
    return pred_all ,act_all
def print_eq(type ,EQ):
    eq = "LMAX = "
    eq += " " + str(EQ['const']) +" + "
    if type == "OLS":
        for key in EQ.keys():
            if key != 'const':
                val = EQ.get(key)
                eq += f" {val} X {key} "
    if type == "R":
        for key in EQ.keys():
            if key != 'const':
                if key != 'Error':
                    val = EQ.get(key)
                    eq += f" {val} X {key} "
        eq += " " + str(EQ["Error"]) + " "
    print(eq)

def TOP_FIVE_Kfold_smOLS_10precent(data ,Y,k,nfolds=10,metric="RSS"): # use kfold corss and OLS
    print(f"start Kfolding now {nfolds}fold K is {k}")
    cv = KFold(n_splits=nfolds,shuffle=False) #set the cross vaildation model
    X = data.drop("Lmax", axis = 1)
    cols  = list(X.columns) # get feldis involed
    all_RSS =[]
    for combo in itertools.combinations(cols,k): # for column and K parma
        v = list(combo) # turn to list
        sele = data[v] # tgrab needed data
        Rss_sum = []
        #start_Rss = 1000000000000000
        for  test_index , train_index  in cv.split(data): # test train split over all
        ## ten
            X_train , X_test , Y_train , Y_test = sele.iloc[train_index] , sele.iloc[test_index] , Y.iloc[train_index] ,Y.iloc[test_index]
            # split the data in the data frame by given index in order
            trained = processSubset(Y_train,X_train,combo) # run regression
            mod = trained.get("model") # get model
            X_test = sm.add_constant(X_test) # add const
            RSS = ((mod.predict(X_test) - Y_test)**2).sum() # find perctors
            MSE = RSS / 6
            BIC =  calc_bic(len(Y_test),MSE,len(trained.get("model").params))
            AIC = calc_aic(len(Y_test),MSE,len(trained.get("model").params))
            modChoice = {"RSS": RSS , "BIC": BIC , "AIC":AIC , "MSE": MSE}
            #print(BIC, RSS)
            Rss_sum.append(modChoice.get(metric)) # add the prdictor to a list
        RSSnp = np.array((Rss_sum)) # to np from array
        finalList ={"combo" : list(combo), "avg" : np.average(RSSnp)}  # put all in a data frame
        all_RSS.append(finalList)
    allMods = pd.DataFrame(all_RSS).sort_values(by=['avg'])
    bestMod = allMods.nsmallest(5,'avg') # retrun lowest 5 avarge pridictor
    return bestMod
def Top_five_LeaveOneOutCV_OLS(X,Y,k):
    print("start loo")
    cols  = list(X.columns)
    loo = LeaveOneOut()
    all_mods = []
    for combo in itertools.combinations(cols,k):
        v = list(combo)
        sele = X[v]
        loo.get_n_splits(sele)
        Rss_sum = 0
        for tr , te in  loo.split(X):
            Xtr , Xte = sele.iloc[tr] , sele.iloc[te]
            Ytr , Yte = Y.iloc[tr] , Y.iloc[te]
            trained = processSubset(Ytr,Xtr,combo)
            mod = trained.get("model") # get model
            variables = (list(mod.params.index))
            values = list(mod.params)
            formula = {}
            for i in range(0,len(values) ): # do it by index
                formula[str(variables[i])] = values[i]
            result = formula['const']
            for key in formula.keys():
                if key != 'const':
                    result += Xte[key] * formula.get(key)
            guess = list(result)[0]
            ans = list(Yte)[0]
            Rss_sum += (guess -ans)**2
        CV = Rss_sum / loo.get_n_splits(sele)
        modDict = {"combo" : list(combo) , "CV_score" : CV}
        all_mods.append(modDict)
    ALLmods = pd.DataFrame(all_mods)
    bestMod = ALLmods.nsmallest(5,'CV_score')
    return bestMod
# running perdicted Rsquared
def SST(Y):
    avg_y = np.mean(Y)
    res = (Y - avg_y)**2
    return np.sum(res)
def predR(X,Y,combo):
    c = list(combo)
    sele = X[c]
    loo = LeaveOneOut()
    loo.get_n_splits(sele)
    PRESS = 0
    for tr ,te in loo.split(sele):
        Xtr , Xte = sele.iloc[tr] , sele.iloc[te]
        Ytr , Yte = Y.iloc[tr] , Y.iloc[te]
        trained = processSubset(Ytr,Xtr,combo)
        mod = trained.get("model")
        variables = (list(mod.params.index))
        values = list(mod.params.values)
        formula = {}
        for i in range(0,len(values) ): # do it by index
            formula[str(variables[i])] = values[i]
        result = formula['const']
        for key in formula.keys():
            if key != 'const':
                result += Xte[key] * formula.get(key)
        guess = list(result)[0]
        ans = list(Yte)[0]
        PRESS += (guess -ans)**2
    sst = SST(Y)
    return 1 - (PRESS/sst)
def plot_Residual(X,Y,combo,title):
    dev = 0
    trained = processSubset(Y,X,combo)
    mod = trained.get("model")
    fig = plt.figure(figsize=(20,12))
    #fig = sm.graphics.plot_regress_exog(mod, str(combo[0]), fig=fig)
    t = title[:-4] + "_partialRegression.png"
    fig = sm.graphics.plot_partregress_grid(mod,fig=fig)
    fig.savefig(t)
