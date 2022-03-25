from sklearn.metrics import confusion_matrix
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import RidgeClassifier
import numpy as np
import pandas as pd
import glob
from sklearn.metrics import confusion_matrix
from statistics import mean
import scipy.io
import os
import sys
# Initialization of directory information:
thisDir = '/projects/b1081/member_directories/aporter/TaskRegScripts/SurfaceResiduals/'
dataDir = thisDir
outDir = thisDir + 'output/'
# Subjects and tasks
taskList=['slowreveal','mixed']
sesList = ['ses-1','ses-2','ses-3','ses-4']


def classifyDS_INET(train_task):
    """
    Classifying different subjects (DS) along the same task

    Parameters
    -------------
    classifier : string
        Provide classifier type for analysis, options SVM=LinearSVC(), Log=LogisticRegression(solver='lbfgs'), Ridge=RidgeClassifier()

    Returns
    -------------
    dfDS : DataFrame
        Dataframe consisting of average accuracy across all subjects

    """
    clf=RidgeClassifier()
    OStotal_acc=[]
    SStotal_acc=[]
    dfDS=pd.DataFrame()
    loo = LeaveOneOut()
    cv_scoreList =[]
    #make list based on available subjects in task folder
    subList=[]
    pattern = dataDir+train_task+"/*"
    files = [os.path.basename(x) for x in glob.glob(pattern)]
    for i in files:
        subList.append(i.split('_', 1)[0])

    data=np.array(subList,dtype='<U61')
    sessions=np.array(sesList,dtype='<U61')
    for  train, test in loo.split(data): #test on all other subj
        SSacc_folds = []
        OSacc_folds =[]
        train_sub = data[test] #training subj
        test_subs = data[train] #testing subjs
        testing_set_task = iNets_OS(train_task, test_subs)
        testing_set_rest = iNets_OS('rest', test_subs)
        #make sure rest and task sets are the same amount
        testing_set_rest=testing_set_rest[np.random.choice(testing_set_rest.shape[0], testing_set_task.shape[0], replace=False), :]
        y_test_task=np.ones(testing_set_task.shape[0])
        y_test_rest=np.zeros(testing_set_rest.shape[0])
        ytest=np.concatenate((y_test_task,y_test_rest))
        Xtest=np.concatenate((testing_set_task,testing_set_rest))
        for train_ses, test_ses in loo.split(sessions): #leave one session out
            train_sub_ses = sessions[train_ses]
            val_sub_ses = sessions[test_ses]
            taskFC = iNets_SS(train_task, train_sub[0], train_sub_ses)
            val_task = iNetOpenSes(train_task, train_sub[0], val_sub_ses[0])
            restFC = iNets_SS('rest', train_sub[0], train_sub_ses)
            val_rest = iNetOpenSes('rest', train_sub[0], val_sub_ses[0])

            #make sure rest and task training sets are the same amount
            restFC_sample=restFC[np.random.choice(restFC.shape[0], taskFC.shape[0], replace=False), :]

            t = np.ones(taskFC.shape[0], dtype = int)
            r=np.zeros(restFC_sample.shape[0], dtype=int)
            y = np.concatenate((t,r))
            x = np.concatenate((taskFC, restFC_sample))
            val_rest_sample=val_rest[np.random.choice(val_rest.shape[0],val_task.shape[0], replace = False, :] #match task amount
            val_t = np.ones(val_task.shape[0], dtype = int)
            val_r=np.zeros(val_rest_sample.shape[0], dtype=int)
            y_val = np.concatenate((val_t,val_r))
            X_val= np.concatenate((val_task,val_rest_sample))
            clf.fit(x,y)
            #Same subject
            SSscores=clf.score(X_val,y_val)
            SSacc_folds.append(SSscores)
            #Other subject
            OSscores=clf.score(Xtest,ytest)
            OSacc_folds.append(OSscores)

        OStotal_acc.append(mean(OSacc_folds))
        SStotal_acc.append(mean(SSacc_folds))
    dfDS['subs']=subList
    dfDS['SS'] = SStotal_acc
    dfDS['OS'] = OStotal_acc
    df=pd.melt(dfDS,id_vars=['subs'],value_vars=['SS','OS'],var_name='Analysis',value_name='acc')
    df.to_csv(outDir+train_task+'_iNets_acc_matched_sample.csv',index=False)




def classifyALL_Binary_INET():
    """
    Classifying different subjects (DS) along the same task

    Parameters
    -------------
    classifier : string
        Provide classifier type for analysis, options SVM=LinearSVC(), Log=LogisticRegression(solver='lbfgs'), Ridge=RidgeClassifier()

    Returns
    -------------
    dfDS : DataFrame
        Dataframe consisting of average accuracy across all subjects

    """
    clf=RidgeClassifier()
    OStotal_acc=[]
    SStotal_acc=[]
    dfDS=pd.DataFrame()
    loo = LeaveOneOut()
    cv_scoreList =[]
    #make list based on available subjects in task folder
    subList=[]
    pattern = dataDir+"mixed/*" #grab all subjects with completed mixed
    files = [os.path.basename(x) for x in glob.glob(pattern)]
    for i in files:
        subList.append(i.split('_', 1)[0])

    data=np.array(subList,dtype='<U61')
    sessions=np.array(sesList,dtype='<U61')
    for  train, test in loo.split(data): #test on all other subj
        SSacc_folds = []
        OSacc_folds =[]
        train_sub = data[test] #training subj
        test_subs = data[train] #testing subjs
        test_mixed = iNets_OS('mixed', test_subs)
        test_slow = iNets_OS('slowreveal', test_subs)
        testing_set_rest = iNets_OS('rest', test_subs)
        testing_set_task = np.concatenate((test_mixed, test_slow))
        y_test_task=np.ones(testing_set_task.shape[0])
        y_test_rest=np.zeros(testing_set_rest.shape[0])
        ytest=np.concatenate((y_test_task,y_test_rest))
        Xtest=np.concatenate((testing_set_task,testing_set_rest))
        for train_ses, test_ses in loo.split(sessions): #leave one session out
            train_sub_ses = sessions[train_ses]
            val_sub_ses = sessions[test_ses]

            slowFC = iNets_SS('slowreveal', train_sub[0], train_sub_ses)
            mixFC = iNets_SS('mixed', train_sub[0], train_sub_ses)
            val_slow = iNetOpenSes('slowreveal', train_sub[0], val_sub_ses[0])
            val_mix = iNetOpenSes('mixed', train_sub[0], val_sub_ses[0])
            restFC = iNets_SS('rest', train_sub[0], train_sub_ses)
            val_rest = iNetOpenSes('rest', train_sub[0], val_sub_ses[0])
            taskFC = np.concatenate((mixFC, slowFC))
            val_task=np.concatenate((val_mix, val_slow))
            t = np.ones(taskFC.shape[0], dtype = int)
            r=np.zeros(restFC.shape[0], dtype=int)
            y = np.concatenate((t,r))
            x = np.concatenate((taskFC, restFC))
            #print(train_sub[0], val_task.shape[0],val_rest.shape[0])
            val_t = np.ones(val_task.shape[0], dtype = int)
            val_r=np.zeros(val_rest.shape[0], dtype=int)
            y_val = np.concatenate((val_t,val_r))
            X_val= np.concatenate((val_task,val_rest))
            clf.fit(x,y)
            #Same subject
            SSscores=clf.score(X_val,y_val)
            SSacc_folds.append(SSscores)
            #Other subject
            OSscores=clf.score(Xtest,ytest)
            OSacc_folds.append(OSscores)

        OStotal_acc.append(mean(OSacc_folds))
        SStotal_acc.append(mean(SSacc_folds))
    dfDS['subs']=subList
    dfDS['SS'] = SStotal_acc
    dfDS['OS'] = OStotal_acc
    df=pd.melt(dfDS,id_vars=['subs'],value_vars=['SS','OS'],var_name='Analysis',value_name='acc')
    df.to_csv(outDir+'ALLBinary_iNets_acc.csv',index=False)



def classifyALL_MC_INET():
    """
    Classifying different subjects (DS) along the same task

    Parameters
    -------------
    classifier : string
        Provide classifier type for analysis, options SVM=LinearSVC(), Log=LogisticRegression(solver='lbfgs'), Ridge=RidgeClassifier()

    Returns
    -------------
    dfDS : DataFrame
        Dataframe consisting of average accuracy across all subjects

    """
    clf=RidgeClassifier()

    all_CM_DS=np.zeros((3,3))
    all_CM_CV=np.zeros((3,3))
    OStotal_acc=[]
    SStotal_acc=[]
    dfDS=pd.DataFrame()
    loo = LeaveOneOut()
    cv_scoreList =[]
    #make list based on available subjects in task folder
    subList=[]
    pattern = dataDir+"mixed/*" #grab all subjects with completed mixed
    files = [os.path.basename(x) for x in glob.glob(pattern)]
    for i in files:
        subList.append(i.split('_', 1)[0])

    data=np.array(subList,dtype='<U61')
    sessions=np.array(sesList,dtype='<U61')
    for  train, test in loo.split(data): #test on all other subj
        SSacc_folds = []
        OSacc_folds =[]
        train_sub = data[test] #training subj
        test_subs = data[train] #testing subjs
        test_mixed = iNets_OS('mixed', test_subs)
        test_slow = iNets_OS('slowreveal', test_subs)
        testing_set_rest = iNets_OS('rest', test_subs)
        #make sure rest and task sets are the same amount
        testing_set_rest=testing_set_rest[np.random.choice(testing_set_rest.shape[0], test_mixed.shape[0], replace=False), :]
        y_test_slow = np.full(test_slow.shape[0],2)
        y_test_mix=np.ones(test_mixed.shape[0])
        y_test_rest=np.zeros(testing_set_rest.shape[0])
        ytest=np.concatenate((y_test_slow,y_test_mix,y_test_rest))
        Xtest=np.concatenate((test_slow,test_mixed,testing_set_rest))
        #same_sub_CM=np.zeros((3,3))
        #diff_sub_CM=np.empty((4,3,3)) #number of sessions is the number of splits
        count = 0
        diff_count=0
        for train_ses, test_ses in loo.split(sessions): #leave one session out
            train_sub_ses = sessions[train_ses]
            val_sub_ses = sessions[test_ses]

            slowFC = iNets_SS('slowreveal', train_sub[0], train_sub_ses)
            mixFC = iNets_SS('mixed', train_sub[0], train_sub_ses)
            val_slow = iNetOpenSes('slowreveal', train_sub[0], val_sub_ses[0])
            val_mix = iNetOpenSes('mixed', train_sub[0], val_sub_ses[0])
            restFC = iNets_SS('rest', train_sub[0], train_sub_ses)
            val_rest = iNetOpenSes('rest', train_sub[0], val_sub_ses[0])
            #make sure rest and task sets are the same amount
            val_rest=val_rest[np.random.choice(val_rest.shape[0], val_mix.shape[0], replace=False), :]
            restFC=restFC[np.random.choice(restFC.shape[0], mixFC.shape[0], replace=False), :]

            s = np.full(slowFC.shape[0], 2)
            t = np.ones(mixFC.shape[0], dtype = int)
            r=np.zeros(restFC.shape[0], dtype=int)
            y = np.concatenate((s,t,r))
            x = np.concatenate((slowFC,mixFC, restFC))
            val_s = np.full(val_slow.shape[0],2)
            val_t = np.ones(val_mix.shape[0], dtype = int)
            val_r=np.zeros(val_rest.shape[0], dtype=int)
            y_val = np.concatenate((val_s,val_t,val_r))
            X_val= np.concatenate((val_slow,val_mix,val_rest))
            clf.fit(x,y)
            #Same subject
            SSscores=clf.score(X_val,y_val)
            SSacc_folds.append(SSscores)
            #Other subject
            OSscores=clf.score(Xtest,ytest)
            OSacc_folds.append(OSscores)
            #y_predict=clf.predict(X_val)
            #cm_same = confusion_matrix(y_val, y_predict)
            #same_sub_CM=same_sub_CM+cm_same

            #order check confusion matrices
            #y_pre=clf.predict(Xtest)
            #cm_diff = confusion_matrix(ytest, y_pre)
            #diff_sub_CM[diff_count]=cm_diff
            #diff_count=diff_count+1
            #DS_cm=diff_sub_CM.mean(axis=0,dtype=int)

        OStotal_acc.append(mean(OSacc_folds))
        SStotal_acc.append(mean(SSacc_folds))
        #DS=DS_cm / DS_cm.astype(np.float).sum(axis=1)
        #CV=same_sub_CM / same_sub_CM.astype(np.float).sum(axis=1)
        #all_CM_DS=DS_cm+all_CM_DS
        #all_CM_CV=same_sub_CM+all_CM_CV
    #finalDS=all_CM_DS / all_CM_DS.astype(np.float).sum(axis=1)
    #finalCV=all_CM_CV / all_CM_CV.astype(np.float).sum(axis=1)
    dfDS['subs']=subList
    dfDS['SS'] = SStotal_acc
    dfDS['OS'] = OStotal_acc
    df=pd.melt(dfDS,id_vars=['subs'],value_vars=['SS','OS'],var_name='Analysis',value_name='acc')
    df.to_csv(outDir+'ALL_MC_iNets_acc.csv',index=False)
    #np.savetxt(outDir+'CM_DS.csv', finalDS)
    #np.savetxt(outDir+'CM_SS.csv', finalCV)


def iNets_SS(train_task, sub, sesList):
    """
    Calculate FC all sessions matrices in list for a sub in a given task
    Parameters
    -----------
    df : str
        Path to file
    Returns
    -----------
    ds : 2D upper triangle FC measures in (roi, days) format

    """
    result_arr = []
    #loop through and append all test sets do each task/rest separate
    for session in sesList:
        all_ses_mats = iNetOpenSes(train_task, sub, session)
        result_arr.append(all_ses_mats)
    result_arr = np.concatenate(result_arr)
    return result_arr

def iNets_OS(train_task, test_subs):
    """
    Calculate FC matrices for all subs in a given task
    Parameters
    -----------
    df : str
        Path to file
    Returns
    -----------
    ds : 2D upper triangle FC measures in (roi, days) format

    """
    result_arr = []
    #loop through and append all test sets do each task/rest separate
    for sub in test_subs:
        all_sub_mats = iNetOpenALL(train_task, sub)
        result_arr.append(all_sub_mats)
    result_arr = np.concatenate(result_arr)
    return result_arr

def iNetOpenALL(train_task, sub):
    """
    Convert matlab files into upper triangle np.arrays for a given sub (all sessions/runs)
    Parameters
    -----------
    df : str
        Path to file
    Returns
    -----------
    ds : 2D upper triangle FC measures in (roi, days) format

    """
    #find all files
    files = glob.glob(dataDir+train_task+'/'+sub+'_ses-*')
    nsess = len(files) #FC matrices
    nrois=333
    ds=np.empty((nsess, int(nrois*(nrois-1)/2)))
    count=0
    for f in files:
    #Consistent parameters to use for editing datasets

        #Load FC file
        fileFC=scipy.io.loadmat(f)

        #Convert to numpy array
        fileFC=np.array(fileFC['parcel_corrmat'])
        #Replace nans and infs with zero
        fileFC=np.nan_to_num(fileFC)
        #Index upper triangle of matrix
        mask=np.triu_indices(nrois,1)

    #Loop through all 10 days to reshape correlations into linear form

        tmp=fileFC[:,:]
        ds[count]=tmp[mask]
        count=count+1
    mask = (ds == 0).all(1)
    column_indices = np.where(mask)[0]
    df = ds[~mask,:]
    return df


def iNetOpenSes(train_task, sub, ses):
    """
    Convert matlab files into upper triangle np.arrays for a given session
    Parameters
    -----------
    df : str
        Path to file
    Returns
    -----------
    ds : 2D upper triangle FC measures in (roi, days) format

    """
    #find all files in a given session
    files = glob.glob(dataDir+train_task+'/'+sub+'_'+ses+'*')
    nsess = len(files) #FC matrices
    nrois=333
    ds=np.empty((nsess, int(nrois*(nrois-1)/2)))
    count=0
    for f in files:
    #Consistent parameters to use for editing datasets

        #Load FC file
        fileFC=scipy.io.loadmat(f)

        #Convert to numpy array
        fileFC=np.array(fileFC['parcel_corrmat'])
        #Replace nans and infs with zero
        fileFC=np.nan_to_num(fileFC)
        #Index upper triangle of matrix
        mask=np.triu_indices(nrois,1)

    #Loop through all 10 days to reshape correlations into linear form

        tmp=fileFC[:,:]
        ds[count]=tmp[mask]
        count=count+1
    mask = (ds == 0).all(1)
    column_indices = np.where(mask)[0]
    df = ds[~mask,:]
    return df


for task in taskList:
    classifyDS_INET(task)
#classifyDS_INET('mixed')
classifyALL_Binary_INET()
classifyALL_MC_INET()
