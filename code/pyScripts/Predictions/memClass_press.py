#!/usr/bin/env python
# coding: utf-8

# In[ ]:
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import RidgeClassifier
import numpy as np
import os
import sys
import pandas as pd
import itertools
import reshape
from statistics import mean
import scipy.io
#import results
import warnings
warnings.filterwarnings("ignore")
# Initialization of directory information:
thisDir = os.path.expanduser('~/Desktop/Porteretal_taskprediction/')
dataDir = thisDir + 'data/corrmats/'
outDir = thisDir + 'output/results/'
figsDir=thisDir + 'output/figures/'
IndNetDir=thisDir+'data/IndNet/'
# Subjects and tasks
taskList=['pres1','pres2','pres3']
#omitting MSC06 for classify All
#subList=['MSC01','MSC02','MSC03','MSC04','MSC05','MSC07','MSC10']
subList=['MSC01','MSC02','MSC03','MSC04','MSC05','MSC06','MSC07','MSC10']
#all possible combinations of subs and tasks
subsComb=(list(itertools.permutations(subList, 2)))
#tasksComb=(list(itertools.permutations(taskList, 2)))
tasksComb=(list(itertools.combinations(taskList, 2)))

task=['glass','semantic', 'motor','mem']
task_sep_Perm=(list(itertools.permutations(task, 2)))

btw=list(itertools.product(list(subsComb),list(task_sep_Perm))) #diff sub diff task
wtn = list(itertools.product(list(subList),list(task_sep_Perm)))

tasksPerm=(list(itertools.combinations(taskList, 2)))

DSvars=list(itertools.product(list(subsComb),list(taskList)))
splitDict=dict([('MSC01',10),('MSC02',10),('MSC03',8),('MSC04',10),('MSC05',10),('MSC06',9),('MSC07',9),('MSC10',10)])
taskDict=dict([('mem',0),('motor',1),('glass',2),('semantic',3)])
SSvars=list(itertools.product(list(subList),list(tasksComb))) #same sub diff task
BSvars=list(itertools.product(list(subsComb),list(tasksPerm))) #diff sub diff task


def classifydiffMem(classifier='string'):
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
    same_sub_per_task=[] #SS same sub
    diff_sub_per_task=[] #OS other sub
    tmp_BS=pd.DataFrame(BSvars, columns=['sub','task'])
    dfDS=pd.DataFrame()
    dfDS[['train_task','test_task']]=pd.DataFrame(tmp_BS['task'].tolist())
    dfDS[['train_sub', 'test_sub']]=pd.DataFrame(tmp_BS['sub'].tolist())

    for index, row in dfDS.iterrows():
        taskFC=reshape.matFiles(dataDir+'mem/'+row['train_task']+'/'+row['train_sub']+'_parcel_corrmat.mat')
        restFC=reshape.matFiles(dataDir+'mem/'+row['test_task']+'/'+row['train_sub']+'_parcel_corrmat.mat')
        test_taskFC=reshape.matFiles(dataDir+'mem/'+row['train_task']+'/'+row['test_sub']+'_parcel_corrmat.mat')
        test_restFC=reshape.matFiles(dataDir+'mem/'+row['test_task']+'/'+row['test_sub']+'_parcel_corrmat.mat')
        SSacc, OSacc=folds(clf, taskFC, restFC, test_taskFC, test_restFC)
        same_sub_per_task.append(SSacc)
        diff_sub_per_task.append(OSacc)
    dfDS['diff_sub']=diff_sub_per_task
    dfDS['same_sub']=same_sub_per_task
    dfDS.to_csv(outDir+'Ridge/single_task/sep_mem_pres_accCG.csv',index=False)
def folds(clf,taskFC, restFC, test_taskFC, test_restFC):
    """
    Cross validation to train and test using nested loops

    Parameters
    -----------
    clf : obj
        Machine learning algorithm
    analysis : str
        Analysis type
    taskFC, restFC, test_taskFC, test_restFC : array_like
        Input arrays, training and testing set of task and rest FC
    Returns
    -----------
    total_score : float
        Average accuracy across folds
    acc_score : list
        List of accuracy for each outer fold
    """

    loo = LeaveOneOut()
    taskSize=taskFC.shape[0]
    restSize=restFC.shape[0]
    t = np.ones(taskSize, dtype = int)
    r=np.zeros(restSize, dtype=int)
    y_test_task=np.ones(test_taskFC.shape[0])
    y_test_rest=np.zeros(test_restFC.shape[0])
    ytest=np.concatenate((y_test_task,y_test_rest))
    Xtest=np.concatenate((test_taskFC,test_restFC))
    #Test same sub
    SS_acc=[]
    #Test other subs
    OS_acc=[]

    #fold each training set
    for train_index, test_index in loo.split(taskFC):
        Xtrain_rest,Xval_rest=restFC[train_index],restFC[test_index]
        Xtrain_task,Xval_task=taskFC[train_index], taskFC[test_index]
        ytrain_rest,yval_rest=r[train_index], r[test_index]
        ytrain_task,yval_task=t[train_index], t[test_index]
        X_tr=np.concatenate((Xtrain_task, Xtrain_rest))
        y_tr = np.concatenate((ytrain_task,ytrain_rest))
        X_val=np.concatenate((Xval_task, Xval_rest))
        y_val = np.concatenate((yval_task,yval_rest))
        clf.fit(X_tr,y_tr)
        #Same subject
        SSscores=clf.score(X_val,y_val)
        SS_acc.append(SSscores)
        #Other subject
        OSscores=clf.score(Xtest,ytest)
        OS_acc.append(OSscores)
    OStotal_acc=mean(OS_acc)
    SStotal_acc=mean(SS_acc)
    return SStotal_acc, OStotal_acc



def groupApp():
    """
    Classifying using all subs testing unseen sub on same task
    Parameters
    -------------


    Returns
    -------------
    dfGroup : DataFrame
        Dataframe consisting of group average accuracy training with subs instead of session

    """
    wtn_scores=[]
    btw_scores = []
    dfGroup=pd.DataFrame(tasksPerm, columns=['train_task','test_task'])
    for index, row in dfGroup.iterrows():
        wtn_acc=model(row['train_task'], row['test_task'])
        wtn_scores.append(wtn_acc)
        #btw_scores.append(btw_acc)
    dfGroup['groupwise']=wtn_scores
    #dfGroup['groupwise']=btw_scores
    #save accuracy
    dfGroup.to_csv(outDir+'Ridge/single_task/memPres_groupwise_acc.csv',index=False)

def model(train_task, test_task):
    """
    Preparing machine learning model with appropriate data

    Parameters
    -------------
    train_task : str
            Task name for training
    test_task : str
            Task name for testing

    Returns
    -------------
    accuracy: for testing on different presentation of different subject

    """

    clf=RidgeClassifier()
    loo = LeaveOneOut()
    #df=pd.DataFrame()
    #nsess x fc x nsub
    ds_T=np.empty((8,55278,8))
    ds_R=np.empty((8,55278,8))
    ds_Test=np.empty((8,55278,8))
    count=0
    #get all subs for a given task
    for sub in subList:
        #training task
        tmp_taskFC=reshape.matFiles(dataDir+'mem/'+train_task+'/'+sub+'_parcel_corrmat.mat')
        tmp_taskFC=tmp_taskFC[:8,:]
        tmp_restFC=reshape.matFiles(dataDir+'mem/'+test_task+'/'+sub+'_parcel_corrmat.mat')
        tmp_restFC=tmp_restFC[:8,:]
        #reshape 2d into 3d nsessxfcxnsubs
        ds_T[:,:,count]=tmp_taskFC
        ds_R[:,:,count]=tmp_restFC
        #testing task
        #test_taskFC=reshape.matFiles(dataDir+'mem/'+test_task+'/'+sub+'_parcel_corrmat.mat')
        #test_taskFC=test_taskFC[:8,:]
        #ds_Test[:,:,count]=test_taskFC
        count=count+1
    sess_wtn_score=[]
    sess_btn_score=[]
    #train leave one sub out
    #test on left out sub
    #test on new task of left out sub
    clf=RidgeClassifier()
    loo = LeaveOneOut()
    for i in range(8):
    #takes one session of data (7 subs)
        taskFC=ds_T[i,:,:]
        taskFC=taskFC.T
        restFC=ds_R[i,:,:]
        restFC=restFC.T
        #testFC = df_Test[i,:,:]
        #testFC=testFC.T
        #yTest = np.ones(testFC.shape[0])
        taskSize=taskFC.shape[0]
        restSize=restFC.shape[0]
        t = np.ones(taskSize, dtype = int)
        r=np.zeros(restSize, dtype=int)

        #btn_scoreList=[]
    #fold each training set (sub)
        for train_index, test_index in loo.split(taskFC):
            Xtrain_rest, Xtest_rest=restFC[train_index], restFC[test_index]
            Xtrain_task, Xtest_task=taskFC[train_index], taskFC[test_index]
            #XNew_test_task, yNew_test_task = testFC[test_index], yTest[test_index]
            #Xval is the new task 10 sessions
            #just the left out sub data
            #testFC=ds_Test[:,:,test_index[0]]
            ytrain_rest,ytest_rest=r[train_index], r[test_index]
            ytrain_task,ytest_task=t[train_index], t[test_index]
            X_tr=np.concatenate((Xtrain_task, Xtrain_rest))
            y_tr = np.concatenate((ytrain_task,ytrain_rest))
            clf.fit(X_tr,y_tr)
            X_test=np.concatenate((Xtest_task, Xtest_rest))
            ytest=np.concatenate((ytest_task, ytest_rest))
            ACCscores=clf.score(X_test,ytest)
            sess_wtn_score.append(ACCscores)
            #testSize=testFC.shape[0]
            #y_val= np.ones(testFC.shape[0], dtype = int)
            #test unseen sub new task all 10 sessions
            #y_pre_val=clf.predict(testFC)
            #btn_score=clf.score(testFC, y_val)
            #sess_btn_score.append(btn_score)
    wtn_score=mean(sess_wtn_score)
    #df['btn_fold']=btn_scoreList
    #btn_score=mean(sess_btn_score)
    #return wtn_scoreList
    return wtn_score#, btn_score




def classifydiffTask():
    """
    Classifying subjects (DS) along the different task

    Parameters
    -------------

    Returns
    -------------
    df : DataFrame
        Dataframe consisting of average accuracy across all subjects

    """
    clf=RidgeClassifier()
    diff_sub_per_task=[] #OS other task
    tmp_BS=pd.DataFrame(btw, columns=['sub','task'])
    dfDS=pd.DataFrame()
    dfDS[['train_task','test_task']]=pd.DataFrame(tmp_BS['task'].tolist())
    dfDS[['train_sub', 'test_sub']]=pd.DataFrame(tmp_BS['sub'].tolist())
    dfDS['Analysis'] = "Different Person"
    tmp_SS=pd.DataFrame(wtn, columns=['sub','task'])
    dfSS=pd.DataFrame()
    dfSS[['train_task','test_task']]=pd.DataFrame(tmp_SS['task'].tolist())
    dfSS[['train_sub']]=pd.DataFrame(tmp_SS['sub'].tolist())
    dfSS['test_sub']=dfSS['train_sub']
    dfSS['Analysis'] = 'Same Person'
    total_comb = pd.concat([dfDS,dfSS])
    for index, row in total_comb.iterrows():
        taskFC=reshape.matFiles(dataDir+row['train_task']+'/'+row['train_sub']+'_parcel_corrmat.mat')
        restFC=reshape.matFiles(dataDir+'rest/'+row['train_sub']+'_parcel_corrmat.mat')
        test_taskFC=reshape.matFiles(dataDir+row['test_task']+'/'+row['test_sub']+'_parcel_corrmat.mat')
        OSacc=across_task_folds(clf, taskFC, restFC, test_taskFC)
        diff_sub_per_task.append(OSacc)
    total_comb['acc']=diff_sub_per_task
    total_comb.to_csv(outDir+'Ridge/single_task/across_task_acc.csv',index=False)
def across_task_folds(clf,taskFC, restFC, Xtest):
    """
    Cross validation to train and test using nested loops

    Parameters
    -----------
    clf : obj
        Machine learning algorithm
    analysis : str
        Analysis type
    taskFC, restFC, test_taskFC, test_restFC : array_like
        Input arrays, training and testing set of task and rest FC
    Returns
    -----------
    total_score : float
        Average accuracy across folds
    acc_score : list
        List of accuracy for each outer fold
    """

    loo = LeaveOneOut()
    taskSize=taskFC.shape[0]
    restSize=restFC.shape[0]
    t = np.ones(taskSize, dtype = int)
    r=np.zeros(restSize, dtype=int)
    ytest=np.ones(Xtest.shape[0])

    #Test task
    OS_acc=[]

    #fold each training set
    for train_index, test_index in loo.split(taskFC):
        Xtrain_rest=restFC[train_index]
        Xtrain_task=taskFC[train_index]
        ytrain_rest=r[train_index]
        ytrain_task=t[train_index]
        X_tr=np.concatenate((Xtrain_task, Xtrain_rest))
        y_tr = np.concatenate((ytrain_task,ytrain_rest))
        clf.fit(X_tr,y_tr)
        #Other task
        OSscores=clf.score(Xtest,ytest)
        OS_acc.append(OSscores)
    OStotal_acc=mean(OS_acc)
    return OStotal_acc
