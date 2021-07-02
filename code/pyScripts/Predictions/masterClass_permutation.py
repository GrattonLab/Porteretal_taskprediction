#!/usr/bin/env python
# coding: utf-8

# In[ ]:
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
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
#outDir = thisDir + 'output/results/'
outDir = thisDir + 'output/results/'
figsDir=thisDir + 'output/figures/'
IndNetDir=thisDir+'data/IndNet/'
# Subjects and tasks
taskList=['glass','semantic', 'motor','mem']
#omitting MSC06 for classify All
#subList=['MSC01','MSC02','MSC03','MSC04','MSC05','MSC07','MSC10']
subList=['MSC01','MSC02','MSC03','MSC04','MSC05','MSC06','MSC07','MSC10']
#all possible combinations of subs and tasks
subsComb=(list(itertools.permutations(subList, 2)))
tasksComb=(list(itertools.permutations(taskList, 2)))
DSvars=list(itertools.product(list(subsComb),list(taskList)))
splitDict=dict([('MSC01',10),('MSC02',10),('MSC03',8),('MSC04',10),('MSC05',10),('MSC06',9),('MSC07',9),('MSC10',10)])
clfDict=dict([('Ridge',RidgeClassifier()),('Log',LogisticRegression(solver='lbfgs')),('SVM',LinearSVC())])
taskDict=dict([('mem',0),('motor',1),('glass',2),('semantic',3)])

def classifyDS(classifier='string'):
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
    clf=clfDict[classifier]
    same_sub_per_task=[] #SS same sub
    diff_sub_per_task=[] #OS other sub
    tmp_df=pd.DataFrame(DSvars, columns=['sub','task'])
    dfDS=pd.DataFrame()
    dfDS[['train_sub','test_sub']]=pd.DataFrame(tmp_df['sub'].tolist())
    dfDS['task']=tmp_df['task']
    for index, row in dfDS.iterrows():
        taskFC=reshape.matFiles(dataDir+row['task']+'/'+row['train_sub']+'_parcel_corrmat.mat')
        restFC=reshape.matFiles(dataDir+'rest/'+row['train_sub']+'_parcel_corrmat.mat')
        test_taskFC=reshape.matFiles(dataDir+row['task']+'/'+row['test_sub']+'_parcel_corrmat.mat')
        test_restFC=reshape.matFiles(dataDir+'rest/'+row['test_sub']+'_parcel_corrmat.mat')
        SSacc, OSacc=folds(clf, taskFC, restFC, test_taskFC, test_restFC)
        same_sub_per_task.append(SSacc)
        diff_sub_per_task.append(OSacc)
    SS=mean(same_sub_per_task)
    OS=mean(diff_sub_per_task)
    return SS, OS

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
        y_tr=np.random.permutation(y_tr)
        clf.fit(X_tr,y_tr)
        SSscores=clf.score(X_val,y_val)
        SS_acc.append(SSscores)
        OSscores=clf.score(Xtest,ytest)
        OS_acc.append(OSscores)
    OStotal_acc=mean(OS_acc)
    SStotal_acc=mean(SS_acc)
    return SStotal_acc, OStotal_acc

def classifyAll(classifier='Ridge'):
    """
    Classifying different subjects along available data rest split into 40 samples to match with task

    Parameters
    -------------

    Returns
    -------------
    df : DataFrame
        Dataframe consisting of average accuracy across all subjects

    """
    clf=clfDict[classifier]
    acc_scores_OS=[]
    acc_scores_SS=[]
    df=pd.DataFrame(subsComb, columns=['train_sub','test_sub'])
    for index, row in df.iterrows():
        SS_score, OS_score=modelAll_binary(clf,train_sub=row['train_sub'], test_sub=row['test_sub'])
        acc_scores_SS.append(SS_score)
        acc_scores_OS.append(OS_score)
    OS=mean(acc_scores_OS)
    SS=mean(acc_scores_SS)
    return SS, OS


def modelAll_binary(clf,train_sub, test_sub):
    """
    Preparing machine learning model with appropriate data

    Parameters
    -------------
    train_sub : str
            Subject name for training
    test_sub : str
            Subject name for testing

    Returns
    -------------
    total_score : float
            Average accuracy of all folds

    """
    #train sub
    memFC=reshape.matFiles(dataDir+'mem/'+train_sub+'_parcel_corrmat.mat')
    semFC=reshape.matFiles(dataDir+'semantic/'+train_sub+'_parcel_corrmat.mat')
    glassFC=reshape.matFiles(dataDir+'glass/'+train_sub+'_parcel_corrmat.mat')
    motFC=reshape.matFiles(dataDir+'motor/'+train_sub+'_parcel_corrmat.mat')
    restFC=reshape.matFiles(dataDir+'rest/corrmats_timesplit/fourths/'+train_sub+'_parcel_corrmat.mat') #keep tasks seperated in order to collect the right amount of days
    restFC=np.reshape(restFC,(10,4,55278)) #reshape to gather correct days
    #test sub
    test_memFC=reshape.matFiles(dataDir+'mem/'+test_sub+'_parcel_corrmat.mat')
    test_semFC=reshape.matFiles(dataDir+'semantic/'+test_sub+'_parcel_corrmat.mat')
    test_glassFC=reshape.matFiles(dataDir+'glass/'+test_sub+'_parcel_corrmat.mat')
    test_motFC=reshape.matFiles(dataDir+'motor/'+test_sub+'_parcel_corrmat.mat')
    test_restFC=reshape.matFiles(dataDir+'rest/corrmats_timesplit/fourths/'+test_sub+'_parcel_corrmat.mat')
    test_taskFC=np.concatenate((test_memFC,test_semFC,test_glassFC,test_motFC))

    CV_score, DS_score=foldsBinary(train_sub, clf, memFC,semFC,glassFC,motFC, restFC, test_taskFC,test_restFC)
    return CV_score, DS_score

def foldsBinary(train_sub, clf, memFC,semFC,glassFC,motFC,restFC, test_taskFC, test_restFC):
    """
    Cross validation to train and test using nested loops

    Parameters
    -----------
    clf : obj
        Machine learning algorithm
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
    test_taskSize=test_taskFC.shape[0]
    test_restSize=test_restFC.shape[0]
    testT= np.ones(test_taskSize, dtype = int)
    testR= np.zeros(test_restSize, dtype = int)
    ytest=np.concatenate((testT,testR))
    Xtest=np.concatenate((test_taskFC,test_restFC))
    CVacc=[]
    DSacc=[]
    #fold each training set
    session=splitDict[train_sub]
    split=np.empty((session, 55278))
    for train_index, test_index in loo.split(split):
        memtrain, memval=memFC[train_index], memFC[test_index]
        semtrain, semval=semFC[train_index], semFC[test_index]
        mottrain, motval=motFC[train_index], motFC[test_index]
        glatrain, glaval=glassFC[train_index], glassFC[test_index]
        Xtrain_task=np.concatenate((memtrain,semtrain,mottrain,glatrain))
        Xtrain_rest, Xval_rest=restFC[train_index,:,:], restFC[test_index,:,:]
        Xval_task=np.concatenate((memval,semval,motval,glaval))
        Xtrain_rest=np.reshape(Xtrain_rest,(-1,55278))
        Xval_rest=np.reshape(Xval_rest,(-1,55278))
        ytrain_task = np.ones(Xtrain_task.shape[0], dtype = int)
        ytrain_rest=np.zeros(Xtrain_rest.shape[0], dtype=int)
        yval_task = np.ones(Xval_task.shape[0], dtype = int)
        yval_rest=np.zeros(Xval_rest.shape[0], dtype=int)
        X_tr=np.concatenate((Xtrain_task, Xtrain_rest))
        X_val=np.concatenate((Xval_task, Xval_rest))
        y_tr = np.concatenate((ytrain_task,ytrain_rest))
        y_val=np.concatenate((yval_task, yval_rest))
        y_tr=np.random.permutation(y_tr)
        clf.fit(X_tr,y_tr)
        CV_score=clf.score(X_val, y_val)
        CVacc.append(CV_score)
        ACCscores=clf.score(Xtest,ytest)
        DSacc.append(ACCscores)
    CV_score=mean(CVacc)
    DS_score=mean(DSacc)
    return  CV_score, DS_score

def multiclassAll(classifier='Ridge'):
    """
    Preparing machine learning model with appropriate data

    Parameters
    -------------
    train_sub : str
            Subject name for training
    test_sub : str
            Subject name for testing

    Returns
    -------------
    total_score : float
            Average accuracy of all folds

    """
    clf=clfDict[classifier]
    data=np.array(['MSC01','MSC02','MSC03','MSC04','MSC05','MSC06','MSC07','MSC10'],dtype='<U61')
    loo = LeaveOneOut()
    SS_per_sub=[]
    OS_per_sub=[]
    for  test, train in loo.split(data): #train on one sub test on the rest
        CV_tmp=pd.DataFrame()
        DS_tmp=pd.DataFrame()
        train_sub=data[train]
        test_sub=data[test]
    #train sub
        memFC=reshape.matFiles(dataDir+'mem/'+train_sub[0]+'_parcel_corrmat.mat')
        semFC=reshape.matFiles(dataDir+'semantic/'+train_sub[0]+'_parcel_corrmat.mat')
        glassFC=reshape.matFiles(dataDir+'glass/'+train_sub[0]+'_parcel_corrmat.mat')
        motFC=reshape.matFiles(dataDir+'motor/'+train_sub[0]+'_parcel_corrmat.mat')
        restFC=reshape.matFiles(dataDir+'rest/'+train_sub[0]+'_parcel_corrmat.mat') #keep tasks seperated in order to collect the right amount of days
        #nsize=restFC.shape[1]
        #restFC=np.reshape(restFC,(10,4,nsize)) #reshape to gather correct days
        #test sub
        testFC,ytest=AllSubFiles(test_sub)
        same_Tsub, diff_Tsub=folds_MC(train_sub, clf, memFC,semFC,glassFC,motFC, restFC, testFC, ytest)
        SS_per_sub.append(same_Tsub)
        OS_per_sub.append(diff_Tsub)
    SS=mean(SS_per_sub)
    OS=mean(OS_per_sub)
    return SS, OS

def folds_MC(train_sub, clf, memFC,semFC,glassFC,motFC, restFC, testFC,ytest):
    """
    Cross validation to train and test using nested loops

    Parameters
    -----------
    clf : obj
        Machine learning algorithm
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
    yrest=np.zeros(restFC.shape[0])
    ymem=np.ones(memFC.shape[0])
    ysem=np.full(semFC.shape[0],2)
    yglass=np.full(glassFC.shape[0],3)
    ymot=np.full(motFC.shape[0],4)
    CVTacc=[]
    DSTacc=[]
    session=splitDict[train_sub[0]]
    split=np.empty((session, 55278))
    for train_index, test_index in loo.split(split):
        memtrain, memval=memFC[train_index], memFC[test_index]
        ymemtrain, ymemval=ymem[train_index], ymem[test_index]
        semtrain, semval=semFC[train_index], semFC[test_index]
        ysemtrain, ysemval=ysem[train_index],ysem[test_index]
        mottrain, motval=motFC[train_index], motFC[test_index]
        ymottrain, ymotval=ymot[train_index],ymot[test_index]
        glatrain, glaval=glassFC[train_index], glassFC[test_index]
        yglatrain,yglaval=yglass[train_index],yglass[test_index]
        resttrain, restval=restFC[train_index], restFC[test_index]
        yresttrain, yrestval=yrest[train_index],yrest[test_index]
        Xtrain=np.concatenate((resttrain,memtrain,semtrain,mottrain,glatrain))
        ytrain=np.concatenate((yresttrain,ymemtrain,ysemtrain,ymottrain,yglatrain))
        yval=np.concatenate((yrestval,ymemval,ysemval,ymotval,yglaval))
        Xval=np.concatenate((restval,memval,semval,motval,glaval))
        ytrain=np.random.permutation(ytrain)
        clf.fit(Xtrain,ytrain)
        score=clf.score(Xval, yval)
        CVTacc.append(score)
        scoreT=clf.score(testFC,ytest)
        DSTacc.append(scoreT)
    same_Tsub=mean(CVTacc)
    diff_Tsub=mean(DSTacc)
    return same_Tsub,diff_Tsub

def AllSubFiles(test_sub):
    """
    Return task and rest FC all subs
    Parameters
    -----------
    test_sub: Array of testing subs
    Returns
    ------------
    taskFC, restFC : Array of task and rest FC of all testing subs
    """
    a_memFC=reshape.matFiles(dataDir+'mem/'+test_sub[0]+'_parcel_corrmat.mat')
    a_semFC=reshape.matFiles(dataDir+'semantic/'+test_sub[0]+'_parcel_corrmat.mat')
    a_glassFC=reshape.matFiles(dataDir+'glass/'+test_sub[0]+'_parcel_corrmat.mat')
    a_motFC=reshape.matFiles(dataDir+'motor/'+test_sub[0]+'_parcel_corrmat.mat')
    a_restFC=reshape.matFiles(dataDir+'rest/'+test_sub[0]+'_parcel_corrmat.mat')

    b_memFC=reshape.matFiles(dataDir+'mem/'+test_sub[1]+'_parcel_corrmat.mat')
    b_semFC=reshape.matFiles(dataDir+'semantic/'+test_sub[1]+'_parcel_corrmat.mat')
    b_glassFC=reshape.matFiles(dataDir+'glass/'+test_sub[1]+'_parcel_corrmat.mat')
    b_motFC=reshape.matFiles(dataDir+'motor/'+test_sub[1]+'_parcel_corrmat.mat')
    b_restFC=reshape.matFiles(dataDir+'rest/'+test_sub[1]+'_parcel_corrmat.mat')

    c_memFC=reshape.matFiles(dataDir+'mem/'+test_sub[2]+'_parcel_corrmat.mat')
    c_semFC=reshape.matFiles(dataDir+'semantic/'+test_sub[2]+'_parcel_corrmat.mat')
    c_glassFC=reshape.matFiles(dataDir+'glass/'+test_sub[2]+'_parcel_corrmat.mat')
    c_motFC=reshape.matFiles(dataDir+'motor/'+test_sub[2]+'_parcel_corrmat.mat')
    c_restFC=reshape.matFiles(dataDir+'rest/'+test_sub[2]+'_parcel_corrmat.mat')

    d_memFC=reshape.matFiles(dataDir+'mem/'+test_sub[3]+'_parcel_corrmat.mat')
    d_semFC=reshape.matFiles(dataDir+'semantic/'+test_sub[3]+'_parcel_corrmat.mat')
    d_glassFC=reshape.matFiles(dataDir+'glass/'+test_sub[3]+'_parcel_corrmat.mat')
    d_motFC=reshape.matFiles(dataDir+'motor/'+test_sub[3]+'_parcel_corrmat.mat')
    d_restFC=reshape.matFiles(dataDir+'rest/'+test_sub[3]+'_parcel_corrmat.mat')

    e_memFC=reshape.matFiles(dataDir+'mem/'+test_sub[4]+'_parcel_corrmat.mat')
    e_semFC=reshape.matFiles(dataDir+'semantic/'+test_sub[4]+'_parcel_corrmat.mat')
    e_glassFC=reshape.matFiles(dataDir+'glass/'+test_sub[4]+'_parcel_corrmat.mat')
    e_motFC=reshape.matFiles(dataDir+'motor/'+test_sub[4]+'_parcel_corrmat.mat')
    e_restFC=reshape.matFiles(dataDir+'rest/'+test_sub[4]+'_parcel_corrmat.mat')

    f_memFC=reshape.matFiles(dataDir+'mem/'+test_sub[5]+'_parcel_corrmat.mat')
    f_semFC=reshape.matFiles(dataDir+'semantic/'+test_sub[5]+'_parcel_corrmat.mat')
    f_glassFC=reshape.matFiles(dataDir+'glass/'+test_sub[5]+'_parcel_corrmat.mat')
    f_motFC=reshape.matFiles(dataDir+'motor/'+test_sub[5]+'_parcel_corrmat.mat')
    f_restFC=reshape.matFiles(dataDir+'rest/'+test_sub[5]+'_parcel_corrmat.mat')

    g_memFC=reshape.matFiles(dataDir+'mem/'+test_sub[6]+'_parcel_corrmat.mat')
    g_semFC=reshape.matFiles(dataDir+'semantic/'+test_sub[6]+'_parcel_corrmat.mat')
    g_glassFC=reshape.matFiles(dataDir+'glass/'+test_sub[6]+'_parcel_corrmat.mat')
    g_motFC=reshape.matFiles(dataDir+'motor/'+test_sub[6]+'_parcel_corrmat.mat')
    g_restFC=reshape.matFiles(dataDir+'rest/'+test_sub[6]+'_parcel_corrmat.mat')

    restFC=np.concatenate((a_restFC,b_restFC,c_restFC,d_restFC,e_restFC,f_restFC,g_restFC))
    test_restSize=restFC.shape[0]
    testR= np.zeros(test_restSize, dtype = int)
    memFC=np.concatenate((a_memFC,b_memFC,c_memFC,d_memFC,e_memFC,f_memFC,g_memFC))
    ymem=np.ones(memFC.shape[0])

    semFC=np.concatenate((a_semFC,b_semFC,c_semFC,d_semFC,e_semFC,f_semFC,g_semFC))
    ysem=np.full(semFC.shape[0],2)

    glassFC=np.concatenate((a_glassFC,b_glassFC,c_glassFC,d_glassFC,e_glassFC,f_glassFC,g_glassFC))
    yglass=np.full(glassFC.shape[0],3)

    motFC=np.concatenate((a_motFC,b_motFC,c_motFC,d_motFC,e_motFC,f_motFC,g_motFC))
    ymot=np.full(motFC.shape[0],4)

    testFC=np.concatenate((restFC,memFC,semFC,motFC, glassFC))
    ytest=np.concatenate((testR,ymem,ysem,ymot,yglass))

    return testFC,ytest


def netFile(netSpec,sub):
    """
    Open appropriate network based file
    """
    #rest will be handled differently because splitting into 4 parts in the timeseries to match
    #zero based indexing
    subDict=dict([('MSC01',0),('MSC02',1),('MSC03',2),('MSC04',3),('MSC05',4),('MSC06',5),('MSC07',6),('MSC10',9)])
    taskDict=dict([('mem','AllMem'),('mixed','AllGlass'),('motor','AllMotor')])
    #fullTask=np.empty((40,120))
    fullRest=np.empty((40,120))
    #memory
    tmp=IndNetDir+'mem/allsubs_mem_corrmats_bysess_orig_INDformat.mat'
    fileFC=scipy.io.loadmat(tmp,struct_as_record=False,squeeze_me=False)
    fileFC=fileFC['sess_task_corrmat']
    fileFC=fileFC[0,0].AllMem
    fileFC=fileFC[0,subDict[sub]]
    fileFC=np.nan_to_num(fileFC)
    nrois=14
    nsess=fileFC.shape[2]
    #Index upper triangle of matrix
    mask=np.triu_indices(nrois,1)
    memFC=np.empty((nsess, int(nrois*(nrois-1)/2)))
    count=0
    #Loop through all 10 days to reshape correlations into linear form
    for sess in range(nsess):
        tmp=fileFC[:,:,sess]
        memFC[count]=tmp[mask]
        count=count+1
    mask = (memFC == 0).all(1)
    column_indices = np.where(mask)[0]
    memFC = memFC[~mask,:]
    #fullTask[:10]=ds
    #motor
    tmp=IndNetDir+'motor/allsubs_motor_corrmats_bysess_orig_INDformat.mat'
    fileFC=scipy.io.loadmat(tmp,struct_as_record=False,squeeze_me=False)
    fileFC=fileFC['sess_task_corrmat']
    fileFC=fileFC[0,0].AllMotor
    fileFC=fileFC[0,subDict[sub]]
    fileFC=np.nan_to_num(fileFC)
    nrois=14
    nsess=fileFC.shape[2]
    #Index upper triangle of matrix
    mask=np.triu_indices(nrois,1)
    motFC=np.empty((nsess, int(nrois*(nrois-1)/2)))
    count=0
    #Loop through all 10 days to reshape correlations into linear form
    for sess in range(nsess):
        tmp=fileFC[:,:,sess]
        motFC[count]=tmp[mask]
        count=count+1
    mask = (motFC == 0).all(1)
    column_indices = np.where(mask)[0]
    motFC = motFC[~mask,:]
    #glass
    tmp=IndNetDir+'mixed/allsubs_mixed_corrmats_bysess_orig_INDformat.mat'
    fileFC=scipy.io.loadmat(tmp,struct_as_record=False,squeeze_me=False)
    fileFC=fileFC['sess_task_corrmat']
    fileFC=fileFC[0,0].AllGlass
    fileFC=fileFC[0,subDict[sub]]
    fileFC=np.nan_to_num(fileFC)
    nrois=14
    nsess=fileFC.shape[2]
    #Index upper triangle of matrix
    mask=np.triu_indices(nrois,1)
    glassFC=np.empty((nsess, int(nrois*(nrois-1)/2)))
    count=0
    #Loop through all 10 days to reshape correlations into linear form
    for sess in range(nsess):
        tmp=fileFC[:,:,sess]
        glassFC[count]=tmp[mask]
        count=count+1
    mask = (glassFC == 0).all(1)
    column_indices = np.where(mask)[0]
    glassFC = glassFC[~mask,:]
    #semantic
    tmp=IndNetDir+'mixed/allsubs_mixed_corrmats_bysess_orig_INDformat.mat'
    fileFC=scipy.io.loadmat(tmp,struct_as_record=False,squeeze_me=False)
    fileFC=fileFC['sess_task_corrmat']
    fileFC=fileFC[0,0].AllSemantic
    fileFC=fileFC[0,subDict[sub]]
    fileFC=np.nan_to_num(fileFC)
    nrois=14
    nsess=fileFC.shape[2]
    #Index upper triangle of matrix
    mask=np.triu_indices(nrois,1)
    semFC=np.empty((nsess, int(nrois*(nrois-1)/2)))
    count=0
    #Loop through all 10 days to reshape correlations into linear form
    for sess in range(nsess):
        tmp=fileFC[:,:,sess]
        semFC[count]=tmp[mask]
        count=count+1
    mask = (semFC == 0).all(1)
    column_indices = np.where(mask)[0]
    semFC = semFC[~mask,:]
    fullTask=np.concatenate((memFC,semFC,glassFC,motFC))
    #will have to write something on converting resting time series data into 4 split pieces
    #######################################################################################
    #open rest
    tmpRest=IndNetDir+'rest/'+sub+'_parcel_corrmat.mat'
    fileFC=scipy.io.loadmat(tmpRest)
    #Convert to numpy array
    fileFC=np.array(fileFC['parcel_corrmat'])
    #Replace nans and infs with zero
    fileFC=np.nan_to_num(fileFC)
    nsess=fileFC.shape[2]
    #Index upper triangle of matrix
    mask=np.triu_indices(nrois,1)
    fullRest=np.empty((nsess, int(nrois*(nrois-1)/2)))
    count=0
    #Loop through all 10 days to reshape correlations into linear form
    for sess in range(nsess):
        tmp=fileFC[:,:,sess]
        fullRest[count]=tmp[mask]
        count=count+1
    mask = (fullRest == 0).all(1)
    column_indices = np.where(mask)[0]
    fullRest = fullRest[~mask,:]
    return fullTask,fullRest

def modelNets(clf,train_sub, test_sub):
    """
    Preparing machine learning model with appropriate data

    Parameters
    -------------
    train_sub : str
            Subject name for training
    test_sub : str
            Subject name for testing

    Returns
    -------------
    total_score : float
            Average accuracy of all folds

    """
    #train sub
    taskFC, restFC=netFile('IndNet',train_sub)
    #test sub
    test_taskFC, test_restFC=netFile('IndNet',test_sub)
    CV, DS=Net_folds(clf, taskFC, restFC, test_taskFC, test_restFC)
    return CV, DS

def Net_folds(clf, taskFC, restFC, test_taskFC, test_restFC):
    """
    Cross validation to train and test using nested loops

    Parameters
    -----------
    clf : obj
        Machine learning algorithm
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
    test_taskSize=test_taskFC.shape[0]
    test_restSize=test_restFC.shape[0]
    testT= np.ones(test_taskSize, dtype = int)
    testR= np.zeros(test_restSize, dtype = int)
    CVacc=[]
    DSacc=[]
    for train_index, test_index in loo.split(taskFC):
        Xtrain_rest, Xval_rest=restFC[train_index], restFC[test_index]
        Xtrain_task, Xval_task=taskFC[train_index], taskFC[test_index]
        ytrain_rest, yval_rest=r[train_index], r[test_index]
        ytrain_task, yval_task=t[train_index], t[test_index]
        X_tr=np.concatenate((Xtrain_task, Xtrain_rest))
        X_val=np.concatenate((Xval_task, Xval_rest))
        y_tr = np.concatenate((ytrain_task,ytrain_rest))
        y_val=np.concatenate((yval_task, yval_rest))
        y_tr=np.random.permutation(y_tr)
        clf.fit(X_tr,y_tr)
        CV_score=clf.score(X_val, y_val)
        CVacc.append(CV_score)
        X_te=np.concatenate((test_taskFC, test_restFC))
        y_te=np.concatenate((testT, testR))
        ACCscores=clf.score(X_te,y_te)
        DSacc.append(ACCscores)
    CV=mean(CVacc)
    DS=mean(DSacc)
    return CV, DS

def classifyIndNet(classifier='Ridge'):
    """
    Classifying different subjects along network level data generated from group atlas rest split into 40 samples to match with task

    Parameters
    -------------

    Returns
    -------------
    df : DataFrame
        Dataframe consisting of average accuracy across all subjects

    """
    clf=clfDict[classifier]
    acc_scores_ds=[]
    acc_scores_cv=[]
    df=pd.DataFrame(subsComb, columns=['train_sub','test_sub'])
    for index, row in df.iterrows():
        CV, DS=modelNets(clf,train_sub=row['train_sub'], test_sub=row['test_sub'])
        acc_scores_ds.append(DS)
        acc_scores_cv.append(CV)
    CV=mean(acc_scores_cv)
    DS=mean(acc_scores_ds)
    return CV, DS


def permutIndNet():
    SSscore=[]
    OSscore=[]
    for i in range(1000):
        SS,OS=classifyIndNet(classifier)
        SSscore.append(SS)
        OSscore.append(OS)
        print(str(i))
    ALL_perms=pd.DataFrame()
    ALL_perms['SS']=SSscore
    ALL_perms['OS']=OSscore
    ALL_perms.to_csv(outDir+classifier+'/ALL_IndNet/permutation.csv',index=False)


def permutMC(classifier):
    SSscore=[]
    OSscore=[]
    for i in range(1000):
        SS,OS=multiclassAll(classifier)
        SSscore.append(SS)
        OSscore.append(OS)
        print(str(i))
    ALL_perms=pd.DataFrame()
    ALL_perms['SS']=SSscore
    ALL_perms['OS']=OSscore
    ALL_perms.to_csv(outDir+classifier+'/ALL_MC/permutation.csv',index=False)


def permutAll_Binary(classifier):
    SSscore=[]
    OSscore=[]
    for i in range(1000):
        SS,OS=classifyAll(classifier)
        SSscore.append(SS)
        OSscore.append(OS)
        print(str(i))
    ALL_perms=pd.DataFrame()
    ALL_perms['SS']=SSscore
    ALL_perms['OS']=OSscore
    ALL_perms.to_csv(outDir+classifier+'/ALL_Binary/permutation.csv',index=False)


def permut_single_task(classifier):
    SSscore=[]
    OSscore=[]
    for i in range(1000):
        SS,OS=classifyDS(classifier)
        SSscore.append(SS)
        OSscore.append(OS)
        print(str(i))
    ALL_perms=pd.DataFrame()
    ALL_perms['SS']=SSscore
    ALL_perms['OS']=OSscore
    ALL_perms.to_csv(outDir+classifier+'/single_task/permutation.csv',index=False)
