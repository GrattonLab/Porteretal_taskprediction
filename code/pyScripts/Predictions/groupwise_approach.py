#old code that could be useful for training on a session level (7 subs train from 1 session)
#!/usr/bin/env python
# coding: utf-8

# In[ ]:
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import RidgeClassifier
import numpy as np
import os
import sys
import pandas as pd
from statistics import mean
import itertools
#import other python scripts for further anlaysis
import reshape
import warnings
warnings.filterwarnings("ignore")
# Initialization of directory information:
thisDir = os.path.expanduser('~/Desktop/Porteretal_taskprediction/')
dataDir = thisDir +'data/corrmats/'
outDir = thisDir + 'output/results/Ridge/'
# Subjects and tasks
taskList=['glass','semantic', 'motor','mem']
#Only using subs with full 10 sessions
subList=['MSC01','MSC02','MSC03','MSC04','MSC05','MSC06','MSC07','MSC10']
#all possible combinations of subs and tasks
subsComb=(list(itertools.permutations(subList, 2)))
tasksComb=(list(itertools.permutations(taskList, 2)))
DSvars=list(itertools.product(list(subsComb),list(taskList)))
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
    acc_scores_per_task=[]
    dfGroup=pd.DataFrame(taskList, columns=['task'])
    for task in taskList:
        score=model(task)
        acc_scores_per_task.append(score)
    dfGroup['acc']=acc_scores_per_task
    #save accuracy
    dfGroup.to_csv(outDir+'single_task/groupwise_acc.csv',index=False)

def model(train_task):
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
    within_score : float
            Average accuracy of all folds leave one sub out of a given task
    within_sens : float
            Sensitivity score for model within task
    within spec : float
            Specificity score for model within task
    btn_score : float
            Average accuracy of all folds leave one sub out of a given task
    btn_sens : float
            Sensitivity score for model within task
    btn spec : float
            Specificity score for model within task

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
        tmp_taskFC=reshape.matFiles(dataDir+train_task+'/'+sub+'_parcel_corrmat.mat')
        tmp_taskFC=tmp_taskFC[:8,:]
        tmp_restFC=reshape.matFiles(dataDir+'rest/'+sub+'_parcel_corrmat.mat')
        tmp_restFC=tmp_restFC[:8,:]
        #reshape 2d into 3d nsessxfcxnsubs
        ds_T[:,:,count]=tmp_taskFC
        ds_R[:,:,count]=tmp_restFC
        #testing task
        #test_taskFC=reshape.matFiles(dataDir+test_task+'/'+sub+'_parcel_corrmat.mat')
        #test_taskFC=test_taskFC[:8,:]
        #ds_Test[:,:,count]=test_taskFC
        count=count+1
    sess_wtn_score=[]
    #sess_btn_score=[]
    #sessionTotal=pd.DataFrame()
    #train leave one sub out
    #test on left out sub
    #test on new task of left out sub
    clf=RidgeClassifier()
    loo = LeaveOneOut()
    wtn_scoreList=[]
    for i in range(8):
    #takes one session of data (7 subs)
        taskFC=ds_T[i,:,:]
        taskFC=taskFC.T
        restFC=ds_R[i,:,:]
        restFC=restFC.T
        taskSize=taskFC.shape[0]
        restSize=restFC.shape[0]
        t = np.ones(taskSize, dtype = int)
        r=np.zeros(restSize, dtype=int)
        #df=pd.DataFrame()

        #btn_scoreList=[]
    #fold each training set (sub)
        for train_index, test_index in loo.split(taskFC):
            Xtrain_rest, Xtest_rest=restFC[train_index], restFC[test_index]
            Xtrain_task, Xtest_task=taskFC[train_index], taskFC[test_index]
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
            wtn_scoreList.append(ACCscores)
            #testSize=testFC.shape[0]
            #y_val= np.ones(testSize, dtype = int)
            #test unseen sub new task all 10 sessions
            #y_pre_val=clf.predict(testFC)
            #btn_ACCscores=clf.score(testFC, y_val)
            #btn_scoreList.append(btn_ACCscores)
    wtn_score=mean(wtn_scoreList)
    #df['btn_fold']=btn_scoreList
    #btn_score=df['btn_fold'].mean()
    #return wtn_scoreList
    return wtn_score#, btn_score




def allTask():
    """
    Leave one subject out, using all tasks but only using one session per model; 28 samples 4Task*7subs*1session

    Parameters
    -------------

    Returns
    -------------
    df : pandas Dataframe
            Accuracy, sensitivity, specificity scores for each kfold

    """
    clf=RidgeClassifier()
    loo = LeaveOneOut()
    final_df=pd.DataFrame()
    ds_mem=np.empty((8,55278,8))
    ds_sem=np.empty((8,55278,8))
    ds_glass=np.empty((8,55278,8))
    ds_motor=np.empty((8,55278,8))
    ds_R=np.empty((8,55278,8))
    count=0
    #get all subs for a given task
    for sub in subList:
        #training task
        tmp_memFC=reshape.matFiles(dataDir+'mem/'+sub+'_parcel_corrmat.mat')
        tmp_memFC=tmp_memFC[:8,:]
        tmp_semFC=reshape.matFiles(dataDir+'semantic/'+sub+'_parcel_corrmat.mat')
        tmp_semFC=tmp_semFC[:8,:]
        tmp_glassFC=reshape.matFiles(dataDir+'glass/'+sub+'_parcel_corrmat.mat')
        tmp_glassFC=tmp_glassFC[:8,:]
        tmp_motorFC=reshape.matFiles(dataDir+'motor/'+sub+'_parcel_corrmat.mat')
        tmp_motorFC=tmp_motorFC[:8,:]
        tmp_restFC=reshape.matFiles(dataDir+'rest/'+sub+'_parcel_corrmat.mat')
        #reshape 2d into 3d nsessxfcxnsubs
        tmp_restFC=tmp_restFC[:8,:]
        ds_mem[:,:,count]=tmp_memFC
        ds_sem[:,:,count]=tmp_semFC
        ds_glass[:,:,count]=tmp_glassFC
        ds_motor[:,:,count]=tmp_motorFC
        ds_R[:,:,count]=tmp_restFC
        count=count+1
    #split up by subs not sess
    sub_splits=np.empty((8,55278))
    accScore_allsess=[]
    #declare session to work in first 10 sessions
    for i in range(8):
        cv_scoreList=[]
        tmp_mem=ds_mem[i,:,:]
        tmp_mem=tmp_mem.T

        tmp_mot=ds_motor[i,:,:]
        tmp_mot=tmp_mot.T

        tmp_sem=ds_sem[i,:,:]
        tmp_sem=tmp_sem.T

        tmp_glass=ds_glass[i,:,:]
        tmp_glass=tmp_glass.T

        ds_rest=ds_R[i,:,:]
        ds_rest=ds_rest.T
        for train_index, test_index in loo.split(sub_splits):
            #train on all sessions 1-6 subs
            #test on all sessions of one sub
            train_mem, test_mem=tmp_mem[train_index,:],tmp_mem[test_index,:]
            train_mot, test_mot=tmp_mot[train_index,:],tmp_mot[test_index,:]
            train_sem, test_sem=tmp_sem[train_index,:],tmp_sem[test_index,:]
            train_glass, test_glass=ds_rest[train_index,:],ds_rest[test_index,:]
            train_rest, test_rest=tmp_glass[train_index,:],tmp_glass[test_index,:]
            #now reshape data
            Xtrain_task=np.concatenate((train_mem,train_mot,train_sem,train_glass))
            Xtest_task=np.concatenate((test_mem,test_mot,test_sem,test_glass))
            #training set
            taskSize=Xtrain_task.shape[0]
            restSize=train_rest.shape[0]
            t = np.ones(taskSize, dtype = int)
            r=np.zeros(restSize, dtype=int)
            x_train=np.concatenate((Xtrain_task,train_rest))
            y_train=np.concatenate((t,r))
            #testing set (left out sub CV)
            testSize=Xtest_task.shape[0]
            test_restSize=test_rest.shape[0]
            test_t = np.ones(testSize, dtype = int)
            test_r=np.zeros(test_restSize, dtype=int)
            x_test=np.concatenate((Xtest_task, test_rest))
            y_test=np.concatenate((test_t,test_r))
            clf.fit(x_train,y_train)
            ACCscores=clf.score(x_test,y_test)
            cv_scoreList.append(ACCscores)
        allsub_per_sess=mean(cv_scoreList)
        accScore_allsess.append(allsub_per_sess)
    final_df['acc_per_sess']=accScore_allsess
    final_df.to_csv(outDir+'/ALL_Binary/groupwise_acc.csv',index=False)

def groupApp_LOSO():
    """
    Classifying using all subs testing unseen sub on same task
    Parameters
    -------------


    Returns
    -------------
    dfGroup : DataFrame
        Dataframe consisting of group average accuracy training with subs instead of session

    """
    acc_scores_per_task=[]
    dfGroup=pd.DataFrame(taskList, columns=['task'])
    for task in taskList:
        score=model_LOSO(task)
        acc_scores_per_task.append(score)
    dfGroup['acc']=acc_scores_per_task
    #save accuracy
    dfGroup.to_csv(outDir+'single_task/groupwise_LOSO_acc.csv',index=False)


def model_LOSO(train_task):
    """
    Train on all sessions, leave one subject out

    Parameters
    -------------
    train_task : str
            Task name for training
    test_task : str
            Task name for testing

    Returns
    -------------
    within_score : float
            Average accuracy of all folds leave one sub out of a given task
    btn_score : float
            Average accuracy of all folds leave one sub out of a given task

    """

    clf=RidgeClassifier()
    loo = LeaveOneOut()
    cv_scoreList =[]
    data=np.array(['MSC01','MSC02','MSC03','MSC04','MSC05','MSC06','MSC07','MSC10'],dtype='<U61')
    for  train, test in loo.split(data): #test on one sub train on all subs
        train_subs = data[train]
        test_sub = data[test]
        test_task = reshape.matFiles(dataDir+train_task+'/'+test_sub[0]+'_parcel_corrmat.mat')
        test_rest = reshape.matFiles(dataDir+'rest/'+test_sub[0]+'_parcel_corrmat.mat')
        task, rest = reshape.AllSubFiles_groupavg(train_subs, train_task)
        #training set
        taskSize=task.shape[0]
        restSize=rest.shape[0]
        t = np.ones(taskSize, dtype = int)
        r=np.zeros(restSize, dtype=int)
        x_train=np.concatenate((task,rest))
        y_train=np.concatenate((t,r))
        #testing set (left out sub CV)
        test_taskSize=test_task.shape[0]
        test_restSize=test_rest.shape[0]
        test_t = np.ones(test_taskSize, dtype = int)
        test_r=np.zeros(test_restSize, dtype=int)
        x_test=np.concatenate((test_task, test_rest))
        y_test=np.concatenate((test_t,test_r))
        #test left out sub 10 sessions
        clf.fit(x_train,y_train)
        ACCscores=clf.score(x_test,y_test)
        cv_scoreList.append(ACCscores)
    cv_score = mean(cv_scoreList)
    return cv_score


def classifyDS(classifier='string'):
    """
    Classifying different subjects (DS) along the same task matched for samples as above

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
    tmp_df=pd.DataFrame(DSvars, columns=['sub','task'])
    dfDS=pd.DataFrame()
    dfDS[['train_sub','test_sub']]=pd.DataFrame(tmp_df['sub'].tolist())
    dfDS['task']=tmp_df['task']
    for index, row in dfDS.iterrows():
        taskFC=reshape.matFiles(dataDir+row['task']+'/'+row['train_sub']+'_parcel_corrmat.mat')
        taskFC=taskFC[:8,:]
        restFC=reshape.matFiles(dataDir+'rest/'+row['train_sub']+'_parcel_corrmat.mat')
        restFC=restFC[:8,:]
        test_taskFC=reshape.matFiles(dataDir+row['task']+'/'+row['test_sub']+'_parcel_corrmat.mat')
        test_taskFC=test_taskFC[:8,:]
        test_restFC=reshape.matFiles(dataDir+'rest/'+row['test_sub']+'_parcel_corrmat.mat')
        test_restFC=test_restFC[:8,:]
        SSacc, OSacc=folds(clf, taskFC, restFC, test_taskFC, test_restFC)
        same_sub_per_task.append(SSacc)
        diff_sub_per_task.append(OSacc)
    dfDS['diff_sub']=diff_sub_per_task
    dfDS['same_sub']=same_sub_per_task
    dfDS.to_csv(outDir+'single_task/acc_matched_groupavg.csv',index=False)

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
