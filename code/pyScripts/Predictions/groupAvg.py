#!/usr/bin/env python
# coding: utf-8

# In[ ]:
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeClassifier
import numpy as np
import os
import sys
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
import itertools
#import other python scripts for further anlaysis
import reshape
#import results
import warnings
warnings.filterwarnings("ignore")
# Initialization of directory information:
thisDir = os.path.expanduser('~/Desktop/MSC_Alexis/analysis/')
dataDir = thisDir + 'data/mvpa_data/'
outDir = thisDir + 'output/results/groupAvg/'
# Subjects and tasks
taskList=['glass','semantic', 'motor','mem']
#Only using subs with full 10 sessions
subList=['MSC01','MSC02','MSC03','MSC04','MSC05','MSC06','MSC07','MSC10']
#all possible combinations of subs and tasks
subsComb=(list(itertools.permutations(subList, 2)))
tasksComb=(list(itertools.permutations(taskList, 2)))
splitDict=dict([('MSC01',10),('MSC02',10),('MSC03',8),('MSC04',10),('MSC05',10),('MSC06',9),('MSC07',9),('MSC10',10)])

def groupApp():
    """
    Classifying using all subs testing unseen sub on same task and diff task

    Parameters
    -------------


    Returns
    -------------
    dfGroup : DataFrame
        Dataframe consisting of group average accuracy training with subs instead of individualized classifier

    """
    cv_acc_per_task=[]
    cv_sen_per_task=[]
    cv_spec_per_task=[]
    SS_acc_per_task=[]
    DS_acc_per_task=[]
    dfGroup=pd.DataFrame(tasksComb, columns=['train_task','test_task'])
    for index, row in dfGroup.iterrows():
        cv_score, cv_sensitivity, cv_specificity, SS_score, DS_score=model(train_task=row['train_task'], test_task=row['test_task'])
        cv_acc_per_task.append(cv_score)
        cv_spec_per_task.append(cv_specificity)
        cv_sen_per_task.append(cv_sensitivity)
        SS_acc_per_task.append(SS_score)
        DS_acc_per_task.append(DS_score)
    dfGroup['CV_acc']=cv_acc_per_task
    dfGroup['SS_acc']=SS_acc_per_task
    dfGroup['DS_acc']=DS_acc_per_task
    #dfGroup['CV_sen']=cv_sen_per_task
    #dfGroup['CV_spec']=cv_spec_per_task
    #save accuracy
    dfGroup.to_csv(outDir+'allSess_acc.csv',index=False)

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
    CV_score : float
            Average accuracy of all folds leave one sub out of a given task
    CV_sens : float
            Sensitivity score for model within task
    CV_spec : float
            Specificity score for model within task
    SS_score : float
            Average accuracy of all folds same subject but tested on a new task
    DS_score : float
            Average accuracy of all folds of a left out subject but tested on a new task

    """
    #nsess x fc x nsub going with all subs
    ds_T=np.empty((10,55278,7))
    ds_R=np.empty((10,55278,7))
    ds_Test=np.empty((10,55278,7))
    count=0
    #get all subs for a given task
    for sub in subList:
        #training task
        tmp_taskFC=reshape.matFiles(dataDir+train_task+'/'+sub+'_parcel_corrmat.mat')

        tmp_restFC=reshape.matFiles(dataDir+'rest/'+sub+'_parcel_corrmat.mat')
        #reshape 2d into 3d nsessxfcxnsubs
        ds_T[:,:,count]=tmp_taskFC
        ds_R[:,:,count]=tmp_restFC
        #testing task
        test_taskFC=reshape.matFiles(dataDir+test_task+'/'+sub+'_parcel_corrmat.mat')
        ds_Test[:,:,count]=test_taskFC
        count=count+1
    clf=RidgeClassifier()
    loo = LeaveOneOut()
    #split up by subs not sess
    sub_splits=np.empty((7,55278))
    df=pd.DataFrame()
    #left out sub
    cv_scoreList=[]
    sen_CV=[]
    spec_CV=[]
    #new task same subjects
    wtn_scoreList=[]
    #new task new subject
    btn_scoreList=[]
    for train_index, test_index in loo.split(sub_splits):
        #train on all sessions 1-6 subs
        #test on all sessions of one sub
        Xtrain_task, Xtest_task=ds_T[:,:,train_index], ds_T[:,:,test_index[0]]
        Xtrain_rest, Xtest_rest=ds_R[:,:,train_index], ds_R[:,:,test_index[0]]
        #test on a new task of 1-6 subs
        #test on a new task of left out sub
        SS_newTask, DS_newTask=ds_Test[:,:,train_index], ds_Test[:,:,test_index[0]]
        #reshape data into a useable format for mL
        Xtrain_task=Xtrain_task.reshape(60,55278)
        Xtrain_rest=Xtrain_rest.reshape(60,55278)
        SS_newTask=SS_newTask.reshape(60,55278)
        #training set
        taskSize=Xtrain_task.shape[0]
        restSize=Xtrain_rest.shape[0]
        t = np.ones(taskSize, dtype = int)
        r=np.zeros(restSize, dtype=int)
        x_train=np.concatenate((Xtrain_task,Xtrain_rest))
        y_train=np.concatenate((t,r))
        #testing set (left out sub CV)
        testSize=Xtest_task.shape[0]
        test_restSize=Xtest_rest.shape[0]
        test_t = np.ones(testSize, dtype = int)
        test_r=np.zeros(test_restSize, dtype=int)
        x_test=np.concatenate((Xtest_task, Xtest_rest))
        y_test=np.concatenate((test_t,test_r))
        #testing set of new task using same subs SS
        SS_taskSize=SS_newTask.shape[0]
        SS_y = np.ones(SS_taskSize, dtype = int)
        #testing set of new task using a diff sub DS
        DS_taskSize=DS_newTask.shape[0]
        DS_y = np.ones(DS_taskSize, dtype = int)
        #test left out sub 10 sessions
        clf.fit(x_train,y_train)
        y_pre=clf.predict(x_test)
        tn, fp, fn, tp=confusion_matrix(y_test, y_pre).ravel()
        specificity= tn/(tn+fp)
        sensitivity= tp/(tp+fn)
        sen_CV.append(sensitivity)
        spec_CV.append(specificity)
        ACCscores=clf.score(x_test,y_test)
        cv_scoreList.append(ACCscores)
        #test new task of same subjects
        clf.predict(SS_newTask)
        wtn_ACCscores=clf.score(SS_newTask, SS_y)
        wtn_scoreList.append(wtn_ACCscores)
        #test new task of new sub
        clf.predict(DS_newTask)
        btn_ACCscores=clf.score(DS_newTask, DS_y)
        btn_scoreList.append(btn_ACCscores)
    df['cv_fold']=cv_scoreList
    df['cv_sens']=sen_CV
    df['cv_spec']=spec_CV
    cv_score=df['cv_fold'].mean()
    cv_sensitivity=df['cv_sens'].mean()
    cv_specificity=df['cv_spec'].mean()
    df['SS_fold']=wtn_scoreList
    SS_score=df['SS_fold'].mean()
    df['DS_fold']=btn_scoreList
    DS_score=df['DS_fold'].mean()
    return cv_score, cv_sensitivity, cv_specificity, SS_score, DS_score


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
    #nsess x fc x nsub
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
    #left out sub

    sen_CV=[]
    spec_CV=[]
    accScore_allsess=[]
    #cv_scoreList=[]
    #declare session to work in first 10 sessions
    for i in range(8):
        cv_scoreList=[]
        df=pd.DataFrame()
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
        """
        if i==0:
            ds_rest=ds_R[:4,:,:]
        else:
            startingpoint=i*4
            endingpoint=startingpoint+4
            ds_rest=ds_R[startingpoint:endingpoint,:,:]
        """
    #now do leave one sub out cv
        for train_index, test_index in loo.split(sub_splits):
            #train on all sessions 1-6 subs
            #test on all sessions of one sub
            train_mem, test_mem=tmp_mem[train_index,:],tmp_mem[test_index,:]
            train_mot, test_mot=tmp_mot[train_index,:],tmp_mot[test_index,:]
            train_sem, test_sem=tmp_sem[train_index,:],tmp_sem[test_index,:]
            train_glass, test_glass=ds_rest[train_index,:],ds_rest[test_index,:]
            train_rest, test_rest=tmp_glass[train_index,:],tmp_glass[test_index,:]
            #rest
            #train_rest, Xtest_rest=ds_rest[:,:,train_index],ds_rest[:,:,test_index[0]]

            #now reshape data
            Xtrain_task=np.concatenate((train_mem,train_mot,train_sem,train_glass))
            Xtest_task=np.concatenate((test_mem,test_mot,test_sem,test_glass))
            #Xtrain_rest=train_rest.reshape(28,55278)
            #test on a new task of 1-6 subs
            #test on a new task of left out sub
            #reshape data into a useable format for mL
            #6 subs with 10 sessions per 4 task

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
            #test left out sub 10 sessions
            clf.fit(x_train,y_train)
            #y_pre=clf.predict(x_test)
            #tn, fp, fn, tp=confusion_matrix(y_test, y_pre).ravel()
            #specificity= tn/(tn+fp)
            #sensitivity= tp/(tp+fn)
            #sen_CV.append(sensitivity)
            #spec_CV.append(specificity)
            ACCscores=clf.score(x_test,y_test)
            cv_scoreList.append(ACCscores)
        df['cv_fold']=cv_scoreList
        allsub_per_sess=df['cv_fold'].mean()
        accScore_allsess.append(allsub_per_sess)
    #final_df['acc_per_sess']=accScore_allsess
    #return final_df
    #final_df.to_csv(outDir+'allTasks_acc.csv',index=False)
    #return Xtest_task, Xtrain_rest, Xtrain_task, Xtest_rest
    #return final_df
    return accScore_allsess

def groupAVG():
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
        #clf=LinearSVC()
        #clf=LogisticRegression(solver = 'lbfgs')
        clf=RidgeClassifier()
        #train sub
        DSacc=[]
        master_df=pd.DataFrame()
        data=np.array(['MSC01','MSC02','MSC03','MSC04','MSC05','MSC06','MSC07','MSC10'],dtype='<U61')
        loo = LeaveOneOut()
        for  train, test in loo.split(data): #train on one sub test on the rest
            tmp=pd.DataFrame()
            train_sub=data[train]
            test_sub=data[test]
        #train sub
            memFC=reshape.matFiles(dataDir+'mem/'+test_sub[0]+'_parcel_corrmat.mat')
            semFC=reshape.matFiles(dataDir+'semantic/'+test_sub[0]+'_parcel_corrmat.mat')
            glassFC=reshape.matFiles(dataDir+'glass/'+test_sub[0]+'_parcel_corrmat.mat')
            motFC=reshape.matFiles(dataDir+'motor/'+test_sub[0]+'_parcel_corrmat.mat')
            test_taskFC=np.concatenate((memFC,semFC,glassFC,motFC))
            test_restFC=reshape.matFiles(dataDir+'rest/'+test_sub[0]+'_parcel_corrmat.mat') #keep tasks seperated in order to collect the right amount of days
            #nsize=restFC.shape[1]
            #restFC=np.reshape(restFC,(10,4,nsize)) #reshape to gather correct days
            #test sub
            taskFC,restFC=AllSubFiles(train_sub)
            diff_score=K_folds(train_sub,clf,testFC, restFC, )
            """
            test_taskSize=test_taskFC.shape[0]
            test_restSize=test_restFC.shape[0]
            testT= np.ones(test_taskSize, dtype = int)
            testR= np.zeros(test_restSize, dtype = int)
            Xtest=np.concatenate((test_taskFC,test_restFC))
            ytest=np.concatenate((testT,testR))
            ytrain_task = np.ones(taskFC.shape[0], dtype = int)
            ytrain_rest=np.zeros(restFC.shape[0], dtype=int)
            Xtrain=np.concatenate((taskFC,restFC))
            ytrain=np.concatenate((ytrain_task,ytrain_rest))
            clf.fit(Xtrain,ytrain)
            score=clf.score(Xtest, ytest)
            DSacc.append(score)
        return DSacc
        """




def AllSubFiles(train_sub):
    """
    Return task and rest FC all subs
    Parameters
    -----------
    test_sub: Array of testing subs
    Returns
    ------------
    taskFC, restFC : Array of task and rest FC of all testing subs
    """
    a_memFC=reshape.matFiles(dataDir+'mem/'+train_sub[0]+'_parcel_corrmat.mat')
    a_semFC=reshape.matFiles(dataDir+'semantic/'+train_sub[0]+'_parcel_corrmat.mat')
    a_glassFC=reshape.matFiles(dataDir+'glass/'+train_sub[0]+'_parcel_corrmat.mat')
    a_motFC=reshape.matFiles(dataDir+'motor/'+train_sub[0]+'_parcel_corrmat.mat')
    a_restFC=reshape.matFiles(dataDir+'rest/'+train_sub[0]+'_parcel_corrmat.mat')

    b_memFC=reshape.matFiles(dataDir+'mem/'+train_sub[1]+'_parcel_corrmat.mat')
    b_semFC=reshape.matFiles(dataDir+'semantic/'+train_sub[1]+'_parcel_corrmat.mat')
    b_glassFC=reshape.matFiles(dataDir+'glass/'+train_sub[1]+'_parcel_corrmat.mat')
    b_motFC=reshape.matFiles(dataDir+'motor/'+train_sub[1]+'_parcel_corrmat.mat')
    b_restFC=reshape.matFiles(dataDir+'rest/'+train_sub[1]+'_parcel_corrmat.mat')

    c_memFC=reshape.matFiles(dataDir+'mem/'+train_sub[2]+'_parcel_corrmat.mat')
    c_semFC=reshape.matFiles(dataDir+'semantic/'+train_sub[2]+'_parcel_corrmat.mat')
    c_glassFC=reshape.matFiles(dataDir+'glass/'+train_sub[2]+'_parcel_corrmat.mat')
    c_motFC=reshape.matFiles(dataDir+'motor/'+train_sub[2]+'_parcel_corrmat.mat')
    c_restFC=reshape.matFiles(dataDir+'rest/'+train_sub[2]+'_parcel_corrmat.mat')

    d_memFC=reshape.matFiles(dataDir+'mem/'+train_sub[3]+'_parcel_corrmat.mat')
    d_semFC=reshape.matFiles(dataDir+'semantic/'+train_sub[3]+'_parcel_corrmat.mat')
    d_glassFC=reshape.matFiles(dataDir+'glass/'+train_sub[3]+'_parcel_corrmat.mat')
    d_motFC=reshape.matFiles(dataDir+'motor/'+train_sub[3]+'_parcel_corrmat.mat')
    d_restFC=reshape.matFiles(dataDir+'rest/'+train_sub[3]+'_parcel_corrmat.mat')

    e_memFC=reshape.matFiles(dataDir+'mem/'+train_sub[4]+'_parcel_corrmat.mat')
    e_semFC=reshape.matFiles(dataDir+'semantic/'+train_sub[4]+'_parcel_corrmat.mat')
    e_glassFC=reshape.matFiles(dataDir+'glass/'+train_sub[4]+'_parcel_corrmat.mat')
    e_motFC=reshape.matFiles(dataDir+'motor/'+train_sub[4]+'_parcel_corrmat.mat')
    e_restFC=reshape.matFiles(dataDir+'rest/'+train_sub[4]+'_parcel_corrmat.mat')

    f_memFC=reshape.matFiles(dataDir+'mem/'+train_sub[5]+'_parcel_corrmat.mat')
    f_semFC=reshape.matFiles(dataDir+'semantic/'+train_sub[5]+'_parcel_corrmat.mat')
    f_glassFC=reshape.matFiles(dataDir+'glass/'+train_sub[5]+'_parcel_corrmat.mat')
    f_motFC=reshape.matFiles(dataDir+'motor/'+train_sub[5]+'_parcel_corrmat.mat')
    f_restFC=reshape.matFiles(dataDir+'rest/'+train_sub[5]+'_parcel_corrmat.mat')

    g_memFC=reshape.matFiles(dataDir+'mem/'+train_sub[6]+'_parcel_corrmat.mat')
    g_semFC=reshape.matFiles(dataDir+'semantic/'+train_sub[6]+'_parcel_corrmat.mat')
    g_glassFC=reshape.matFiles(dataDir+'glass/'+train_sub[6]+'_parcel_corrmat.mat')
    g_motFC=reshape.matFiles(dataDir+'motor/'+train_sub[6]+'_parcel_corrmat.mat')
    g_restFC=reshape.matFiles(dataDir+'rest/'+train_sub[6]+'_parcel_corrmat.mat')


    taskFC=np.concatenate((a_memFC,a_semFC,a_glassFC,a_motFC,b_memFC,b_semFC,b_glassFC,b_motFC,c_memFC,c_semFC,c_glassFC,c_motFC,d_memFC,d_semFC,d_glassFC,d_motFC,e_memFC,e_semFC,e_glassFC,e_motFC,f_memFC,f_semFC,f_glassFC,f_motFC,g_memFC,g_semFC,g_glassFC,g_motFC))
    restFC=np.concatenate((a_restFC,b_restFC,c_restFC,d_restFC,e_restFC,f_restFC,g_restFC))

    return taskFC, restFC
