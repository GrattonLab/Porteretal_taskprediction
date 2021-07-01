#!/usr/bin/env python
# coding: utf-8

# In[ ]:

from sklearn.model_selection import train_test_split
import reshape
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeClassifier
import numpy as np
import os
import sys
import pandas as pd
import classification
from sklearn.model_selection import cross_val_score
import itertools
import warnings
warnings.filterwarnings("ignore")
# Initialization of directory information:
thisDir = os.path.expanduser('~/Desktop/MSC_Alexis/analysis/')
dataDir = thisDir + 'data/mvpa_data/'
outDir = thisDir + 'output/results/permutation/'
# Subjects and tasks
taskList=['glass','semantic', 'motor','mem']
subList=['MSC01','MSC02','MSC03','MSC04','MSC05','MSC06','MSC07','MSC10']
#all possible combinations of subs and tasks
subsComb=(list(itertools.permutations(subList, 2)))
tasksComb=(list(itertools.permutations(taskList, 2)))
#DS combination
DSvars=list(itertools.product(list(subsComb),list(taskList)))
##SS combination
SSvars=list(itertools.product(list(subList),list(tasksComb)))
#BS combination
BSvars=list(itertools.product(list(subsComb),list(tasksComb)))

def SSmBS():
        dfBS=classifyBS()
        dfSS=classifySS()
        dfBS.rename(columns={'train_sub':'sub'},inplace=True)
        diff_task=dfBS.merge(dfSS,how='left',on=['train_task','test_task','sub'],suffixes=('','_SS'))
        #SS-OS
        diff_task['diff'] =diff_task.acc_SS-diff_task.acc
        #subset so we only average the SS-OS per train/test tasks
        DT=diff_task[['train_task','test_task','diff']]
        #take average
        diff=DT.groupby(['train_task','test_task']).mean()
        diff.reset_index(inplace=True)
        return diff
def CVmDS():
    """
    Classifying different subjects (DS) along the same task

    Parameters
    -------------

    Returns
    -------------
    dfDS : DataFrame
        Dataframe consisting of average accuracy across all subjects

    """
    acc_scores_per_task=[]
    tmp_df=pd.DataFrame(DSvars, columns=['sub','task'])
    dfDS=pd.DataFrame()
    dfDS[['train_sub','test_sub']]=pd.DataFrame(tmp_df['sub'].tolist())
    dfDS['task']=tmp_df['task']
    for index, row in dfDS.iterrows():
        score=model('DS', train_sub=row['train_sub'], test_sub=row['test_sub'], train_task=row['task'], test_task=row['task'])
        acc_scores_per_task.append(score)
    #dfDS['acc']=acc_scores_per_task
    dfDS['diff']=acc_scores_per_task
    diff=dfDS.groupby('task').mean()
    diff.reset_index(inplace=True)
    return diff


def classifySS():
    """
    Classifying the same subject (SS) along a different task

    Parameters
    -------------


    Returns
    -------------
    dfSS : DataFrame
        Dataframe consisting of average accuracy across all subjects

    """
    acc_scores_per_task=[]
    tmp_df=pd.DataFrame(SSvars, columns=['sub','task'])
    dfSS=pd.DataFrame()
    dfSS[['train_task','test_task']]=pd.DataFrame(tmp_df['task'].tolist())
    dfSS['sub']=tmp_df['sub']
    for index, row in dfSS.iterrows():
        score=model('SS', train_sub=row['sub'], test_sub=row['sub'], train_task=row['train_task'], test_task=row['test_task'])
        acc_scores_per_task.append(score)
    dfSS['acc']=acc_scores_per_task
    return dfSS
def classifyBS():
    """
    Classifying different subjects (BS) along different tasks

    Parameters
    -------------

    Returns
    -------------
    dfBS : DataFrame
        Dataframe consisting of average accuracy across all subjects

    """
    #BS=pd.DataFrame(columns=['train_task','test_task','train_sub','test_sub'])
    acc_scores_per_task=[]
    tmp_df=pd.DataFrame(BSvars, columns=['sub','task'])
    dfBS=pd.DataFrame()
    dfBS[['train_task','test_task']]=pd.DataFrame(tmp_df['task'].tolist())
    dfBS[['train_sub', 'test_sub']]=pd.DataFrame(tmp_df['sub'].tolist())
    for index, row in dfBS.iterrows():
        score=model('BS', train_sub=row['train_sub'], test_sub=row['test_sub'], train_task=row['train_task'], test_task=row['test_task'])
        acc_scores_per_task.append(score)
    dfBS['acc']=acc_scores_per_task
    return dfBS




def model(analysis, train_sub, test_sub, train_task, test_task):
    """
    Preparing machine learning model with appropriate data

    Parameters
    -------------
    analysis : string
            The type of analysis to be conducted
    train_sub : str
            Subject name for training
    test_sub : str
            Subject name for testing
    train_task : str
            Task name for training
    test_task : str
            Task name for testing

    Returns
    -------------
    total_score : float
            Average accuracy of all folds

    """

    clf=RidgeClassifier()
    taskFC=reshape.matFiles(dataDir+train_task+'/'+train_sub+'_parcel_corrmat.mat')

    #if your subs are the same
    if train_sub==test_sub:
        restFC=reshape.matFiles(dataDir+'rest/corrmats_timesplit/half/'+train_sub+'_parcel_corrmat.mat')
        restFC, test_restFC=train_test_split(restFC, test_size=.5)
        test_taskFC=reshape.matFiles(dataDir+test_task+'/'+test_sub+'_parcel_corrmat.mat')
        total_score=CV_folds(clf, analysis, taskFC, restFC, test_taskFC, test_restFC)
    else:
        restFC=reshape.matFiles(dataDir+'rest/'+train_sub+'_parcel_corrmat.mat')
        test_taskFC=reshape.matFiles(dataDir+test_task+'/'+test_sub+'_parcel_corrmat.mat')
        test_restFC=reshape.matFiles(dataDir+'rest/'+test_sub+'_parcel_corrmat.mat')
        total_score=CV_folds(clf, analysis, taskFC, restFC, test_taskFC, test_restFC)
    return total_score



def CV_folds(clf, analysis, taskFC, restFC, test_taskFC, test_restFC):
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
    t_tmp= np.ones(taskSize, dtype = int)
    r_tmp=np.zeros(restSize, dtype=int)
    #Concatenate rest and task labels
    Y=np.concatenate((t_tmp,r_tmp))
    #Permute the data
    Y_perm=np.random.permutation(Y)
    #For the purpose of this script split them back into a pseudo rest and task array
    t, r =np.array_split(Y_perm, 2)
    if analysis=='SS':
        df=pd.DataFrame()
        acc_score=[]
        for train_index, test_index in loo.split(taskFC):
            Xtrain_rest, Xval_rest=restFC[train_index], restFC[test_index]
            Xtrain_task, Xval_task=taskFC[train_index], taskFC[test_index]
            ytrain_rest=r[train_index]
            ytrain_task=t[train_index]
            X_tr=np.concatenate((Xtrain_task, Xtrain_rest))
            y_tr = np.concatenate((ytrain_task,ytrain_rest))

            #same sub same task
            X_val=np.concatenate((Xval_task,Xval_rest))
            y_val=np.array([1,0])
            #y_tr=np.random.permutation(y_tr)
            clf.fit(X_tr,y_tr)
            clf.predict(X_val)
            #same sub
            SS=clf.score(X_val,y_val)
            tmpdf=pd.DataFrame()
            acc_scores_per_fold=[]
            for t_index, te_index in loo.split(test_taskFC):
                Xtest_task=test_taskFC[te_index]
                Xtest_rest=test_restFC[te_index]
                X_Test = np.concatenate((Xtest_task, Xtest_rest))
                #This way we are including the correct rest and task labels
                y_Test = np.array([1, 0])
                #test set
                clf.predict(X_Test)
                #Get accuracy of model
                ACCscores=clf.score(X_Test,y_Test)
                acc_scores_per_fold.append(ACCscores)
            tmpdf['inner_fold']=acc_scores_per_fold
            score=tmpdf['inner_fold'].mean()
            acc_score.append(score)
        df['outer_fold']=acc_score
        total_score=df['outer_fold'].mean()
    elif analysis=='BS':
        df=pd.DataFrame()
        acc_score=[]
        #fold each training set
        for train_index, test_index in loo.split(taskFC):
            Xtrain_rest=restFC[train_index]
            Xtrain_task=taskFC[train_index]
            ytrain_rest=r[train_index]
            ytrain_task=t[train_index]
            X_tr=np.concatenate((Xtrain_task, Xtrain_rest))
            y_tr = np.concatenate((ytrain_task,ytrain_rest))
            clf.fit(X_tr,y_tr)
            tmpdf=pd.DataFrame()
            acc_scores_per_fold=[]
            #fold each testing set
            for t_index, te_index in loo.split(test_taskFC):
                Xtest_rest=test_restFC[te_index]
                Xtest_task=test_taskFC[te_index]
                X_te=np.concatenate((Xtest_task, Xtest_rest))
                y_te=np.array([1, 0])
                #test set
                clf.predict(X_te)
                #Get accuracy of model
                ACCscores=clf.score(X_te,y_te)
                acc_scores_per_fold.append(ACCscores)
            tmpdf['inner_fold']=acc_scores_per_fold
            score=tmpdf['inner_fold'].mean()
            acc_score.append(score)
        df['outer_fold']=acc_score
        total_score=df['outer_fold'].mean()
    else:
    #CVmDS
        OSdf=pd.DataFrame()
        SSdf=pd.DataFrame()
        SS=[]
        OS=[]
        acc_score=[]
        diff_score=[]
        #fold each training set
        for train_index, test_index in loo.split(taskFC):
            Xtrain_rest, Xval_rest=restFC[train_index], restFC[test_index]
            Xtrain_task,Xval_task=taskFC[train_index],taskFC[test_index]
            ytrain_rest=r[train_index]
            ytrain_task=t[train_index]
            X_tr=np.concatenate((Xtrain_task, Xtrain_rest))
            y_tr = np.concatenate((ytrain_task,ytrain_rest))
            #y_tr=np.random.permutation(y_tr)
            #make sure the label is correct for this part
            X_val=np.concatenate((Xval_task,Xval_rest))
            y_val=np.array([1,0])
            clf.fit(X_tr,y_tr)

            clf.predict(X_val)
            tmpSS=clf.score(X_val,y_val)
            SS.append(tmpSS)
            tmpdf=pd.DataFrame()
            acc_scores_per_fold=[]
            #fold each testing set
            for t_index, te_index in loo.split(test_taskFC):
                Xtest_rest=test_restFC[te_index]
                Xtest_task=test_taskFC[te_index]
                X_te=np.concatenate((Xtest_task, Xtest_rest))
                y_te=np.array([1, 0])
                    #test set
                clf.predict(X_te)
                    #Get accuracy of model
                ACCscores=clf.score(X_te,y_te)
                acc_scores_per_fold.append(ACCscores)
            tmpdf['inner_fold']=acc_scores_per_fold
            score=tmpdf['inner_fold'].mean()
            acc_score.append(score)
        SSdf['outer_fold']=SS
        SS_score=SSdf['outer_fold'].mean()
        OSdf['outer_fold']=acc_score
        OS_score=OSdf['outer_fold'].mean()
        total_score=SS_score-OS_score
    return total_score


def classifyAll():
    """
    Classifying different subjects along available data rest split into 30 samples to match with task

    Parameters
    -------------

    Returns
    -------------
    df : DataFrame
        Dataframe consisting of average accuracy across all subjects

    """
    acc_scores_per_sub=[]
    acc_scores_cv=[]
    df=pd.DataFrame(subsComb, columns=['train_sub','test_sub'])
    for index, row in df.iterrows():
        diff_score, same_score=modelAll(train_sub=row['train_sub'], test_sub=row['test_sub'])
        acc_scores_per_sub.append(diff_score)
        acc_scores_cv.append(same_score)
    #all subs avg acc score within and between
    df['SS']=acc_scores_cv
    df['OS']=acc_scores_per_sub
    df['diff']=df.SS-df.OS
    SS=df['SS'].mean()
    OS=df['OS'].mean()
    DT=df['diff'].mean()
    return DT, SS, OS



def modelAll(train_sub, test_sub):
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

    clf=RidgeClassifier()

    #df=pd.DataFrame()
    #train sub
    memFC=classification.matFiles(dataDir+'mem/'+train_sub+'_parcel_corrmat.mat')
    glassFC=classification.matFiles(dataDir+'glass/'+train_sub+'_parcel_corrmat.mat')
    semFC=classification.matFiles(dataDir+'semantic/'+train_sub+'_parcel_corrmat.mat')
    motFC=classification.matFiles(dataDir+'motor/'+train_sub+'_parcel_corrmat.mat')
    restFC=classification.matFiles(dataDir+'rest/corrmats_timesplit/fourths/'+train_sub+'_parcel_corrmat.mat')
    taskFC=np.concatenate((memFC,semFC,glassFC,motFC))
    #test sub
    test_memFC=classification.matFiles(dataDir+'mem/'+test_sub+'_parcel_corrmat.mat')
    test_glassFC=classification.matFiles(dataDir+'glass/'+test_sub+'_parcel_corrmat.mat')
    test_semFC=classification.matFiles(dataDir+'semantic/'+test_sub+'_parcel_corrmat.mat')
    test_motFC=classification.matFiles(dataDir+'motor/'+test_sub+'_parcel_corrmat.mat')
    test_restFC=classification.matFiles(dataDir+'rest/corrmats_timesplit/fourths/'+test_sub+'_parcel_corrmat.mat')
    test_taskFC=np.concatenate((test_memFC,test_semFC,test_glassFC,test_motFC))
    diff_score, same_score, acc_score=CV_foldsAll(train_sub, clf, taskFC, restFC, test_taskFC, test_restFC)
    return diff_score, same_score

def CV_foldsAll(train_sub, clf, taskFC, restFC, test_taskFC, test_restFC):
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

    kf = KFold(n_splits=5)

    taskSize=taskFC.shape[0]
    restSize=restFC.shape[0]
    t = np.ones(taskSize, dtype = int)
    r=np.zeros(restSize, dtype=int)

    #permute the labels
    #Concatenate rest and task labels
    Y=np.concatenate((t,r))
    #Permute the data
    Y_perm=np.random.permutation(Y)
    #For the purpose of this script split them back into a pseudo rest and task array
    tP, rP =np.array_split(Y_perm, 2)
    #################################################################################
    #test set
    test_taskSize=test_taskFC.shape[0]
    test_restSize=test_restFC.shape[0]
    testT= np.ones(test_taskSize, dtype = int)
    testR= np.zeros(test_restSize, dtype = int)
    CVacc=[]
    CVdf=pd.DataFrame()
    df=pd.DataFrame()
    acc_score=[]
    #fold each training set
    for train_index, test_index in kf.split(taskFC):
        Xtrain_rest, Xval_rest=restFC[train_index], restFC[test_index]
        Xtrain_task, Xval_task=taskFC[train_index], taskFC[test_index]
        #use permuted for the training, use the correct labels for the test
        ytrain_rest, yval_rest=rP[train_index], r[test_index]
        ytrain_task, yval_task=tP[train_index], t[test_index]
        X_tr=np.concatenate((Xtrain_task, Xtrain_rest))
        X_val=np.concatenate((Xval_task, Xval_rest))
        y_tr = np.concatenate((ytrain_task,ytrain_rest))
        y_val=np.concatenate((yval_task, yval_rest))
        clf.fit(X_tr,y_tr)
        #cross validation
        clf.predict(X_val)
        #get accuracy
        CV_score=clf.score(X_val, y_val)
        CVacc.append(CV_score)
        tmpdf=pd.DataFrame()
        acc_scores_per_fold=[]
        #fold each testing set
        for t_index, te_index in kf.split(test_taskFC):
            Xtest_rest=test_restFC[te_index]
            Xtest_task=test_taskFC[te_index]
            X_te=np.concatenate((Xtest_task, Xtest_rest))

            ytest_task=testT[te_index]
            ytest_rest=testR[te_index]
            y_te=np.concatenate((ytest_task, ytest_rest))
            #test set
            clf.predict(X_te)
            #Get accuracy of model
            ACCscores=clf.score(X_te,y_te)
            acc_scores_per_fold.append(ACCscores)
        tmpdf['inner_fold']=acc_scores_per_fold
        score=tmpdf['inner_fold'].mean()
        acc_score.append(score)
    CVdf['acc']=CVacc
    df['cv']=CVacc
    df['outer_fold']=acc_score
    same_sub_score=df['cv'].mean()
    diff_sub_score=df['outer_fold'].mean()
    return diff_sub_score, same_sub_score, acc_score
def classifyScores():
    ALLscores=[]
    SSscore=[]
    OSscore=[]
    for i in range(1000):
        score, SS, OS=classifyAll()
        ALLscores.append(score)
        SSscore.append(SS)
        OSscore.append(OS)
    ALL_perms=pd.DataFrame()
    ALL_perms['diff_acc']=ALLscores
    ALL_perms['SS']=SSscore
    ALL_perms['OS']=OSscore
    ALL_perms.to_csv(outDir+'ALL_acc.csv',index=False)
