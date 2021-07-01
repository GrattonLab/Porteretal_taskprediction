#!/usr/bin/env python
# coding: utf-8

# In[ ]:
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
import numpy as np
import os
import sys
import pandas as pd
from statistics import mean
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
import itertools
from sklearn.preprocessing import StandardScaler #data scaling
from sklearn import decomposition #PCA
#import other python scripts for further anlaysis
import reshape
#import results
import warnings
warnings.filterwarnings("ignore")
# Initialization of directory information:
thisDir = os.path.expanduser('~/Desktop/MSC_Alexis/analysis/')
dataDir = thisDir + 'data/mvpa_data/'
#outDir = thisDir + 'output/results/'
outDir = thisDir + 'output/results/permutation/'
# Subjects and tasks
taskList=['glass','semantic', 'motor','mem']
#omitting MSC06 for classify All
#subList=['MSC01','MSC02','MSC03','MSC04','MSC05','MSC07','MSC10']
subList=['MSC01','MSC02','MSC03','MSC04','MSC05','MSC06','MSC07','MSC10']
splitDict=dict([('MSC01',10),('MSC02',10),('MSC03',8),('MSC04',10),('MSC05',10),('MSC06',9),('MSC07',9),('MSC10',10)])
#all possible combinations of subs and tasks
subsComb=(list(itertools.permutations(subList, 2)))
tasksComb=(list(itertools.permutations(taskList, 2)))

def classifyAll():
    """
    Classifying different subjects along available data rest split into 40 samples to match with task

    Parameters
    -------------

    Returns
    -------------
    df : DataFrame
        Dataframe consisting of average accuracy across all subjects

    """
    final=pd.DataFrame()
    netRoi=np.logspace(1, 4.2, num=1000,dtype=int)#randomly generate values 10 to 55000
    #netRoi=dict([('unassign',14808),('default', 10824),('visual',8736),('fp', 4620),('dan',5264),('van',3151),('salience', 494),('co', 4060),('sm', 2375),('sm-lat', 316),('auditory', 564),('pmn',45),('pon',21)])

    for network in netRoi:
        df=modelAll(network)
        df['network']=network
        #df['feature']=netRoi[network]
        final=pd.concat([final,df])
    #final.to_csv(outDir+'ALL/shuffle_ROIacc.csv',index=False)
    final.to_csv(outDir+'ALL/null_ROIacc.csv',index=False)

def modelAll(network):
    """
    Preparing machine learning model with appropriate data

    Parameters
    -------------

    Returns
    -------------
    total_score : float
            Average accuracy of all folds

    """
    clf=RidgeClassifier()
    master_df=pd.DataFrame()
    data=np.array(['MSC01','MSC02','MSC03','MSC04','MSC05','MSC06','MSC07','MSC10'],dtype='<U61')
    loo = LeaveOneOut()
    for  test, train in loo.split(data): #train on one sub test on the rest
        tmp=pd.DataFrame()
        train_sub=data[train]
        test_sub=data[test]
    #train sub
        memFC=reshape.permROI(dataDir+'mem/'+train_sub[0]+'_parcel_corrmat.mat')
        semFC=reshape.permROI(dataDir+'semantic/'+train_sub[0]+'_parcel_corrmat.mat')
        glassFC=reshape.permROI(dataDir+'glass/'+train_sub[0]+'_parcel_corrmat.mat')
        motFC=reshape.permROI(dataDir+'motor/'+train_sub[0]+'_parcel_corrmat.mat')
        restFC=reshape.permROI(dataDir+'rest/corrmats_timesplit/fourths/'+train_sub[0]+'_parcel_corrmat.mat') #keep tasks seperated in order to collect the right amount of days
        restFC=np.reshape(restFC,(10,4,55278)) #reshape to gather correct days
        #return taskFC,restFC, test_taskFC,test_restFC
        diff=K_folds(network,train_sub[0], clf, memFC,semFC,glassFC,motFC, restFC)
        tmp['train']=train_sub
        tmp['acc']=diff
        master_df=pd.concat([master_df,tmp])
    return master_df

def K_folds(network, train_sub, clf, memFC,semFC,glassFC,motFC,restFC):
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
    session=splitDict[train_sub]
    split=np.empty((session, 55278))
    refROI=pd.read_csv(thisDir+'output/results/permutation/ALL/ref_acc.csv') #for taking differece
    ref=refROI[refROI['train']==train_sub].acc.values
    ref_sub=ref[0]
    #ref_sub=round(ref_sub,2)
    kf = KFold(n_splits=5)
    CVacc=[]
    for train_index, test_index in kf.split(split):
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
        #XtrainNew_task, XtrainNew_rest=reshape.permuteIndicesRandom(Xtrain_task,Xtrain_rest,network)
        #XtrainNew_task, XtrainNew_rest=reshape.permuteIndices(Xtrain_task,Xtrain_rest,network)
        X_tr=np.concatenate((Xtrain_task, Xtrain_rest))
        X_val=reshape.permuteIndicesRandom(Xval_task,Xval_rest,network)
        #X_val=reshape.permuteIndices(Xval_task,Xval_rest,network)
        y_tr = np.concatenate((ytrain_task,ytrain_rest))
        y_val=np.concatenate((yval_task, yval_rest))
        clf.fit(X_tr,y_tr)
        #cross validation
        CV_score=clf.score(X_val, y_val)
        CVacc.append(CV_score)
    same_sub_score=mean(CVacc)
    #same=round(same_sub_score,2)
    diff=same_sub_score - ref_sub
    d=round(diff,2)
    return d



def modelAll_byRow(train_sub):
    """
    Row based permutation ROI analysis; calculates the difference in performance after permuting row

    Parameters
    -------------

    Returns
    -------------
    total_score : float
            Average accuracy of all folds

    """

    clf=RidgeClassifier()
    #resort for gordon script in matlab to match original corrmat
    Parcel_params = reshape.loadParcelParams('Gordon333','/Users/Alexis/Desktop/MSC_Alexis/analysis/data/Parcel_info/')
    roi_sort = np.squeeze(Parcel_params['roi_sort'])
#train sub
    memFC=reshape.permROI(dataDir+'mem/'+train_sub+'_parcel_corrmat.mat')#permROI is necessary in order to have the same indexing as the other script get indices
    semFC=reshape.permROI(dataDir+'semantic/'+train_sub+'_parcel_corrmat.mat')
    glassFC=reshape.permROI(dataDir+'glass/'+train_sub+'_parcel_corrmat.mat')
    motFC=reshape.permROI(dataDir+'motor/'+train_sub+'_parcel_corrmat.mat')
    restFC=reshape.permROI(dataDir+'rest/corrmats_timesplit/fourths/'+train_sub+'_parcel_corrmat.mat') #keep tasks seperated in order to collect the right amount of days
    #taskFC=np.concatenate((memFC,semFC,glassFC,motFC))
    restFC=np.reshape(restFC,(10,4,55278)) #reshape to gather correct days
    results=K_byRow(train_sub, clf, memFC,semFC,glassFC,motFC, restFC)
    data={'acc':results,'roi':roi_sort}
    df=pd.DataFrame(data)
    df.sort_values(by='roi',inplace=True)
    array=df['acc'].to_numpy()#convert back to numpy array for saving
    array.tofile('/Users/Alexis/Desktop/MSC_Alexis/analysis/output/results/permutation/ALL/subs/'+train_sub+'_Row.csv', sep = ',')

def K_byRow(train_sub, clf, memFC,semFC,glassFC,motFC, restFC):
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

    session=splitDict[train_sub]
    split=np.empty((session, 55278))
    refROI=pd.read_csv(thisDir+'output/results/permutation/ALL/ref_acc.csv') #for taking differece
    ref=refROI[refROI['train']==train_sub].acc.values
    ref_sub=ref[0] #use this to calculate difference
    results=np.empty((333))#store results to use for plotting take diff from ref_sub per row and store here
    kf = KFold(n_splits=5)
    for rowID, null in enumerate(results):
        CVacc=[]
        #fold each training set
        for train_index, test_index in kf.split(split):
            memtrain, memval=memFC[train_index], memFC[test_index]
            semtrain, semval=semFC[train_index], semFC[test_index]
            mottrain, motval=motFC[train_index], motFC[test_index]
            glatrain, glaval=glassFC[train_index], glassFC[test_index]
            Xtrain_task=np.concatenate((memtrain,semtrain,mottrain,glatrain))
            Xval_task=np.concatenate((memval,semval,motval,glaval))
            Xtrain_rest, Xval_rest=restFC[train_index,:,:], restFC[test_index,:,:]
            Xtrain_rest=np.reshape(Xtrain_rest,(-1,55278))
            Xval_rest=np.reshape(Xval_rest,(-1,55278))
            ytrain_task = np.ones(Xtrain_task.shape[0], dtype = int)
            ytrain_rest=np.zeros(Xtrain_rest.shape[0], dtype=int)
            yval_task = np.ones(Xval_task.shape[0], dtype = int)
            yval_rest=np.zeros(Xval_rest.shape[0], dtype=int)
            y_tr = np.concatenate((ytrain_task,ytrain_rest))
            X_val=reshape.permuteIndices_byRow(Xval_task,Xval_rest,rowID)
            y_val=np.concatenate((yval_task, yval_rest))
            #X_task, X_rest=reshape.permuteIndices_byRow(Xtrain_task,Xtrain_rest,rowID)
            X_train=np.concatenate((Xtrain_task, Xtrain_rest))
            clf.fit(X_train,y_tr)
            #cross validation
            CV_score=clf.score(X_val, y_val)
            CVacc.append(CV_score)
        same_sub_score=mean(CVacc)
        diff=same_sub_score - ref_sub
        #d=round(diff,4)
        results[rowID]=diff
    return results



def NULLmodelAll_byRow(train_sub):
    """
    Null permuting 332 rows ignoring 1 row

    Parameters
    -------------

    Returns
    -------------
    total_score : float
            Average accuracy of all folds

    """
    clf=RidgeClassifier()

#train sub
    memFC=reshape.permROI(dataDir+'mem/'+train_sub+'_parcel_corrmat.mat')#permROI is necessary in order to have the same indexing as the other script get indices
    semFC=reshape.permROI(dataDir+'semantic/'+train_sub+'_parcel_corrmat.mat')
    glassFC=reshape.permROI(dataDir+'glass/'+train_sub+'_parcel_corrmat.mat')
    motFC=reshape.permROI(dataDir+'motor/'+train_sub+'_parcel_corrmat.mat')
    restFC=reshape.permROI(dataDir+'rest/corrmats_timesplit/fourths/'+train_sub+'_parcel_corrmat.mat') #keep tasks seperated in order to collect the right amount of days
    #taskFC=np.concatenate((memFC,semFC,glassFC,motFC))
    #to have more control over sessions
    #taskFC=np.dstack((memFC,semFC,glassFC,motFC))#10x55278x4
    restFC=np.reshape(restFC,(10,4,55278)) #reshape to gather correct days

    results=NULLK_byRow(train_sub, clf, memFC,semFC,glassFC,motFC, restFC)
    return results

def NULLK_byRow(train_sub, clf, memFC,semFC,glassFC,motFC,restFC):
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
    refROI=pd.read_csv(thisDir+'output/results/permutation/ALL/ref_acc.csv') #for taking differece
    refROI.drop(columns=['diff_sub'],inplace=True)
    ref=refROI[refROI['train']==train_sub].same_sub.values
    ref_sub=ref[0] #use this to calculate difference
    results=np.empty((333))#store results to use for plotting take diff from ref_sub per row and store here
    kf = KFold(n_splits=5,shuffle=True)
    if train_sub=='MSC03':
        split=np.empty((8,55278))
        #xtrainSize=24
        #xtestSize=4
    elif train_sub=='MSC06' or train_sub=='MSC07':
        split=np.empty((9,55278))
    else:
        split=np.empty((10,55278))

    for rowID, null in enumerate(results):
        CVacc=[]
        df=pd.DataFrame()
        #fold each training set

        for train_index, test_index in kf.split(split):
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
            X_val=reshape.NULLpermuteIndices_byRow(Xval_task,Xval_rest,rowID)#permute all ignore rowID
            X_tr=np.concatenate((Xtrain_task,Xtrain_rest))
            y_tr = np.concatenate((ytrain_task,ytrain_rest))
            y_val=np.concatenate((yval_task, yval_rest))
            clf.fit(X_tr,y_tr)
            #cross validation
            CV_score=clf.score(X_val, y_val)
            CVacc.append(CV_score)
        df['cv']=CVacc
        #Different sub outer acc
        same_sub_score=df['cv'].mean()
        diff=same_sub_score-ref_sub
        results[rowID]=diff
    return results
