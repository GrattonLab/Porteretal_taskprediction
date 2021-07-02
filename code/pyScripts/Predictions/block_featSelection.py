#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[2]:
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import RidgeClassifier
import numpy as np
import os
import sys
import pandas as pd
from statistics import mean
from sklearn.model_selection import cross_val_score
from sklearn import metrics
import itertools
#import other python scripts for further anlaysis
import reshape
# Initialization of directory information:
thisDir = os.path.expanduser('~/Desktop/Porteretal_taskprediction/')
dataDir = thisDir +'data/corrmats/'
outDir = thisDir + 'output/results/Ridge/'
# Subjects and tasks
taskList=['semantic','glass', 'motor','mem']
subList=['MSC01','MSC02','MSC03','MSC04','MSC05','MSC06','MSC07','MSC10']
#all possible combinations of subs and tasks
subsComb=(list(itertools.permutations(subList, 2)))
splitDict=dict([('MSC01',10),('MSC02',10),('MSC03',8),('MSC04',10),('MSC05',10),('MSC06',9),('MSC07',9),('MSC10',10)])

#all combination of network to network
networks=['unassign','default','visual','fp','dan','van','salience','co','sm','sm-lat','auditory','pmn','pon']
#no repeats
netComb=(list(itertools.combinations(networks, 2)))

def blockNetAll():
    netDF=pd.DataFrame(netComb, columns=['Network_A','Network_B'])
    finalDF=pd.DataFrame()
    for index, row in netDF.iterrows(): #between networks
        tmp_df=classifyAll(network=row['Network_A'], subnetwork=row['Network_B'])
        tmp_df=tmp_df.groupby(['train_sub']).mean()
        tmp_df.rename(columns={'cv_acc':'Same Subject','acc':'Different Subject'},inplace=True)
        tmp_df.reset_index(inplace=True)
        tmp_df=pd.melt(tmp_df, id_vars=['train_sub'], value_vars=['Same Subject','Different Subject'],var_name='Analysis',value_name='acc')
        tmp_df['Network_A']=row['Network_A']
        tmp_df['Network_B']=row['Network_B']
        finalDF=pd.concat([finalDF, tmp_df])
    for i in networks: #within networks
        tmp_df=classifyAll(network=i,subnetwork=i)
        tmp_df=tmp_df.groupby(['train_sub']).mean()
        tmp_df.rename(columns={'cv_acc':'Same Subject','acc':'Different Subject'},inplace=True)
        tmp_df.reset_index(inplace=True)
        tmp_df=pd.melt(tmp_df, id_vars=['train_sub'], value_vars=['Same Subject','Different Subject'],var_name='Analysis',value_name='acc')
        tmp_df['Network_A']=i
        tmp_df['Network_B']=i
        finalDF=pd.concat([finalDF, tmp_df])
    finalDF.to_csv(thisDir+'ALL_Binary/blockNet_acc.csv')
def classifyAll(network,subnetwork=None):
    """
    Classifying different subjects along available data rest split into 40 samples to match with task

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
        diff_score, same_score=modelAll(network,subnetwork,train_sub=row['train_sub'], test_sub=row['test_sub'])
        acc_scores_per_sub.append(diff_score)
        acc_scores_cv.append(same_score)
    df['same_sub']=acc_scores_cv
    df['diff_sub']=acc_scores_per_sub
    return df

def modelAll(network,subnetwork,train_sub, test_sub):
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
    clf=RidgeClassifier(max_iter=10000)
    df=pd.DataFrame()
    #train sub
    memFC=reshape.network_to_network(dataDir+'mem/'+train_sub+'_parcel_corrmat.mat',network,subnetwork)
    semFC=reshape.network_to_network(dataDir+'semantic/'+train_sub+'_parcel_corrmat.mat',network,subnetwork)
    glassFC=reshape.network_to_network(dataDir+'glass/'+train_sub+'_parcel_corrmat.mat',network,subnetwork)
    motFC=reshape.network_to_network(dataDir+'motor/'+train_sub+'_parcel_corrmat.mat',network,subnetwork)
    restFC=reshape.network_to_network(dataDir+'rest/corrmats_timesplit/fourths/'+train_sub+'_parcel_corrmat.mat',network,subnetwork)
    netSize=reshape.determineNetSize(network,subnetwork)
    restFC=np.reshape(restFC,(10,4,netSize))
    #taskFC=np.concatenate((memFC,semFC,glassFC,motFC))
    #test sub
    test_memFC=reshape.network_to_network(dataDir+'mem/'+test_sub+'_parcel_corrmat.mat',network,subnetwork)
    test_semFC=reshape.network_to_network(dataDir+'semantic/'+test_sub+'_parcel_corrmat.mat',network,subnetwork)
    test_glassFC=reshape.network_to_network(dataDir+'glass/'+test_sub+'_parcel_corrmat.mat',network,subnetwork)
    test_motFC=reshape.network_to_network(dataDir+'motor/'+test_sub+'_parcel_corrmat.mat',network,subnetwork)
    test_restFC=reshape.network_to_network(dataDir+'rest/corrmats_timesplit/fourths/'+test_sub+'_parcel_corrmat.mat',network,subnetwork)
    test_taskFC=np.concatenate((test_memFC,test_semFC,test_glassFC,test_motFC))
    #return taskFC,restFC, test_taskFC,test_restFC
    diff_score, same_score=K_folds(netSize,train_sub, clf, memFC,semFC,glassFC,motFC,restFC, test_taskFC, test_restFC)
    return diff_score, same_score


def K_folds(netSize,train_sub, clf, memFC,semFC,glassFC,motFC, restFC, test_taskFC, test_restFC):
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
    df=pd.DataFrame()
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
        Xtrain_rest=np.reshape(Xtrain_rest,(-1,netSize))
        Xval_rest=np.reshape(Xval_rest,(-1,netSize))
        ytrain_task = np.ones(Xtrain_task.shape[0], dtype = int)
        ytrain_rest=np.zeros(Xtrain_rest.shape[0], dtype=int)
        yval_task = np.ones(Xval_task.shape[0], dtype = int)
        yval_rest=np.zeros(Xval_rest.shape[0], dtype=int)
        X_tr=np.concatenate((Xtrain_task, Xtrain_rest))
        X_val=np.concatenate((Xval_task, Xval_rest))
        y_tr = np.concatenate((ytrain_task,ytrain_rest))
        y_val=np.concatenate((yval_task, yval_rest))
        scaler = preprocessing.StandardScaler().fit(X_tr)
        scaler.transform(X_tr)
        scaler.transform(X_val)
        clf.fit(X_tr,y_tr)
        #cross validation
        y_pred=clf.predict(X_val)
        #Test labels and predicted labels to calculate sensitivity specificity
        #get accuracy
        CV_score=clf.score(X_val, y_val)
        CVacc.append(CV_score)
        #fold each testing set
        scaler.transform(Xtest)
        DS_score=clf.score(Xtest,ytest)
        DSacc.append(DS_score)
    diff_sub_score=mean(DSacc)
    same_sub_score=mean(CVacc)
    return diff_sub_score, same_sub_score


def blockNetDS():
    """
    Classifying different subjects (DS) along the same task

    Parameters
    -------------


    Returns
    -------------
    dfDS : DataFrame
        Dataframe consisting of average accuracy across all subjects

    """
    netDF=pd.DataFrame(netComb, columns=['Network_A','Network_B'])
    finalDF=pd.DataFrame()
    for i in networks:#within network
        tmp_df=classifyDS(network=i,subnetwork=i)
        tmp_df['Network_A']=i
        tmp_df['Network_B']=i
        finalDF=pd.concat([finalDF, tmp_df])
    for index, row in netDF.iterrows(): #between networks
        tmp_df=classifyDS(network=row['Network_A'], subnetwork=row['Network_B'])
        tmp_df['Network_A']=row['Network_A']
        tmp_df['Network_B']=row['Network_B']
        finalDF=pd.concat([finalDF, tmp_df])
    finalDF.to_csv(outDir+'single_task/blockNet_acc.csv')

def classifyDS(network,subnetwork=None):
    clf=RidgeClassifier()
    DS=pd.DataFrame()
    data=np.array(['MSC01','MSC02','MSC03','MSC04','MSC05','MSC06','MSC07','MSC10'],dtype='<U61')
    loo = LeaveOneOut()
    for t in taskList:
        for  test, train in loo.split(data): #train on one sub test on the rest
            tmp=pd.DataFrame()
            train_sub=data[train]
            test_sub=data[test]
        #train sub
            taskFC=reshape.network_to_network(dataDir+t+'/'+train_sub[0]+'_parcel_corrmat.mat',network, subnetwork)
            restFC=reshape.network_to_network(dataDir+'rest/'+train_sub[0]+'_parcel_corrmat.mat',network, subnetwork) #keep tasks seperated in order to collect the right amount of days
            test_taskFC,test_restFC=AllSubFiles_DS(test_sub,t,network, subnetwork)
            same_sub, diff_sub=folds(clf, taskFC,restFC, test_taskFC,test_restFC)
            tmp['train']=train_sub
            tmp['task']=t
            tmp['same_sub']=same_sub
            tmp['diff_sub']=diff_sub
            DS=pd.concat([DS,tmp])
    return DS


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
    test_taskSize=test_taskFC.shape[0]
    test_restSize=test_restFC.shape[0]
    ttest = np.ones(test_taskSize, dtype = int)
    rtest=np.zeros(test_restSize, dtype=int)
    X_test=np.concatenate((test_taskFC, test_restFC))
    y_test = np.concatenate((ttest,rtest))
    df=pd.DataFrame()
    CVacc=[]
    DSacc=[]

    #fold each training set
    for train_index, test_index in loo.split(taskFC):
        Xtrain_rest,Xval_rest=restFC[train_index],restFC[test_index]
        Xtrain_task,Xval_task=taskFC[train_index],taskFC[test_index]
        ytrain_rest,yval_rest=r[train_index],r[test_index]
        ytrain_task,yval_task=t[train_index],t[test_index]
        X_tr=np.concatenate((Xtrain_task, Xtrain_rest))
        y_tr = np.concatenate((ytrain_task,ytrain_rest))
        Xval=np.concatenate((Xval_task,Xval_rest))
        yval=np.concatenate((yval_task,yval_rest))
        scaler = preprocessing.StandardScaler().fit(X_tr)
        scaler.transform(X_tr)
        clf.fit(X_tr,y_tr)
        scaler.transform(Xval)
        scaler.transform(X_test)
        same=clf.score(Xval,yval)
        diff=clf.score(X_test,y_test)

        CVacc.append(same)
        DSacc.append(diff)
    same_sub=mean(CVacc)
    diff_sub=mean(DSacc)

    return same_sub, diff_sub


def AllSubFiles_DS(test_sub,task,network, subnetwork):
    """
    Return task and rest FC all subs
    Parameters
    -----------
    test_sub: Array of testing subs
    Returns
    ------------
    taskFC, restFC : Array of task and rest FC of all testing subs
    """
    a_taskFC=reshape.network_to_network(dataDir+task+'/'+test_sub[0]+'_parcel_corrmat.mat',network, subnetwork)
    a_restFC=reshape.network_to_network(dataDir+'rest/'+test_sub[0]+'_parcel_corrmat.mat',network, subnetwork)

    b_taskFC=reshape.network_to_network(dataDir+task+'/'+test_sub[1]+'_parcel_corrmat.mat',network, subnetwork)
    b_restFC=reshape.network_to_network(dataDir+'rest/'+test_sub[1]+'_parcel_corrmat.mat',network, subnetwork)

    c_taskFC=reshape.network_to_network(dataDir+task+'/'+test_sub[2]+'_parcel_corrmat.mat',network, subnetwork)
    c_restFC=reshape.network_to_network(dataDir+'rest/'+test_sub[2]+'_parcel_corrmat.mat',network, subnetwork)

    d_taskFC=reshape.network_to_network(dataDir+task+'/'+test_sub[3]+'_parcel_corrmat.mat',network, subnetwork)
    d_restFC=reshape.network_to_network(dataDir+'rest/'+test_sub[3]+'_parcel_corrmat.mat',network, subnetwork)

    e_taskFC=reshape.network_to_network(dataDir+task+'/'+test_sub[4]+'_parcel_corrmat.mat',network, subnetwork)
    e_restFC=reshape.network_to_network(dataDir+'rest/'+test_sub[4]+'_parcel_corrmat.mat',network, subnetwork)

    f_taskFC=reshape.network_to_network(dataDir+task+'/'+test_sub[5]+'_parcel_corrmat.mat',network, subnetwork)
    f_restFC=reshape.network_to_network(dataDir+'rest/'+test_sub[5]+'_parcel_corrmat.mat',network, subnetwork)

    g_taskFC=reshape.network_to_network(dataDir+task+'/'+test_sub[6]+'_parcel_corrmat.mat',network, subnetwork)
    g_restFC=reshape.network_to_network(dataDir+'rest/'+test_sub[6]+'_parcel_corrmat.mat',network, subnetwork)


    taskFC=np.concatenate((a_taskFC,b_taskFC,c_taskFC,d_taskFC,e_taskFC,f_taskFC,g_taskFC))
    restFC=np.concatenate((a_restFC,b_restFC,c_restFC,d_restFC,e_restFC,f_restFC,g_restFC))

    return taskFC, restFC
