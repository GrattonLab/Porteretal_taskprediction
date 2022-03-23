from sklearn.metrics import confusion_matrix
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
#from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import numpy as np
import os
import sys
import pandas as pd
import itertools
import reshape
from statistics import mean
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
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
    SS_TPV_per_task=[]
    SS_RPV_per_task=[]
    OS_TPV_per_task=[]
    OS_RPV_per_task=[]
    tmp_df=pd.DataFrame(DSvars, columns=['sub','task'])
    dfDS=pd.DataFrame()
    dfDS[['train_sub','test_sub']]=pd.DataFrame(tmp_df['sub'].tolist())
    dfDS['task']=tmp_df['task']
    for index, row in dfDS.iterrows():
        taskFC=reshape.matFiles(dataDir+row['task']+'/'+row['train_sub']+'_parcel_corrmat.mat')
        restFC=reshape.matFiles(dataDir+'rest/'+row['train_sub']+'_parcel_corrmat.mat')
        test_taskFC=reshape.matFiles(dataDir+row['task']+'/'+row['test_sub']+'_parcel_corrmat.mat')
        test_restFC=reshape.matFiles(dataDir+'rest/'+row['test_sub']+'_parcel_corrmat.mat')
        SSacc, SS_TPV, SS_RPV, OSacc, OS_TPV, OS_RPV=folds(clf, taskFC, restFC, test_taskFC, test_restFC)
        same_sub_per_task.append(SSacc)
        SS_TPV_per_task.append(SS_TPV)
        SS_RPV_per_task.append(SS_RPV)
        diff_sub_per_task.append(OSacc)
        OS_TPV_per_task.append(OS_TPV)
        OS_RPV_per_task.append(OS_RPV)
    dfDS['diff_sub']=diff_sub_per_task
    dfDS['same_sub']=same_sub_per_task
    dfDS['OS_TPV']=OS_TPV_per_task
    dfDS['OS_RPV']=OS_RPV_per_task
    dfDS['SS_TPV']=SS_TPV_per_task
    dfDS['SS_RPV']=SS_RPV_per_task
    dfDS.to_csv(outDir+classifier+'/single_task/acc.csv',index=False)

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
    SS_TPV_score=[]
    SS_RPV_score=[]
    #Test other subs
    OS_acc=[]
    OS_TPV_score=[]
    OS_RPV_score=[]

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
        SS_y_predict=clf.predict(X_val)
        SStn, SSfp, SSfn, SStp=confusion_matrix(y_val, SS_y_predict).ravel()
        same_sub_TPV=SStp/(SStp+SSfp)
        same_sub_RPV=SStn/(SStn+SSfn)
        SSscores=clf.score(X_val,y_val)
        SS_acc.append(SSscores)
        SS_TPV_score.append(same_sub_TPV)
        SS_RPV_score.append(same_sub_RPV)
        #Other subject
        OS_y_predict=clf.predict(Xtest)
        OStn, OSfp, OSfn, OStp=confusion_matrix(ytest, OS_y_predict).ravel()
        other_sub_TPV=OStp/(OStp+OSfp)
        other_sub_RPV=OStn/(OStn+OSfn)
        OSscores=clf.score(Xtest,ytest)
        OS_acc.append(OSscores)
        OS_TPV_score.append(other_sub_TPV)
        OS_RPV_score.append(other_sub_RPV)
    OStotal_acc=mean(OS_acc)
    OStotal_TPV=mean(OS_TPV_score)
    OStotal_RPV=mean(OS_RPV_score)
    SStotal_acc=mean(SS_acc)
    SStotal_TPV=mean(SS_TPV_score)
    SStotal_RPV=mean(SS_RPV_score)
    return SStotal_acc, SStotal_TPV, SStotal_RPV, OStotal_acc, OStotal_TPV, OStotal_RPV
