#!/usr/bin/env python
# coding: utf-8

# In[ ]:
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
    TPV_scores_OS=[]
    RPV_scores_OS=[]
    acc_scores_SS=[]
    TPV_scores_SS=[]
    RPV_scores_SS=[]
    df=pd.DataFrame(subsComb, columns=['train_sub','test_sub'])
    for index, row in df.iterrows():
        SS_score, SS_TPV_total,SS_RPV_total, OS_score, OS_TPV_total,OS_RPV_total=modelAll_binary(clf,train_sub=row['train_sub'], test_sub=row['test_sub'])
        acc_scores_SS.append(SS_score)
        TPV_scores_SS.append(SS_TPV_total)
        RPV_scores_SS.append(SS_RPV_total)
        acc_scores_OS.append(OS_score)
        TPV_scores_OS.append(OS_TPV_total)
        RPV_scores_OS.append(OS_RPV_total)
    df['same_sub']=acc_scores_SS
    df['SS_tpv']=TPV_scores_SS
    df['SS_rpv']=RPV_scores_SS
    df['diff_sub']=acc_scores_OS
    df['OS_tpv']=TPV_scores_OS
    df['OS_rpv']=RPV_scores_OS
    df.to_csv(outDir+classifier+'/ALL_Binary/acc.csv',index=False)


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

    CV_score, CV_TPV_total,CV_RPV_total, DS_score, DS_TPV_total,DS_RPV_total=foldsBinary(train_sub, clf, memFC,semFC,glassFC,motFC, restFC, test_taskFC,test_restFC)
    return CV_score, CV_TPV_total,CV_RPV_total, DS_score, DS_TPV_total,DS_RPV_total

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
    CV_TPV=[]
    CV_RPV=[]
    DSacc=[]
    DS_TPV=[]
    DS_RPV=[]
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
        clf.fit(X_tr,y_tr)
        y_pred=clf.predict(X_val)
        #Test labels and predicted labels to calculate sensitivity specificity
        tn, fp, fn, tp=confusion_matrix(y_val, y_pred).ravel()
        CV_TPV_score=tp/(tp+fp)
        CV_RPV_score=tn/(tn+fn)
        CV_score=clf.score(X_val, y_val)
        CVacc.append(CV_score)
        CV_TPV.append(CV_TPV_score)
        CV_RPV.append(CV_RPV_score)
        y_pred_testset=clf.predict(Xtest)
        DStn, DSfp, DSfn, DStp=confusion_matrix(ytest, y_pred_testset).ravel()
        DS_TPV_score=DStp/(DStp+DSfp)
        DS_RPV_score=DStn/(DStn+DSfn)
        ACCscores=clf.score(Xtest,ytest)
        DSacc.append(ACCscores)
        DS_TPV.append(DS_TPV_score)
        DS_RPV.append(DS_RPV_score)
    CV_score=mean(CVacc)
    CV_TPV_total=mean(CV_TPV)
    CV_RPV_total=mean(CV_RPV)
    DS_score=mean(DSacc)
    DS_TPV_total=mean(DS_TPV)
    DS_RPV_total=mean(DS_RPV)
    return  CV_score, CV_TPV_total,CV_RPV_total, DS_score, DS_TPV_total,DS_RPV_total




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
    all_CM_DS=np.zeros((5,5))
    all_CM_CV=np.zeros((5,5))
    fig=plt.figure(figsize=(25,20))#, constrained_layout=True)
    fig.text(.08, 1.05, 'a',fontsize=28)
    fig.text(.45, 1.05, 'Same Person',fontsize=28)
    fig.text(.08, .48, 'b',fontsize=28)
    fig.text(.43, .48, 'Different Person',fontsize=28)
    #plt.rcParams['figure.constrained_layout.use'] = True
#Add grid space for subplots 1 rows by 3 columns
    #gs = gridspec.GridSpec(nrows=4, ncols=4)
    gs00 = fig.add_gridspec(nrows=2, ncols=4,top=1, bottom=.55,wspace=0.1, hspace=0.13)
    gs01 = fig.add_gridspec(nrows=2, ncols=4, top=.45, bottom=0,wspace=0.1, hspace=0.13)
    b=0
    #train sub
    master_df=pd.DataFrame()
    data=np.array(['MSC01','MSC02','MSC03','MSC04','MSC05','MSC06','MSC07','MSC10'],dtype='<U61')
    loo = LeaveOneOut()
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
        same_Tsub, diff_Tsub,sameF,diffF,same_sub_CM, DS_cm=folds_MC(train_sub, clf, memFC,semFC,glassFC,motFC, restFC, testFC, ytest)
        DS=DS_cm / DS_cm.astype(np.float).sum(axis=1)
        CV=same_sub_CM / same_sub_CM.astype(np.float).sum(axis=1)
        if b<4:
            a=0
            ax1=fig.add_subplot(gs00[a,b])
            ax=ConfusionMatrixDisplay(CV,display_labels=["Rest","Memory","Semantic","Motor", "Coherence"]).plot(cmap=plt.cm.Blues,colorbar=False,ax=ax1)
            ax1.set_xlabel(' ')
            plt.title(train_sub[0],fontsize=25)

            ax2=fig.add_subplot(gs01[a,b])
            ax=ConfusionMatrixDisplay(DS,display_labels=["Rest","Memory","Semantic","Motor", "Coherence"]).plot(cmap=plt.cm.Blues,colorbar=False,ax=ax2)
            ax2.set_xlabel(' ')
            plt.title(train_sub[0],fontsize=25)
            if b==0:
                ax1.set_ylabel('True Label',fontsize=25)
                ax2.set_ylabel('True Label',fontsize=25)
                ax1.tick_params(labelsize=25)
                ax2.tick_params(labelsize=25)
                plt.setp(ax1.get_xticklabels(), visible=False)
                plt.setp(ax2.get_xticklabels(), visible=False)
            else:
                ax1.set_ylabel(' ')
                ax2.set_ylabel(' ')
                plt.setp(ax1.get_xticklabels(), visible=False)
                plt.setp(ax2.get_xticklabels(), visible=False)
                plt.setp(ax1.get_yticklabels(), visible=False)
                plt.setp(ax2.get_yticklabels(), visible=False)
            b=b+1
        else:
            a=1
            c=b-4
            ax1=fig.add_subplot(gs00[a,c])
            ax=ConfusionMatrixDisplay(CV,display_labels=["Rest","Memory","Semantic","Motor", "Coherence"]).plot(cmap=plt.cm.Blues,colorbar=False,ax=ax1)
            ax1.set_xlabel(' ')
            plt.title(train_sub[0],fontsize=25)

            ax2=fig.add_subplot(gs01[a,c])
            ax=ConfusionMatrixDisplay(DS,display_labels=["Rest","Memory","Semantic","Motor", "Coherence"]).plot(cmap=plt.cm.Blues,colorbar=False,ax=ax2)
            plt.xlabel('Predicted Label',fontsize=25)
            plt.title(train_sub[0],fontsize=25)
            if c==0:
                ax1.set_ylabel('True Label',fontsize=25)
                ax2.set_ylabel('True Label',fontsize=25)
                plt.xticks(rotation=90)
                ax1.tick_params(labelsize=25)
                ax2.tick_params(labelsize=25)
                plt.setp(ax1.get_xticklabels(), visible=False)
                #plt.setp(ax2.get_xticklabels(), visible=False)
            else:
                ax1.set_ylabel('')
                ax2.set_ylabel('')
                plt.setp(ax1.get_xticklabels(), visible=False)
                #plt.setp(ax2.get_xticklabels(), visible=False)
                plt.setp(ax1.get_yticklabels(), visible=False)
                plt.setp(ax2.get_yticklabels(), visible=False)
                ax2.tick_params(labelsize=25)
                plt.xticks(rotation=90)
            b=b+1

        plt.savefig(figsDir+'MC_allSubs.png', bbox_inches='tight')

        all_CM_DS=DS_cm+all_CM_DS
        all_CM_CV=same_sub_CM+all_CM_CV
        CV_tmp['Task']=['rest','mem','sem','mot','glass']
        CV_tmp['f1']=sameF
        CV_tmp['train']=train_sub[0]
        CV_tmp['acc']=same_Tsub

        CV_tmp['Analysis']='Same Person'

        DS_tmp['Task']=['rest','mem','sem','mot','glass']
        DS_tmp['f1']=diffF
        DS_tmp['train']=train_sub[0]
        DS_tmp['acc']=diff_Tsub
        DS_tmp['Analysis']='Different Person'

        master_df=pd.concat([master_df,CV_tmp,DS_tmp])

    master_df.to_csv(outDir+classifier+'/ALL_MC/acc.csv',index=False)
    finalDS=all_CM_DS / all_CM_DS.astype(np.float).sum(axis=1)
    finalCV=all_CM_CV / all_CM_CV.astype(np.float).sum(axis=1)
    fig=plt.figure(figsize=(15,10), constrained_layout=True)
    plt.rcParams['figure.constrained_layout.use'] = True
#Add grid space for subplots 1 rows by 3 columns
    gs = gridspec.GridSpec(nrows=2, ncols=2)
    ax0=fig.add_subplot(gs[1,0])
    ax=ConfusionMatrixDisplay(finalCV,display_labels=["Rest","Memory","Semantic","Motor", "Coherence"]).plot(cmap=plt.cm.Blues,ax=ax0,colorbar=False)
    plt.ylabel('True Label',fontsize=25)
    plt.xlabel('Predicted Label',fontsize=25)
    plt.title('Average Multiclass Within Person',fontsize=25)
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=20)
    ax1=fig.add_subplot(gs[1,1])
    ax=ConfusionMatrixDisplay(finalDS,display_labels=["Rest","Memory","Semantic","Motor", "Coherence"]).plot(cmap=plt.cm.Blues,ax=ax1,colorbar=False)
    plt.ylabel(' ',fontsize=25)
    plt.xlabel('Predicted Label',fontsize=25)
    plt.title('Average Multiclass Across Person',fontsize=25)
    plt.yticks([],[])
    plt.xticks(fontsize=17)
    plt.savefig(figsDir+'ALL_MC_average.png', bbox_inches='tight')

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
    count=0
    #fold each training set
    session=splitDict[train_sub[0]]
    split=np.empty((session, 55278))
    sameF1=np.empty((session,5))
    diffF1=np.empty((session,5))
    same_sub_CM=np.zeros((5,5))
    diff_sub_CM=np.empty((session,5,5))
    diff_count=0
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
        clf.fit(Xtrain,ytrain)
        score=clf.score(Xval, yval)
        CVTacc.append(score)
        y_predict=clf.predict(Xval)
        score_same = f1_score(yval, y_predict, average=None)
        cm_same = confusion_matrix(yval, y_predict)
        same_sub_CM=same_sub_CM+cm_same
        #order is position rest, mem, sem, mot, glass
        sameF1[count]=score_same
        scoreT=clf.score(testFC,ytest)
        DSTacc.append(scoreT)
        y_pre=clf.predict(testFC)
        score_diff = f1_score(ytest, y_pre, average=None)
        cm_diff = confusion_matrix(ytest, y_pre)
        diff_sub_CM[diff_count]=cm_diff
        diff_count=diff_count+1
        #order is position rest, mem, sem, mot, glass
        diffF1[count]=score_diff
        count=count+1
    same_f=sameF1.mean(axis=0)
    same_Tsub=mean(CVTacc)
    diff_Tsub=mean(DSTacc)
    diff_f=diffF1.mean(axis=0)
    DS_cm=diff_sub_CM.mean(axis=0,dtype=int)
    return same_Tsub,diff_Tsub,same_f, diff_f, same_sub_CM, DS_cm

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
    CV, DS, SS_TPV, SS_RPV, OS_TPV, OS_RPV=Net_folds(clf, taskFC, restFC, test_taskFC, test_restFC)
    return CV, DS, SS_TPV, SS_RPV, OS_TPV, OS_RPV

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
    CVTPV=[]
    CVRPV=[]
    DSacc=[]
    DSTPV=[]
    DSRPV=[]
    for train_index, test_index in loo.split(taskFC):
        Xtrain_rest, Xval_rest=restFC[train_index], restFC[test_index]
        Xtrain_task, Xval_task=taskFC[train_index], taskFC[test_index]
        ytrain_rest, yval_rest=r[train_index], r[test_index]
        ytrain_task, yval_task=t[train_index], t[test_index]
        X_tr=np.concatenate((Xtrain_task, Xtrain_rest))
        X_val=np.concatenate((Xval_task, Xval_rest))
        y_tr = np.concatenate((ytrain_task,ytrain_rest))
        y_val=np.concatenate((yval_task, yval_rest))
        clf.fit(X_tr,y_tr)
        y_pred=clf.predict(X_val)
        tn, fp, fn, tp=confusion_matrix(y_val, y_pred).ravel()
        pos=tp+fp
        neg=tn+fn
        if pos == 0:
            TPV=0 #can't divide by zero
        else:
            TPV=tp/(tp+fp)
        if neg == 0:
            RPV=0
        else:
            RPV=tn/(tn+fn)

        #get accuracy
        CV_score=clf.score(X_val, y_val)
        CVacc.append(CV_score)
        CVTPV.append(TPV)
        CVRPV.append(RPV)
        X_te=np.concatenate((test_taskFC, test_restFC))
        y_te=np.concatenate((testT, testR))
        y_pred_testset=clf.predict(X_te)
        #Test labels and predicted labels to calculate sensitivity specificity
        DStn, DSfp, DSfn, DStp=confusion_matrix(y_te, y_pred_testset).ravel()
        DS_TPV=DStp/(DStp+DSfp)
        DS_RPV=DStn/(DStn+DSfn)
        DSTPV.append(DS_TPV)
        DSRPV.append(DS_RPV)
        ACCscores=clf.score(X_te,y_te)
        DSacc.append(ACCscores)
    CV=mean(CVacc)
    DS=mean(DSacc)
    SS_TPV=mean(CVTPV)
    SS_RPV=mean(CVRPV)
    OS_TPV=mean(DSTPV)
    OS_RPV=mean(DSRPV)
    return CV, DS, SS_TPV, SS_RPV, OS_TPV, OS_RPV

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
    tpv_ds=[]
    rpv_ds=[]
    acc_scores_cv=[]
    tpv_cv=[]
    rpv_cv=[]
    df=pd.DataFrame(subsComb, columns=['train_sub','test_sub'])
    for index, row in df.iterrows():
        CV, DS, SS_TPV, SS_RPV, OS_TPV, OS_RPV=modelNets(clf,train_sub=row['train_sub'], test_sub=row['test_sub'])
        acc_scores_ds.append(DS)
        acc_scores_cv.append(CV)
        tpv_ds.append(OS_TPV)
        rpv_ds.append(OS_RPV)
        tpv_cv.append(SS_TPV)
        rpv_cv.append(SS_RPV)
    df['cv']=acc_scores_cv
    df['cv_tpv']=tpv_cv
    df['cv_rpv']=rpv_cv
    df['ds']=acc_scores_ds
    df['ds_tpv']=tpv_ds
    df['ds_rpv']=rpv_ds
    df.to_csv(outDir+classifier+'/ALL_IndNet/acc.csv',index=False)
