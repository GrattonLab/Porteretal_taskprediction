#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#comparison Same sub same task - same sub diff task etc etc
import reshape
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import RidgeClassifier
import numpy as np
import os
import sys
import pandas as pd
import classification
from statistics import mean
from sklearn.model_selection import cross_val_score
import itertools
import reshape
import warnings
warnings.filterwarnings("ignore")
# Initialization of directory information:
thisDir = os.path.expanduser('~/Desktop/MSC_Alexis/analysis/')
dataDir = thisDir + 'data/mvpa_data/'
outDir = thisDir + 'output/results/permutation/'
splitDict=dict([('MSC01',10),('MSC02',10),('MSC03',8),('MSC04',10),('MSC05',10),('MSC06',9),('MSC07',9),('MSC10',10)])

taskDict=dict([('mem',0),('motor',1),('glass',2),('semantic',3)])

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

def DSmBS():
    DSmBS_perms=pd.DataFrame()
    for i in range(1000):
        diff=classify_DSmBS()
        DSmBS_perms=pd.concat([DSmBS_perms, diff])
    DSmBS_perms.to_csv(outDir+'STmDT_DSmBS_acc.csv',index=False)
def classify_DSmBS():
    BS=classifyBS()
    DS=classifyDS()
    diff_task=BS.merge(DS,how='left',on=['train_task','train_sub','test_sub'],suffixes=('','_DS'))
    diff_task['diff']=diff_task['acc_DS']-diff_task['acc']
    #diff sub same task - diff sub diff task
    STmDT=diff_task[['train_task','test_task','diff']]
    #take average
    diff=STmDT.groupby(['train_task','test_task']).mean()
    diff.reset_index(inplace=True)
    return diff
def CVmSS():
    CVmSS_perms=pd.DataFrame()
    for i in range(1000):
        dfSS=classifySS()
        #take average
        CVmSS_perms=pd.concat([CVmSS_perms,dfSS])
        print(str(i))
    CVmSS_perms.to_csv(outDir+'TaskOnly_STmDT_CVmSS_acc.csv',index=False)
def classifyDS():
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
    dfDS['train_task']=tmp_df['task']
    for index, row in dfDS.iterrows():
        score=model('DS', train_sub=row['train_sub'], test_sub=row['test_sub'], train_task=row['train_task'], test_task=row['train_task'])
        acc_scores_per_task.append(score)
    dfDS['acc']=acc_scores_per_task
    return dfDS

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
    dfSS['diff']=acc_scores_per_task
    #subset so we only average the SS-OS per train/test tasks
    STmDT=dfSS[['train_task','test_task','diff']]
    #take average
    diff=STmDT.groupby(['train_task','test_task']).mean()
    diff.reset_index(inplace=True)
    return diff

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
    taskFC=classification.matFiles(dataDir+train_task+'/'+train_sub+'_parcel_corrmat.mat')

    #if your subs are the same
    if train_sub==test_sub:
        restFC=classification.matFiles(dataDir+'rest/corrmats_timesplit/half/'+train_sub+'_parcel_corrmat.mat')
        restFC, test_restFC=train_test_split(restFC, test_size=.5)
        test_taskFC=classification.matFiles(dataDir+test_task+'/'+test_sub+'_parcel_corrmat.mat')
        total_score=CV_folds(clf, analysis, taskFC, restFC, test_taskFC, test_restFC)
    else:
        restFC=classification.matFiles(dataDir+'rest/'+train_sub+'_parcel_corrmat.mat')
        test_taskFC=classification.matFiles(dataDir+test_task+'/'+test_sub+'_parcel_corrmat.mat')
        test_restFC=classification.matFiles(dataDir+'rest/'+test_sub+'_parcel_corrmat.mat')
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
    yTest= np.ones(test_taskFC.shape[0], dtype = int)
    r_tmp=np.zeros(restSize, dtype=int)
    #Concatenate rest and task labels
    #Y=np.concatenate((t_tmp,r_tmp))
    #Permute the data
    Y_perm=np.random.permutation(Y)
    #For the purpose of this script split them back into a pseudo rest and task array
    t, r =np.array_split(Y_perm, 2)
    if analysis=='SS':
        SS_ST_acc_scores=[]
        OS_ST_acc_scores=[]
        ST_tmp=pd.DataFrame()
        df=pd.DataFrame()
        acc_score=[]
        acc_scores_per_fold=[]
        for train_index, test_index in loo.split(taskFC):
            Xtrain_rest, Xval_rest=restFC[train_index], restFC[test_index]
            Xtrain_task, Xval_task=taskFC[train_index], taskFC[test_index]
            ytrain_rest=r[train_index]
            ytrain_task=t[train_index]
            X_tr=np.concatenate((Xtrain_task, Xtrain_rest))
            y_tr = np.concatenate((ytrain_task,ytrain_rest))

            #same sub
            #X_val=np.concatenate((Xval_task,Xval_rest))
            y_val=np.array([1])

            clf.fit(X_tr,y_tr)
            #clf.predict(X_val)
            #same sub same task:ST
            SS=clf.score(Xval_task,y_val)
            SS_ST_acc_scores.append(SS)
            OS=clf.score(test_taskFC,yTest)
            OS_ST_acc_scores.append(OS)
            #tmpdf=pd.DataFrame()
            #acc_scores_per_fold=[]
            """
            for t_index, te_index in loo.split(test_taskFC):
                Xtest_task=test_taskFC[te_index]
                Xtest_rest=test_restFC[te_index]
                X_Test = np.concatenate((Xtest_task, Xtest_rest))
                #This way we are including the correct rest and task labels
                y_Test = np.array([1, 0])
                #same sub diff task DT
                clf.predict(X_Test)
                #Get accuracy of model
                ACCscores=clf.score(X_Test,y_Test)
                acc_scores_per_fold.append(ACCscores)
            tmpdf['inner_fold']=acc_scores_per_fold
            score=tmpdf['inner_fold'].mean()
            acc_score.append(score)
            """
        #Same sub same task

        ST_tmp['outer_fold']=SS_ST_acc_scores
        SS_ST=ST_tmp['outer_fold'].mean()

        #same sub diff task
        df['outer_fold']=OS_ST_acc_scores
        SS_DT=df['outer_fold'].mean()

        #take the difference
        total_score=SS_ST-SS_DT
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
    elif analysis=='DS':
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
    return total_score


def multiclass_perm():
    master_df=pd.DataFrame()
    for i in range(1000):
        tmp=multiclassAll()
        master_df=pd.concat([master_df,tmp])
        print(str(i))
    master_df.to_csv(outDir+'ALL/multiclass_Diffperm.csv',index=False)

def multiclassAll():
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
    master_df=pd.DataFrame()
    d=pd.DataFrame()
    data=np.array(['MSC01','MSC02','MSC03','MSC04','MSC05','MSC06','MSC07','MSC10'],dtype='<U61')
    loo = LeaveOneOut()
    for  test, train in loo.split(data): #train on one sub test on the rest
        tmp=pd.DataFrame()
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
        diff=K_folds_MC(train_sub, clf, memFC,semFC,glassFC,motFC, restFC, testFC, ytest)
        tmp['train']=train_sub
        tmp['diff']=diff
        master_df=pd.concat([master_df,tmp])
    return master_df
    #master_df.to_csv(outDir+'ALL/TrueDiff_multiclass_acc.csv',index=False)

def K_folds_MC(train_sub, clf, memFC,semFC,glassFC,motFC, restFC, testFC,ytest):
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
    number=memFC.shape[1]

    yrest=np.zeros(restFC.shape[0])
    ymem=np.ones(memFC.shape[0])
    ysem=np.full(semFC.shape[0],2)
    yglass=np.full(glassFC.shape[0],3)
    ymot=np.full(motFC.shape[0],4)
    CVTacc=[]

    df=pd.DataFrame()
    DSTacc=[]

    #fold each training set
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
        ytrain=np.random.permutation(ytrain) #permute training labels
        clf.fit(Xtrain,ytrain)
        score=clf.score(Xval, yval)
        CVTacc.append(score)
        scoreT=clf.score(testFC,ytest)
        DSTacc.append(scoreT)


    same_Tsub=mean(CVTacc)
    diff_Tsub=mean(DSTacc)
    diff=same_Tsub-diff_Tsub
    return diff


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
    #rest, mem, sem, mot, glass

    testFC=np.concatenate((restFC,memFC,semFC,motFC, glassFC))
    ytest=np.concatenate((testR,ymem,ysem,ymot,yglass))
    #taskFC=np.concatenate((a_memFC,a_semFC,a_glassFC,a_motFC,b_memFC,b_semFC,b_glassFC,b_motFC,c_memFC,c_semFC,c_glassFC,c_motFC,d_memFC,d_semFC,d_glassFC,d_motFC,e_memFC,e_semFC,e_glassFC,e_motFC,f_memFC,f_semFC,f_glassFC,f_motFC,g_memFC,g_semFC,g_glassFC,g_motFC))

    #testFC=np.concatenate((taskFC,restFC))
    #ytest=np.concatenate((ytask,testR))
    return testFC,ytest
    #return taskFC, restFC,ytask, testR


def SS_sep_perm():
    master_df=pd.DataFrame()
    for i in range(1000):
        tmp=classifySS_sep()
        master_df=pd.concat([master_df,tmp])
        print(str(i))
    #master_df.to_csv(outDir+'SS/sepTask_perm.csv',index=False) #perm is the differencescore
    master_df.to_csv(outDir+'SS/acc.csv',index=False) #acc is the null


def classifySS_sep():
    """
    Classifying the same subject (SS) along a different task

    Parameters
    -------------


    Returns
    -------------
    dfSS : DataFrame
        Dataframe consisting of average accuracy across all subjects

    """
    clf=RidgeClassifier()
    STmDT=[]

    tmp_df=pd.DataFrame(SSvars, columns=['sub','task'])
    dfSS=pd.DataFrame()
    dfSS[['train_task','test_task']]=pd.DataFrame(tmp_df['task'].tolist())
    dfSS['sub']=tmp_df['sub']
    for index, row in dfSS.iterrows():
        taskFC=reshape.matFiles(dataDir+row['train_task']+'/'+row['sub']+'_parcel_corrmat.mat')
        restFC=reshape.matFiles(dataDir+'rest/corrmats_timesplit/fourths/'+row['sub']+'_parcel_corrmat.mat') #keep tasks seperated in order to collect the right amount of days
        nsize=restFC.shape[1]
        restFC=np.reshape(restFC,(10,4,nsize)) #reshape to gather correct days
        trainRest=taskDict[row['train_task']]
        testRest=taskDict[row['test_task']]
        Xtrain_rest, Xval_rest=restFC[:,trainRest,:], restFC[:,testRest,:]
        testFC=reshape.matFiles(dataDir+row['test_task']+'/'+row['sub']+'_parcel_corrmat.mat')
        ytest=np.ones(testFC.shape[0])
        #same, diff=SS_folds(clf,taskFC,restFC,testFC,ytest)
        diff=SS_folds(row['sub'],clf, taskFC,Xtrain_rest, testFC, Xval_rest)
        STmDT.append(diff)
    dfSS['diff']=STmDT
    dfSS.to_csv(outDir+'SS/sepTask_Trueacc.csv',index=False) #true difference score
    #return dfSS

def SS_folds(train_sub,clf,taskFC, restFC, test_taskFC, testRestFC):
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
    Y=np.concatenate((t,r))
    #Y_perm=np.random.permutation(Y)
    #For the purpose of this script split them back into a pseudo rest and task array
    #t, r =np.array_split(Y_perm, 2)
    Test_taskSize=test_taskFC.shape[0]
    Test_restSize=testRestFC.shape[0]
    Tt = np.ones(Test_taskSize, dtype = int)
    Tr=np.zeros(Test_restSize, dtype=int)
    CVTacc=[]
    DSTacc=[]
    #CVRacc=[]
    #DSRacc=[]
    session=splitDict[train_sub]
    split=np.empty((session, 55278))
    #fold each training set
    for train_index, test_index in loo.split(split):
        Xtrain_rest,Xval_rest=restFC[train_index],restFC[test_index]
        Xtrain_task,Xval_task=taskFC[train_index],taskFC[test_index]
        ytrain_rest=r[train_index]
        ytrain_task=t[train_index]
        yval_task=np.array([1])
        X_tr=np.concatenate((Xtrain_task, Xtrain_rest))
        y_tr = np.concatenate((ytrain_task,ytrain_rest))
        Xtest_task, Xtest_rest=test_taskFC[test_index], testRestFC[test_index]
        yTest_Task=Tt[test_index]
        yTest_Rest=Tr[test_index]
        clf.fit(X_tr,y_tr)
        sameT=clf.score(Xval_task,yval_task)
        diffT=clf.score(Xtest_task,yTest_Task)
        #sameR=clf.score(Xval_rest,yval_rest)
        #diffR=clf.score(Xtest_rest,yTest_Rest)
        CVTacc.append(sameT)
        DSTacc.append(diffT)
        #CVRacc.append(sameR)
        #DSRacc.append(diffR)
    same_Tsub=mean(CVTacc)
    diff_Tsub=mean(DSTacc)
    diff=same_Tsub-diff_Tsub
    #same_Rsub=mean(CVRacc)
    #diff_Rsub=mean(DSRacc)
    return diff
