from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeClassifier
import numpy as np
import pandas as pd
import itertools
import scipy.io
import os
import sys
#import other python scripts for further anlaysis
import reshape
#import results
import warnings
warnings.filterwarnings("ignore")
# Initialization of directory information:
thisDir = os.path.expanduser('~/Desktop/MSC_Alexis/analysis/')
dataDir = thisDir + 'data/mvpa_data/'
#outDir = thisDir + 'output/results/'
outDir = thisDir + 'output/results/'
# Subjects and tasks
taskList=['glass','semantic', 'motor','mem']
subList=['MSC01','MSC02','MSC03','MSC04','MSC05','MSC06','MSC07','MSC10']
#all possible combinations of subs and tasks
subsComb=(list(itertools.permutations(subList, 2)))
tasksComb=(list(itertools.permutations(taskList, 2)))


def classifyBTW():
    acc_scores_per_sub=[]
    sen_scores_per_sub=[]
    spec_scores_per_sub=[]
    acc_scores_cv=[]
    sen_scores_cv=[]
    spec_scores_cv=[]
    df=pd.DataFrame(subsComb, columns=['train_sub','test_sub'])
    for index, row in df.iterrows():
        diff_score, same_score, CV_sens_score, CV_spec_score, DS_sens_score, DS_spec_score=btwNet(train_sub=row['train_sub'], test_sub=row['test_sub'])
        acc_scores_per_sub.append(diff_score)
        acc_scores_cv.append(same_score)
        sen_scores_cv.append(CV_sens_score)
        spec_scores_cv.append(CV_spec_score)
        sen_scores_per_sub.append(DS_sens_score)
        spec_scores_per_sub.append(DS_spec_score)
    df['cv_acc']=acc_scores_cv
    df['cv_sen']=sen_scores_cv
    df['cv_spec']=spec_scores_cv
    df['acc']=acc_scores_per_sub
    df['ds_sen']=sen_scores_per_sub
    df['ds_spec']=spec_scores_per_sub
    df.to_csv(outDir+'wtn_btw_netSelection/btw_acc.csv',index=False)
def classifyWTN():
    """
    Classifying different subjects along available data rest split into 40 samples to match with task using only within networks

    Parameters
    -------------

    Returns
    -------------
    df : DataFrame
        Dataframe consisting of average accuracy across all subjects

    """
    acc_scores_per_sub=[]
    sen_scores_per_sub=[]
    spec_scores_per_sub=[]
    acc_scores_cv=[]
    sen_scores_cv=[]
    spec_scores_cv=[]
    df=pd.DataFrame(subsComb, columns=['train_sub','test_sub'])
    for index, row in df.iterrows():
        diff_score, same_score, CV_sens_score, CV_spec_score, DS_sens_score, DS_spec_score=wtnNet(train_sub=row['train_sub'], test_sub=row['test_sub'])
        acc_scores_per_sub.append(diff_score)
        acc_scores_cv.append(same_score)
        sen_scores_cv.append(CV_sens_score)
        spec_scores_cv.append(CV_spec_score)
        sen_scores_per_sub.append(DS_sens_score)
        spec_scores_per_sub.append(DS_spec_score)
    df['cv_acc']=acc_scores_cv
    df['cv_sen']=sen_scores_cv
    df['cv_spec']=spec_scores_cv
    df['acc']=acc_scores_per_sub
    df['ds_sen']=sen_scores_per_sub
    df['ds_spec']=spec_scores_per_sub
    df.to_csv(outDir+'wtn_btw_netSelection/wtn_acc.csv',index=False)
def btwNet(train_sub, test_sub):
    clf=RidgeClassifier()
    df=pd.DataFrame()
    #train sub
    memFC=reshape.btwBlock(dataDir+'mem/'+train_sub+'_parcel_corrmat.mat')
    semFC=reshape.btwBlock(dataDir+'semantic/'+train_sub+'_parcel_corrmat.mat')
    glassFC=reshape.btwBlock(dataDir+'glass/'+train_sub+'_parcel_corrmat.mat')
    motFC=reshape.btwBlock(dataDir+'motor/'+train_sub+'_parcel_corrmat.mat')
    restFC=reshape.btwBlock(dataDir+'rest/corrmats_timesplit/fourths/'+train_sub+'_parcel_corrmat.mat')
    restFC=np.reshape(restFC,(10,4,49740)) #taskFC=np.concatenate((memFC,semFC,glassFC,motFC))
    nSize=49740
    #test sub
    test_memFC=reshape.btwBlock(dataDir+'mem/'+test_sub+'_parcel_corrmat.mat')
    test_semFC=reshape.btwBlock(dataDir+'semantic/'+test_sub+'_parcel_corrmat.mat')
    test_glassFC=reshape.btwBlock(dataDir+'glass/'+test_sub+'_parcel_corrmat.mat')
    test_motFC=reshape.btwBlock(dataDir+'motor/'+test_sub+'_parcel_corrmat.mat')
    test_restFC=reshape.btwBlock(dataDir+'rest/corrmats_timesplit/fourths/'+test_sub+'_parcel_corrmat.mat')
    test_taskFC=np.concatenate((test_memFC,test_semFC,test_glassFC,test_motFC))
    diff_score, same_score,CV_sens_score, CV_spec_score, DS_sens_score, DS_spec_score=K_folds(nSize,train_sub, clf, memFC,semFC,glassFC,motFC, restFC, test_taskFC, test_restFC)
    return diff_score, same_score, CV_sens_score, CV_spec_score, DS_sens_score, DS_spec_score
    #return taskFC,restFC,test_taskFC,test_restFC

def wtnNet(train_sub, test_sub):
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
    df=pd.DataFrame()
    #train sub
    memFC=reshape.subBlock(dataDir+'mem/'+train_sub+'_parcel_corrmat.mat')
    semFC=reshape.subBlock(dataDir+'semantic/'+train_sub+'_parcel_corrmat.mat')
    glassFC=reshape.subBlock(dataDir+'glass/'+train_sub+'_parcel_corrmat.mat')
    motFC=reshape.subBlock(dataDir+'motor/'+train_sub+'_parcel_corrmat.mat')
    restFC=reshape.subBlock(dataDir+'rest/corrmats_timesplit/fourths/'+train_sub+'_parcel_corrmat.mat')
    restFC=np.reshape(restFC,(10,4,4410))
    nSize=4410
    #taskFC=np.concatenate((memFC,semFC,glassFC,motFC))
    #test sub
    test_memFC=reshape.subBlock(dataDir+'mem/'+test_sub+'_parcel_corrmat.mat')
    test_semFC=reshape.subBlock(dataDir+'semantic/'+test_sub+'_parcel_corrmat.mat')
    test_glassFC=reshape.subBlock(dataDir+'glass/'+test_sub+'_parcel_corrmat.mat')
    test_motFC=reshape.subBlock(dataDir+'motor/'+test_sub+'_parcel_corrmat.mat')
    test_restFC=reshape.subBlock(dataDir+'rest/corrmats_timesplit/fourths/'+test_sub+'_parcel_corrmat.mat')
    test_taskFC=np.concatenate((test_memFC,test_semFC,test_glassFC,test_motFC))
    diff_score, same_score,CV_sens_score, CV_spec_score, DS_sens_score, DS_spec_score=K_folds(nSize,train_sub, clf, memFC,semFC,glassFC,motFC, restFC, test_taskFC, test_restFC)
    return diff_score, same_score, CV_sens_score, CV_spec_score, DS_sens_score, DS_spec_score

def K_folds(nSize,train_sub, clf, memFC,semFC,glassFC,motFC, restFC, test_taskFC, test_restFC):
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
    """
    taskSize=taskFC.shape[0]
    restSize=restFC.shape[0]
    t = np.ones(taskSize, dtype = int)
    r=np.zeros(restSize, dtype=int)
    """
    test_taskSize=test_taskFC.shape[0]
    test_restSize=test_restFC.shape[0]
    testT= np.ones(test_taskSize, dtype = int)
    testR= np.zeros(test_restSize, dtype = int)
    CVacc=[]
    CVspec=[]
    CVsen=[]
    df=pd.DataFrame()
    acc_score=[]
    DSspec=[]
    DSsen=[]
    #fold each training set
    if train_sub=='MSC03':
        split=np.empty((8,nSize))
        #xtrainSize=24
        #xtestSize=4
    elif train_sub=='MSC06' or train_sub=='MSC07':
        split=np.empty((9,nSize))
    else:
        split=np.empty((10,nSize))
    for train_index, test_index in kf.split(split):
        memtrain, memval=memFC[train_index], memFC[test_index]
        semtrain, semval=semFC[train_index], semFC[test_index]
        mottrain, motval=motFC[train_index], motFC[test_index]
        glatrain, glaval=glassFC[train_index], glassFC[test_index]
        Xtrain_task=np.concatenate((memtrain,semtrain,mottrain,glatrain))
        Xtrain_rest, Xval_rest=restFC[train_index,:,:], restFC[test_index,:,:]
        Xval_task=np.concatenate((memval,semval,motval,glaval))
        Xtrain_rest=np.reshape(Xtrain_rest,(-1,nSize))
        Xval_rest=np.reshape(Xval_rest,(-1,nSize))
        ytrain_task = np.ones(Xtrain_task.shape[0], dtype = int)
        ytrain_rest=np.zeros(Xtrain_rest.shape[0], dtype=int)
        yval_task = np.ones(Xval_task.shape[0], dtype = int)
        yval_rest=np.zeros(Xval_rest.shape[0], dtype=int)
        X_tr=np.concatenate((Xtrain_task, Xtrain_rest))
        X_val=np.concatenate((Xval_task, Xval_rest))
        y_tr = np.concatenate((ytrain_task,ytrain_rest))
        y_val=np.concatenate((yval_task, yval_rest))
        clf.fit(X_tr,y_tr)
        #cross validation
        y_pred=clf.predict(X_val)
        #Test labels and predicted labels to calculate sensitivity specificity
        tn, fp, fn, tp=confusion_matrix(y_val, y_pred).ravel()
        CV_specificity= tn/(tn+fp)
        CV_sensitivity= tp/(tp+fn)
        #get accuracy
        CV_score=clf.score(X_val, y_val)
        CVacc.append(CV_score)
        CVspec.append(CV_specificity)
        CVsen.append(CV_sensitivity)
        tmpdf=pd.DataFrame()
        acc_scores_per_fold=[]
        sen_scores_per_fold=[]
        spec_scores_per_fold=[]
        #fold each testing set
        for t_index, te_index in kf.split(test_taskFC):
            Xtest_rest=test_restFC[te_index]
            Xtest_task=test_taskFC[te_index]
            X_te=np.concatenate((Xtest_task, Xtest_rest))
            ytest_task=testT[te_index]
            ytest_rest=testR[te_index]
            y_te=np.concatenate((ytest_task, ytest_rest))
            #test set
            y_pred_testset=clf.predict(X_te)
            #Test labels and predicted labels to calculate sensitivity specificity
            DStn, DSfp, DSfn, DStp=confusion_matrix(y_te, y_pred_testset).ravel()
            DS_specificity= DStn/(DStn+DSfp)
            DS_sensitivity= DStp/(DStp+DSfn)
            #Get accuracy of model
            ACCscores=clf.score(X_te,y_te)
            acc_scores_per_fold.append(ACCscores)
            sen_scores_per_fold.append(DS_sensitivity)
            spec_scores_per_fold.append(DS_specificity)
        tmpdf['inner_fold']=acc_scores_per_fold
        tmpdf['DS_sen']=sen_scores_per_fold
        tmpdf['DS_spec']=spec_scores_per_fold
        score=tmpdf['inner_fold'].mean()
        sen=tmpdf['DS_sen'].mean()
        spec=tmpdf['DS_spec'].mean()
        acc_score.append(score)
        DSspec.append(spec)
        DSsen.append(sen)
    df['cv']=CVacc
    df['CV_sen']=CVsen
    df['CV_spec']=CVspec
    #Different sub outer acc
    df['outer_fold']=acc_score
    df['DS_sen']=DSsen
    df['DS_spec']=DSspec
    same_sub_score=df['cv'].mean()
    diff_sub_score=df['outer_fold'].mean()
    CV_sens_score=df['CV_sen'].mean()
    CV_spec_score=df['CV_spec'].mean()
    DS_sens_score=df['DS_sen'].mean()
    DS_spec_score=df['DS_spec'].mean()
    return diff_sub_score, same_sub_score, CV_sens_score, CV_spec_score, DS_sens_score, DS_spec_score
