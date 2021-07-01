
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import RidgeClassifier
import numpy as np
import pandas as pd
import itertools
import scipy.io
from statistics import mean
subList=['MSC01','MSC02','MSC03','MSC04','MSC05','MSC06','MSC07','MSC10']
#all possible combinations of subs and tasks
subsComb=(list(itertools.permutations(subList, 2)))
#permutation analysis using IndNet
outDir='/Users/Alexis/Desktop/MSC_Alexis/analysis/output/results/permutation/ALL/IndNet'
projDir='/Users/Alexis/Desktop/MSC_Alexis/analysis/data/IndNet'
#group based network
#dataDir='/projects/b1081/MSC/TaskFC/FC_Parcels_IndNet/'
#task/allsubs_mem_corrmats_bysess_orig_INDformat.mat

#individual specific Network
#indDir='/projects/b1081/MSC/TaskFC/FC_Parcels_IndNet/'
def netFile(netSpec,sub):
    #rest will be handled differently because splitting into 4 parts in the timeseries to match
    #zero based indexing
    subDict=dict([('MSC01',0),('MSC02',1),('MSC03',2),('MSC04',3),('MSC05',4),('MSC06',5),('MSC07',6),('MSC10',9)])
    taskDict=dict([('mem','AllMem'),('mixed','AllGlass'),('motor','AllMotor')])
    #fullTask=np.empty((40,120))
    fullRest=np.empty((40,120))
    #memory
    tmp=projDir+'/mem/allsubs_mem_corrmats_bysess_orig_INDformat.mat'
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
    tmp=projDir+'/motor/allsubs_motor_corrmats_bysess_orig_INDformat.mat'
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
    tmp=projDir+'/mixed/allsubs_mixed_corrmats_bysess_orig_INDformat.mat'
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
    tmp=projDir+'/mixed/allsubs_mixed_corrmats_bysess_orig_INDformat.mat'
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
    tmpRest=projDir+'/rest/'+sub+'_parcel_corrmat.mat'
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
    df=pd.DataFrame()
    #train sub
    taskFC, restFC=netFile('IndNet',train_sub)
    #test sub
    test_taskFC, test_restFC=netFile('IndNet',test_sub)
    CV, DS=K_folds(clf, taskFC, restFC, test_taskFC, test_restFC)
    return CV, DS

def K_folds(clf, taskFC, restFC, test_taskFC, test_restFC):
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
    #diff=CV-DS
    return CV, DS

def classifyIndNet():
    """
    Classifying different subjects along network level data generated from group atlas rest split into 40 samples to match with task

    Parameters
    -------------

    Returns
    -------------
    df : DataFrame
        Dataframe consisting of average accuracy across all subjects

    """
    acc_scores_ds=[]
    acc_scores_cv=[]
    SSmOS=[]
    df=pd.DataFrame(subsComb, columns=['train_sub','test_sub'])
    for index, row in df.iterrows():
        CV,DS=modelAll(train_sub=row['train_sub'], test_sub=row['test_sub'])
        acc_scores_ds.append(DS)
        acc_scores_cv.append(CV)
        #SSmOS.append(diff)
    df['cv']=acc_scores_cv
    df['ds']=acc_scores_ds
    df['diff']=df.cv-df.ds
    SS=mean(acc_scores_cv)
    OS=mean(acc_scores_ds)
    diff=df['diff'].mean()
    return SS, OS, diff
    #df.to_csv(outDir+'IndNet.csv',index=False)

def permutation():
    diffScore=[]
    SSscore=[]
    OSscore=[]
    for i in range(1000):
        SS,OS,diff=classifyIndNet()
        diffScore.append(diff)
        SSscore.append(SS)
        OSscore.append(OS)
        print(str(i))
    ALL_perms=pd.DataFrame()
    ALL_perms['diff_acc']=diffScore
    ALL_perms['SS']=SSscore
    ALL_perms['OS']=OSscore
    ALL_perms.to_csv(outDir+'/permutation.csv',index=False)
