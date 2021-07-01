
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

outDir='/Users/Alexis/Desktop/MSC_Alexis/analysis/output/results/acc/'
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
    #train sub
    taskFC, restFC=netFile('IndNet',train_sub)
    #test sub
    test_taskFC, test_restFC=netFile('IndNet',test_sub)
    CV, DS, SS_TPV, SS_RPV, OS_TPV, OS_RPV=K_folds(clf, taskFC, restFC, test_taskFC, test_restFC)
    return CV, DS, SS_TPV, SS_RPV, OS_TPV, OS_RPV

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
    CVTPV=[]
    CVRPV=[]
    #CVspec=[]
    #CVsen=[]
    #df=pd.DataFrame()
    DSacc=[]
    DSTPV=[]
    DSRPV=[]
    #DSspec=[]
    #DSsen=[]
    #fold each training set
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
        #cross validation
        y_pred=clf.predict(X_val)

        #Test labels and predicted labels to calculate sensitivity specificity
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
        #CVspec.append(CV_specificity)
        #CVsen.append(CV_sensitivity)
        #tmpdf=pd.DataFrame()
        #acc_scores_per_fold=[]
        #sen_scores_per_fold=[]
        #spec_scores_per_fold=[]
        #fold each testing set
        X_te=np.concatenate((test_taskFC, test_restFC))
        y_te=np.concatenate((testT, testR))
        y_pred_testset=clf.predict(X_te)
        #Test labels and predicted labels to calculate sensitivity specificity
        DStn, DSfp, DSfn, DStp=confusion_matrix(y_te, y_pred_testset).ravel()
        DS_TPV=DStp/(DStp+DSfp)
        DS_RPV=DStn/(DStn+DSfn)
        DSTPV.append(DS_TPV)
        DSRPV.append(DS_RPV)
        #DS_specificity= DStn/(DStn+DSfp)
        #DS_sensitivity= DStp/(DStp+DSfn)
        #Get accuracy of model
        ACCscores=clf.score(X_te,y_te)
        DSacc.append(ACCscores)


    #df['cv']=CVacc
    CV=mean(CVacc)
    DS=mean(DSacc)
    SS_TPV=mean(CVTPV)
    SS_RPV=mean(CVRPV)
    OS_TPV=mean(DSTPV)
    OS_RPV=mean(DSRPV)
    #df['CV_sen']=CVsen
    #df['CV_spec']=CVspec
    #Different sub outer acc
    #df['outer_fold']=acc_score
    #df['DS_sen']=DSsen
    #df['DS_spec']=DSspec
    #same_sub_score=df['cv'].mean()
    #diff_sub_score=df['outer_fold'].mean()
    #CV_sens_score=df['CV_sen'].mean()
    #CV_spec_score=df['CV_spec'].mean()
    #DS_sens_score=df['DS_sen'].mean()
    #DS_spec_score=df['DS_spec'].mean()
    return CV, DS, SS_TPV, SS_RPV, OS_TPV, OS_RPV
    #return diff_sub_score, same_sub_score, CV_sens_score, CV_spec_score, DS_sens_score, DS_spec_score



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
    tpv_ds=[]
    rpv_ds=[]
    #sen_scores_per_sub=[]
    #spec_scores_per_sub=[]
    acc_scores_cv=[]
    tpv_cv=[]
    rpv_cv=[]
    #sen_scores_cv=[]
    #spec_scores_cv=[]
    df=pd.DataFrame(subsComb, columns=['train_sub','test_sub'])
    for index, row in df.iterrows():
        CV, DS, SS_TPV, SS_RPV, OS_TPV, OS_RPV=modelAll(train_sub=row['train_sub'], test_sub=row['test_sub'])
        acc_scores_ds.append(DS)
        acc_scores_cv.append(CV)
        tpv_ds.append(OS_TPV)
        rpv_ds.append(OS_RPV)
        tpv_cv.append(SS_TPV)
        rpv_cv.append(SS_RPV)
        #sen_scores_cv.append(CV_sens_score)
        #spec_scores_cv.append(CV_spec_score)
        #sen_scores_per_sub.append(DS_sens_score)
        #spec_scores_per_sub.append(DS_spec_score)
    df['cv']=acc_scores_cv
    df['cv_tpv']=tpv_cv
    df['cv_rpv']=rpv_cv
    #df['cv_sen']=sen_scores_cv
    #df['cv_spec']=spec_scores_cv
    df['ds']=acc_scores_ds
    df['ds_tpv']=tpv_ds
    df['ds_rpv']=rpv_ds
    #df['ds_sen']=sen_scores_per_sub
    #df['ds_spec']=spec_scores_per_sub
    df.to_csv(outDir+'ALL/IndNet/acc.csv',index=False)
