from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import RidgeClassifier
import numpy as np
import os
import sys
import pandas as pd
import itertools
from statistics import mean
import scipy.io
import warnings
warnings.filterwarnings("ignore")

thisDir = os.path.expanduser('main/project/path/')
dataDir = thisDir + 'data/file/location'
outDir = thisDir + 'path/for/output/results'

# input task and subject

taskFC=matFiles(dataDir+TASK+'/'+SUBJECT+'_parcel_corrmat.mat')
restFC=matFiles(dataDir+'rest/'+SUBJECT+'_parcel_corrmat.mat')
clf=RidgeClassifier()
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
  SSscores=clf.score(X_val,y_val)
  SS_acc.append(SSscores)
SStotal_acc=mean(SS_acc)
print("Subject scored " + SStotal + "%")


  def matFiles(df='path'):
      """
      Convert matlab files into upper triangle np.arrays
      Parameters
      -----------
      df : str
          Path to file
      Returns
      -----------
      ds : 2D upper triangle FC measures in (roi, days) format

      """
      #Consistent parameters to use for editing datasets
      nrois=300
      #Load FC file
      fileFC=scipy.io.loadmat(df)

      #Convert to numpy array
      fileFC=np.array(fileFC['parcel_corrmat'])
      #Replace nans and infs with zero
      fileFC=np.nan_to_num(fileFC)
      nsess=fileFC.shape[2]
      #Index upper triangle of matrix
      mask=np.triu_indices(nrois,1)
      ds=np.empty((nsess, int(nrois*(nrois-1)/2)))
      count=0
      #Loop through all 10 days to reshape correlations into linear form
      for sess in range(nsess):
          tmp=fileFC[:,:,sess]
          ds[count]=tmp[mask]
          count=count+1
      mask = (ds == 0).all(1)
      column_indices = np.where(mask)[0]
      df = ds[~mask,:]
      return df
