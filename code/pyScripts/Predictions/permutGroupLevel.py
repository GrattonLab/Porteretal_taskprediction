import sys
import os
from scipy import stats
import pandas as pd
import numpy as np
import seaborn as sns
from statistics import mean
import warnings
warnings.filterwarnings("ignore")
thisDir = os.path.expanduser('~/Desktop/Porteretal_taskprediction/')
dataDir = thisDir +  'data/corrmats/'

outDir = thisDir + 'output/results/Ridge/'
taskList=['mem','motor','semantic','glass']



def shuffle(df, n=1, axis=0):
    df['new']=np.random.permutation(df['acc'].values)
    null=df[df['Analysis']=='SS'].new.values-df[df['Analysis']=='OS'].new.values
    value=mean(null)
    return value
    
def perm_btw():
    ALL=pd.read_csv(outDir+'ALL_Binary/acc.csv',usecols=['same_sub','diff_sub'])
    ALL.rename(columns={'same_sub':'SS','diff_sub':'OS'},inplace=True)
    ALL=pd.melt(ALL,value_vars=['SS','OS'],var_name='Analysis',value_name='acc')
    Diff=ALL[ALL['Analysis']=='SS'].acc.values-ALL[ALL['Analysis']=='OS'].acc.values
    TrueDiff=mean(Diff)
    distribution=[]
    for i in range(1000):
        null=shuffle(ALL)
        distribution.append(null)
    null=pd.DataFrame(distribution,columns=['distribution'])

    keep=null[null['distribution']>TrueDiff]
    count=keep['distribution'].count()
    p=(count+1)/1001
    print('p value for all vs rest (binary) is '+str(p))
    DS=pd.read_csv(outDir+'single_task/acc.csv',usecols=['task','diff_sub','same_sub'])
    DS.rename(columns={'same_sub':'SS','diff_sub':'OS'},inplace=True)
    single_task=pd.melt(DS,value_vars=['SS','OS'],var_name='Analysis',value_name='acc',id_vars='task')
    for task in taskList:
        tmp=single_task[single_task['task']==task]
        Diff=tmp[tmp['Analysis']=='SS'].acc.values-tmp[tmp['Analysis']=='OS'].acc.values
        TrueDiff=mean(Diff)
        distribution=[]
        for i in range(1000):
            null=shuffle(tmp)
            distribution.append(null)
        null=pd.DataFrame(distribution,columns=['distribution'])
        keep=null[null['distribution']>TrueDiff]
        count=keep['distribution'].count()
        p=(count+1)/1001
        print('p value for '+task+ ' task is ' +str(p))


    MC=pd.read_csv(outDir+'ALL_MC/acc.csv',usecols=['train','acc','Analysis'])
    MC.drop_duplicates(inplace=True)
    MC=MC.replace('Same Person','SS')
    MC=MC.replace('Different Person','OS')
    Diff=MC[MC['Analysis']=='SS'].acc.values-MC[MC['Analysis']=='OS'].acc.values
    TrueDiff=mean(Diff)
    distribution=[]
    for i in range(1000):
        null=shuffle(MC)
        distribution.append(null)
    null=pd.DataFrame(distribution,columns=['distribution'])

    keep=null[null['distribution']>TrueDiff]
    count=keep['distribution'].count()
    p=(count+1)/1001
    print('p value for multiclass is ' +str(p))

    IndNet=pd.read_csv(outDir+'ALL_IndNet/acc.csv',usecols=['cv','ds'])
    IndNet.rename(columns={'cv':'SS','ds':'OS'},inplace=True)
    Same=pd.melt(IndNet,value_vars=['SS','OS'],var_name='Analysis',value_name='acc')
    Diff=Same[Same['Analysis']=='SS'].acc.values-Same[Same['Analysis']=='OS'].acc.values
    TrueDiff=mean(Diff)
    distribution=[]
    for i in range(1000):
        null=shuffle(Same)
        distribution.append(null)
    null=pd.DataFrame(distribution,columns=['distribution'])

    keep=null[null['distribution']>TrueDiff]
    count=keep['distribution'].count()
    p=(count+1)/1001
    print('p value at network level is ' +str(p))
