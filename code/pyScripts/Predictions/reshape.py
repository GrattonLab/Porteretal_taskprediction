#!/usr/bin/env python
# coding: utf-8

# In[ ]:
#import matlab.engine
import scipy.io
import pandas as pd
import numpy as np
import sys
import os
import glob
import matplotlib.pyplot as plt
thisDir = os.path.expanduser('~/Desktop/Porteretal_taskprediction/')
splitDict=dict([('MSC01',10),('MSC02',10),('MSC03',8),('MSC04',10),('MSC05',10),('MSC06',9),('MSC07',9),('MSC10',10)])
dataDir = thisDir + 'data/corrmats/'
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
    nrois=333
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
def concateFC(taskFC, restFC):
    """
    Concatenates task and rest FC arrays and creates labels
    Parameters
    -----------
    taskFC, restFC : array_like
        Numpy arrays of FC upper triangle for rest and task
    Returns
    -----------
    x, y : array_like
        Arrays containing task and restFC concatenated together and labels for each
    """
    x=np.concatenate((taskFC, restFC))
    taskSize=taskFC.shape[0]
    restSize=restFC.shape[0]
    t = np.ones(taskSize, dtype = int)
    r=np.zeros(restSize, dtype=int)
    y = np.concatenate((t,r))
    return x, y
def network_to_network(df='path', networkA='networkA',networkB='networkB'):
    """
    A more efficient script for getting network to network connections
    str options for networks ['unassign',
    'default',
    'visual',
    'fp',
    'dan',
    'van',
    'salience',
    'co',
    'sm',
    'sm-lat',
    'auditory',
    'pmn',
    'pon']
    Parameters
    -----------
    df : str
        Path to file
    Returns
    ------------
    dsNet : Array of task or rest FC with only blocks
    """
 #roi count for building arrays
    networks=['unassign','default','visual','fp','dan','van','salience','co','sm','sm-lat','auditory','pmn','pon']
    fileFC=scipy.io.loadmat(df)
    fileFC=np.array(fileFC['parcel_corrmat'])
    fileFC=np.nan_to_num(fileFC)
    #nsess=checkSession(df)
    nsess=fileFC.shape[2]
    netSize=determineNetSize(networkA,networkB)
    dsNet=np.empty((nsess, netSize))
    dsNet_count=0
    for sess in range(nsess):
        ds=fileFC[:,:,sess]
        Parcel_params = loadParcelParams('Gordon333')
        roi_sort = np.squeeze(Parcel_params['roi_sort'])
        corrmat=ds[roi_sort,:][:,roi_sort]
        nrois=list(range(333))
        nets=[]
        position=0
        count=0
        networks=Parcel_params['networks']
        t=Parcel_params['transitions']
    #have to add extra value otherwise error
        transitions=np.append(t,333)
        while count<333:
            if count<=transitions[position]:
                nets.append(networks[position])
                count=count+1
            else:
                position=position+1
        #transform data to locate network
        df=pd.DataFrame(corrmat, index=[nets, nrois], columns=[nets, nrois])
        df_ut = df.where(np.triu(np.ones(df.shape),1).astype(np.bool))
        #initlize empty
        btwBlocks=np.array([])
        tmp=df_ut.loc[networkA,networkB]
        tmp=tmp.values
        clean_array=tmp[~np.isnan(tmp)]
        dsNet[dsNet_count]=clean_array
        dsNet_count=dsNet_count+1
    mask = (dsNet == 0).all(1)
    column_indices = np.where(mask)[0]
    df = dsNet[~mask,:]
    return df


def determineNetSize(networkA,networkB):
    df=dataDir+'mem/MSC01_parcel_corrmat.mat' #use as temp for knowing size
    fileFC=scipy.io.loadmat(df)
    fileFC=np.array(fileFC['parcel_corrmat'])
    fileFC=np.nan_to_num(fileFC)
    ds=fileFC[:,:,0]
    Parcel_params =loadParcelParams('Gordon333')
    roi_sort = np.squeeze(Parcel_params['roi_sort'])
    corrmat=ds[roi_sort,:][:,roi_sort]
    nrois=list(range(333))
    nets=[]
    position=0
    count=0
    networks=Parcel_params['networks']
    t=Parcel_params['transitions']
#have to add extra value otherwise error
    transitions=np.append(t,333)
    while count<333:
        if count<=transitions[position]:
            nets.append(networks[position])
            count=count+1
        else:
            position=position+1
    #transform data to locate network
    df=pd.DataFrame(corrmat, index=[nets, nrois], columns=[nets, nrois])
    #avoid duplicates by taking upper triangle k=1 so we don't take the first value
    df_ut = df.where(np.triu(np.ones(df.shape),1).astype(np.bool))
    tmp=df_ut.loc[networkA,networkB]
    tmp=tmp.values
    clean_array=tmp[~np.isnan(tmp)]
    array_size=clean_array.shape[0]
    return array_size
def subNets(df='path', networkLabel='networklabel',otherNets=None):
    """
    Same as reshape but subset by network
    str options for networks ['unassign',
    'default',
    'visual',
    'fp',
    'dan',
    'van',
    'salience',
    'co',
    'sm',
    'sm-lat',
    'auditory',
    'pmn',
    'pon']
    Parameters
    -----------
    df : str
        Path to file
    networkLabel : str
        String to indicate which network to subset
    otherNets : str; optional
        If looking at specific network to network connection include other network
    Returns
    ----------
    dsNet : Array of task or rest FC containing only subnetworks
    """
 #roi count for building arrays
    netRoi=dict([('unassign',14808),('default', 10824),('visual',8736),('fp', 4620),('dan',5264),('van',3151),('salience', 494),('co', 4060),('sm', 2375),('sm-lat', 316),('auditory', 564),('pmn',45),('pon',21)])
    fileFC=scipy.io.loadmat(df)
    fileFC=np.array(fileFC['parcel_corrmat'])
    fileFC=np.nan_to_num(fileFC)
    nsess=fileFC.shape[2]
    if otherNets is None:
        dsNet=np.empty((nsess, netRoi[networkLabel]))
    else:
        interBlock=dict([('co',780),('fp',960),('default',1640)])
        #netLength=netRoi[networkLabel]+netRoi[otherNets]
        #netLength=netBlock
        dsNet=np.empty((nsess, interBlock[networkLabel]))
    dsNet_count=0
    for sess in range(nsess):
        ds=fileFC[:,:,sess]
        Parcel_params = loadParcelParams('Gordon333')
        roi_sort = np.squeeze(Parcel_params['roi_sort'])
        corrmat=ds[roi_sort,:][:,roi_sort]
        nrois=list(range(333))
        nets=[]
        position=0
        count=0
        networks=Parcel_params['networks']
        t=Parcel_params['transitions']
    #have to add extra value otherwise error
        transitions=np.append(t,333)
        while count<333:
            if count<=transitions[position]:
                nets.append(networks[position])
                count=count+1
            else:
                position=position+1
        #transform data to locate network
        df=pd.DataFrame(corrmat, index=[nets, nrois], columns=[nets, nrois])
        #avoid duplicates by taking upper triangle k=1 so we don't take the first value
        df_ut = df.where(np.triu(np.ones(df.shape),1).astype(np.bool))
        if otherNets is None:
            df_new=df_ut.loc[[networkLabel]]
        else:
            df_new=df_ut.loc[networkLabel]
            df_new=df_new[otherNets]
        #convert to array
        array=df_new.values
        #remove nans
        clean_array = array[~np.isnan(array)]
        dsNet[dsNet_count]=clean_array
        dsNet_count=dsNet_count+1
    mask = (dsNet == 0).all(1)
    #column_indices = np.where(mask)[0]
    df = dsNet[~mask,:]
    return dsNet
#btwn network selection
def btwBlock(df='path'):
    """
    Same as subNets but subset between networks
    str options for networks ['unassign',
    'default',
    'visual',
    'fp',
    'dan',
    'van',
    'salience',
    'co',
    'sm',
    'sm-lat',
    'auditory',
    'pmn',
    'pon']
    Parameters
    -----------
    df : str
        Path to file
    Returns
    ------------
    dsNet : Array of task or rest FC with only blocks
    """
 #roi count for building arrays
    networks=['unassign','default','visual','fp','dan','van','salience','co','sm','sm-lat','auditory','pmn','pon']
    fileFC=scipy.io.loadmat(df)
    fileFC=np.array(fileFC['parcel_corrmat'])
    fileFC=np.nan_to_num(fileFC)
    nsess=fileFC.shape[2]
    dsNet=np.empty((nsess, 49740))
    dsNet_count=0
    for sess in range(nsess):
        ds=fileFC[:,:,sess]
        Parcel_params = loadParcelParams('Gordon333')
        roi_sort = np.squeeze(Parcel_params['roi_sort'])
        corrmat=ds[roi_sort,:][:,roi_sort]
        nrois=list(range(333))
        nets=[]
        position=0
        count=0
        networks=Parcel_params['networks']
        t=Parcel_params['transitions']
    #have to add extra value otherwise error
        transitions=np.append(t,333)
        while count<333:
            if count<=transitions[position]:
                nets.append(networks[position])
                count=count+1
            else:
                position=position+1
        #transform data to locate network
        df=pd.DataFrame(corrmat, index=[nets, nrois], columns=[nets, nrois])
        #avoid duplicates by taking upper triangle k=1 so we don't take the first value
        df_ut = df.where(np.triu(np.ones(df.shape),1).astype(np.bool))
        #initlize empty
        for n in networks:
            df_ut.loc[n,n]=np.nan
        tmp=df_ut.values
        clean_array=tmp[~np.isnan(tmp)]
        dsNet[dsNet_count]=clean_array
        dsNet_count=dsNet_count+1
    return dsNet
    #return df_ut

#looking at the blocks of networks ex: default to default connections. This scripts grabs all network blocks
def subBlock(df='path'):
    """
    Same as subNets but subset by block level
    str options for networks ['unassign',
    'default',
    'visual',
    'fp',
    'dan',
    'van',
    'salience',
    'co',
    'sm',
    'sm-lat',
    'auditory',
    'pmn',
    'pon']
    Parameters
    -----------
    df : str
        Path to file
    Returns
    ------------
    dsNet : Array of task or rest FC with only blocks
    """
 #roi count for building arrays
    netRoi=dict([('default', 10824),('visual',8736),('fp', 4620),('dan',5264),('van',3151),('salience', 484),('co', 4060),('sm', 2375),('sm-lat', 316),('auditory', 564),('pmn',45),('pon',21)])
    fileFC=scipy.io.loadmat(df)
    fileFC=np.array(fileFC['parcel_corrmat'])
    fileFC=np.nan_to_num(fileFC)
    nsess=fileFC.shape[2]
    dsNet=np.empty((nsess, 4410))
    dsNet_count=0
    for sess in range(nsess):
        ds=fileFC[:,:,sess]
        Parcel_params = loadParcelParams('Gordon333')
        roi_sort = np.squeeze(Parcel_params['roi_sort'])
        corrmat=ds[roi_sort,:][:,roi_sort]
        nrois=list(range(333))
        nets=[]
        position=0
        count=0
        networks=Parcel_params['networks']
        t=Parcel_params['transitions']
    #have to add extra value otherwise error
        transitions=np.append(t,333)
        while count<333:
            if count<=transitions[position]:
                nets.append(networks[position])
                count=count+1
            else:
                position=position+1
        #transform data to locate network
        df=pd.DataFrame(corrmat, index=[nets, nrois], columns=[nets, nrois])
        #avoid duplicates by taking upper triangle k=1 so we don't take the first value
        df_ut = df.where(np.triu(np.ones(df.shape),1).astype(np.bool))
        #initlize empty
        allBlocks=np.array([])
        for network in netRoi:
            df_new=df_ut.loc[[network]]
            tmp=df_new[network].values
            clean_array = tmp[~np.isnan(tmp)]
            #stack all blocks horizontally
            allBlocks=np.append(allBlocks,clean_array)
        dsNet[dsNet_count]=allBlocks
        dsNet_count=dsNet_count+1
    return dsNet

def randFeats(df, idx):
    """
    Random feature selection based on random indexing

    Parameters
    ----------
    df : str
        path to file
    idx : int
        number to index from
    Returns
    ----------
    featDS : Array of task or rest with random features selected
    """
    data=matFiles(df)
    feat=idx.shape[0]
    nsess=data.shape[0]
    featDS=np.empty((nsess, feat))
    for sess in range(nsess):
        f=data[sess][idx]
        featDS[sess]=f
    return featDS

def loadParcelParams(roiset):
    """ This function loads information about the ROIs and networks.
    For now, this is only set up to work with 333 Gordon 2014 Cerebral Cortex regions
    Inputs:
    roiset = string naming roi type to get parameters for (e.g. 'Gordon333')
    datadir = string path to the location where ROI files are stored
    Returns:
    Parcel_params: a dictionary with ROI information stored in it
    """
    import scipy.io as spio
    datadir=thisDir+'data/Parcel_info/'
    #initialize a dictionary where info will be stored
    Parcel_params = {}

    # put some info into the dict that will work for all roi sets
    Parcel_params['roiset'] = roiset
    dataIn_types = {'dmat','mods_array','roi_sort','net_colors'}
    for dI in dataIn_types:
          dataIn = spio.loadmat(datadir + roiset + '_' + dI + '.mat')
          Parcel_params[dI] = np.array(dataIn[dI])
    Parcel_params['roi_sort'] = Parcel_params['roi_sort'] - 1 #orig indexing in matlab, need to subtract 1

    #transition points and centers for plotting
    transitions,centers = compute_trans_centers(Parcel_params['mods_array'],Parcel_params['roi_sort'])
    Parcel_params['transitions'] = transitions
    Parcel_params['centers'] = centers

    # some ROI specific info that needs to be added by hand
    # add to this if you have a new ROI set that you're using
    if roiset == 'Gordon333':
        Parcel_params['dist_thresh'] = 20 #exclusion distance to not consider in metrics
        Parcel_params['num_rois'] = 333
        Parcel_params['networks'] = ['unassign','default','visual','fp','dan','van','salience',
                                         'co','sm','sm-lat','auditory','pmn','pon']
    else:
        raise ValueError("roiset input is recognized.")

    return Parcel_params
def compute_trans_centers(mods_array,roi_sort):
    """ Function that computes transitions and centers of networks for plotting names
    Inputs:
    mods_array: a numpy vector with the network assignment for each ROI (indexed as a number)
    roi_sort: ROI sorting ordered to show each network in sequence
    Returns:
    transitions: a vector with transition points between networks
    centers: a vector with center points for each network
    """

    mods_sorted = np.squeeze(mods_array[roi_sort])
    transitions = np.nonzero((np.diff(mods_sorted,axis=0)))[0]+1 #transition happens 1 after

    trans_plusends = np.hstack((0,transitions,mods_array.size)) #add ends
    centers = trans_plusends[:-1] + ((trans_plusends[1:] - trans_plusends[:-1])/2)

    return transitions,centers

def figure_corrmat(corrmat,Parcel_params, clims=(-.4,1)):
    """ This function will make a nice looking plot of a correlation matrix for a given parcellation,
    labeling and demarkating networks.
    Inputs:
    corrmat: an roi X roi matrix for plotting
    Parcel_params: a dictionary with ROI information
    clims: (optional) limits to place on corrmat colormap
    Returns:
    fig: a figure handle for figure that was made
    """

    # some variables for ease
    roi_sort = np.squeeze(Parcel_params['roi_sort'])

    # main figure plotting
    fig, ax = plt.subplots()
    im = ax.imshow(corrmat[roi_sort,:][:,roi_sort],cmap='seismic',vmin=clims[0],vmax=clims[1], interpolation='none')
    plt.colorbar(im)

    # add some lines between networks
    for tr in Parcel_params['transitions']:
        ax.axhline(tr,0,Parcel_params['num_rois'],color='k')
        ax.axvline(tr,0,Parcel_params['num_rois'],color='k')

    # alter how the tick marks are shown to plot network names
    ax.set_xticks(Parcel_params['centers'])
    ax.set_yticks(Parcel_params['centers'])
    ax.set_xticklabels(Parcel_params['networks'],fontsize=8)
    ax.set_yticklabels(Parcel_params['networks'],fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation = 45, ha = 'right', rotation_mode = 'anchor')

    plt.show()

    return fig

def getFrames(sub='MSC01', num=5, task='mem'):
    eng=matlab.engine.start_matlab()
    parcel=eng.reframe(sub, num, task)
    fileFC=np.asarray(parcel)
    #Replace nans and infs with zero
    fileFC=np.nan_to_num(fileFC)
    #Consistent parameters to use for editing datasets
    nrois=333
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
    return ds


def permute_importance(df, network):
    fileFC=scipy.io.loadmat(df)
    fileFC=np.array(fileFC['parcel_corrmat'])
    fileFC=np.nan_to_num(fileFC)
    nrois=333
    nsess=fileFC.shape[2]
    ds_full=np.empty((nsess, int(nrois*(nrois-1)/2)))
    dsNet_count=0
    for sess in range(nsess):
        ds=fileFC[:,:,sess]
        Parcel_params = loadParcelParams('Gordon333')
        roi_sort = np.squeeze(Parcel_params['roi_sort'])
        corrmat=ds[roi_sort,:][:,roi_sort]
        nrois=list(range(333))
        nets=[]
        position=0
        count=0
        networks=Parcel_params['networks']
        t=Parcel_params['transitions']
    #have to add extra value otherwise error
        transitions=np.append(t,333)
        while count<333:
            if count<=transitions[position]:
                nets.append(networks[position])
                count=count+1
            else:
                position=position+1
        #transform data to locate network
        df=pd.DataFrame(corrmat, index=[nets, nrois], columns=[nets, nrois])
        #avoid duplicates by taking upper triangle k=1 so we don't take the first value
        df_ut = df.where(np.triu(np.ones(df.shape),1).astype(np.bool))
        #permute block
        df_ut.loc[network,network]=np.random.permutation(df_ut.loc[network,network])
        tmp=df_ut.values
        clean_array=tmp[~np.isnan(tmp)]
        ds_full[dsNet_count]=clean_array
        dsNet_count=dsNet_count+1
    return ds_full



def checkSession(df='path'):
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
    nrois=333
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
    sess=df.shape[0]
    return sess




def permROI(df='path'):
    """
    Formats arrays to be in the same format as the reorganized index
    Parameters
    -----------
    df : str
        Path to file
    Returns
    ------------
    dsNet : Array of task or rest FC with only blocks
    """
 #roi count for building arrays
    fileFC=scipy.io.loadmat(df)
    fileFC=np.array(fileFC['parcel_corrmat'])
    fileFC=np.nan_to_num(fileFC)
    nsess=fileFC.shape[2]
    dsNet=np.empty((nsess, 55278))
    dsNet_count=0
    for sess in range(nsess):
        ds=fileFC[:,:,sess]
        Parcel_params = loadParcelParams('Gordon333')
        roi_sort = np.squeeze(Parcel_params['roi_sort'])
        corrmat=ds[roi_sort,:][:,roi_sort]
        nrois=list(range(333))
        nets=[]
        position=0
        count=0
        networks=Parcel_params['networks']
        t=Parcel_params['transitions']
    #have to add extra value otherwise error
        transitions=np.append(t,333)
        while count<333:
            if count<=transitions[position]:
                nets.append(networks[position])
                count=count+1
            else:
                position=position+1
        #transform data to locate network
        df=pd.DataFrame(corrmat, index=[nets, nrois], columns=[nets, nrois])
        df_ut = df.where(np.triu(np.ones(df.shape),1).astype(np.bool))
        df_new=pd.melt(df_ut,ignore_index=False)
        df_new.dropna(inplace=True)
        df_new.reset_index(inplace=True)
        clean=df_new.value.values
        dsNet[dsNet_count]=clean
        dsNet_count=dsNet_count+1
    mask = (dsNet == 0).all(1)
    column_indices = np.where(mask)[0]
    df = dsNet[~mask,:]
    return df




def getIndices():
    """
    Get mask and use as dictionary
    Parameters
    -----------
    df : str
        Path to file
    Returns
    ------------
    dsNet : Array of task or rest FC with only blocks
    """
 #roi count for building arrays
    df=dataDir +'mem/MSC01_parcel_corrmat.mat' #temp file for getting indices
    fileFC=scipy.io.loadmat(df)
    fileFC=np.array(fileFC['parcel_corrmat'])
    fileFC=np.nan_to_num(fileFC)
    ds=fileFC[:,:,0]
    Parcel_params = loadParcelParams('Gordon333')
    roi_sort = np.squeeze(Parcel_params['roi_sort'])
    corrmat=ds[roi_sort,:][:,roi_sort]
    nrois=list(range(333))
    nets=[]
    position=0
    count=0
    networks=Parcel_params['networks']
    t=Parcel_params['transitions']
#have to add extra value otherwise error
    transitions=np.append(t,333)
    while count<333:
        if count<=transitions[position]:
            nets.append(networks[position])
            count=count+1
        else:
            position=position+1
    #transform data to locate network
    df=pd.DataFrame(corrmat, index=[nets, nrois], columns=[nets, nrois])
    df_ut = df.where(np.triu(np.ones(df.shape),1).astype(np.bool))
    df_new=pd.melt(df_ut,ignore_index=False)
    df_new.dropna(inplace=True)
    df_new.reset_index(inplace=True)
    df_new.drop(columns=['value'],inplace=True)
    return df_new

def permuteIndices(Xtrain_task,Xtrain_rest,network):
    """
    Permute rows of networks and switch tast and rest of that particular network
    Parameters
    -----------
    taskFC: numpy array
        nsess x ROI
    restFC: numpy array
        nsess x ROI
    network: str
        particular network of interest
    Returns
    ------------
    taskFC, restFC : Array of task and rest FC permuting specific rows
    """
    XtrainNew_task=Xtrain_task.copy()
    XtrainNew_rest=Xtrain_rest.copy()
    indices=getIndices()
    index=indices.index
    condition=indices['level_0']==network
    ROI=index[condition]
    ROI_list=ROI.tolist()
    tmpTask=XtrainNew_task[:,ROI_list]
    tmpRest=XtrainNew_rest[:,ROI_list]
    #permute values
    tmpTask_permute=np.random.permutation(tmpTask)
    tmpRest_permute=np.random.permutation(tmpRest)
    #Now switch
    XtrainNew_task[:,ROI_list]=tmpRest_permute #now we purposefully swap the permuted labels to the other task/rest FC
    XtrainNew_rest[:,ROI_list]=tmpTask_permute
    X=np.concatenate((XtrainNew_task, XtrainNew_rest))
    return X

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
    a_memFC=matFiles(dataDir+'mem/'+test_sub[0]+'_parcel_corrmat.mat')
    a_semFC=matFiles(dataDir+'semantic/'+test_sub[0]+'_parcel_corrmat.mat')
    a_glassFC=matFiles(dataDir+'glass/'+test_sub[0]+'_parcel_corrmat.mat')
    a_motFC=matFiles(dataDir+'motor/'+test_sub[0]+'_parcel_corrmat.mat')
    a_restFC=matFiles(dataDir+'rest/corrmats_timesplit/fourths/'+test_sub[0]+'_parcel_corrmat.mat')

    b_memFC=matFiles(dataDir+'mem/'+test_sub[1]+'_parcel_corrmat.mat')
    b_semFC=matFiles(dataDir+'semantic/'+test_sub[1]+'_parcel_corrmat.mat')
    b_glassFC=matFiles(dataDir+'glass/'+test_sub[1]+'_parcel_corrmat.mat')
    b_motFC=matFiles(dataDir+'motor/'+test_sub[1]+'_parcel_corrmat.mat')
    b_restFC=matFiles(dataDir+'rest/corrmats_timesplit/fourths/'+test_sub[1]+'_parcel_corrmat.mat')

    c_memFC=matFiles(dataDir+'mem/'+test_sub[2]+'_parcel_corrmat.mat')
    c_semFC=matFiles(dataDir+'semantic/'+test_sub[2]+'_parcel_corrmat.mat')
    c_glassFC=matFiles(dataDir+'glass/'+test_sub[2]+'_parcel_corrmat.mat')
    c_motFC=matFiles(dataDir+'motor/'+test_sub[2]+'_parcel_corrmat.mat')
    c_restFC=matFiles(dataDir+'rest/corrmats_timesplit/fourths/'+test_sub[2]+'_parcel_corrmat.mat')

    d_memFC=matFiles(dataDir+'mem/'+test_sub[3]+'_parcel_corrmat.mat')
    d_semFC=matFiles(dataDir+'semantic/'+test_sub[3]+'_parcel_corrmat.mat')
    d_glassFC=matFiles(dataDir+'glass/'+test_sub[3]+'_parcel_corrmat.mat')
    d_motFC=matFiles(dataDir+'motor/'+test_sub[3]+'_parcel_corrmat.mat')
    d_restFC=matFiles(dataDir+'rest/corrmats_timesplit/fourths/'+test_sub[3]+'_parcel_corrmat.mat')

    e_memFC=matFiles(dataDir+'mem/'+test_sub[4]+'_parcel_corrmat.mat')
    e_semFC=matFiles(dataDir+'semantic/'+test_sub[4]+'_parcel_corrmat.mat')
    e_glassFC=matFiles(dataDir+'glass/'+test_sub[4]+'_parcel_corrmat.mat')
    e_motFC=matFiles(dataDir+'motor/'+test_sub[4]+'_parcel_corrmat.mat')
    e_restFC=matFiles(dataDir+'rest/corrmats_timesplit/fourths/'+test_sub[4]+'_parcel_corrmat.mat')

    f_memFC=matFiles(dataDir+'mem/'+test_sub[5]+'_parcel_corrmat.mat')
    f_semFC=matFiles(dataDir+'semantic/'+test_sub[5]+'_parcel_corrmat.mat')
    f_glassFC=matFiles(dataDir+'glass/'+test_sub[5]+'_parcel_corrmat.mat')
    f_motFC=matFiles(dataDir+'motor/'+test_sub[5]+'_parcel_corrmat.mat')
    f_restFC=matFiles(dataDir+'rest/corrmats_timesplit/fourths/'+test_sub[5]+'_parcel_corrmat.mat')

    g_memFC=matFiles(dataDir+'mem/'+test_sub[6]+'_parcel_corrmat.mat')
    g_semFC=matFiles(dataDir+'semantic/'+test_sub[6]+'_parcel_corrmat.mat')
    g_glassFC=matFiles(dataDir+'glass/'+test_sub[6]+'_parcel_corrmat.mat')
    g_motFC=matFiles(dataDir+'motor/'+test_sub[6]+'_parcel_corrmat.mat')
    g_restFC=matFiles(dataDir+'rest/corrmats_timesplit/fourths/'+test_sub[6]+'_parcel_corrmat.mat')


    taskFC=np.concatenate((a_memFC,a_semFC,a_glassFC,a_motFC,b_memFC,b_semFC,b_glassFC,b_motFC,c_memFC,c_semFC,c_glassFC,c_motFC,d_memFC,d_semFC,d_glassFC,d_motFC,e_memFC,e_semFC,e_glassFC,e_motFC,f_memFC,f_semFC,f_glassFC,f_motFC,g_memFC,g_semFC,g_glassFC,g_motFC))
    restFC=np.concatenate((a_restFC,b_restFC,c_restFC,d_restFC,e_restFC,f_restFC,g_restFC))
    xtrain = np.concatenate((taskFC, restFC))
    rest_y = np.zeros(restFC.shape[0])
    task_y = np.ones(taskFC.shape[0])
    ytrain=np.concatenate((task_y, rest_y))
    #return xtrain, ytrain
    return taskFC, restFC

def AllSubFiles_groupavg(test_sub, task):
    """
    Return task and rest FC all subs
    Parameters
    -----------
    test_sub: Array of testing subs
    Returns
    ------------
    taskFC, restFC : Array of task and rest FC of all testing subs
    """
    a_memFC=matFiles(dataDir+task+'/'+test_sub[0]+'_parcel_corrmat.mat')
    a_restFC=matFiles(dataDir+'rest/'+test_sub[0]+'_parcel_corrmat.mat')

    b_memFC=matFiles(dataDir+task+'/'+test_sub[1]+'_parcel_corrmat.mat')
    b_restFC=matFiles(dataDir+'rest/'+test_sub[1]+'_parcel_corrmat.mat')

    c_memFC=matFiles(dataDir+task+'/'+test_sub[2]+'_parcel_corrmat.mat')
    c_restFC=matFiles(dataDir+'rest/'+test_sub[2]+'_parcel_corrmat.mat')

    d_memFC=matFiles(dataDir+task+'/'+test_sub[3]+'_parcel_corrmat.mat')
    d_restFC=matFiles(dataDir+'rest/'+test_sub[3]+'_parcel_corrmat.mat')

    e_memFC=matFiles(dataDir+task+'/'+test_sub[4]+'_parcel_corrmat.mat')
    e_restFC=matFiles(dataDir+'rest/'+test_sub[4]+'_parcel_corrmat.mat')

    f_memFC=matFiles(dataDir+task+'/'+test_sub[5]+'_parcel_corrmat.mat')
    f_restFC=matFiles(dataDir+'rest/'+test_sub[5]+'_parcel_corrmat.mat')

    g_memFC=matFiles(dataDir+task+'/'+test_sub[6]+'_parcel_corrmat.mat')
    g_restFC=matFiles(dataDir+'rest/'+test_sub[6]+'_parcel_corrmat.mat')



    taskFC=np.concatenate((a_memFC,b_memFC,c_memFC,d_memFC,e_memFC,f_memFC,g_memFC))
    restFC=np.concatenate((a_restFC,b_restFC,c_restFC,d_restFC,e_restFC,f_restFC,g_restFC))

    return taskFC, restFC


def incremental_AllSubFiles_groupavg(test_sub, task,size):
    """
    Return task and rest FC all subs
    Parameters
    -----------
    test_sub: Array of testing subs
    Returns
    ------------
    taskFC, restFC : Array of task and rest FC of all testing subs
    """
    a_memFC=matFiles(dataDir+task+'/'+test_sub[0]+'_parcel_corrmat.mat')
    a_restFC=matFiles(dataDir+'rest/'+test_sub[0]+'_parcel_corrmat.mat')
    number_of_rows = a_memFC.shape[0]
    random_indices = np.random.choice(number_of_rows, size=size, replace=False)
    a_memFC = a_memFC[random_indices, :]

    number_of_rows = a_restFC.shape[0]
    random_indices = np.random.choice(number_of_rows, size=size, replace=False)
    a_restFC = a_restFC[random_indices, :]

    b_memFC=matFiles(dataDir+task+'/'+test_sub[1]+'_parcel_corrmat.mat')
    b_restFC=matFiles(dataDir+'rest/'+test_sub[1]+'_parcel_corrmat.mat')

    number_of_rows = b_memFC.shape[0]
    random_indices = np.random.choice(number_of_rows, size=size, replace=False)
    b_memFC = b_memFC[random_indices, :]

    number_of_rows = b_restFC.shape[0]
    random_indices = np.random.choice(number_of_rows, size=size, replace=False)
    b_restFC = b_restFC[random_indices, :]

    c_memFC=matFiles(dataDir+task+'/'+test_sub[2]+'_parcel_corrmat.mat')
    c_restFC=matFiles(dataDir+'rest/'+test_sub[2]+'_parcel_corrmat.mat')

    number_of_rows = c_memFC.shape[0]
    random_indices = np.random.choice(number_of_rows, size=size, replace=False)
    c_memFC = c_memFC[random_indices, :]

    number_of_rows = c_restFC.shape[0]
    random_indices = np.random.choice(number_of_rows, size=size, replace=False)
    c_restFC = c_restFC[random_indices, :]

    d_memFC=matFiles(dataDir+task+'/'+test_sub[3]+'_parcel_corrmat.mat')
    d_restFC=matFiles(dataDir+'rest/'+test_sub[3]+'_parcel_corrmat.mat')

    number_of_rows = d_memFC.shape[0]
    random_indices = np.random.choice(number_of_rows, size=size, replace=False)
    d_memFC = d_memFC[random_indices, :]

    number_of_rows = d_restFC.shape[0]
    random_indices = np.random.choice(number_of_rows, size=size, replace=False)
    d_restFC = d_restFC[random_indices, :]

    e_memFC=matFiles(dataDir+task+'/'+test_sub[4]+'_parcel_corrmat.mat')
    e_restFC=matFiles(dataDir+'rest/'+test_sub[4]+'_parcel_corrmat.mat')

    number_of_rows = e_memFC.shape[0]
    random_indices = np.random.choice(number_of_rows, size=size, replace=False)
    e_memFC = e_memFC[random_indices, :]

    number_of_rows = e_restFC.shape[0]
    random_indices = np.random.choice(number_of_rows, size=size, replace=False)
    e_restFC = e_restFC[random_indices, :]

    f_memFC=matFiles(dataDir+task+'/'+test_sub[5]+'_parcel_corrmat.mat')
    f_restFC=matFiles(dataDir+'rest/'+test_sub[5]+'_parcel_corrmat.mat')

    number_of_rows = f_memFC.shape[0]
    random_indices = np.random.choice(number_of_rows, size=size, replace=False)
    f_memFC = f_memFC[random_indices, :]

    number_of_rows = f_restFC.shape[0]
    random_indices = np.random.choice(number_of_rows, size=size, replace=False)
    f_restFC = f_restFC[random_indices, :]

    g_memFC=matFiles(dataDir+task+'/'+test_sub[6]+'_parcel_corrmat.mat')
    g_restFC=matFiles(dataDir+'rest/'+test_sub[6]+'_parcel_corrmat.mat')

    number_of_rows = g_memFC.shape[0]
    random_indices = np.random.choice(number_of_rows, size=size, replace=False)
    g_memFC = g_memFC[random_indices, :]

    number_of_rows = g_restFC.shape[0]
    random_indices = np.random.choice(number_of_rows, size=size, replace=False)
    g_restFC = g_restFC[random_indices, :]

    taskFC=np.concatenate((a_memFC,b_memFC,c_memFC,d_memFC,e_memFC,f_memFC,g_memFC))
    restFC=np.concatenate((a_restFC,b_restFC,c_restFC,d_restFC,e_restFC,f_restFC,g_restFC))

    return taskFC, restFC


def permuteIndicesRandom(Xtrain_task,Xtrain_rest,network):
    """
    Permute rows of networks and switch tast and rest of that particular network
    Parameters
    -----------
    taskFC: numpy array
        nsess x ROI
    restFC: numpy array
        nsess x ROI
    network: str
        particular network of interest
    Returns
    ------------
    taskFC, restFC : Array of task and rest FC permuting specific rows
    """
    XtrainNew_task=Xtrain_task.copy()
    XtrainNew_rest=Xtrain_rest.copy()
    #netRoi=dict([('unassign',14808),('default', 10824),('visual',8736),('fp', 4620),('dan',5264),('van',3151),('salience', 494),('co', 4060),('sm', 2375),('sm-lat', 316),('auditory', 564),('pmn',45),('pon',21)])
    #number=netRoi[network]
    #idx=np.random.randint(55278, size=(number))#will generate random index to sample from
    idx=np.random.randint(55278, size=(network))
    tmpTask=XtrainNew_task[:,idx]
    tmpRest=XtrainNew_rest[:,idx]
    #permute values
    tmpTask_permute=np.random.permutation(tmpTask)
    tmpRest_permute=np.random.permutation(tmpRest)
    #Now switch
    XtrainNew_task[:,idx]=tmpRest_permute #now we purposefully swap the permuted labels to the other task/rest FC
    XtrainNew_rest[:,idx]=tmpTask_permute
    X=np.concatenate((XtrainNew_task, XtrainNew_rest))
    return X




def permuteIndices_byRow(Xtrain_task,Xtrain_rest,rowID):
    """
    Permute rows of networks and switch tast and rest
    Parameters
    -----------
    taskFC: numpy array
        nsess x ROI
    restFC: numpy array
        nsess x ROI
    network: str
        particular network of interest
    Returns
    ------------
    taskFC, restFC : Array of task and rest FC permuting specific rows
    """
    XtrainNew_task=Xtrain_task.copy()
    XtrainNew_rest=Xtrain_rest.copy()
    indices=getIndices()
    index=indices.index
    condition=indices['level_1']==rowID
    ROI=index[condition]
    ROI_list=ROI.tolist()
    tmpTask=XtrainNew_task[:,ROI_list]
    tmpRest=XtrainNew_rest[:,ROI_list]
    tmpTask_permute=np.random.permutation(tmpTask)
    tmpRest_permute=np.random.permutation(tmpRest)
    #Now switch
    XtrainNew_task[:,ROI_list]=tmpRest_permute #now we purposefully swap the permuted labels to the other task/rest FC
    XtrainNew_rest[:,ROI_list]=tmpTask_permute
    X=np.concatenate((XtrainNew_task, XtrainNew_rest))
    return X




def NULLpermuteIndices_byRow(Xtrain_task,Xtrain_rest,rowID):
    """
    Permute rows of networks and switch tast and rest
    Parameters
    -----------
    taskFC: numpy array
        nsess x ROI
    restFC: numpy array
        nsess x ROI
    network: str
        particular network of interest
    Returns
    ------------
    taskFC, restFC : Array of task and rest FC permuting specific rows
    """
    XtrainNew_task=Xtrain_task.copy()
    XtrainNew_rest=Xtrain_rest.copy()
    indices=getIndices()
    index=indices.index
    condition=indices['level_1']!=rowID
    ROI=index[condition]
    ROI_list=ROI.tolist()
    tmpTask=XtrainNew_task[:,ROI_list]
    tmpRest=XtrainNew_rest[:,ROI_list]
    #permute values
    tmpTask_permute=np.random.permutation(tmpTask)
    tmpRest_permute=np.random.permutation(tmpRest)
    #Now switch
    XtrainNew_task[:,ROI_list]=tmpRest_permute #now we purposefully swap the permuted labels to the other task/rest FC
    XtrainNew_rest[:,ROI_list]=tmpTask_permute
    X=np.concatenate((XtrainNew_task, XtrainNew_rest))
    return X
def iNets_SS(train_task, sub, sesList):
    """
    Calculate FC all sessions matrices in list for a subs in a given task
    Parameters
    -----------
    df : str
        Path to file
    Returns
    -----------
    ds : 2D upper triangle FC measures in (roi, days) format

    """
    result_arr = []
    #loop through and append all test sets do each task/rest separate
    for session in sesList:
        all_ses_mats = iNetOpenSes(train_task, sub, session)
        result_arr.append(all_ses_mats)
    result_arr = np.concatenate(result_arr)
    return result_arr

def iNets_OS(train_task, test_subs):
    """
    Calculate FC matrices for all subs in a given task
    Parameters
    -----------
    df : str
        Path to file
    Returns
    -----------
    ds : 2D upper triangle FC measures in (roi, days) format

    """
    result_arr = []
    #loop through and append all test sets do each task/rest separate
    for sub in test_subs:
        all_sub_mats = iNetOpenALL(train_task, sub)
        result_arr.append(all_sub_mats)
    result_arr = np.concatenate(result_arr)
    return result_arr

def iNetOpenALL(train_task, sub):
    """
    Convert matlab files into upper triangle np.arrays for a given sub (all sessions/runs)
    Parameters
    -----------
    df : str
        Path to file
    Returns
    -----------
    ds : 2D upper triangle FC measures in (roi, days) format

    """
    #find all files
    files = glob.glob(dataDir+'iNetworks/'+train_task+'/'+sub+'_ses-*')
    nsess = len(files) #FC matrices
    nrois=333
    ds=np.empty((nsess, int(nrois*(nrois-1)/2)))
    count=0
    for f in files:
    #Consistent parameters to use for editing datasets

        #Load FC file
        fileFC=scipy.io.loadmat(f)

        #Convert to numpy array
        fileFC=np.array(fileFC['parcel_corrmat'])
        #Replace nans and infs with zero
        fileFC=np.nan_to_num(fileFC)
        #Index upper triangle of matrix
        mask=np.triu_indices(nrois,1)

    #Loop through all 10 days to reshape correlations into linear form

        tmp=fileFC[:,:]
        ds[count]=tmp[mask]
        count=count+1
    mask = (ds == 0).all(1)
    column_indices = np.where(mask)[0]
    df = ds[~mask,:]
    return df


def iNetOpenSes(train_task, sub, ses):
    """
    Convert matlab files into upper triangle np.arrays for a given session
    Parameters
    -----------
    df : str
        Path to file
    Returns
    -----------
    ds : 2D upper triangle FC measures in (roi, days) format

    """
    #find all files in a given session
    files = glob.glob(dataDir+'iNetworks/'+train_task+'/'+sub+'_'+ses+'*')
    nsess = len(files) #FC matrices
    nrois=333
    ds=np.empty((nsess, int(nrois*(nrois-1)/2)))
    count=0
    for f in files:
    #Consistent parameters to use for editing datasets

        #Load FC file
        fileFC=scipy.io.loadmat(f)

        #Convert to numpy array
        fileFC=np.array(fileFC['parcel_corrmat'])
        #Replace nans and infs with zero
        fileFC=np.nan_to_num(fileFC)
        #Index upper triangle of matrix
        mask=np.triu_indices(nrois,1)

    #Loop through all 10 days to reshape correlations into linear form

        tmp=fileFC[:,:]
        ds[count]=tmp[mask]
        count=count+1
    mask = (ds == 0).all(1)
    column_indices = np.where(mask)[0] #remove inf or nans
    df = ds[~mask,:]

    return df



def groupIND(task='mixed', train_sub = 'string'):
    """
    Formats arrays to be in the same format as the reorganized index
    Parameters
    -----------
    df : str
        Path to file
    Returns
    ------------
    dsNet : Array of task or rest FC with only blocks
    """
    if task == 'rest':
        fileFC=scipy.io.loadmat(dataDir+task+'/corrmats_timesplit/fourths/'+train_sub+'_parcel_corrmat.mat')
        fileFC=np.array(fileFC['parcel_corrmat'])
        fileFC=np.nan_to_num(fileFC)
        nsess = nsess=fileFC.shape[2]
    else:
        fileFC=scipy.io.loadmat(dataDir+task+'/'+train_sub+'_parcel_corrmat.mat')
        fileFC=np.array(fileFC['parcel_corrmat'])
        fileFC=np.nan_to_num(fileFC)
        nsess=splitDict[train_sub]
    dsNet=np.empty((nsess, 91))
    dsNet_count=0
    mask=np.triu_indices(91,1)
    for sess in range(nsess):
        ds=fileFC[:,:,sess]
        Parcel_params = loadParcelParams('Gordon333')
        roi_sort = np.squeeze(Parcel_params['roi_sort'])
        corrmat=ds[roi_sort,:][:,roi_sort]
        nrois=list(range(333))
        nets=[]
        position=0
        count=0
        networks=Parcel_params['networks']
        t=Parcel_params['transitions']
    #have to add extra value otherwise error
        transitions=np.append(t,333)
        while count<333:
            if count<=transitions[position]:
                nets.append(networks[position])
                count=count+1
            else:
                position=position+1
        #transform data to locate network
        df=pd.DataFrame(corrmat, index=[nets, nrois], columns=[nets, nrois])
        #Replace middle diagonal with Nans
        df = df.where(df.values != np.diag(df),np.nan,df.where(df.values != np.flipud(df).diagonal(0),0,inplace=True))
        layers1 = df.groupby(level=[0], axis = 1).mean()
        final_df = layers1.groupby(level=0).mean()
        df_ut = final_df.where(np.triu(np.ones(final_df.shape)).astype(np.bool))
        df_new=pd.melt(df_ut,ignore_index=False)
        df_new.dropna(inplace=True)
        df_new.reset_index(inplace=True)
        clean=df_new.value.values
        if clean.shape[0] == 0:
            #print(f"{dsNet_count} is empty skipping session")
            continue
        dsNet[dsNet_count]=clean
        dsNet_count=dsNet_count+1
    mask = (dsNet == 0).all(1)
    column_indices = np.where(mask)[0]
    df = dsNet[~mask,:]
    #x = df[np.logical_not(np.isnan(df))]
    #df=np.reshape(x,(-1,91))
    df=np.nan_to_num(df)
    return df

def groupIND_OS(train_task, test_subs):
    """
    Calculate FC matrices for all subs in a given task
    Parameters
    -----------
    df : str
        Path to file
    Returns
    -----------
    ds : 2D upper triangle FC measures in (roi, days) format

    """
    result_arr = []
    #loop through and append all test sets do each task/rest separate
    for sub in test_subs:
        all_sub_mats = groupIND(train_task, sub)
        result_arr.append(all_sub_mats)
    result_arr = np.concatenate(result_arr)
    return result_arr
