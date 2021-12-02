def model_LOSO(train_task):
    """
    Train on all sessions, leave one subject out

    Parameters
    -------------
    train_task : str
            Task name for training
    test_task : str
            Task name for testing

    Returns
    -------------
    within_score : float
            Average accuracy of all folds leave one sub out of a given task
    btn_score : float
            Average accuracy of all folds leave one sub out of a given task

    """

    clf=RidgeClassifier()
    loo = LeaveOneOut()
    #df=pd.DataFrame()
    #nsess x fc x nsub
    ds_T=np.empty((8,55278,8))
    ds_R=np.empty((8,55278,8))
    ds_Test=np.empty((8,55278,8))
    count=0
    #get all subs for a given task
    for sub in subList:
        #training task
        tmp_taskFC=reshape.matFiles(dataDir+train_task+'/'+sub+'_parcel_corrmat.mat')
        tmp_taskFC=tmp_taskFC[:8,:]
        tmp_restFC=reshape.matFiles(dataDir+'rest/'+sub+'_parcel_corrmat.mat')
        tmp_restFC=tmp_restFC[:8,:]
        #reshape 2d into 3d nsessxfcxnsubs
        ds_T[:,:,count]=tmp_taskFC
        ds_R[:,:,count]=tmp_restFC
        count=count+1
    sess_wtn_score=[]
    clf=RidgeClassifier()
    loo = LeaveOneOut()
    wtn_scoreList=[]
    #split up by subs not sess
    sub_splits=np.empty((8,55278))

    #fold each training set (sub)
    for train_index, test_index in loo.split(sub_splits):
        #train on all sessions 1-6 subs
        #test on all sessions of one sub
        Xtrain_task, Xtest_task=ds_T[:,:,train_index], ds_T[:,:,test_index[0]]
        Xtrain_rest, Xtest_rest=ds_R[:,:,train_index], ds_R[:,:,test_index[0]]
        #reshape data into a useable format for mL
        Xtrain_task=Xtrain_task.reshape(60,55278)
        Xtrain_rest=Xtrain_rest.reshape(60,55278)
        #training set
        taskSize=Xtrain_task.shape[0]
        restSize=Xtrain_rest.shape[0]
        t = np.ones(taskSize, dtype = int)
        r=np.zeros(restSize, dtype=int)
        x_train=np.concatenate((Xtrain_task,Xtrain_rest))
        y_train=np.concatenate((t,r))
        #testing set (left out sub CV)
        testSize=Xtest_task.shape[0]
        test_restSize=Xtest_rest.shape[0]
        test_t = np.ones(testSize, dtype = int)
        test_r=np.zeros(test_restSize, dtype=int)
        x_test=np.concatenate((Xtest_task, Xtest_rest))
        y_test=np.concatenate((test_t,test_r))
        #testing set of new task using same subs SS
        SS_taskSize=SS_newTask.shape[0]
        SS_y = np.ones(SS_taskSize, dtype = int)
        #testing set of new task using a diff sub DS
        DS_taskSize=DS_newTask.shape[0]
        DS_y = np.ones(DS_taskSize, dtype = int)
        #test left out sub 10 sessions
        clf.fit(x_train,y_train)
        y_pre=clf.predict(x_test)
        ACCscores=clf.score(x_test,y_test)
        cv_scoreList.append(ACCscores)
    cv_score = mean(cv_scoreList)
    return cv_score
