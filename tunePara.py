def tunePara(stepSize,errorThreshold):
    '''
    tunePara([0.02],[10e-8])
    gives a relatively better solution
    
    0 th acc =  0.677966101695  step size =  0.02
    optimal =  0.677966101695
    whole =  0.884353741497

    '''
    import SGD
    import getFeature
    import loadData
    import numpy as np
    from sklearn import cross_validation
    from sklearn.linear_model import SGDClassifier
    
    
    getFeature.featureExtract('badges.modified.data.all',\
                            'badges_all.arff','badgeAll')
    X,Y,fn = loadData.loadARFF2py('badges_all.arff')
    X_train, X_test, y_train, y_test = cross_validation.train_test_split\
                                    (X, Y, test_size=0.2, random_state=0)
    #X_train, X_test, y_train, y_test = cross_validation.train_test_split\
    #                                (X, Y, test_size=0.4) 
    for i in range(len(stepSize)):                               
        w,theta = SGD.SGD_clf(X_train,y_train,\
                            step_size = stepSize[i],threshold = errorThreshold[i])
        acc = SGD.accScore_SGD(X_test,y_test,w,theta)
        print i,'th acc = ',acc, ' step size = ',stepSize[i]
    
    clf = SGDClassifier(loss="hinge", penalty="l2", random_state=0)
    clf.fit(X_train, y_train)
    acc2 = clf.score(X_test,y_test)
    acc3 = clf.score(X,Y)
    
    print 'optimal = ',acc2
    print 'whole = ',acc3