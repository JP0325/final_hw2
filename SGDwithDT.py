import SGD
#import loadData
import numpy as np
from sklearn import tree

def SGD_DT(X_train,Y_train,X_test,Y_test,stepSize,errorThreshold):
    '''
    step_size = 0.02,threshold = 10e-8
    tuneSGDofDT(0.01,10e-9) is very impressive
    tuneSGDofDT(0.005,10e-9) is even better
    '''
    #X,Y,fn = loadData.loadARFF2py('badges_all.arff')
    
    X_train, y_train = X_train,Y_train
    X_test,y_test = X_test,Y_test
    
    numTree = 100
    SR = 0.5   #sample rate
    
    numExample = len(X_train)
    numTest = len(X_test)
    # create a 2D array newFeature to store the output of each DT
    # newFeature is 294*100 (at most, all 294 data and 100 tree)
    newFeature = np.zeros(shape=(numExample,numTree))
    X_test_transform = np.zeros(shape=(numTest,numTree))
    
    for i in range(numTree):
        # for each tree, i gonna sample by different seed(i),
        # and set the seed can make this script repeatable by anytime
        # each time just pick 147 different examples to train the DT
        np.random.seed(i)    
        randPick = np.random.choice(numExample,size = numExample*SR,replace = False)
        X_trainForTree = X_train[randPick]
        y_trainForTree = y_train[randPick]
        # train small tree with sampled data
        clf = tree.DecisionTreeClassifier(criterion='entropy',random_state = i,max_depth = 4)   
        clf = clf.fit(X_trainForTree, y_trainForTree)
        for n in range(numExample):
            lab_output_train = int(clf.predict([X_train[n]]))
            if lab_output_train == 1: 
                newFeature[n,i] = lab_output_train
        # need to transfer the feature of test data, just let them go through DTs        
        for m in range(numTest):
            lab_output_test = int(clf.predict([X_test[m]]))
            if lab_output_test == 1:
                X_test_transform[m,i] = lab_output_test
    
    
    w, theta = SGD.SGD_clf(newFeature,y_train,step_size = stepSize,threshold = errorThreshold)
    acc = SGD.accScore_SGD(X_test_transform,y_test,w,theta)
    return acc
    #print 'acc1 = ',acc, ' step = ',stepSize
    
    #clf = SGDClassifier(loss="hinge", penalty="l2")
    #clf.fit(newFeature, y_train)
    #acc2 = clf.score(X_test_transform,y_test)
    #acc3 = clf.score(np.vstack([newFeature,X_test_transform]),np.append(y_train,y_test))
    #
    #print 'optimal = ',acc2
    #print 'whole = ',acc3
    #print '---------------------'
